import unittest
import json
import tempfile

import numpy as np
import pandas as pd

from benchmarks.run import augmentation_targets, evaluate_cell, smoke_split, split_fingerprint
from experiments import RunArtifacts


class BenchmarkRunnerTests(unittest.TestCase):
    def test_matched_budget_uses_real_deficits_and_deterministic_integer_quotas(self):
        labels = np.array([0] * 10 + [1] * 4 + [2] * 2)
        targets = augmentation_targets(labels, {"mode": "matched", "total": 5})
        self.assertEqual(targets, {0: 10, 1: 6, 2: 5})
        self.assertEqual(sum(targets.values()) - len(labels), 5)
        self.assertEqual(augmentation_targets(labels, {"mode": "matched", "total": 100}),
                         {0: 10, 1: 10, 2: 10})
        self.assertIsNone(augmentation_targets(labels, {"mode": "native"}))

    def test_smoke_never_consumes_official_outer_test(self):
        labels = np.array([0, 0, 0, 1] * 100)
        dataset = {"X": pd.DataFrame({"x": np.arange(400)}), "y": labels,
                   "row_ids": np.array([f"real-{i}" for i in range(400)])}
        original = {"fold": 1, "train_indices": np.arange(300),
                    "test_indices": np.arange(300, 400)}
        smoke = smoke_split(dataset, original, 42, 100, 40, .2)
        for key in ("train_indices", "fit_indices", "validation_indices", "test_indices"):
            self.assertTrue(set(smoke[key]).issubset(set(original["train_indices"])))
        self.assertFalse(set(smoke["train_indices"]) & set(smoke["test_indices"]))
        self.assertFalse(set(smoke["fit_indices"]) & set(smoke["validation_indices"]))
        self.assertEqual(set(smoke["fit_indices"]) | set(smoke["validation_indices"]),
                         set(smoke["train_indices"]))
        self.assertEqual(set(labels[smoke["fit_indices"]]), {0, 1})
        self.assertEqual(set(labels[smoke["validation_indices"]]), {0, 1})
        repeated = smoke_split(dataset, original, 42, 100, 40, .2)
        self.assertEqual(split_fingerprint(dataset, smoke), split_fingerprint(dataset, repeated))
        changed = {**smoke, "test_indices": smoke["test_indices"][::-1]}
        self.assertNotEqual(split_fingerprint(dataset, smoke), split_fingerprint(dataset, changed))

    def test_outer_features_and_labels_cannot_select_parameters_or_cost_threshold(self):
        rng = np.random.default_rng(19)
        labels = np.tile([0, 0, 0, 1], 30).astype(np.int64)
        dataset = {"name": "isolation", "X": pd.DataFrame({"x": labels + rng.normal(size=120),
                    "category": pd.Categorical(np.where(np.arange(120) % 2, "a", "b"))}),
                   "y": labels, "row_ids": np.arange(120)}
        split = {"fold": 0, "train_indices": np.arange(100), "fit_indices": np.arange(80),
                 "validation_indices": np.arange(80, 100), "test_indices": np.arange(100, 120)}
        changed = {**dataset, "X": dataset["X"].copy(), "y": labels.copy()}
        changed["X"].loc[100:, "x"] += 10000
        changed["y"][100:] = 1 - labels[100:]
        protocol = {"primary_metric": "aps_cost", "threshold": {"name": "aps_cost"}, "positive_class": 1}
        settings = {"device": "cpu", "mlp_epochs": 2, "ugo_max_iterations": 1,
                    "augmentation": {"mode": "matched", "total": 10}, "method_parameters": {}}
        with tempfile.TemporaryDirectory() as root:
            selections = []
            for data in (dataset, changed):
                artifacts = RunArtifacts(root, {"random_state": 19})
                row = evaluate_cell(data, split, "logistic_regression", "baseline", "baseline",
                                    [{"C": .1}, {"C": 10}], settings, protocol, artifacts, {"seed": 19})
                folder = artifacts.fold_path("isolation", 0) / "seed-19" / "logistic_regression" / "baseline"
                selections.append(json.loads((folder / "selection.json").read_text()))
                self.assertEqual(row["augmentation"]["actual_count"], 0)
                self.assertEqual(row["augmentation"]["requested_count"], 0)
            self.assertEqual(selections[0], selections[1])


if __name__ == "__main__":
    unittest.main()
