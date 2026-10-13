import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.model_selection import train_test_split

from benchmarks.protocols import load_dataset, load_protocol
from benchmarks.run import evaluate_cell, prepare_partitions, train_method
from datasets.scripts.climb import categorical_columns
from experiments import RunArtifacts
from samplers.hdbscan_support import prepare_h1_support

ROOT = Path(__file__).resolve().parents[1]


def deterministic_state(value):
    if isinstance(value, dict):
        return {key: deterministic_state(item) for key, item in value.items()
                if not key.endswith("seconds") and key not in {"inference_count"}}
    if isinstance(value, list):
        return [deterministic_state(item) for item in value]
    return value


class ShoppersH1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(ROOT / "configs/benchmarks/shoppers_controlled.yaml")
        real = load_dataset(cls.protocol, ROOT / ".cache/benchmarks", download=False)
        chosen, _ = train_test_split(np.arange(len(real["y"])), train_size=400,
                                    stratify=real["y"], random_state=19)
        cls.data = {"name": "shoppers", "X": real["X"].iloc[chosen].reset_index(drop=True),
                    "y": real["y"][chosen].copy(), "row_ids": real["row_ids"][chosen].copy()}
        train, test = train_test_split(np.arange(len(chosen)), test_size=.2,
                                      stratify=cls.data["y"], random_state=19)
        fit, validation = train_test_split(train, test_size=.25, stratify=cls.data["y"][train], random_state=19)
        cls.split = {"fold": 0, "train_indices": train, "fit_indices": fit,
                     "validation_indices": validation, "test_indices": test}
        cls.settings = {"device": "cpu", "mlp_epochs": 2, "ugo_max_iterations": 4,
                        "augmentation": {"mode": "matched", "total": 20}, "method_parameters": {}}

    def test_no_clusters_preserves_real_data_and_skips_unchanged_final_training(self):
        partitions = prepare_partitions(self.data, self.split)
        X, y = partitions["X_fit"], self.data["y"][self.split["fit_indices"]]
        settings = {**self.settings, "method_parameters": {"ugo_h1": {
            "alpha": 1., "n_clusters": 3, "repeats": 3,
            "h1_min_cluster_size": 10000, "h1_min_samples": 1}}}
        model, audit = train_method("logistic_regression", "ugo_h1", {"C": 1.}, X, y,
            categorical_columns(partitions["inner"]), 19, settings)
        self.assertEqual(audit["augmentation"]["actual_count"], 0)
        self.assertEqual(audit["classifier_fit_count"], 1)
        self.assertTrue(audit["unchanged_final_fit_skipped"])
        self.assertEqual(audit["stop_reason"], "h1_no_candidates_accepted")
        budget = audit["support_summary"]["proposal_budget"]
        self.assertEqual(budget["attempted"], 20)
        self.assertEqual(budget["rejected"], 20)
        self.assertEqual(budget["accepted"], 0)
        self.assertEqual(audit["support_summary"]["h1_support"]["usable_clusters"], 0)
        # The reused classifier is exactly the real-only learner, not a refit or fallback.
        from classifiers.factory import fit_estimator, make_estimator
        reference = fit_estimator(make_estimator("logistic_regression", 19, parameters={"C": 1.}, device="cpu"), X, y)
        np.testing.assert_array_equal(model.predict_proba(X), reference.predict_proba(X))

    def test_candidate_decisions_use_final_coordinates_and_keep_original_group(self):
        partitions = prepare_partitions(self.data, self.split)
        X, y = partitions["X_fit"], self.data["y"][self.split["fit_indices"]]
        parameters = {"alpha": 1., "n_clusters": 3, "repeats": 3,
                      "h1_min_cluster_size": 5, "h1_min_samples": 1}
        settings = {**self.settings, "method_parameters": {"ugo_h1": parameters}}
        _, audit = train_method("logistic_regression", "ugo_h1", {"C": 1.}, X, y,
            categorical_columns(partitions["inner"]), 19, settings)
        support = audit["support_summary"]
        detector = prepare_h1_support(X, y, categorical_columns(partitions["inner"]), parameters)
        rows = support["candidate_log"]
        decisions = detector.predict(np.array([row["final_coordinates"] for row in rows]))
        self.assertEqual([row["accepted"] for row in rows], (decisions["valid"] & (decisions["labels"] != -1)).tolist())
        for row in rows:
            self.assertEqual(row["group_id"], support["group_support"]["row_group_ids"][row["seed_index"]])
            self.assertLess(row["seed_index"], len(y))
            self.assertLess(row["partner_index"], len(y))
            if row["relocation_index"] >= 0:
                np.testing.assert_array_equal(row["final_coordinates"], X[row["relocation_index"]])
        self.assertLessEqual(support["proposal_budget"]["attempted"], 20)
        self.assertEqual(support["proposal_budget"]["accepted"], audit["augmentation"]["actual_count"])
        self.assertEqual(sum(group["actual_generated"] for group in support["group_support"]["groups"]),
                         audit["augmentation"]["actual_count"])

    def test_test_poisoning_cannot_change_selection_or_frozen_real_only_support(self):
        changed = {**self.data, "X": self.data["X"].copy(), "y": self.data["y"].copy()}
        test = self.split["test_indices"]
        changed["X"].loc[test, "Administrative_Duration"] = 1e12
        changed["y"][test] = 1 - changed["y"][test]
        metadata = {"seed": 19, "sampler_parameters": {"grouping": "kmeans", "allocation": "stability",
                    "n_clusters": 3, "repeats": 3}, "sampler_search": {"group_alpha": [0., .5],
                    "h1_min_cluster_size": [5, 10000], "h1_min_samples": [1]}}
        snapshots = []
        with tempfile.TemporaryDirectory() as root:
            for index, data in enumerate((self.data, changed)):
                artifacts = RunArtifacts(Path(root) / str(index), {"random_state": 19})
                evaluate_cell(data, self.split, "logistic_regression", "ugo_h1", "H1",
                    [{"C": .1}, {"C": 1.}], self.settings, self.protocol, artifacts, metadata)
                folder = artifacts.fold_path("shoppers", 0) / "seed-19" / "logistic_regression" / "H1"
                snapshots.append({"selection": json.loads((folder / "selection.json").read_text()),
                                  "outer": json.loads((folder / "outer.json").read_text())})
        # Held-out group recall is a report, not learned state, and changes when test labels change.
        for snapshot in snapshots:
            snapshot["outer"].pop("group_recall")
        self.assertEqual(deterministic_state(snapshots[0]), deterministic_state(snapshots[1]))
        fit_ids = set(self.data["row_ids"][self.split["fit_indices"]])
        val_ids = set(self.data["row_ids"][self.split["validation_indices"]])
        test_ids = set(self.data["row_ids"][test])
        for row, real_ids in ((snapshots[0]["selection"]["inner"], fit_ids),
                             (snapshots[0]["outer"], set(self.data["row_ids"][self.split["train_indices"]]))):
            detector = row["support_summary"]["h1_support"]
            fitted = set(detector["fit_real_row_ids"])
            self.assertTrue(fitted <= real_ids)
            self.assertFalse(fitted & test_ids)
            if real_ids == fit_ids:
                self.assertFalse(fitted & val_ids)
            self.assertEqual(detector["synthetic_training_count"], 0)
        canonical = snapshots[0]["selection"]["u2_selection"]["inner"]
        self.assertEqual(snapshots[0]["selection"]["inner"]["parameters"], canonical["parameters"])
        self.assertEqual(snapshots[0]["selection"]["inner"]["sampler_parameters"]["group_alpha"],
                         canonical["sampler_parameters"]["group_alpha"])


if __name__ == "__main__":
    unittest.main()
