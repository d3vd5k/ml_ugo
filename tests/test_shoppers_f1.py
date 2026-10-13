import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from benchmarks.flow_partitions import flow_for_partition
from benchmarks.protocols import load_dataset, load_protocol
from benchmarks.run import evaluate_cell, prepare_partitions, train_method
from datasets.scripts.climb import categorical_columns
from experiments import RunArtifacts

ROOT = Path(__file__).resolve().parents[1]


def learned_state(value):
    if isinstance(value, dict):
        return {key: learned_state(item) for key, item in value.items() if not key.endswith("seconds")}
    if isinstance(value, list):
        return [learned_state(item) for item in value]
    return value


class ShoppersF1Tests(unittest.TestCase):
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
        cls.parameters = {"grouping": "kmeans", "allocation": "stability", "alpha": 1.,
                          "n_clusters": 3, "repeats": 3, "flow_epochs": 3, "f1_tau": .8}
        cls.settings = {"device": "cpu", "mlp_epochs": 2, "ugo_max_iterations": 4,
                        "augmentation": {"mode": "matched", "total": 20},
                        "method_parameters": {"ugo_f1": cls.parameters}}

    def test_original_decisions_and_no_refill_preserve_ancestry_and_frozen_model(self):
        partitions = prepare_partitions(self.data, self.split)
        gate = flow_for_partition(self.data, self.split, partitions, "inner", 19, self.parameters, "cpu")
        before = {key: value.clone() for key, value in gate.core.model.state_dict().items()}
        X, y = partitions["X_fit"], self.data["y"][self.split["fit_indices"]]
        _, audit = train_method("logistic_regression", "ugo_f1", {"C": 1.}, X, y,
            categorical_columns(partitions["inner"]), 19, self.settings, flow_support=gate)
        support = audit["support_summary"]
        rows = support["candidate_log"]
        scores = gate.predict(np.asarray([row["original_coordinates"] for row in rows]),
                              np.asarray([row["class"] for row in rows]), .8)
        self.assertEqual([row["accepted"] for row in rows], scores["accepted"].tolist())
        for row in rows:
            self.assertEqual(row["final_coordinates"], row["original_coordinates"])
            self.assertEqual(row["relocation_index"], -1)
            self.assertEqual(row["group_id"], support["group_support"]["row_group_ids"][row["seed_index"]])
            self.assertLess(row["seed_index"], len(y))
            self.assertLess(row["partner_index"], len(y))
        budget = support["proposal_budget"]
        self.assertEqual(budget["attempted"], 20)
        self.assertEqual(budget["accepted"], int(scores["accepted"].sum()))
        self.assertEqual(budget["rejected"] + budget["accepted"], budget["attempted"])
        self.assertEqual(audit["augmentation"]["actual_count"], budget["accepted"])
        self.assertEqual(sum(group["actual_generated"] for group in support["group_support"]["groups"]), budget["accepted"])
        self.assertNotIn("h1_support", support)
        for key, value in gate.core.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_all_rejected_round_keeps_real_only_classifier_and_skips_refit(self):
        partitions = prepare_partitions(self.data, self.split)
        gate = flow_for_partition(self.data, self.split, partitions, "inner", 19, self.parameters, "cpu")
        # A valid extreme Gaussian component exercises the zero-acceptance boundary
        # through the actual flow/log-ratio gate, not a mocked acceptance callback.
        with torch.no_grad():
            gate.core.model.means[1].fill_(1e6)
            gate.core.model.log_stds.zero_()
        X, y = partitions["X_fit"], self.data["y"][self.split["fit_indices"]]
        model, audit = train_method("logistic_regression", "ugo_f1", {"C": 1.}, X, y,
            categorical_columns(partitions["inner"]), 19, self.settings, flow_support=gate)
        self.assertEqual(audit["stop_reason"], "f1_no_candidates_accepted")
        self.assertEqual(audit["augmentation"]["actual_count"], 0)
        self.assertEqual(audit["classifier_fit_count"], 1)
        self.assertTrue(audit["unchanged_final_fit_skipped"])
        self.assertEqual(audit["support_summary"]["proposal_budget"]["rejected"], 20)
        from classifiers.factory import fit_estimator, make_estimator
        reference = fit_estimator(make_estimator("logistic_regression", 19, parameters={"C": 1.}, device="cpu"), X, y)
        np.testing.assert_array_equal(model.predict_proba(X), reference.predict_proba(X))

    def test_reserved_calibration_and_validation_features_cannot_change_flow_fit(self):
        original = prepare_partitions(self.data, self.split)
        gate = flow_for_partition(self.data, self.split, original, "inner", 19, self.parameters, "cpu")
        reserved_ids = set(gate.calibration_row_ids.tolist()) | set(gate.validation_row_ids.tolist())
        changed = {**self.data, "X": self.data["X"].copy()}
        reserved = np.array([i for i, identity in enumerate(self.data["row_ids"]) if identity in reserved_ids])
        changed["X"].loc[reserved, "Administrative_Duration"] = 1e12
        other = flow_for_partition(changed, self.split, prepare_partitions(changed, self.split),
                                   "outer", 19, self.parameters, "cpu")
        self.assertEqual(gate.core.audit["model_checksum"], other.core.audit["model_checksum"])
        from experiments import _preprocessing_state
        np.testing.assert_equal(_preprocessing_state(gate.core.preprocessor), _preprocessing_state(other.core.preprocessor))
        fit_ids = set(gate.core.fit_row_ids.tolist())
        self.assertFalse(fit_ids & reserved_ids)
        self.assertFalse(fit_ids & set(self.data["row_ids"][self.split["test_indices"]].tolist()))
        self.assertEqual(fit_ids | set(gate.calibration_row_ids.tolist()),
                         set(self.data["row_ids"][self.split["fit_indices"]].tolist()))

    def test_outer_test_poisoning_cannot_change_selection_or_generation(self):
        changed = {**self.data, "X": self.data["X"].copy(), "y": self.data["y"].copy()}
        test = self.split["test_indices"]
        changed["X"].loc[test, "Administrative_Duration"] = 1e12
        changed["y"][test] = 1 - changed["y"][test]
        metadata = {"seed": 19, "sampler_parameters": self.parameters,
                    "sampler_search": {"group_alpha": [0., .5], "f1_tau": [.5, .8]}}
        snapshots = []
        with tempfile.TemporaryDirectory() as root:
            for index, data in enumerate((self.data, changed)):
                artifacts = RunArtifacts(Path(root) / str(index), {"random_state": 19})
                evaluate_cell(data, self.split, "logistic_regression", "ugo_f1", "F1", [{"C": .1}, {"C": 1.}],
                              self.settings, self.protocol, artifacts, metadata)
                folder = artifacts.fold_path("shoppers", 0) / "seed-19" / "logistic_regression" / "F1"
                snapshots.append({"selection": json.loads((folder / "selection.json").read_text()),
                                  "outer": json.loads((folder / "outer.json").read_text())})
        for snapshot in snapshots:
            snapshot["outer"].pop("group_recall")
        self.assertEqual(learned_state(snapshots[0]), learned_state(snapshots[1]))
