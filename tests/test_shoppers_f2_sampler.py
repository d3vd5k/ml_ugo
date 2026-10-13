"""Actual Shoppers F2 sampler integration, ancestry, caps and no-refill tests."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from benchmarks.flow_partitions import flow_for_partition
from benchmarks.protocols import load_dataset, load_protocol
from benchmarks.run import prepare_partitions, train_method
from datasets.scripts.climb import categorical_columns
from samplers.controls import resample
from samplers.flow_support import CalibratedFlowGate, calibrate_flow
from samplers.protected import prepare_protected_groups


ROOT = Path(__file__).resolve().parents[1]


class ShoppersF2SamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol = load_protocol(ROOT / "configs/benchmarks/shoppers_controlled.yaml")
        real = load_dataset(protocol, ROOT / ".cache/benchmarks", download=False)
        # Include rare unknown calibration categories without excluding rows.
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
                        "method_parameters": {"ugo_f1": cls.parameters,
                                              "ugo_f2": {**cls.parameters, "f2_q": .1}}}

    def new_gate(self, q=.1, extreme=False):
        partitions = prepare_partitions(self.data, self.split)
        base = flow_for_partition(self.data, self.split, partitions, "inner", 19, self.parameters, "cpu")
        if extreme:
            with torch.no_grad():
                base.core.model.means[1].fill_(1e6)
                base.core.model.log_stds.zero_()
        indices = np.flatnonzero(np.isin(self.data["row_ids"], base.calibration_row_ids))
        calibration = calibrate_flow(base.core, self.data["X"].iloc[indices],
                                     self.data["y"][indices], self.data["row_ids"][indices])
        return partitions, CalibratedFlowGate(base, calibration, q=q)

    def test_frozen_source_ancestry_original_coordinates_and_caps(self):
        partitions, gate = self.new_gate()
        before = {key: value.clone() for key, value in gate.core.model.state_dict().items()}
        source = hashlib.sha256((ROOT / "samplers/flow_gmm.py").read_bytes()).hexdigest()
        X, y = partitions["X_fit"], self.data["y"][self.split["fit_indices"]]
        targets = {int(c): int((y == c).sum()) + (20 if c == 1 else 0) for c in np.unique(y)}
        groups = prepare_protected_groups(X, y, 19, targets, self.parameters)
        groups["real_row_ids"] = self.data["row_ids"][self.split["fit_indices"]].copy()
        _, audit = train_method("logistic_regression", "ugo_f2", {"C": 1.}, X, y,
            categorical_columns(partitions["inner"]), 19, self.settings, group_partition=groups, flow_support=gate)
        support = audit["support_summary"]
        rows = support["candidate_log"]
        self.assertTrue(rows)
        coordinates = np.asarray([row["original_coordinates"] for row in rows])
        labels = np.asarray([row["class"] for row in rows])
        scores = gate.predict(coordinates, labels, .8)
        f1 = gate.base_gate.predict(coordinates, labels, .8)
        self.assertEqual([row["accepted"] for row in rows], scores["accepted"].tolist())
        self.assertFalse(np.any(scores["accepted"] & ~f1["accepted"]))
        for index, row in enumerate(rows):
            self.assertEqual(row["final_coordinates"], row["original_coordinates"])
            self.assertEqual(row["relocation_index"], -1)
            self.assertLess(row["seed_index"], len(y))
            self.assertLess(row["partner_index"], len(y))
            self.assertEqual(y[row["seed_index"]], row["class"])
            self.assertEqual(y[row["partner_index"]], row["class"])
            self.assertEqual(row["seed_row_id"], str(groups["real_row_ids"][row["seed_index"]]))
            self.assertEqual(row["partner_row_id"], str(groups["real_row_ids"][row["partner_index"]]))
            self.assertEqual(row["group_id"], support["group_support"]["row_group_ids"][row["seed_index"]])
            self.assertEqual(row["compatibility_accepted"], bool(scores["compatibility_accepted"][index]))
            self.assertEqual(row["likelihood_accepted"], bool(scores["likelihood_accepted"][index]))
            threshold = float(scores["likelihood_threshold"][index])
            self.assertEqual(row["likelihood_threshold"], "-inf" if np.isneginf(threshold) else threshold)
            self.assertEqual(row["likelihood_threshold_kind"], "negative_infinity" if np.isneginf(threshold) else "finite")
            self.assertEqual(row["support_quantile"], .1)
            self.assertTrue(row["support_filter"])
            for key in ("log_likelihood_intended", "log_likelihood_opposing", "log_ratio", "log_determinant", "compatibility_score"):
                expected = float(scores[key][index])
                self.assertEqual(row[key], expected if np.isfinite(expected) else None)
        budget = support["proposal_budget"]
        self.assertEqual(budget["cap"], 20)
        self.assertLessEqual(budget["attempted"], 20)
        self.assertEqual(budget["attempted"], len(rows))
        self.assertEqual(budget["accepted"], int(scores["accepted"].sum()))
        self.assertEqual(budget["accepted"] + budget["rejected"], budget["attempted"])
        self.assertEqual(audit["augmentation"]["actual_count"], budget["accepted"])
        for allocation in support["group_support"]["allocation_records"]:
            for group in allocation["group_trace"]:
                self.assertLessEqual(group["used_group"] + group["quota"], 2 * group["n_real"])
        for key, value in gate.core.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertEqual(source, gate.audit["source_binding"]["source_sha256"])
        self.assertEqual(gate.audit["synthetic_training_count"], 0)
        self.assertNotIn("h1_support", support)
        self.assertFalse(set(gate.core.fit_row_ids) & set(gate.calibration_row_ids))
        self.assertFalse(set(gate.core.fit_row_ids) & set(gate.validation_row_ids))
        self.assertFalse(set(gate.core.fit_row_ids) & set(self.data["row_ids"][self.split["test_indices"]]))
        json.dumps(gate.audit, allow_nan=False)

    def test_actual_all_rejected_skips_unchanged_retraining_without_refill(self):
        partitions, gate = self.new_gate(extreme=True)
        X, y = partitions["X_fit"], self.data["y"][self.split["fit_indices"]]
        model, audit = train_method("logistic_regression", "ugo_f2", {"C": 1.}, X, y,
            categorical_columns(partitions["inner"]), 19, self.settings, flow_support=gate)
        self.assertEqual(audit["stop_reason"], "f2_no_candidates_accepted")
        self.assertEqual(audit["augmentation"]["actual_count"], 0)
        self.assertEqual(audit["classifier_fit_count"], 1)
        self.assertTrue(audit["unchanged_final_fit_skipped"])
        budget = audit["support_summary"]["proposal_budget"]
        self.assertEqual(budget["attempted"], 20)
        self.assertEqual(budget["rejected"], 20)
        self.assertEqual(len(audit["rounds"]), 1)
        from classifiers.factory import fit_estimator, make_estimator
        reference = fit_estimator(make_estimator("logistic_regression", 19, parameters={"C": 1.}, device="cpu"), X, y)
        np.testing.assert_array_equal(model.predict_proba(X), reference.predict_proba(X))

    def test_off_is_exact_f1_and_dispatch_rejects_mismatch_and_composition(self):
        partitions, gate = self.new_gate(q=None)
        X, y = partitions["X_fit"], self.data["y"][self.split["fit_indices"]]
        settings = {**self.settings, "method_parameters": {
            "ugo_f1": self.parameters, "ugo_f2": {**self.parameters, "f2_q": None, "support_filter": False}}}
        f1_model, f1 = train_method("logistic_regression", "ugo_f1", {"C": 1.}, X, y,
            categorical_columns(partitions["inner"]), 19, settings, flow_support=gate.base_gate)
        off_model, off = train_method("logistic_regression", "ugo_f2", {"C": 1.}, X, y,
            categorical_columns(partitions["inner"]), 19, settings, flow_support=gate)
        self.assertEqual(f1["support_summary"]["proposal_budget"], off["support_summary"]["proposal_budget"])
        for original, calibrated in zip(f1["support_summary"]["candidate_log"], off["support_summary"]["candidate_log"]):
            for key, value in original.items():
                self.assertEqual(calibrated[key], value)
            self.assertIsNone(calibrated["likelihood_threshold"])
            self.assertFalse(calibrated["support_filter"])
        np.testing.assert_array_equal(f1_model.predict_proba(X), off_model.predict_proba(X))
        common = {"device": "cpu", "model_fn": lambda X, y: None,
                  "categorical_columns": categorical_columns(partitions["inner"])}
        invalid = [("ugo_f2", {"flow_support": None}),
                   ("ugo_f2", {"flow_support": gate.base_gate}),
                   ("ugo_f1", {"flow_support": gate}),
                   ("ugo_groups_only", {"flow_support": gate}),
                   ("ugo_f2", {"flow_support": gate, "h1_support": object()}),
                   ("ugo_f2", {"flow_support": gate, "constraint_geometry": object()}),
                   ("ugo_f2", {"flow_support": gate, "seed_score": lambda *args: None}),
                   ("ugo_f2", {"flow_support": gate, "parameters": {"f2_q": 0.}}),
                   ("ugo_f2", {"flow_support": gate, "parameters": {"support_filter": True}}),
                   ("ugo_f1", {"flow_support": gate.base_gate, "parameters": {"f2_q": None}})]
        for method, values in invalid:
            with self.subTest(method=method, values=values), self.assertRaises(ValueError):
                resample(method, X, y, 19, **common, **values)


if __name__ == "__main__":
    unittest.main()
