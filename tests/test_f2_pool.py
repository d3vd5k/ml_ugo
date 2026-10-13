import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from benchmarks.f2_pool import (assert_f1_reproduction, choose_q, indices_for,
                                matched_f1_thinning, save_f2_decisions,
                                source_manifest, SMOKE_SOURCE, FULL_SOURCE)
from benchmarks.f1_pool import original_pool
from tests.test_f1_pool import source_audit


class F2PoolTests(unittest.TestCase):
    def test_thinning_matches_original_groups_and_f1_eligibility_zero_full_partial(self):
        groups = np.array([-37] * 4 + [812] * 4 + [2000003] * 5 + [6] * 2)
        f1 = np.array([1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0], dtype=bool)
        target = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0], dtype=bool)
        old = [array.copy() for array in (groups, f1, target)]
        for seed in range(5):
            mask = matched_f1_thinning(groups, f1, target, seed)
            self.assertFalse((mask & ~f1).any())
            for group in np.unique(groups):
                self.assertEqual(mask[groups == group].sum(), target[groups == group].sum())
            np.testing.assert_array_equal(mask[groups == 812], f1[groups == 812])
            self.assertFalse(mask[groups == -37].any())
        for array, previous in zip((groups, f1, target), old):
            np.testing.assert_array_equal(array, previous)

    def test_thinning_is_stable_across_source_order_and_group_selection(self):
        groups = np.repeat([-5, 413, 99], 20)
        ids = np.arange(60) * 13 + 7
        f1 = np.arange(60) % 3 != 0
        target = f1 & (np.arange(60) % 4 == 0)
        permutation = np.random.default_rng(7).permutation(60)
        for seed in range(5):
            expected = matched_f1_thinning(groups, f1, target, seed, ids)
            reordered = matched_f1_thinning(groups[permutation], f1[permutation], target[permutation], seed, ids[permutation])
            np.testing.assert_array_equal(reordered, expected[permutation])
            for group in np.unique(groups):
                subset = groups == group
                np.testing.assert_array_equal(matched_f1_thinning(groups[subset], f1[subset], target[subset], seed, ids[subset]), expected[subset])
            np.testing.assert_array_equal(expected, matched_f1_thinning(groups, f1, target, seed, ids))
        outcomes = {tuple(matched_f1_thinning(groups, f1, target, seed, ids)) for seed in range(5)}
        self.assertGreater(len(outcomes), 1)

    def test_thinning_rejects_f2_outside_f1_and_bad_alignment(self):
        with self.assertRaises(ValueError):
            matched_f1_thinning([7, 7], [1, 0], [0, 1], 0)
        with self.assertRaises(ValueError):
            matched_f1_thinning([7], [1, 0], [0, 1], 0)
        with self.assertRaises(ValueError):
            matched_f1_thinning([7, 7], [1, 1], [0, 1], 0, [3, 3])
        np.testing.assert_array_equal(matched_f1_thinning([], [], [], 0), [])

    def test_q_selection_is_validation_ap_with_explicit_ties(self):
        def diagnostics(values):
            return [{"q": q, "metrics": {"average_precision": ap, "accuracy": accuracy}}
                    for q, ap, accuracy in values]
        self.assertIsNone(choose_q(diagnostics([(.1, .8, 1), (.01, .8, .1), (None, .8, 0)])))
        self.assertEqual(choose_q(diagnostics([(.1, .8, 1), (.05, .8, .1), (None, .7, 1)])), .05)
        self.assertEqual(choose_q(diagnostics([(None, .8, 1), (.1, .9, 0)])), .1)

    def test_exact_off_probability_and_mask_checks_detect_drift(self):
        class Model:
            classes_ = np.array([0, 1])
            def predict_proba(self, X):
                return np.array([[.75, .25], [.1, .9]])
        pool = {"proposal_ids": np.array([9, 71])}
        saved = pd.DataFrame({"probability_0": [.75, .1], "probability_1": [.25, .9]})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "F1_pool/inner").mkdir(parents=True)
            (root / "F1_pool/inner/candidate_log.json").write_text(json.dumps({"rows": [
                {"proposal_id": 9, "accepted": True}, {"proposal_id": 71, "accepted": False}]}))
            scores = {"accepted": np.array([True, False])}
            assert_f1_reproduction(root, "inner", pool, scores, Model(), np.zeros((2, 1)), saved)
            changed = saved.copy()
            changed.loc[0, "probability_1"] = np.nextafter(.25, 1)
            with self.assertRaises(ValueError):
                assert_f1_reproduction(root, "inner", pool, scores, Model(), np.zeros((2, 1)), changed)
            with self.assertRaises(ValueError):
                assert_f1_reproduction(root, "inner", pool, {"accepted": np.array([False, False])}, Model(), np.zeros((2, 1)), saved)

    def test_decisions_preserve_original_parents_groups_scores_and_no_relocation(self):
        audit = source_audit()
        audit["support_summary"]["candidate_log"].append({
            **audit["support_summary"]["candidate_log"][1], "proposal_id": 91})
        pool = original_pool(audit, 1)
        before = pool["X"].copy()
        scores = {"accepted": np.array([True, False, False]), "valid": np.ones(3, dtype=bool),
                  "reasons": np.array(["accepted", "flow_low_likelihood", "flow_incompatible"], dtype=object),
                  "compatibility_accepted": np.array([True, True, False]),
                  "likelihood_accepted": np.array([True, False, True]),
                  "likelihood_threshold": np.array([-10., -10., -10.]),
                  "log_likelihood_intended": np.array([-3., -11., -4.]),
                  "log_likelihood_opposing": np.array([-5., -13., -2.]),
                  "log_ratio": np.array([2., 2., -2.]), "log_determinant": np.array([1., 1., 1.])}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_f2_decisions(folder, pool, scores, {"variant_id": "F2_pool"}, .5, .05)
            rows = json.loads((folder / "candidate_log.json").read_text())["rows"]
            for row, original in zip(rows, pool["rows"]):
                self.assertEqual(row["proposal_id"], original["proposal_id"])
                self.assertEqual(row["original_source_group_id"], original["group_id"])
                self.assertEqual(row["real_parents"], [original["seed_row_id"], original["partner_row_id"]])
                np.testing.assert_array_equal(np.asarray(row["final_coordinates"], dtype=np.float32),
                                              np.asarray(original["original_coordinates"], dtype=np.float32))
                self.assertFalse(row["relocation_applied"])
                self.assertEqual(row["likelihood_threshold"], -10.)
            with np.load(folder / "decisions.npz") as saved:
                np.testing.assert_array_equal(saved["accepted"], scores["accepted"])
                np.testing.assert_array_equal(saved["proposal_ids"], pool["proposal_ids"])
            scores["likelihood_threshold"] = np.full(3, -np.inf)
            save_f2_decisions(folder, pool, scores, {"variant_id": "F2_pool"}, .5, 0.)
            active = json.loads((folder / "candidate_log.json").read_text())["rows"]
            self.assertTrue(all(row["support_filter"] for row in active))
            self.assertTrue(all(row["likelihood_threshold"] == "-inf" for row in active))
            with np.load(folder / "decisions.npz") as saved:
                self.assertTrue(np.isneginf(saved["likelihood_threshold"]).all())
            scores["likelihood_threshold"] = np.full(3, np.nan)
            save_f2_decisions(folder, pool, scores, {"variant_id": "F2_support_off"}, .5, None)
            off = json.loads((folder / "candidate_log.json").read_text())["rows"]
            self.assertTrue(all(not row["support_filter"] and row["likelihood_threshold"] is None for row in off))
        np.testing.assert_array_equal(before, pool["X"])

    def test_indices_preserve_saved_order_and_reject_unknowns(self):
        dataset = {"row_ids": np.array(["z", "m", "a"])}
        np.testing.assert_array_equal(indices_for(dataset, ["a", "z"]), [2, 0])
        with self.assertRaises(ValueError):
            indices_for(dataset, ["a", "a"])
        with self.assertRaises(ValueError):
            indices_for(dataset, ["unknown"])

    def test_source_smoke_safety_rejects_full_or_test_source(self):
        with self.assertRaises(ValueError):
            source_manifest(Path(FULL_SOURCE), True)
        with self.assertRaises(ValueError):
            source_manifest(Path(SMOKE_SOURCE), False)
        manifest = source_manifest(Path(SMOKE_SOURCE), True)
        self.assertEqual(manifest["protocol_snapshot"]["dataset"], "shoppers")

    def test_source_replay_rejects_wrong_raw_identity_and_effective_settings(self):
        from copy import deepcopy
        from benchmarks.f2_pool import load_source_cell
        from benchmarks.protocols import load_dataset

        source = Path(SMOKE_SOURCE)
        manifest = source_manifest(source, True)
        dataset = load_dataset(manifest["protocol_snapshot"], manifest["arguments"]["data_home"], download=False)
        wrong_identity = deepcopy(manifest)
        wrong_identity["datasets"]["shoppers"]["fingerprint"] = "different_raw_data"
        with self.assertRaises(ValueError):
            load_source_cell(source, dataset, "logistic_regression", 0, 0, wrong_identity)
        wrong_settings = deepcopy(manifest)
        wrong_settings["arguments"]["effective_settings"]["augmentation"]["total"] += 1
        with self.assertRaises(ValueError):
            load_source_cell(source, dataset, "logistic_regression", 0, 0, wrong_settings)



if __name__ == "__main__":
    unittest.main()
