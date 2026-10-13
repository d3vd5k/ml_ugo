import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmarks.f1_pool import (TAU_GRID, accounting, flow_decisions, neighbor_decisions,
                                original_pool, save_decisions, save_source)


def source_audit():
    rows = [{"proposal_id": 7, "round": 1, "seed_index": 1, "partner_index": 2,
             "relocation_index": 1, "group_id": 4, "class": 1, "accepted": True,
             "seed_row_id": "minority-a", "partner_row_id": "minority-b",
             "original_coordinates": [.1], "final_coordinates": [2.], "reason": "accepted"},
            {"proposal_id": 8, "round": 2, "seed_index": 2, "partner_index": 1,
             "relocation_index": -1, "group_id": 5, "class": 1, "accepted": False,
             "seed_row_id": "minority-b", "partner_row_id": "minority-a",
             "original_coordinates": [2.5], "final_coordinates": [2.5], "reason": "rejected"}]
    return {"support_summary": {"candidate_log": rows, "group_support": {
        "row_group_ids": [-1, 4, 5], "groups": [
            {"id": 4, "class": 1, "real_support": 1, "stability": .5},
            {"id": 5, "class": 1, "real_support": 1, "stability": .7}],
        "allocation_records": [{"group_trace": [{"group_id": 4, "quota": 1}, {"group_id": 5, "quota": 1}]}]}},
        "augmentation": {"requested_count": 2, "actual_count": 1}, "stop_reason": "quota_exhausted"}


class F1PoolTests(unittest.TestCase):
    def test_original_pool_includes_canonical_rejections_and_never_relocates(self):
        audit = source_audit()
        pool = original_pool(audit, 1)
        np.testing.assert_allclose(pool["X"], [[.1], [2.5]])
        np.testing.assert_array_equal(pool["proposal_ids"], [7, 8])
        np.testing.assert_array_equal(pool["group_ids"], [4, 5])
        self.assertFalse(pool["rows"][1]["accepted"])
        pool["X"][0] = -10
        self.assertEqual(audit["support_summary"]["candidate_log"][0]["original_coordinates"], [.1])

    def test_equal_prior_log_ratio_grid_reuses_scores_and_excludes_invalid(self):
        ratios = np.array([-1000., 0., .5, 1., 1000., 1000., np.nan])
        valid = np.array([True] * 5 + [False, True])
        scores = {"valid": valid, "log_ratio": ratios, "reasons": np.array(["accepted"] * 5 + ["invalid_onehot", "nonfinite_score"], dtype=object)}
        old_reasons = scores["reasons"].copy()
        previous = np.ones(len(ratios), dtype=bool)
        for tau in TAU_GRID:
            decisions = flow_decisions(scores, tau)
            expected = valid & np.isfinite(ratios) & (ratios >= np.log(tau / (1 - tau)))
            np.testing.assert_array_equal(decisions["accepted"], expected)
            self.assertFalse((decisions["accepted"] & ~previous).any())
            self.assertEqual(decisions["reasons"][5], "invalid_onehot")
            self.assertEqual(decisions["support_filter"], "off")
            self.assertIs(decisions["log_ratio"], ratios)
            previous = decisions["accepted"]
        np.testing.assert_array_equal(scores["reasons"], old_reasons)

    def test_neighbor_rejection_uses_original_coordinates_without_mutation(self):
        pool = original_pool(source_audit(), 1)
        before = pool["X"].copy()
        scores, seconds = neighbor_decisions(np.array([[0.], [2.], [3.]], dtype=np.float32),
                                              np.array([0, 1, 1]), pool, {})
        np.testing.assert_array_equal(scores["accepted"], [False, True])
        np.testing.assert_array_equal(pool["X"], before)
        self.assertNotEqual(scores["reasons"][0], "accepted")
        self.assertGreaterEqual(seconds, 0.)

    def test_immutable_source_saves_all_original_parents_groups_and_both_coordinates(self):
        audit = source_audit()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "source"
            canonical, original, reference = save_source(folder, audit,
                np.array([[0.], [2.], [3.]], dtype=np.float32), np.array([0, 1, 1]),
                np.array(["majority", "minority-a", "minority-b"]), {})
            self.assertEqual(len(canonical["X"]), 1)
            self.assertEqual(len(original["X"]), 2)
            with np.load(reference["proposals_path"], allow_pickle=False) as saved:
                np.testing.assert_allclose(saved["original_coordinates"], [[.1], [2.5]])
                np.testing.assert_allclose(saved["canonical_u2_coordinates"], [[2.], [2.5]])
                np.testing.assert_array_equal(saved["real_parent_row_ids"], [["minority-a", "minority-b"], ["minority-b", "minority-a"]])
                np.testing.assert_array_equal(saved["original_source_group_ids"], [4, 5])
                np.testing.assert_array_equal(saved["proposal_ids"], [7, 8])
            for path_key, hash_key in (("pool_path", "pool_sha256"), ("proposals_path", "proposals_sha256")):
                self.assertEqual(hashlib.sha256(Path(reference[path_key]).read_bytes()).hexdigest(), reference[hash_key])

    def test_decisions_keep_flow_original_final_same_and_canonical_relocation_separate(self):
        pool = original_pool(source_audit(), 1)
        scores = {"reasons": np.array(["accepted", "flow_incompatible"]), "valid": np.array([True, True]),
                  "log_likelihood_intended": np.array([-10., -20.]), "log_likelihood_opposing": np.array([-12., -19.]),
                  "log_determinant": np.array([-2., -3.]), "log_ratio": np.array([2., -1.]),
                  "compatibility_score": np.array([.88, .27])}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            save_decisions(folder, pool, np.array([True, False]), {"variant_id": "F1_pool"}, scores, .7)
            rows = json.loads((folder / "candidate_log.json").read_text())["rows"]
        for row in rows:
            np.testing.assert_allclose(row["original_coordinates"], row["final_coordinates"])
            self.assertEqual(row["relocation_index"], -1)
            self.assertFalse(row["relocation_applied"])
            self.assertEqual(row["support_filter"], "off")
        self.assertEqual(rows[0]["canonical_u2_relocation_index"], 1)
        self.assertEqual(rows[0]["canonical_u2_coordinates"], [2.])
        self.assertEqual(rows[0]["real_parents"], ["minority-a", "minority-b"])
        self.assertEqual(rows[0]["latent_log_likelihood_intended"], -8.)
        self.assertEqual(rows[1]["reason"], "flow_incompatible")

    def test_original_group_accounting_accepts_outside_canonical_u2_without_refill(self):
        audit = source_audit()
        pool = original_pool(audit, 1)
        report = accounting(pool, np.array([False, True]), audit,
                            np.array([[0.], [2.], [3.]], dtype=np.float32), np.array([0, 1, 1]))
        self.assertEqual(report["proposal_count"], 2)
        self.assertEqual(report["augmentation"]["actual_count"], 1)
        self.assertEqual(report["unspent_quota"], 1)
        self.assertTrue(report["no_quota_refill"])
        group = report["group_acceptance"][1]
        self.assertEqual(group["u2_accepted_count"], 0)
        self.assertEqual(group["accepted_count"], 1)
        self.assertEqual(group["acceptance_of_proposals"], 1.)
        self.assertNotIn("acceptance_of_u2", group)

    def test_detector_representation_failures_are_not_original_domain_failures(self):
        audit = source_audit()
        scores = {"valid": np.array([False, False]),
                  "reasons": np.array(["unknown_flow_category", "nonfinite_score"])}
        report = accounting(original_pool(audit, 1), np.array([False, False]), audit,
                            np.array([[0.], [2.], [3.]], dtype=np.float32), np.array([0, 1, 1]), scores)
        self.assertEqual(report["feature_invalid_count"], 0)
        self.assertEqual(report["score_invalid_count"], 2)
        self.assertEqual(report["unknown_detector_category_count"], 1)
        self.assertEqual(report["non_finite_score_count"], 1)


if __name__ == "__main__":
    unittest.main()
