import unittest

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from samplers.constraints import ConstraintConfig, accept_constraints
from samplers.controls import resample
from samplers.geometry import prepare_constraint_geometry
from samplers.ugo import Candidates, RealSupport


def proposal(X, y, seed, partner, interpolation):
    features = torch.as_tensor(X, dtype=torch.float32)
    labels = torch.as_tensor(y, dtype=torch.long)
    support = RealSupport(features, labels, torch.unique(labels), torch.ones(len(y), dtype=torch.bool),
                          {}, {label: torch.nonzero(labels == label, as_tuple=True)[0]
                               for label in np.unique(y)}, [])
    point = (X[seed].astype(float) + interpolation * (X[partner].astype(float) - X[seed])).astype(np.float32)
    candidates = Candidates(torch.as_tensor(point[None]), labels[[seed]], torch.tensor([seed]),
                            torch.tensor([partner]), torch.tensor([-1]))
    return support, candidates


class ConstraintRepairTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-2.], [-1.8], [-1.6], [0.], [1.], [2.]], dtype=np.float32)
        self.y = np.array([0, 0, 0, 1, 1, 1])
        self.geometry = prepare_constraint_geometry(self.X, self.y, parameters={"graph_neighbors": 2,
                                                                                 "graph_radius_quantile": 1.})

    def test_shortening_accepts_first_feasible_step_not_the_initial_or_over_cap_point(self):
        for cap, count, steps in ((2, 0, 2), (3, 1, 3), (4, 1, 3)):
            with self.subTest(cap=cap):
                support, candidates = proposal(self.X, self.y, 3, 5, .75)
                accepted, audit = accept_constraints(support, candidates, [.75], self.geometry,
                                                     ConstraintConfig(eta=.2, rho=.8, max_shortening=cap))
                self.assertEqual(len(accepted.X), count)
                self.assertFalse(audit[0]["initial_metrics"]["displacement_pass"])
                self.assertEqual(audit[0]["shortening_steps"], steps)
                self.assertEqual(audit[0]["final_lambda"], .75 / 2**steps)
                if count:
                    np.testing.assert_array_equal(accepted.X.numpy(), [[.1875]])
                    self.assertLessEqual(audit[0]["final_metrics"]["displacement"], .2 * 1.6)
                    self.assertTrue(audit[0]["final_metrics"]["passes"])

    def test_rejection_keeps_failed_candidate_out_without_repair_or_quota_padding(self):
        support, candidates = proposal(self.X, self.y, 3, 5, .75)
        accepted, audit = accept_constraints(support, candidates, [.75], self.geometry,
                                             ConstraintConfig(eta=.2, rho=.8, repair="reject"))
        self.assertEqual(len(accepted.X), 0)
        self.assertFalse(audit[0]["accepted"])
        self.assertEqual(audit[0]["shortening_steps"], 0)
        self.assertEqual(audit[0]["relocation_index"], -1)
        self.assertGreater(audit[0]["final_metrics"]["displacement"], .2 * 1.6)

    def test_relocation_is_rejected_if_its_real_endpoint_still_breaks_seed_radius(self):
        support, candidates = proposal(self.X, self.y, 3, 5, .75)
        accepted, audit = accept_constraints(support, candidates, [.75], self.geometry,
                                             ConstraintConfig(eta=.2, rho=.8, repair="relocate"))
        self.assertEqual(len(accepted.X), 0)
        self.assertGreaterEqual(audit[0]["relocation_index"], 0)
        self.assertFalse(audit[0]["final_metrics"]["displacement_pass"])
        self.assertTrue(audit[0]["final_metrics"]["ratio_pass"])

    def test_ratio_alone_drives_shortening_and_relocation_can_be_an_exact_real_duplicate(self):
        X = np.array([[0.], [.05], [-2.], [2.]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        geometry = prepare_constraint_geometry(X, y, parameters={"graph_neighbors": 1, "graph_radius_quantile": 1.})
        for repair, cap, count in (("reject", 4, 0), ("shorten", 1, 0), ("shorten", 2, 1), ("relocate", 4, 1)):
            support, candidates = proposal(X, y, 2, 3, .475)
            accepted, audit = accept_constraints(support, candidates, [.475], geometry,
                                                 ConstraintConfig(eta=1., rho=.5, repair=repair, max_shortening=cap))
            self.assertEqual(len(accepted.X), count)
            self.assertTrue(audit[0]["initial_metrics"]["displacement_pass"])
            self.assertFalse(audit[0]["initial_metrics"]["ratio_pass"])
            if count:
                self.assertLessEqual(audit[0]["final_metrics"]["ratio"], .5)
                if repair == "relocate":
                    np.testing.assert_array_equal(accepted.X.numpy(), X[[2]])
                    self.assertEqual(audit[0]["relocation_index"], 2)
                else:
                    np.testing.assert_array_equal(accepted.X.numpy(), np.array([[-1.525]], dtype=np.float32))
                    self.assertEqual(audit[0]["shortening_steps"], 2)

    def test_zero_opposing_distance_never_becomes_a_positive_radius_after_retries(self):
        X = np.array([[0.], [-1.], [0.], [1.]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        geometry = prepare_constraint_geometry(X, y)
        support, candidates = proposal(X, y, 2, 3, .5)
        accepted, audit = accept_constraints(support, candidates, [.5], geometry, ConstraintConfig())
        self.assertEqual(len(accepted.X), 0)
        self.assertEqual(audit[0]["final_metrics"]["seed_opposing_distance"], 0.)
        self.assertGreater(audit[0]["final_metrics"]["displacement"], 0.)
        self.assertEqual(audit[0]["shortening_steps"], 4)

    def test_supported_but_categorically_unique_seeds_underspend_instead_of_cross_category_interpolation(self):
        X = np.column_stack((np.r_[np.linspace(-4, -2, 32), np.linspace(0, 1, 10)],
                             np.r_[np.zeros(32), np.arange(100, 110)])).astype(np.float32)
        y = np.array([0] * 32 + [1] * 10)
        def train(features, labels):
            return LogisticRegression().fit(features.cpu().numpy(), labels.cpu().numpy())
        kwargs = dict(model_fn=train, device="cpu", categorical_columns=(1,), target_counts={0: 32, 1: 15},
                      parameters={"alpha": 1.}, max_iterations=4)
        baseline = resample("ugo_corrected", X, y, 11, **kwargs)
        self.assertEqual(int((baseline.y == 1).sum()), 15)
        result = resample("ugo_constraints_only", X, y, 11, **kwargs)
        self.assertEqual(len(result.y), len(y))
        self.assertEqual(result.stop_reason, "no_valid_partner")
        audit = result.support_summary["constraint_support"]
        self.assertTrue(all(not record["accepted"] and record["reason"] == "no_compatible_distinct_partner"
                            for record in audit["attempts"]))
        self.assertEqual(len(audit["attempts"]), 4)
        self.assertEqual(result.rounds[0]["no_compatible_partner"], 4)
        self.assertEqual(result.rounds[0]["accepted"], 0)
        np.testing.assert_array_equal(result.X.cpu().numpy(), X)

    def test_generated_rows_preserve_categories_connected_ancestry_and_requested_caps(self):
        numeric = np.r_[np.linspace(-4, -2, 32), np.linspace(.5, .6, 6), np.linspace(5, 5.1, 6)]
        X = np.column_stack((numeric, np.eye(2)[np.arange(len(numeric)) % 2])).astype(np.float32)
        y = np.array([0] * 32 + [1] * 12)
        def train(features, labels):
            return LogisticRegression().fit(features.cpu().numpy(), labels.cpu().numpy())
        result = resample("ugo_constraints_only", X, y, 13, categorical_columns=(2, 1),
                          categorical_groups=((1, 2),), model_fn=train, device="cpu", max_iterations=4,
                          target_counts={0: 32, 1: 17}, parameters={"alpha": 1., "graph_neighbors": 2,
                                                                 "graph_radius_quantile": 1.})
        self.assertEqual(int((result.y == 1).sum()), 17)
        audit = result.support_summary["constraint_support"]
        rows = result.X.cpu().numpy()
        for entry in result.provenance:
            seed, partner = entry["seed_index"], entry["partner_index"]
            self.assertLess(seed, len(y))
            self.assertLess(partner, len(y))
            self.assertEqual(y[seed], y[partner])
            self.assertEqual(audit["row_group_ids"][seed], audit["row_group_ids"][partner])
            np.testing.assert_array_equal(rows[entry["row_index"], 1:], X[seed, 1:])
            self.assertTrue(entry["final_metrics"]["passes"])
            self.assertLessEqual(entry["final_metrics"]["displacement"], .5 * entry["final_metrics"]["seed_opposing_distance"])

    def test_native_budget_keeps_phi_cap_and_cannot_reuse_minority_only_matched_geometry(self):
        numeric = np.r_[np.linspace(-4, -2, 32), np.linspace(.5, .6, 6), np.linspace(5, 5.1, 6)]
        X = np.column_stack((numeric, np.eye(2)[np.arange(len(numeric)) % 2])).astype(np.float32)
        y = np.array([0] * 32 + [1] * 12)
        def train(features, labels):
            return LogisticRegression().fit(features.cpu().numpy(), labels.cpu().numpy())
        parameters = {"alpha": 1., "phi": 1.5, "graph_neighbors": 2, "graph_radius_quantile": 1.}
        kwargs = dict(categorical_columns=(1, 2), categorical_groups=((1, 2),), model_fn=train,
                      device="cpu", parameters=parameters)
        native = resample("ugo_constraints_only", X, y, 13, **kwargs)
        native_cap = int(1.5 * max(np.bincount(y)) * len(np.unique(y))) - len(y)
        self.assertEqual(len(native.y) - len(y), native_cap)
        self.assertGreater(len(native.y) - len(y), int(np.bincount(y).max() - np.bincount(y).min()))
        matched = prepare_constraint_geometry(X, y, (1, 2), ((1, 2),), parameters, {0: 32, 1: 17})
        with self.assertRaises(ValueError):
            resample("ugo_constraints_only", X, y, 13, constraint_geometry=matched, **kwargs)



if __name__ == "__main__":
    unittest.main()
