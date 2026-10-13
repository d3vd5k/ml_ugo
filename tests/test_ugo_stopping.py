import unittest

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from samplers.controls import resample
from samplers.stopping import stopping_budget, stopping_configuration
from samplers.ugo import UGOConfig, UGOPipeline


def clustered(counts):
    return (np.concatenate([10 * c + np.linspace(0, 1, n) for c, n in enumerate(counts)])[:, None].astype(np.float32),
            np.repeat(np.arange(len(counts)), counts))


class StoppingBudgetTests(unittest.TestCase):
    def test_floor_total_then_largest_remainder_is_label_tied_not_input_tied(self):
        y = np.repeat([0, 1, 2], 8)
        for labels, targets in ((y, {0: 9, 1: 9, 2: 9}), (y[::-1], {2: 9, 1: 9, 0: 9})):
            maximum, selected, checkpoints, caps = stopping_budget(labels, targets, {"budget_fraction": .5})
            self.assertEqual((maximum, selected, checkpoints), (3, 1, (0, 1)))
            self.assertEqual(caps, {0: 9, 1: 8, 2: 8})
        self.assertEqual(stopping_budget(y, {0: 15, 1: 11, 2: 9}, {"budget_fraction": .5, "policy": "fixed_grid"}),
                         (11, 5, (0, 2, 5), {0: 11, 1: 9, 2: 9}))
        self.assertEqual(stopping_budget(y, {0: 15, 1: 11, 2: 9}, {"budget_fraction": .5})[3],
                         {0: 11, 1: 10, 2: 8})
        self.assertEqual(stopping_budget(y, {0: 15, 1: 11, 2: 9}, {})[3], {0: 15, 1: 11, 2: 9})

    def test_native_is_phi_cap_not_a_balance_target(self):
        y = np.repeat([0, 1], [12, 8])
        self.assertEqual(stopping_budget(y, None, {"phi": 1.5, "budget_fraction": .5}),
                         (16, 8, (0, 4, 8), None))
        self.assertEqual(stopping_budget(y, {0: 12, 1: 12}, {})[0], 4)
        self.assertEqual(stopping_budget(y, None, {"phi": .5}), (0, 0, (0,), None))

    def test_invalid_configuration_is_rejected(self):
        invalid = [{"policy": "welch"}, {"budget_fraction": True}, {"budget_fraction": np.bool_(False)},
                   {"budget_fraction": .3}, {"budget_fraction": float("nan")},
                   {"budget_fractions": None}, {"budget_fractions": [0, 1, .5]},
                   {"budget_fractions": [0, .5, .5, 1]}, {"budget_fractions": [0, float("nan"), 1]},
                   {"budget_fractions": [0, float("inf"), 1]}, {"budget_fractions": [False, .5, 1]},
                   {"budget_fractions": [0, .5, True]}, {"budget_fractions": [.25, 1]},
                   {"budget_fractions": [0, .5]}, {"budget_fractions": [0, "half", 1]}]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                stopping_configuration(values)
        for targets in ({0: 7}, {2: 10}, {0: 8.5}):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                stopping_budget(np.repeat([0, 1], 8), targets, {})
        with self.assertRaises(TypeError):
            stopping_budget(np.repeat([0, 1], 8), None, {"unrelated": 1})


class StoppingEngineTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = clustered([12, 8])
        self.fits = []

    def train_equal(self, X, y):
        self.fits.append(len(y))
        return DummyClassifier(strategy="uniform").fit(X.cpu().numpy(), y.cpu().numpy())

    def sample(self, policy="checkpoint", fraction=1., **kwargs):
        parameters = {"policy": policy, "budget_fraction": fraction, "lambda_step": .4, "k_pre": 1}
        parameters.update(kwargs.pop("parameters", {}))
        return resample("ugo_stopping_only", self.X, self.y, 19, model_fn=self.train_equal,
                        device="cpu", target_counts=kwargs.pop("target_counts", {0: 12, 1: 20}),
                        max_iterations=kwargs.pop("max_iterations", 20), parameters=parameters, **kwargs)

    def test_zero_does_not_fit_a_generator_or_emit_any_round_or_provenance(self):
        for policy in ("checkpoint", "fixed_grid"):
            for targets in (None, {0: 12, 1: 20}):
                with self.subTest(policy=policy, targets=targets):
                    result = self.sample(policy, 0., target_counts=targets)
                    self.assertEqual(len(result.y), len(self.y))
                    self.assertEqual(result.rounds, [])
                    self.assertEqual(result.provenance, [])
                    self.assertEqual(result.support_summary["stopping"]["requested_count"], 0)
                    np.testing.assert_array_equal(result.X.numpy(), self.X)
        self.assertEqual(self.fits, [])

    def test_equal_difficulty_stops_corrected_but_not_stopping_only(self):
        corrected = resample("ugo_corrected", self.X, self.y, 19, model_fn=self.train_equal,
                             device="cpu", target_counts={0: 12, 1: 20}, parameters={"k_pre": 1})
        self.assertEqual(corrected.stop_reason, "class_difficulty_balance")
        self.assertEqual(len(corrected.y), 20)
        for policy in ("checkpoint", "fixed_grid"):
            result = self.sample(policy)
            self.assertEqual(len(result.y), 32)
            self.assertTrue(all(row["p_value"] == 1. for row in result.rounds))
            self.assertTrue(result.support_summary["stopping"]["welch_bypassed"])
            self.assertEqual(result.support_summary["stopping"]["per_class_caps"], {0: 12, 1: 20})

    def test_checkpoint_clips_proposals_and_retraining_cadence_not_fixed_grid(self):
        checkpoint = self.sample()
        self.assertEqual([r["proposed"] for r in checkpoint.rounds], [3, 3, 6])
        self.assertEqual([r["total_generated"] for r in checkpoint.rounds], [3, 6, 12])
        self.assertEqual(self.fits, [20, 23, 26])
        self.fits.clear()
        fixed = self.sample("fixed_grid")
        self.assertEqual([r["proposed"] for r in fixed.rounds], [8, 4])
        self.assertEqual(self.fits, [20, 28])
        self.assertEqual(checkpoint.support_summary["stopping"]["checkpoint_counts"], (0, 3, 6, 12))

    def test_checkpoint_actual_multiclass_prefix_replay_uses_cumulative_class_caps(self):
        X, y = clustered([8, 8, 8])
        targets = {0: 16, 1: 20, 2: 12}
        def sample(fraction):
            return resample("ugo_stopping_only", X, y, 91, device="cpu", model_fn=self.train_equal,
                            max_iterations=30, target_counts=targets,
                            parameters={"policy": "checkpoint", "budget_fraction": fraction,
                                        "lambda_step": .4, "k_pre": 1})
        full = sample(1.)
        self.assertEqual(len(full.y), 48)
        for fraction in (0., .25, .5):
            replay = sample(fraction)
            wanted = stopping_budget(y, targets, {"budget_fraction": fraction})[1]
            self.assertEqual(len(replay.y), len(y) + wanted)
            torch.testing.assert_close(replay.X, full.X[:len(y) + wanted], rtol=0, atol=0)
            torch.testing.assert_close(replay.y, full.y[:len(y) + wanted], rtol=0, atol=0)
            self.assertEqual(replay.provenance, full.provenance[:wanted])
            caps = replay.support_summary["stopping"]["per_class_caps"]
            self.assertEqual({c: int((replay.y == c).sum()) for c in caps}, caps)
        for count, caps in full.support_summary["stopping"]["checkpoint_class_caps"].items():
            prefix = full.y[:len(y) + count]
            self.assertEqual({c: int((prefix == c).sum()) for c in caps}, caps)

    def test_alabama_paradox_grid_has_monotone_caps_and_exact_actual_replay(self):
        X, y = clustered([8] * 6)
        deficits = [1500, 1500, 900, 500, 500, 200]
        targets = {c: 8 + deficit for c, deficit in enumerate(deficits)}
        fractions = (0., 25 / 5100, 26 / 5100, 1.)
        values = {"budget_fractions": fractions, "phi": 200., "k_pre": 1, "lambda_step": .4}
        checkpoint_caps, fixed_caps = [], []
        for fraction in fractions:
            for policy, collected in (("checkpoint", checkpoint_caps), ("fixed_grid", fixed_caps)):
                _, selected, _, caps = stopping_budget(y, targets, dict(values, policy=policy, budget_fraction=fraction))
                self.assertEqual(sum(caps[c] - 8 for c in caps), selected)
                collected.append(caps)
        self.assertTrue(any(fixed_caps[1][c] > fixed_caps[2][c] for c in targets))
        for before, after in zip(checkpoint_caps, checkpoint_caps[1:]):
            self.assertTrue(all(before[c] <= after[c] for c in targets))
        self.assertEqual(checkpoint_caps[-1], targets)
        def sample(fraction):
            return resample("ugo_stopping_only", X, y, 13, device="cpu", model_fn=self.train_equal,
                            max_iterations=30, target_counts=targets,
                            parameters=dict(values, budget_fraction=fraction))
        full = sample(fractions[2])
        replay = sample(fractions[1])
        self.assertEqual(len(full.y), 48 + 26)
        self.assertEqual(len(replay.y), 48 + 25)
        torch.testing.assert_close(replay.X, full.X[:73], rtol=0, atol=0)
        self.assertEqual(replay.provenance, full.provenance[:25])

    def test_native_and_matched_both_respect_original_phi_limit(self):
        native = self.sample("fixed_grid", .5, target_counts=None, parameters={"phi": 1.5})
        self.assertEqual(len(native.y) - 20, 8)
        self.assertEqual(native.support_summary["stopping"]["maximum_count"], 16)
        self.assertIsNone(native.support_summary["stopping"]["per_class_caps"])
        # A requested matched budget larger than phi still cannot raise the native cap.
        matched = self.sample(target_counts={0: 12, 1: 100}, parameters={"phi": 1.})
        self.assertEqual(len(matched.y) - 20, 4)
        self.assertEqual(matched.stop_reason, "generation_cap")
        self.assertEqual(matched.support_summary["stopping"]["requested_count"], 92)

    def test_iteration_cap_underspends_without_padding(self):
        result = self.sample(max_iterations=1)
        self.assertEqual(len(result.y) - 20, 3)
        self.assertEqual(len(result.provenance), 3)
        self.assertEqual(result.stop_reason, "iteration_cap")
        self.assertEqual(result.rounds[-1]["unspent_budget"], 9)

    def test_no_real_support_underspends_without_fit_or_padding(self):
        X = np.array([[0.], [0.], [10.], [10.]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        result = resample("ugo_stopping_only", X, y, 1, device="cpu", model_fn=self.train_equal,
                          target_counts={0: 4, 1: 4}, parameters={"k_pre": 1})
        self.assertEqual(result.stop_reason, "no_eligible_group")
        self.assertEqual(len(result.y), 4)
        self.assertEqual(result.provenance, [])
        self.assertEqual(self.fits, [])
        self.assertEqual(result.support_summary["stopping"]["requested_count"], 4)

    def test_real_ancestry_categorical_donors_and_real_only_scoring_with_actual_learner(self):
        X, y = clustered([20, 12])
        X = np.column_stack((X, np.tile([0., 7.], 16))).astype(np.float32)
        fit_sizes = []
        def train(features, labels):
            fit_sizes.append(len(labels))
            return LogisticRegression(random_state=7).fit(features.numpy(), labels.numpy())
        result = resample("ugo_stopping_only", X, y, 7, model_fn=train, device="cpu", max_iterations=10,
                          categorical_columns=(1,), target_counts={0: 20, 1: 20},
                          parameters={"k_pre": 1, "lambda_step": .3})
        self.assertEqual(len(result.y), 40)
        np.testing.assert_array_equal(result.X[:32].numpy(), X)
        self.assertEqual(fit_sizes, [32, 34, 36])
        for row in result.rounds:
            self.assertEqual(row["real_group_support"], {"0": 20, "1": 12})
        for record in result.provenance:
            row, seed, partner = (record[k] for k in ("row_index", "seed_index", "partner_index"))
            self.assertLess(seed, len(y))
            self.assertLess(partner, len(y))
            self.assertEqual(y[seed], y[partner])
            self.assertEqual(int(result.y[row]), y[seed])
            donor = record["relocation_index"] if record["relocation_index"] >= 0 else partner
            self.assertEqual(float(result.X[row, 1]), X[donor, 1])

    def test_existing_corrected_native_and_matched_semantics_are_unchanged(self):
        config = UGOConfig(random_state=19, device="cpu", alpha=1., k_pre=1, lambda_step=.4)
        direct = UGOPipeline(config).run(torch.as_tensor(self.X), torch.as_tensor(self.y), self.train_equal)
        wrapped = resample("ugo_corrected", self.X, self.y, 19, device="cpu", model_fn=self.train_equal,
                           parameters={"alpha": 1., "k_pre": 1, "lambda_step": .4})
        torch.testing.assert_close(wrapped.X, direct.X, rtol=0, atol=0)
        self.assertEqual(wrapped.provenance, direct.provenance)
        corrected = resample("ugo_corrected", self.X, self.y, 19, device="cpu", model_fn=self.train_equal,
                             target_counts={0: 12, 1: 20}, parameters={"alpha": 1., "k_pre": 1, "lambda_step": .4})
        fixed = self.sample("fixed_grid", parameters={"alpha": 1.})
        torch.testing.assert_close(corrected.X, fixed.X, rtol=0, atol=0)
        self.assertEqual(corrected.provenance, fixed.provenance)
        self.assertEqual(corrected.stop_reason, fixed.stop_reason)


if __name__ == "__main__":
    unittest.main()
