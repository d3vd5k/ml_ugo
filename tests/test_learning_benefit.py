"""Full logistic gradient, shared-pool selection and real Shoppers U2 regressions."""
from copy import deepcopy

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F

from benchmarks.protocols import load_dataset, load_protocol
from benchmarks.run import prepare_partitions
from datasets.scripts.climb import categorical_columns
from samplers.learning_benefit import (CandidatePoolSelector, balanced_ce_loss,
    balanced_weights, logistic_ce_gradients, model_state_hash, select_pool_mask)
from samplers.protected import prepare_protected_groups, protected_ugo
from samplers.ugo import (Candidates, GroupBudget, UGOConfig, accept_relocation,
    build_support, propose_interpolation, score_seeds)

ROOT = Path(__file__).resolve().parents[1]


def train(X, y):
    return LogisticRegression(C=.7, solver="lbfgs", max_iter=1000).fit(np.asarray(X), np.asarray(y))


def fixture():
    X = np.array([[-3., 1.], [-2.8, 0.], [-2., -.5], [-1.5, .2],
                  [1.2, .5], [1.8, -.5], [2.4, .2], [3., 1.]], dtype=np.float32)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    G = X + np.array([.1, .3])
    cfg = UGOConfig(k_pre=2, alpha=1., device="cpu")
    real = build_support(torch.from_numpy(X), torch.from_numpy(y), cfg)
    model = train(X, y)
    plan = GroupBudget(1, torch.tensor([4, 4, 5, 5, 6, 6, 7, 7]), 8)
    record = {"requested": 8, "allocated": 8, "group_trace": [
        {"group_id": 10, "quota": 4, "used_seed": {4: 0, 5: 0}},
        {"group_id": 11, "quota": 4, "used_seed": {6: 0, 7: 0}}]}
    return X, y, G, cfg, real, model, plan, record


class LogisticGradientTests(unittest.TestCase):
    def test_full_candidate_and_balanced_guidance_gradients_match_torch_autograd(self):
        X, y, G, _, _, model, _, _ = fixture()
        expected_w, expected_b = logistic_ce_gradients(model, X, y)
        w = torch.tensor(model.coef_[0], dtype=torch.float64, requires_grad=True)
        b = torch.tensor(model.intercept_[0], dtype=torch.float64, requires_grad=True)
        for i, (row, label) in enumerate(zip(X, y)):
            loss = F.binary_cross_entropy_with_logits(torch.tensor(row, dtype=torch.float64) @ w + b,
                                                     torch.tensor(float(label), dtype=torch.float64))
            gw, gb = torch.autograd.grad(loss, (w, b))
            np.testing.assert_allclose(gw.numpy(), expected_w[i], atol=1e-13)
            self.assertAlmostEqual(gb.item(), expected_b[i], places=13)
        # Imbalanced real guidance: class means have equal total weight.
        indices = np.array([0, 0, 1, 2, 3, 4, 5])
        selector = CandidatePoolSelector("utility", G[indices], y[indices], 9, batch_size=2)
        selector.score_seeds(model, torch.from_numpy(X), torch.from_numpy(y))
        logits = torch.tensor(G[indices], dtype=torch.float64) @ w + b
        losses = F.binary_cross_entropy_with_logits(logits, torch.tensor(y[indices], dtype=torch.float64), reduction="none")
        loss = (losses * torch.tensor(balanced_weights(y[indices]))).sum()
        gw, gb = torch.autograd.grad(loss, (w, b))
        np.testing.assert_allclose(selector.vG_coefficient, gw.numpy(), atol=1e-13)
        self.assertAlmostEqual(selector.vG_bias, gb.item(), places=13)
        self.assertAlmostEqual(selector.checkpoints[0]["guidance_loss"], loss.item(), places=13)
        np.testing.assert_allclose(selector.candidate_scores(X, y),
            expected_w @ gw.numpy() + expected_b * gb.item(), atol=1e-13)

    def test_state_all_attributes_unchanged_and_checkpoint_refreshes(self):
        X, y, G, cfg, real, model, plan, record = fixture()
        model.custom_audit_attribute = {"nested": np.arange(5)}
        before = model_state_hash(model)
        selector = CandidatePoolSelector("utility", G, y, 11, batch_size=3)
        np.testing.assert_array_equal(selector.score_seeds(model, real.X, real.y), score_seeds(model, real.X, real.y))
        candidates = selector.propose(real, plan, np.random.default_rng(3), cfg)
        selector.select(real, accept_relocation(real, candidates, cfg), cfg, record)
        self.assertEqual(before, model_state_hash(model))
        self.assertEqual(selector.rounds[0]["state_before_hash"], selector.rounds[0]["state_after_hash"])
        old_gradient = selector.vG_coefficient.copy()
        changed = deepcopy(model)
        changed.coef_ += .7
        selector.score_seeds(changed, real.X, real.y)
        self.assertEqual(len(selector.checkpoints), 2)
        self.assertFalse(np.array_equal(old_gradient, selector.vG_coefficient))
        np.testing.assert_array_equal(selector.model.coef_, changed.coef_)
        self.assertEqual(before, model_state_hash(model))


    def test_one_step_verification_predicts_true_balanced_loss_reduction(self):
        X, y, G, cfg, real, model, plan, record = fixture()
        selector = CandidatePoolSelector("utility", G, y, 11)
        selector.score_seeds(model, real.X, real.y)
        candidates = selector.propose(real, plan, np.random.default_rng(9), cfg)
        selector.select(real, accept_relocation(real, candidates, cfg), cfg, record)
        result = selector.rounds[0]["verification"][0]
        self.assertEqual(result["eta"], 1e-4)
        self.assertGreaterEqual(result["spearman"], .95)
        self.assertGreaterEqual(result["sign_agreement"], .95)
        self.assertGreaterEqual(result["pearson"], .95)
        self.assertLess(result["max_absolute_error"], 1e-7)
        self.assertEqual(len(result["sample_indices"]), 20)

    def test_failed_linear_step_records_fixed_smaller_eta_without_tuning(self):
        X, y, G, _, real, model, _, _ = fixture()
        # A deliberately huge hypothetical step exposes second-order loss terms.
        selector = CandidatePoolSelector("utility", G, y, 11, eta=100.)
        selector.score_seeds(model, real.X, real.y)
        candidates = np.array([[10., 2.], [-10., -2.], [30., 3.], [-30., -3.]])
        labels = np.array([0, 1, 0, 1])
        results = selector._verify(candidates, labels, selector.candidate_scores(candidates, labels))
        self.assertEqual([row["eta"] for row in results], [100., 1e-5])
        np.testing.assert_array_equal(results[0]["sample_indices"], results[1]["sample_indices"])


class PoolSelectionTests(unittest.TestCase):
    def test_negative_scores_are_selected_and_counted_to_fill_attainable_quotas(self):
        X, y, G, cfg, real, model, plan, record = fixture()
        selector = CandidatePoolSelector("utility", G, y, 7, k=2)
        selector.score_seeds(model, real.X, real.y)
        selector.vG_coefficient = model.coef_[0].copy()
        selector.vG_bias = 0.
        candidates = selector.propose(real, plan, np.random.default_rng(5), cfg)
        selected = selector.select(real, accept_relocation(real, candidates, cfg), cfg, record)
        row = selector.rounds[0]
        self.assertTrue(np.all(row["scores"] < 0))
        self.assertEqual(len(selected.X), 4)
        self.assertEqual(row["negative_selected_count"], 4)
        self.assertEqual(row["positive_selected_count"], 0)

    def test_negative_top_fill_and_exact_group_quota_counts(self):
        anchors = np.array([4, 4, 5, 5, 6, 6, 7, 7, 8])
        groups = np.array([10]*4 + [11]*5)
        scores = np.array([-4., -2., -3., -5., 7., 6., -2., -8., 0.])
        final = np.arange(18).reshape(-1, 2)
        keep = np.array([True]*8 + [False])
        quotas = {10: 2, 11: 9}
        utility = select_pool_mask(final, anchors, groups, scores, keep, quotas, "utility", 9)
        np.testing.assert_array_equal(np.flatnonzero(utility), [1, 2, 4, 6])
        for seed in range(20):
            random = select_pool_mask(final, anchors, groups, scores, keep, quotas, "random", seed)
            self.assertEqual(random.sum(), utility.sum())
            self.assertEqual(len(np.unique(anchors[random])), int(random.sum()))
            self.assertEqual(np.sum(random & (groups == 10)), 2)
            self.assertEqual(np.sum(random & (groups == 11)), 2)

    def test_distinct_planned_anchors_original_rng_geometry_and_shortfalls(self):
        X, y, G, cfg, real, model, plan, record = fixture()
        pools = []
        for name in ("random", "utility"):
            selector = CandidatePoolSelector(name, G, y, 7, k=3)
            selector.score_seeds(model, real.X, real.y)
            rng = np.random.default_rng(12)
            candidates = selector.propose(real, plan, rng, cfg)
            expected_rng = np.random.default_rng(12)
            expected = propose_interpolation(real, plan, expected_rng, cfg, seed_indices=torch.tensor(np.repeat([4,5,6,7], 3)))
            np.testing.assert_array_equal(candidates.X, expected.X)
            np.testing.assert_array_equal(candidates.partner_indices, expected.partner_indices)
            self.assertEqual(rng.random(), expected_rng.random())
            original = candidates.X.numpy().copy()
            selected = selector.select(real, accept_relocation(real, candidates, cfg), cfg, record)
            row = selector.rounds[0]
            np.testing.assert_array_equal(row["original_coordinates"], original)
            self.assertEqual(row["requested_budget"], 8)
            self.assertEqual(row["planned_budget"], 8)
            self.assertEqual(row["accepted"], 4)
            self.assertEqual(row["group_shortfalls"], {10: 2, 11: 2})
            self.assertEqual(row["shortfall"], 4)
            self.assertEqual(len(np.unique(selected.seed_indices)), 4)
            self.assertEqual(len(candidates.X), 12)
            pools.append(row)
        for key in ("original_coordinates", "final_coordinates", "scores", "keep", "random_selected", "utility_selected"):
            np.testing.assert_array_equal(pools[0][key], pools[1][key])
        self.assertEqual(pools[0]["accepted"], pools[1]["accepted"])
        self.assertGreater(pools[0]["diagnostic_only_random_seconds"], 0)
        self.assertEqual(pools[1]["diagnostic_only_random_seconds"], 0)

    def test_final_relocation_scoring_real_matches_first_owner_and_accumulated_dedup(self):
        X, y, G, cfg, real, model, plan, record = fixture()
        selector = CandidatePoolSelector("utility", G, y, 7, k=1)
        selector.score_seeds(model, real.X, real.y)
        candidates = selector.propose(real, plan, np.random.default_rng(12), cfg)
        # Force one genuine baseline relocation; then two cross-anchor duplicates.
        candidates.X[0] = real.X[0]
        selector._pending["original_coordinates"] = candidates.X.numpy().copy()
        accepted = accept_relocation(real, candidates, cfg)
        self.assertGreaterEqual(int(accepted.relocation_indices[0]), 0)
        accepted.X[1] = accepted.X[0]
        accepted.X[2] = accepted.X[0]
        accepted.X[3] = torch.tensor([2.9, .8])
        selected = selector.select(real, accepted, cfg, record)
        row = selector.rounds[0]
        np.testing.assert_allclose(row["scores"], selector.candidate_scores(row["final_coordinates"], row["labels"]))
        self.assertNotAlmostEqual(row["scores"][0], selector.candidate_scores(row["original_coordinates"][:1], row["labels"][:1])[0])
        np.testing.assert_array_equal(row["keep"], [True, False, False, True])
        self.assertTrue(row["real_matches"][0])
        self.assertTrue(row["selected"][0])  # Real equality is NOT an extra rejection rule.
        self.assertEqual(row["pool_duplicate_count"], 2)
        self.assertEqual(len(selected.X), 2)
        # Same final pool in the next round: previously selected synthetic coords excluded.
        selector.score_seeds(model, real.X, real.y)
        next_candidates = selector.propose(real, plan, np.random.default_rng(12), cfg)
        next_candidates.X[:] = torch.from_numpy(row["final_coordinates"])
        selector.select(real, next_candidates, cfg, record)
        second = selector.rounds[-1]
        self.assertEqual(second["accumulated_duplicate_count"], 2)
        self.assertEqual(second["accepted"], 0)
        self.assertTrue(np.all(second["proposal_ids"] > row["proposal_ids"].max()))


class U2HookTests(unittest.TestCase):
    def test_plain_only_guard(self):
        X, y, G, _, _, _, _, _ = fixture()
        selector = CandidatePoolSelector("utility", G, y, 7)
        parameters = {"n_clusters": 2, "sample_fraction": 1., "repeats": 2, "alpha": 1.}
        common = dict(X=X, y=y, seed=7, categorical_columns=(), max_iterations=3, device="cpu",
                      model_fn=train, targets={0:4, 1:8}, parameters=parameters)
        for incompatible in ({"seed_score": score_seeds}, {"h1_support": object()},
                             {"flow_support": object()}, {"reject_neighbors": True}):
            with self.assertRaises(ValueError):
                protected_ugo(**common, candidate_selector=selector, **incompatible)
        for stop_key in ("stop", "policy", "budget_fraction", "budget_fractions"):
            with self.assertRaises(ValueError):
                protected_ugo(**{**common, "parameters":{**parameters, stop_key: object()}},
                              candidate_selector=selector)


    def test_native_welch_stopping_without_generation_or_extra_retrain(self):
        X, y, G, _, _, _, _, _ = fixture()
        calls = []
        def fit(features, labels):
            calls.append(len(labels))
            return train(features, labels)
        selector = CandidatePoolSelector("utility", G, y, 7)
        with patch("samplers.ugo._welch_anova_p_value", return_value=.9):
            result = protected_ugo(X, y, 7, (), 4, "cpu", fit, {0:4, 1:8},
                {"n_clusters":2, "repeats":2, "sample_fraction":1., "alpha":.05}, candidate_selector=selector)
        self.assertEqual(result.stop_reason, "class_difficulty_balance")
        self.assertEqual(calls, [8])
        self.assertEqual(selector.rounds, [])
        self.assertEqual(len(selector.checkpoints), 1)

    def test_actual_real_shoppers_sampler_gradients_caps_ancestry_and_guidance_is_not_seed(self):
        protocol = load_protocol(ROOT / "configs/benchmarks/shoppers_controlled.yaml")
        real_data = load_dataset(protocol, ROOT / ".cache/benchmarks", download=False)
        chosen, _ = train_test_split(np.arange(len(real_data["y"])), train_size=400,
                                    stratify=real_data["y"], random_state=19)
        data = {"name":"shoppers", "X":real_data["X"].iloc[chosen].reset_index(drop=True),
                "y":real_data["y"][chosen], "row_ids":real_data["row_ids"][chosen]}
        train_indices, test = train_test_split(np.arange(400), test_size=.2, stratify=data["y"], random_state=19)
        fit, guidance = train_test_split(train_indices, test_size=.25, stratify=data["y"][train_indices], random_state=19)
        split = {"fold":0, "train_indices":train_indices, "test_indices":test,
                 "fit_indices":fit, "validation_indices":guidance}
        partitions = prepare_partitions(data, split)
        X, y = partitions["X_fit"], data["y"][fit]
        G, gy = partitions["X_validation"], data["y"][guidance]
        params = {"grouping":"kmeans", "allocation":"stability", "n_clusters":3,
                  "repeats":3, "alpha":1., "seed_cap":2, "group_cap_multiplier":1.5}
        targets = {int(c):int(np.sum(y == c)) + (20 if c == 1 else 0) for c in np.unique(y)}
        prepared = prepare_protected_groups(X, y, 19, targets, params)
        prepared["real_row_ids"] = data["row_ids"][fit]
        initial = train(X, y)
        pools, results = [], []
        for name in ("random", "utility"):
            selector = CandidatePoolSelector(name, G, gy, 19, batch_size=16)
            fitted_models, fitting_sizes = [], []
            def fit_model(features, labels):
                fitted_models.append(train(features, labels))
                fitting_sizes.append(len(labels))
                return fitted_models[-1]
            result = protected_ugo(X, y, 19, categorical_columns(partitions["inner"]), 4,
                                   "cpu", fit_model, targets, params, prepared, candidate_selector=selector)
            self.assertTrue(selector.rounds)
            self.assertGreater(len(result.y), len(y))
            np.testing.assert_array_equal(result.X[:len(y)], X)
            self.assertLessEqual(len(result.y) - len(y), 20)
            self.assertEqual(len(fitted_models), len(selector.checkpoints))
            self.assertEqual(fitting_sizes[0], len(y))
            for i, row in enumerate(selector.rounds):
                self.assertEqual(row["state_before_hash"], model_state_hash(fitted_models[i]))
                self.assertEqual(len(np.unique(row["anchor_indices"][row["selected"]])), row["accepted"])
                self.assertLessEqual(row["accepted"], row["planned_budget"])
                self.assertEqual(row["random_selected"].sum(), row["utility_selected"].sum())
                for group, quota in row["group_quotas"].items():
                    viable = len(np.unique(row["anchor_indices"][row["keep"] & (row["group_ids"] == group)]))
                    count = int(np.sum(row["selected"] & (row["group_ids"] == group)))
                    self.assertEqual(count, min(quota, viable))
                # Torch autograd on actual preprocessed real Shoppers model examples.
                w = torch.tensor(row["coefficient"][0], dtype=torch.float64, requires_grad=True)
                b = torch.tensor(row["intercept"][0], dtype=torch.float64, requires_grad=True)
                for pos in range(min(3, len(row["scores"]))):
                    loss = F.binary_cross_entropy_with_logits(
                        torch.tensor(row["final_coordinates"][pos], dtype=torch.float64) @ w + b,
                        torch.tensor(float(row["labels"][pos]), dtype=torch.float64))
                    gw, gb = torch.autograd.grad(loss, (w, b))
                    utility = float(gw @ torch.tensor(row["vG_coefficient"]) + gb * row["vG_bias"])
                    self.assertAlmostEqual(utility, row["scores"][pos], places=11)
            multiplicity = result.support_summary["group_support"]["seed_multiplicity"]
            self.assertLessEqual(max(multiplicity.values()), 2)
            for group in result.support_summary["group_support"]["groups"]:
                self.assertLessEqual(group["actual_generated"], int(1.5 * group["real_support"]))
            for parent in result.provenance:
                self.assertLess(parent["seed_index"], len(y))
                self.assertLess(parent["partner_index"], len(y))
                self.assertNotIn(str(data["row_ids"][fit[parent["seed_index"]]]), set(map(str, data["row_ids"][guidance])))
            # Guidance never enters real support/partner arrays or allocator membership.
            self.assertEqual(result.support_summary["real_count"], len(fit))
            self.assertTrue(set(prepared["real_row_ids"]).isdisjoint(data["row_ids"][guidance]))

            pools.append(selector.rounds[0])
            results.append(result)
        for key in ("original_coordinates", "final_coordinates", "scores", "keep", "proposal_ids"):
            np.testing.assert_array_equal(pools[0][key], pools[1][key])
        self.assertEqual(pools[0]["accepted"], pools[1]["accepted"])
        np.testing.assert_allclose(pools[0]["coefficient"], initial.coef_)
        baseline = protected_ugo(X, y, 19, categorical_columns(partitions["inner"]), 4,
                                  "cpu", train, targets, params, prepared)
        original_record = baseline.support_summary["group_support"]["allocation_records"][0]
        self.assertEqual(pools[0]["requested_budget"], original_record["requested"])
        self.assertEqual(pools[0]["planned_budget"], original_record["allocated"])
        self.assertEqual(pools[0]["group_quotas"],
                         {row["group_id"]:row["quota"] for row in original_record["group_trace"]})
        for result in results:
            np.testing.assert_array_equal(result.support_summary["group_support"]["row_group_ids"],
                baseline.support_summary["group_support"]["row_group_ids"])



if __name__ == "__main__":
    unittest.main()
