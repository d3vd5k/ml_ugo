"""Original-access guidance and native-classifier/LR-surrogate behavioral regressions."""
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split
import torch

from benchmarks.protocols import load_dataset, load_protocol, make_splits
from benchmarks.run import prepare_partitions
from benchmarks.learning_benefit import parent_training
from samplers.validation_guided import ValidationGuidedLearner
from classifiers.factory import fit_estimator, make_estimator
from datasets.scripts.climb import categorical_columns
from samplers.learning_benefit import balanced_ce_loss, model_state_hash, select_pool_mask
from samplers.validation_guided import ValidationGuidedSelector
from samplers.protected import prepare_protected_groups, protected_ugo
from samplers.ugo import UGOConfig, GroupBudget, build_support, accept_relocation, score_seeds

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = {
    "logistic_regression": {"C": .7},
    "random_forest": {"n_estimators": 12, "max_depth": 4},
    "lightgbm": {"n_estimators": 12, "num_leaves": 7, "min_child_samples": 5},
}


class AnalyticalGuidanceTests(unittest.TestCase):
    def test_full_weight_and_bias_dot_matches_independent_gradient_and_sgd(self):
        X = np.array([[-3., 1.], [-2., -.5], [-1., .2], [1., .5], [2., -.5], [3., 1.]], dtype=np.float32)
        y = np.array([0, 0, 0, 1, 1, 1])
        G = X[[0, 1, 1, 2, 3, 4]] + [.3, .2]
        gy = y[[0, 1, 1, 2, 3, 4]]
        learner = ValidationGuidedLearner("logistic_regression", {"C": .7}, 3, {"C": .7})
        native = learner.fit(X, y)
        selector = ValidationGuidedSelector("utility", G, gy, 3, algorithm="logistic_regression", scorer_for_model=learner.scorer_for_model)
        selector.score_seeds(native, torch.from_numpy(X), torch.from_numpy(y))
        error = expit(G @ native.coef_[0] + native.intercept_[0]) - gy
        weights = np.array([1 / (2 * np.sum(gy == label)) for label in gy])
        vg_w = (error * weights) @ G
        vg_b = np.sum(error * weights)
        candidate_error = expit(X @ native.coef_[0] + native.intercept_[0]) - y
        expected = (candidate_error[:, None] * X) @ vg_w + candidate_error * vg_b
        np.testing.assert_allclose(selector.vG_coefficient, vg_w, atol=1e-13)
        self.assertAlmostEqual(selector.vG_bias, vg_b, places=13)
        np.testing.assert_allclose(selector.candidate_scores(X, y), expected, atol=1e-13)
        self.assertGreater(np.max(np.abs(candidate_error * vg_b)), 1e-6)
        before = balanced_ce_loss(native, G, gy)
        eta = 1e-5
        for i in range(len(y)):
            stepped = deepcopy(native)
            stepped.coef_ -= eta * candidate_error[i] * X[i:i+1]
            stepped.intercept_ -= eta * candidate_error[i]
            self.assertAlmostEqual((before - balanced_ce_loss(stepped, G, gy)) / eta, expected[i], delta=2e-6)

    def test_negative_topfill_quota_and_one_per_anchor_boundaries(self):
        anchors = np.array([4, 4, 5, 5, 6, 6, 7, 7, 8])
        groups = np.array([10] * 4 + [11] * 5)
        scores = np.array([-4., -2., -3., -5., 7., 6., -2., -8., 0.])
        final = np.arange(18).reshape(-1, 2)
        keep = np.array([True] * 8 + [False])
        quotas = {10: 2, 11: 9}
        utility = select_pool_mask(final, anchors, groups, scores, keep, quotas, "utility", 9)
        np.testing.assert_array_equal(np.flatnonzero(utility), [1, 2, 4, 6])
        for seed in range(5):
            random = select_pool_mask(final, anchors, groups, scores, keep, quotas, "random", seed)
            self.assertEqual(random.sum(), utility.sum())
            self.assertEqual(len(np.unique(anchors[random])), random.sum())
            self.assertEqual(np.sum(random & (groups == 10)), 2)
            self.assertEqual(np.sum(random & (groups == 11)), 2)
        self.assertFalse(select_pool_mask(final, anchors, groups, scores, keep, {10: 0, 11: 0}, "utility", 9).any())


    def test_all_negative_candidates_fill_attainable_budget_without_regeneration(self):
        X = np.array([[-3., 1.], [-2.8, 0.], [-2., -.5], [-1.5, .2], [1.2, .5], [1.8, -.5], [2.4, .2], [3., 1.]], dtype=np.float32)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        cfg = UGOConfig(k_pre=2, alpha=1., device="cpu")
        real = build_support(torch.from_numpy(X), torch.from_numpy(y), cfg)
        learner = ValidationGuidedLearner("logistic_regression", {"C": .7}, 3, {"C": .7})
        model = learner.fit(X, y)
        selector = ValidationGuidedSelector("utility", X + [.1, .3], y, 3, algorithm="logistic_regression", scorer_for_model=learner.scorer_for_model)
        selector.score_seeds(model, real.X, real.y)
        selector.vG_coefficient = model.coef_[0].copy()
        selector.vG_bias = 0.
        plan = GroupBudget(1, torch.tensor([4, 4, 5, 5, 6, 6, 7, 7]), 8)
        allocation = {"requested": 8, "allocated": 8, "group_trace": [{"group_id": 10, "quota": 4, "used_seed": {4: 0, 5: 0}}, {"group_id": 11, "quota": 4, "used_seed": {6: 0, 7: 0}}]}
        candidates = selector.propose(real, plan, np.random.default_rng(5), cfg)
        selected = selector.select(real, accept_relocation(real, candidates, cfg), cfg, allocation)
        row = selector.rounds[0]
        self.assertTrue(np.all(row["scores"] < 0))
        self.assertEqual(len(candidates.X), 20)
        self.assertEqual(len(selected.X), 4)
        self.assertEqual(row["negative_selected_count"], 4)
        self.assertEqual(row["shortfall"], 4)
        self.assertEqual(row["random_selected"].sum(), 4)


class ShoppersNativeGuidanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(ROOT / "configs/benchmarks/shoppers_controlled.yaml")
        cls.data = load_dataset(cls.protocol, ROOT / ".cache/benchmarks", download=False)
        cls.canonical = make_splits(cls.data, cls.protocol, 0)[0]
        fit, _ = train_test_split(cls.canonical["fit_indices"], train_size=320, stratify=cls.data["y"][cls.canonical["fit_indices"]], random_state=19)
        guidance, _ = train_test_split(cls.canonical["validation_indices"], train_size=80, stratify=cls.data["y"][cls.canonical["validation_indices"]], random_state=19)
        cls.split = {**cls.canonical, "fit_indices": fit, "validation_indices": guidance}
        cls.partitions = prepare_partitions(cls.data, cls.split)
        cls.X = cls.partitions["X_fit"]
        cls.y = cls.data["y"][fit]
        cls.G = cls.partitions["X_validation"]
        cls.gy = cls.data["y"][guidance]
        cls.categorical = categorical_columns(cls.partitions["inner"])
        cls.params = {"grouping": "kmeans", "allocation": "stability", "n_clusters": 3, "repeats": 3, "alpha": 1., "seed_cap": 2, "group_cap_multiplier": 1.5}
        cls.targets = {int(c): int(np.sum(cls.y == c)) + (20 if c == 1 else 0) for c in np.unique(cls.y)}
        cls.prepared = prepare_protected_groups(cls.X, cls.y, 19, cls.targets, cls.params)

    def test_original_access_canonical_and_smoke_roles(self):
        from benchmarks.validation_guided import validation_guided_split
        split = validation_guided_split(self.data, self.protocol, 0)
        for role in ("fit", "train", "validation", "test"):
            np.testing.assert_array_equal(split[f"{role}_indices"], self.canonical[f"{role}_indices"])
        self.assertEqual(len(split["fit_indices"]), 7891)
        self.assertEqual(len(split["train_indices"]), 9864)
        self.assertEqual(len(split["guidance_indices"]), 1973)
        np.testing.assert_array_equal(split["guidance_indices"], split["validation_indices"])
        for split in (split, validation_guided_split(self.data, self.protocol, 0, smoke=True, train_rows=400, test_rows=120)):
            G = split["guidance_indices"]
            self.assertFalse(np.intersect1d(G, split["fit_indices"]).size)
            self.assertTrue(np.isin(G, split["train_indices"]).all())
            self.assertFalse(np.intersect1d(split["train_indices"], split["canonical_test_indices"]).size)
            self.assertFalse(np.intersect1d(G, split["test_indices"]).size)

    def test_fitting_only_preprocessing_excludes_test_poison_for_both_roles(self):
        from benchmarks.validation_guided import validation_guided_split, prepare_validation_partitions
        from experiments import _preprocessing_state
        split = validation_guided_split(self.data, self.protocol, 0, smoke=True, train_rows=400, test_rows=120)
        partitions = prepare_validation_partitions(self.data, split)
        outer_positions = {int(row): i for i, row in enumerate(split["train_indices"])}
        np.testing.assert_array_equal(partitions["X_outer_guidance"], partitions["X_train"][[outer_positions[int(row)] for row in split["guidance_indices"]]])
        np.testing.assert_array_equal(partitions["X_inner_guidance"], partitions["X_validation"])
        poisoned = {**self.data, "X": self.data["X"].copy(), "y": self.data["y"].copy()}
        excluded = np.union1d(split["test_indices"], split["canonical_test_indices"])
        numeric = poisoned["X"].select_dtypes(include=np.number).columns
        poisoned["X"].loc[poisoned["X"].index[excluded], numeric] = 1e6
        poisoned["y"][excluded] = 1 - poisoned["y"][excluded]
        other = prepare_validation_partitions(poisoned, split)
        for preprocessor, Xkey, Gkey in (("inner", "X_fit", "X_inner_guidance"), ("outer", "X_train", "X_outer_guidance")):
            np.testing.assert_array_equal(other[Xkey], partitions[Xkey])
            np.testing.assert_array_equal(other[Gkey], partitions[Gkey])
            np.testing.assert_equal(_preprocessing_state(other[preprocessor]), _preprocessing_state(partitions[preprocessor]))
            np.testing.assert_array_equal(partitions[Gkey], partitions[preprocessor].transform(self.data["X"].iloc[split["guidance_indices"]]).astype(np.float32))
            for algorithm in ALGORITHMS:
                with self.subTest(role=preprocessor, algorithm=algorithm):
                    indices = split["fit_indices" if preprocessor == "inner" else "train_indices"]
                    original_model = self.learner(algorithm).fit(partitions[Xkey], self.data["y"][indices])
                    poisoned_model = self.learner(algorithm).fit(other[Xkey], poisoned["y"][indices])
                    np.testing.assert_array_equal(original_model.predict_proba(partitions[Gkey]), poisoned_model.predict_proba(partitions[Gkey]))

    def test_guidance_poison_is_excluded_inner_but_deliberately_changes_outer_fit(self):
        from benchmarks.validation_guided import validation_guided_split, prepare_validation_partitions
        from experiments import _preprocessing_state
        split = validation_guided_split(self.data, self.protocol, 0, smoke=True, train_rows=400, test_rows=120)
        partitions = prepare_validation_partitions(self.data, split)
        poisoned = {**self.data, "X": self.data["X"].copy()}
        numeric = poisoned["X"].select_dtypes(include=np.number).columns
        poisoned["X"].loc[poisoned["X"].index[split["guidance_indices"]], numeric] = 1e6
        other = prepare_validation_partitions(poisoned, split)
        np.testing.assert_array_equal(other["X_fit"], partitions["X_fit"])
        np.testing.assert_equal(_preprocessing_state(other["inner"]), _preprocessing_state(partitions["inner"]))
        self.assertFalse(np.array_equal(other["X_train"], partitions["X_train"]))

    def test_guidance_labels_change_utility_not_native_difficulty_or_geometry(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                learner = self.learner(algorithm)
                native = learner.fit(self.X, self.y)
                rows = []
                difficulties = []
                for gy in (self.gy, 1 - self.gy):
                    selector = ValidationGuidedSelector("utility", self.G, gy, 19, algorithm=algorithm, scorer_for_model=learner.scorer_for_model)
                    difficulties.append(selector.score_seeds(native, torch.from_numpy(self.X), torch.from_numpy(self.y)))
                    protected_ugo(self.X, self.y, 19, self.categorical, 1, "cpu", lambda X, y: native, self.targets, self.params, self.prepared, candidate_selector=selector)
                    rows.append(selector.rounds[0])
                np.testing.assert_array_equal(*difficulties)
                for key in ("original_coordinates", "final_coordinates", "planned_anchor_indices", "coefficient", "intercept"):
                    np.testing.assert_array_equal(rows[0][key], rows[1][key])
                self.assertFalse(np.allclose(rows[0]["scores"], rows[1]["scores"]))

    def learner(self, algorithm):
        return ValidationGuidedLearner(algorithm, ALGORITHMS[algorithm], 19, {"C": .7})

    def selector(self, algorithm, learner, branch="utility"):
        return ValidationGuidedSelector(branch, self.G, self.gy, 19, algorithm=algorithm, scorer_for_model=learner.scorer_for_model, batch_size=16)

    def test_native_anchor_difficulty_and_frozen_scoring_for_all_classifiers(self):
        cfg = UGOConfig(k_pre=2, alpha=1., device="cpu")
        real = build_support(torch.from_numpy(self.X), torch.from_numpy(self.y), cfg)
        anchors = np.flatnonzero(self.y == 1)[:4]
        plan = GroupBudget(1, torch.tensor(np.repeat(anchors, 2)), 8)
        allocation = {"requested": 8, "allocated": 8, "group_trace": [{"group_id": 10, "quota": 8, "used_seed": {int(a): 0 for a in anchors}}]}
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                learner = self.learner(algorithm)
                native = learner.fit(self.X, self.y)
                scorer = learner.scorer_for_model(native)
                native_before, scorer_before = model_state_hash(native), model_state_hash(scorer)
                selector = self.selector(algorithm, learner)
                actual = selector.score_seeds(native, real.X, real.y)
                np.testing.assert_array_equal(actual, score_seeds(native, real.X, real.y))
                if algorithm != "logistic_regression":
                    self.assertFalse(np.allclose(actual, score_seeds(scorer, real.X, real.y)))
                candidates = selector.propose(real, plan, np.random.default_rng(5), cfg)
                selector.select(real, accept_relocation(real, candidates, cfg), cfg, allocation)
                self.assertEqual(native_before, model_state_hash(native))
                self.assertEqual(scorer_before, model_state_hash(scorer))
                self.assertEqual(selector.rounds[0]["downstream_state_before_hash"], selector.rounds[0]["downstream_state_after_hash"])
                np.testing.assert_array_equal(native.predict_proba(self.G), fit_estimator(make_estimator(algorithm, 19, ALGORITHMS[algorithm], device="cpu"), self.X, self.y).predict_proba(self.G))

    def test_surrogate_current_prefix_refresh_and_shared_initial_cache(self):
        additions = self.X[np.flatnonzero(self.y == 1)[:12]] + .1
        prefix_X = np.concatenate((self.X, additions))
        prefix_y = np.r_[self.y, np.ones(len(additions), dtype=self.y.dtype)]
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                learner = self.learner(algorithm)
                initial = learner.fit(self.X, self.y, kind="initial")
                first = learner.scorer_for_model(initial)
                later = learner.fit(prefix_X, prefix_y)
                second = learner.scorer_for_model(later)
                expected = fit_estimator(make_estimator("logistic_regression", 19, {"C": .7}, device="cpu"), prefix_X, prefix_y)
                np.testing.assert_allclose(second.coef_, expected.coef_, atol=1e-12)
                np.testing.assert_allclose(second.intercept_, expected.intercept_, atol=1e-12)
                self.assertFalse(np.array_equal(first.coef_, second.coef_))
                self.assertIs(learner.scorer_for_model(initial), first)
                self.assertIs(learner.scorer_for_model(later), second)
                for branch in ("random", "utility"):
                    selector = self.selector(algorithm, learner, branch)
                    selector.score_seeds(initial, torch.from_numpy(self.X), torch.from_numpy(self.y))
                    np.testing.assert_array_equal(selector.model.coef_, first.coef_)
                    selector.score_seeds(later, torch.from_numpy(self.X), torch.from_numpy(self.y))
                    np.testing.assert_array_equal(selector.model.coef_, second.coef_)
                self.assertEqual(learner.surrogate_fits, 0 if algorithm == "logistic_regression" else 2)

    def test_actual_native_trajectories_shared_pool_caps_ancestry_and_parent_mass(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                learner = self.learner(algorithm)
                initial = learner.fit(self.X, self.y, kind="initial")
                pools = []
                for branch in ("random", "utility"):
                    selector = self.selector(algorithm, learner, branch)
                    models, prefixes = [], []
                    def model_fn(X, y):
                        model = initial if not models else learner.fit(X, y)
                        models.append(model)
                        prefixes.append((np.asarray(X).copy(), np.asarray(y).copy()))
                        return model
                    result = protected_ugo(self.X, self.y, 19, self.categorical, 4, "cpu", model_fn, self.targets, self.params, self.prepared, candidate_selector=selector)
                    self.assertGreaterEqual(len(selector.rounds), 2)
                    self.assertGreater(len(prefixes[1][1]), len(self.y))
                    self.assertFalse(np.array_equal(selector.rounds[0]["coefficient"], selector.rounds[1]["coefficient"]))
                    self.assertGreater(len(result.y), len(self.y))
                    self.assertLessEqual(len(result.y) - len(self.y), 20)
                    final = learner.fit(result.X, result.y)
                    direct = fit_estimator(make_estimator(algorithm, 19, ALGORITHMS[algorithm], device="cpu"), result.X, result.y)
                    np.testing.assert_array_equal(final.predict_proba(self.G), direct.predict_proba(self.G))
                    for i, row in enumerate(selector.rounds):
                        scorer = learner.scorer_for_model(models[i])
                        expected = fit_estimator(make_estimator("logistic_regression", 19, {"C": .7}, device="cpu"), *prefixes[i])
                        np.testing.assert_allclose(row["coefficient"], expected.coef_, atol=1e-12)
                        np.testing.assert_array_equal(row["coefficient"], scorer.coef_)
                        self.assertEqual(row["random_selected"].sum(), row["utility_selected"].sum())
                        selected = row["selected"]
                        self.assertEqual(len(np.unique(row["anchor_indices"][selected])), selected.sum())
                        self.assertTrue(np.all(row["anchor_indices"] < len(self.y)))
                        self.assertTrue(np.all(row["partner_indices"] < len(self.y)))
                        for group, quota in row["group_quotas"].items():
                            viable = len(np.unique(row["anchor_indices"][row["keep"] & (row["group_ids"] == group)]))
                            self.assertEqual(np.sum(selected & (row["group_ids"] == group)), min(quota, viable))
                        px, py, weights = parent_training(*prefixes[i], self.X, self.y, row["anchor_indices"][selected], row["partner_indices"][selected])
                        self.assertEqual(len(py), len(prefixes[i][1]) + 2 * selected.sum())
                        self.assertEqual(weights.sum(), len(prefixes[i][1]) + selected.sum())
                        np.testing.assert_array_equal(weights[len(prefixes[i][1]):], .5)
                        parent = learner.fit(px, py, weights, kind="parentdup")
                        parent_direct = fit_estimator(make_estimator(algorithm, 19, ALGORITHMS[algorithm], device="cpu"), px, py, sample_weight=weights)
                        np.testing.assert_array_equal(parent.predict_proba(self.G), parent_direct.predict_proba(self.G))
                        parent_scorer = learner.scorer_for_model(parent)
                        expected_parent_scorer = fit_estimator(make_estimator("logistic_regression", 19, {"C": .7}, device="cpu"), px, py, sample_weight=weights)
                        np.testing.assert_allclose(parent_scorer.coef_, expected_parent_scorer.coef_, atol=1e-12)
                    multiplicity = result.support_summary["group_support"]["seed_multiplicity"]
                    self.assertLessEqual(max(multiplicity.values()), 2)
                    for group in result.support_summary["group_support"]["groups"]:
                        self.assertLessEqual(group["actual_generated"], int(1.5 * group["real_support"]))
                    pools.append(selector.rounds[0])
                for key in ("original_coordinates", "final_coordinates", "scores", "keep", "proposal_ids", "planned_anchor_indices"):
                    np.testing.assert_array_equal(pools[0][key], pools[1][key])
                self.assertEqual(pools[0]["accepted"], pools[1]["accepted"])
                baseline = protected_ugo(self.X, self.y, 19, self.categorical, 1, "cpu", lambda X, y: initial, self.targets, self.params, self.prepared)
                original = baseline.support_summary["group_support"]["allocation_records"][0]
                self.assertEqual(pools[0]["requested_budget"], original["requested"])
                self.assertEqual(pools[0]["planned_budget"], original["allocated"])
                self.assertEqual(pools[0]["group_quotas"], {r["group_id"]: r["quota"] for r in original["group_trace"]})


if __name__ == "__main__":
    unittest.main()
