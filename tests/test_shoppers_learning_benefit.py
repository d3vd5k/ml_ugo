from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from benchmarks.learning_benefit import (Learner, learning_split, metrics_for,
    parent_training, prepare_learning_partitions, run_trajectory)
from benchmarks.protocols import load_dataset, load_protocol, make_splits
from benchmarks.run import augmentation_targets, train_method
from datasets.scripts.climb import categorical_columns
from experiments import _preprocessing_state
from samplers.learning_benefit import CandidatePoolSelector, logistic_ce_gradients
from samplers.protected import prepare_protected_groups

ROOT = Path(__file__).resolve().parents[1]


class ShoppersLearningBenefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(ROOT / "configs/benchmarks/shoppers_controlled.yaml")
        cls.data = load_dataset(cls.protocol, ROOT / ".cache/benchmarks", download=False)
        cls.split = learning_split(cls.data, cls.protocol, 0, True, 800, 120)
        cls.partitions = prepare_learning_partitions(cls.data, cls.split)
        cls.X = cls.partitions["X_fit"]
        cls.y = cls.data["y"][cls.split["fit_indices"]]
        cls.validation = (cls.partitions["X_validation"], cls.data["y"][cls.split["validation_indices"]])
        cls.guidance_y = cls.data["y"][cls.split["guidance_indices"]]
        cls.categorical = categorical_columns(cls.partitions["preprocessor"])
        cls.parameters = {"grouping": "kmeans", "allocation": "stability", "group_alpha": .5}
        cls.settings = {"device": "cpu", "mlp_epochs": 160, "augmentation": {"mode": "matched", "total": 1000},
                        "ugo_max_iterations": 3, "method_parameters": {"ugo_groups_only": cls.parameters}}
        cls.prepared = prepare_protected_groups(cls.X, cls.y, 0, augmentation_targets(cls.y, cls.settings["augmentation"]), cls.parameters)
        cls.learner = Learner({}, 0, cls.validation)
        cls.initial = cls.learner.fit(cls.X, cls.y, kind="initial")

    def run_branch(self, branch="utility", guidance_y=None, validation=None, rounds=1):
        selector = CandidatePoolSelector(branch, self.partitions["X_guidance"],
            self.guidance_y if guidance_y is None else guidance_y, 0)
        learner = Learner({}, 0, self.validation if validation is None else validation)
        model, result, _, calls = run_trajectory(self.X, self.y, 0, self.categorical,
            self.settings, self.parameters, self.prepared, learner,
            initial_model=self.initial, selector=selector, rounds=rounds)
        return selector, model, result, calls

    def test_smoke_never_consumes_official_outer_test(self):
        canonical = make_splits(self.data, self.protocol, 0)[0]
        sampled = np.concatenate([self.split[f"{role}_indices"]
                                  for role in ("fit", "guidance", "validation", "test")])
        self.assertEqual(len(np.intersect1d(sampled, canonical["test_indices"])), 0)
        self.assertTrue(np.isin(sampled, canonical["train_indices"]).all())
        self.assertEqual(len(np.unique(sampled)), len(sampled))


    def test_canonical_split_and_fit_only_preprocessing_exclude_poisoned_roles(self):
        split = learning_split(self.data, self.protocol, 0)
        canonical = make_splits(self.data, self.protocol, 0)[0]
        np.testing.assert_array_equal(split["test_indices"], canonical["test_indices"])
        np.testing.assert_array_equal(split["validation_indices"], canonical["validation_indices"])
        np.testing.assert_array_equal(np.sort(np.r_[split["fit_indices"], split["guidance_indices"]]),
                                      np.sort(canonical["fit_indices"]))
        for role in ("guidance", "validation", "test"):
            self.assertEqual(len(np.intersect1d(split["fit_indices"], split[f"{role}_indices"])), 0)
        poisoned = {**self.data, "X": self.data["X"].copy(), "y": self.data["y"].copy()}
        excluded = np.r_[self.split["guidance_indices"], self.split["validation_indices"], self.split["test_indices"]]
        numeric = poisoned["X"].select_dtypes(include=np.number).columns
        poisoned["X"].loc[poisoned["X"].index[excluded], numeric] = 1e6
        poisoned["y"][excluded] = 1 - poisoned["y"][excluded]
        other = prepare_learning_partitions(poisoned, self.split)
        np.testing.assert_array_equal(other["X_fit"], self.X)
        np.testing.assert_equal(_preprocessing_state(other["preprocessor"]), _preprocessing_state(self.partitions["preprocessor"]))
        other_model = self.learner.fit(other["X_fit"], poisoned["y"][self.split["fit_indices"]])
        np.testing.assert_array_equal(other_model.coef_, self.initial.coef_)
        groups = prepare_protected_groups(other["X_fit"], self.y, 0,
            augmentation_targets(self.y, self.settings["augmentation"]), self.parameters)
        np.testing.assert_array_equal(groups["row_group_ids"], self.prepared["row_group_ids"])

    def test_same_frozen_pool_matched_counts_and_real_only_ancestry(self):
        random, _, rr, _ = self.run_branch("random")
        utility, _, ur, _ = self.run_branch("utility")
        self.assertGreater(len(random.rounds), 0)
        self.assertEqual(len(rr.y), len(ur.y))
        r, u = random.rounds[0], utility.rounds[0]
        for key in ("original_coordinates", "final_coordinates", "scores", "coefficient", "intercept", "planned_anchor_indices"):
            np.testing.assert_array_equal(r[key], u[key])
        np.testing.assert_array_equal(r["coefficient"], self.initial.coef_)
        for record in (r, u):
            self.assertEqual(record["state_before_hash"], record["state_after_hash"])
            selected = record["selected"]
            self.assertEqual(len(np.unique(record["anchor_indices"][selected])), int(selected.sum()))
            self.assertTrue(np.all(record["anchor_indices"] < len(self.y)))
            self.assertTrue(np.all(record["partner_indices"] < len(self.y)))
            self.assertTrue(np.all(record["relocation_indices"] < len(self.y)))
            for group, quota in record["group_quotas"].items():
                self.assertLessEqual(np.sum(selected & (record["group_ids"] == group)), quota)
        np.testing.assert_array_equal(self.initial.coef_, r["coefficient"])

    def test_validation_test_poisoning_does_not_change_locked_generation(self):
        original, model, result, _ = self.run_branch()
        poisoned_validation = (np.full_like(self.validation[0], 1e6), 1 - self.validation[1])
        poisoned, other_model, other_result, _ = self.run_branch(validation=poisoned_validation)
        np.testing.assert_array_equal(result.X, other_result.X)
        np.testing.assert_array_equal(model.coef_, other_model.coef_)
        for key in ("scores", "selected", "final_coordinates", "vG_coefficient"):
            np.testing.assert_array_equal(original.rounds[0][key], poisoned.rounds[0][key])
        # Test queries are not even accepted by run_trajectory or its model_fn.
        extreme_test = np.full_like(self.partitions["X_test"], 1e6)
        metrics_for(model, extreme_test, 1 - self.data["y"][self.split["test_indices"]], self.protocol, .5)
        np.testing.assert_array_equal(model.coef_, other_model.coef_)

    def test_guidance_labels_legitimately_change_scores_not_pool_or_fitted_checkpoint(self):
        original, _, _, _ = self.run_branch()
        poisoned, _, _, _ = self.run_branch(guidance_y=1 - self.guidance_y)
        for key in ("original_coordinates", "final_coordinates", "coefficient", "intercept", "planned_anchor_indices"):
            np.testing.assert_array_equal(original.rounds[0][key], poisoned.rounds[0][key])
        self.assertFalse(np.allclose(original.rounds[0]["scores"], poisoned.rounds[0]["scores"]))
        self.assertFalse(np.array_equal(original.rounds[0]["utility_selected"], poisoned.rounds[0]["utility_selected"]))

    def test_equal_weight_parentdup_real_rows_and_actual_fits(self):
        selector, actual_model, result, _ = self.run_branch()
        record = selector.rounds[0]
        chosen = record["selected"]
        X, y, weights = parent_training(self.X, self.y, self.X, self.y,
            record["anchor_indices"][chosen], record["partner_indices"][chosen])
        count = int(chosen.sum())
        self.assertEqual(len(y), len(self.y) + 2 * count)
        self.assertEqual(float(weights.sum()), float(len(result.y)))
        np.testing.assert_array_equal(weights[len(self.y):], np.full(2 * count, .5))
        np.testing.assert_array_equal(X[len(self.y):][::2], self.X[record["anchor_indices"][chosen]])
        parent_model = self.learner.fit(X, y, weights, kind="parentdup")
        self.assertFalse(np.array_equal(parent_model.coef_, actual_model.coef_))
        self.assertTrue(np.isfinite(metrics_for(parent_model, *self.validation, self.protocol)[0]["balanced_cross_entropy"]))

    def test_fresh_current_scores_every_round_and_unchanged_u2_baseline(self):
        selector, _, result, calls = self.run_branch(rounds=3)
        self.assertGreaterEqual(len(selector.rounds), 2)
        self.assertEqual(calls[0], len(self.y))
        self.assertGreater(calls[1], calls[0])
        first, second = selector.rounds[:2]
        self.assertFalse(np.array_equal(first["coefficient"], second["coefficient"]))
        checkpoint = deepcopy(self.initial)
        checkpoint.coef_ = second["coefficient"]
        checkpoint.intercept_ = second["intercept"]
        gradients, bias = logistic_ce_gradients(checkpoint, second["final_coordinates"], second["labels"])
        np.testing.assert_allclose(second["scores"], gradients @ second["vG_coefficient"] + bias * second["vG_bias"])
        learner = Learner({}, 0, self.validation)
        direct, sampled, _, _ = run_trajectory(self.X, self.y, 0, self.categorical, self.settings,
            self.parameters, self.prepared, learner)
        canonical, audit = train_method("logistic_regression", "ugo_groups_only", {}, self.X, self.y,
            self.categorical, 0, self.settings, self.validation, group_partition=self.prepared)
        np.testing.assert_array_equal(direct.coef_, canonical.coef_)
        self.assertEqual(len(sampled.y) - len(self.y), audit["augmentation"]["actual_count"])
        self.assertEqual([{k: v for k, v in p.items() if k != "proposal_id"} for p in sampled.provenance], audit["provenance"])




if __name__ == "__main__":
    unittest.main()
