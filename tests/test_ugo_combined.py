import unittest
from collections import Counter

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from samplers.controls import resample
from samplers.protected import prepare_protected_groups


PARAMETERS = {"grouping": "kmeans", "allocation": "stability", "n_clusters": 2,
              "sample_fraction": 1., "group_alpha": 0., "k_pre": 1,
              "lambda_select": .5, "lambda_step": 1 / 3}


def islands():
    X = np.r_[np.linspace(-10, -9, 16), [0., .1, .2, .3, 10., 10.1, 10.2, 10.3]]
    return X.astype(np.float32)[:, None], np.repeat([0, 1], [16, 8])


class FixedRealScores:
    """A supplied OOF-like vector, independent of the current augmented fit."""

    def __init__(self, X, y, scores):
        self.X, self.y = X.copy(), y.copy()
        self.scores = np.asarray(scores, dtype=np.float32)
        self.calls = []

    def __call__(self, model, X, y):
        np.testing.assert_array_equal(X.cpu().numpy(), self.X)
        np.testing.assert_array_equal(y.cpu().numpy(), self.y)
        self.calls.append(model.predict_proba(X.cpu().numpy()).copy())
        return torch.as_tensor(self.scores.copy(), device=X.device, dtype=X.dtype)


class CombinedSamplerTests(unittest.TestCase):
    def logistic_trainer(self, fits, C=1.):
        def train(X, y):
            fits.append(len(y))
            return LogisticRegression(C=C, max_iter=500, random_state=13).fit(
                X.cpu().numpy(), y.cpu().numpy())
        return train

    def sample(self, X, y, scores, *, parameters=None, targets=None, iterations=1,
               trainer=None, prepared=None, method="ugo_combined"):
        parameters = {**PARAMETERS, **(parameters or {})}
        if prepared is None:
            prepared = prepare_protected_groups(X, y, 13, targets, parameters)
        return resample(method, X, y, 13, device="cpu", max_iterations=iterations,
                        model_fn=trainer or self.logistic_trainer([]), target_counts=targets,
                        parameters=parameters, seed_score=scores, group_partition=prepared)

    def assert_ancestry(self, result, X, y, prepared, categorical=()):
        rows, labels = result.X.cpu().numpy(), result.y.cpu().numpy()
        np.testing.assert_array_equal(rows[:len(X)], X)
        np.testing.assert_array_equal(labels[:len(y)], y)
        self.assertEqual(len(result.provenance), len(rows) - len(X))
        self.assertEqual([entry["row_index"] for entry in result.provenance],
                         list(range(len(X), len(rows))))
        descendants = Counter(entry["seed_index"] for entry in result.provenance)
        self.assertTrue(all(count <= 4 for count in descendants.values()))
        group_counts = Counter(entry["group_id"] for entry in result.provenance)
        for group in prepared["groups"]:
            self.assertLessEqual(group_counts[group["id"]], 2 * len(group["indices"]))
        for entry in result.provenance:
            seed, partner, relocation = (entry[key] for key in
                                         ("seed_index", "partner_index", "relocation_index"))
            self.assertIn(seed, range(len(X)))
            self.assertIn(partner, range(len(X)))
            self.assertNotEqual(seed, partner)
            self.assertEqual(y[seed], entry["class"])
            self.assertEqual(y[partner], entry["class"])
            self.assertEqual(labels[entry["row_index"]], entry["class"])
            self.assertEqual(entry["group_id"], prepared["row_group_ids"][seed])
            if relocation >= 0:
                self.assertIn(relocation, range(len(X)))
                self.assertEqual(y[relocation], entry["class"])
                np.testing.assert_array_equal(rows[entry["row_index"]], X[relocation])
            elif categorical:
                np.testing.assert_array_equal(rows[entry["row_index"], list(categorical)],
                                              X[partner, list(categorical)])
        self.assertNotIn("constraint_support", result.support_summary)
        self.assertNotIn("stopping", result.support_summary)
        self.assertNotIn("stopping_selection", result.support_summary)
        return descendants, group_counts

    def test_oof_difficulty_changes_actual_quotas_and_ancestry_against_u2_logistic(self):
        X, y = islands()
        targets = {0: 16, 1: 16}
        prepared = prepare_protected_groups(X, y, 13, targets, PARAMETERS)
        np.testing.assert_array_equal(prepared["groups"][0]["indices"], [16, 17, 18, 19])
        np.testing.assert_array_equal(prepared["groups"][1]["indices"], [20, 21, 22, 23])
        scores = FixedRealScores(X, y, np.r_[np.full(16, -.8), np.arange(1, 9) / 10])
        combined_fits, u2_fits = [], []
        combined = self.sample(X, y, scores, targets=targets, prepared=prepared,
                               trainer=self.logistic_trainer(combined_fits, C=.001))
        u2 = self.sample(X, y, None, targets=targets, prepared=prepared,
                         trainer=self.logistic_trainer(u2_fits, C=.001), method="ugo_groups_only")
        # Eight within-class midranks give group means .25 and .75. With
        # alpha=0 and stability=1, largest remainders of 8*w/sum(w) are 2, 6.
        for result, expected_h, expected_quota, expected_seeds in (
            (combined, [.25, .75], [2, 6], Counter({19: 1, 18: 1, 23: 3, 22: 3})),
            (u2, [.75, .25], [6, 2], Counter({16: 3, 17: 3, 20: 1, 21: 1})),
        ):
            trace = result.support_summary["group_support"]["allocation_records"][0]["group_trace"]
            np.testing.assert_allclose([g["h"] for g in trace], expected_h)
            self.assertEqual([g["quota"] for g in trace], expected_quota)
            descendants, group_counts = self.assert_ancestry(result, X, y, prepared)
            self.assertEqual(descendants, expected_seeds)
            self.assertEqual([group_counts[g["id"]] for g in prepared["groups"]], expected_quota)
            self.assertEqual(result.support_summary["actual_counts"], targets)
        self.assertEqual(combined_fits, [24])
        self.assertEqual(u2_fits, [24])
        self.assertEqual(len(scores.calls), 1)
        for a, b in zip(combined.support_summary["group_support"]["groups"],
                        u2.support_summary["group_support"]["groups"]):
            self.assertEqual((a["real_support"], a["stability"], a["supported"]), (4, 1., True))
            np.testing.assert_array_equal(a["indices"], b["indices"])

    def test_fixed_real_scores_survive_changed_downstream_models_and_augmented_refits(self):
        X, y = islands()
        scores = FixedRealScores(X, y, np.r_[np.full(16, -.8), np.arange(1, 9) / 10])
        fits = []
        def changing_train(rows, labels):
            fits.append(len(labels))
            strategy = "prior" if len(fits) == 1 else "uniform"
            return DummyClassifier(strategy=strategy).fit(rows.cpu().numpy(), labels.cpu().numpy())
        result = self.sample(X, y, scores, targets={0: 16, 1: 20}, iterations=3,
                             parameters={"lambda_step": .125}, trainer=changing_train)
        self.assertEqual(fits, [24, 27, 30])
        self.assertEqual(len(scores.calls), 3)
        self.assertFalse(np.array_equal(scores.calls[0], scores.calls[1]))
        np.testing.assert_array_equal(scores.calls[1], scores.calls[2])
        self.assertEqual(result.stop_reason, "iteration_cap")
        self.assertEqual(len(result.y), 33)
        self.assertEqual([r["accepted"] for r in result.rounds], [3, 3, 3])
        for row in result.rounds:
            self.assertEqual(row["p_value"], 0.)
            self.assertAlmostEqual(row["class_mean_difficulty"]["1"], .45, places=6)
            self.assertEqual(row["real_group_support"], {"0": 16, "1": 8})
        audit = result.support_summary["group_support"]
        for record in audit["allocation_records"]:
            self.assertEqual([g["quota"] for g in record["group_trace"]], [1, 2])
            np.testing.assert_allclose([g["h"] for g in record["group_trace"]], [.25, .75])
        self.assertEqual(Counter(p["seed_index"] for p in result.provenance),
                         Counter({19: 3, 23: 3, 22: 3}))
        self.assertEqual(audit["synthetic_training_count"], 0)

    def test_saturation_redistributes_then_underspends_without_padding_or_descendant_seeds(self):
        X = np.r_[np.linspace(-10, -9, 24), [0., .1], np.linspace(10, 10.5, 6)].astype(np.float32)[:, None]
        y = np.repeat([0, 1], [24, 8])
        parameters = {**PARAMETERS, "group_alpha": 1., "lambda_step": .375}
        targets = {0: 24, 1: 38}
        prepared = prepare_protected_groups(X, y, 13, targets, parameters)
        scores = FixedRealScores(X, y, np.r_[np.full(24, -.8), [.9, 1.], np.arange(1, 7) / 10])
        fits = []
        result = self.sample(X, y, scores, parameters=parameters, targets=targets,
                             iterations=10, prepared=prepared, trainer=self.logistic_trainer(fits))
        records = result.support_summary["group_support"]["allocation_records"]
        # The small group's initial proportional share exceeds its 2*n=4 cap;
        # its excess must go to the six-row group, not disappear or get padded.
        self.assertEqual([[g["quota"] for g in r["group_trace"]] for r in records],
                         [[4, 8], [0, 4], [0, 0]])
        self.assertEqual([r["accepted"] for r in records], [12, 4, 0])
        self.assertEqual([r["unspent"] for r in records], [0, 8, 12])
        self.assertEqual(fits, [32, 44, 48])
        self.assertEqual(result.stop_reason, "no_eligible_group")
        self.assertEqual(result.support_summary["actual_counts"], {0: 24, 1: 24})
        self.assertEqual(len(result.y), 48)
        descendants, groups = self.assert_ancestry(result, X, y, prepared)
        self.assertEqual(descendants, Counter({25: 4, 31: 4, 30: 4, 29: 4}))
        self.assertEqual(groups, Counter({0: 4, 1: 12}))

    def test_conflicting_small_island_is_rescued_and_actually_sampled(self):
        X = np.r_[np.linspace(-10, -9, 8), [-.001, .001, .009, .011],
                  [0., .01, 10., 10.1, 10.2, 10.3]].astype(np.float32)[:, None]
        y = np.repeat([0, 1], [12, 6])
        targets = {0: 12, 1: 12}
        parameters = {**PARAMETERS, "group_alpha": 1.}
        prepared = prepare_protected_groups(X, y, 13, targets, parameters)
        # Independent nearest-real-row oracle: these minority rows conflict
        # with majority neighbors, but each belongs to a supported island.
        for index in (12, 13):
            distances = np.abs(X[:, 0] - X[index, 0])
            distances[index] = np.inf
            self.assertEqual(y[np.argmin(distances)], 0)
        scores = FixedRealScores(X, y, np.r_[np.full(12, -.8), [.9, 1., .1, .2, .3, .4]])
        fits = []
        result = self.sample(X, y, scores, targets=targets, parameters=parameters,
                             prepared=prepared, trainer=self.logistic_trainer(fits))
        audit = result.support_summary["group_support"]
        np.testing.assert_array_equal(audit["protected_conflict_indices"], [12, 13])
        self.assertEqual(audit["groups"][0]["real_support"], 2)
        self.assertTrue(audit["groups"][0]["supported"])
        descendants, groups = self.assert_ancestry(result, X, y, prepared)
        self.assertEqual(descendants[13], 4)
        self.assertEqual(groups, Counter({0: 4, 1: 2}))
        self.assertEqual(fits, [18])
        self.assertEqual(len(result.y), 24)

    def test_real_onehot_donors_and_cross_group_interpolation_are_not_u3_constrained(self):
        coordinates, y = islands()
        numerators = np.r_[np.ones(20), np.full(4, 3.)].astype(np.float32)
        X = np.column_stack((coordinates, numerators, np.ones(24, dtype=np.float32),
                             np.eye(2, dtype=np.float32)[np.arange(24) % 2]))
        parameters = {**PARAMETERS, "k_pre": 7, "lambda_select": 1., "lambda_step": 1.}
        targets = {0: 16, 1: 24}
        prepared = prepare_protected_groups(X, y, 13, targets, parameters)
        scores = FixedRealScores(X, y, np.r_[np.full(16, -.8), np.arange(1, 9) / 10])
        fits = []
        result = resample("ugo_combined", X, y, 13, device="cpu", max_iterations=1,
                          model_fn=self.logistic_trainer(fits), target_counts=targets,
                          categorical_columns=(3, 4), categorical_groups=((3, 4),),
                          parameters=parameters, seed_score=scores, group_partition=prepared)
        self.assert_ancestry(result, X, y, prepared, categorical=(3, 4))
        generated = result.X.cpu().numpy()[len(X):]
        self.assertEqual(len(generated), 16)
        self.assertTrue(np.isin(generated[:, 3:], [0., 1.]).all())
        np.testing.assert_array_equal(generated[:, 3:].sum(axis=1), np.ones(16))
        cross = [p for p in result.provenance if
                 prepared["row_group_ids"][p["seed_index"]] != prepared["row_group_ids"][p["partner_index"]]]
        self.assertTrue(cross)
        unconstrained = 0
        for entry in cross:
            if entry["relocation_index"] >= 0:
                continue
            row = result.X[entry["row_index"]].cpu().numpy()
            group = next(g for g in prepared["groups"] if g["id"] == entry["group_id"])
            real = X[group["indices"]]
            ratios = real[:, 1] / real[:, 2]
            if not (real[:, 0].min() <= row[0] <= real[:, 0].max()) and not (
                    ratios.min() <= row[1] / row[2] <= ratios.max()):
                unconstrained += 1
        self.assertGreater(unconstrained, 0)
        self.assertEqual(fits, [24])

    def test_original_welch_equal_scores_stop_but_differing_scores_reach_iteration_cap(self):
        X, y = islands()
        targets = {0: 16, 1: 20}
        for scores, expected, accepted, fits_expected in (
            (np.full(24, .2), "class_difficulty_balance", 0, [24]),
            (np.r_[np.full(16, -.8), np.full(8, .2)], "iteration_cap", 6, [24, 27]),
        ):
            with self.subTest(stop=expected):
                callback = FixedRealScores(X, y, scores)
                fits = []
                result = self.sample(X, y, callback, targets=targets, iterations=2,
                                     parameters={"lambda_step": .125}, trainer=self.logistic_trainer(fits))
                self.assertEqual(result.stop_reason, expected)
                self.assertEqual(len(result.y) - len(y), accepted)
                self.assertEqual(fits, fits_expected)
                self.assertTrue(all(r["p_value"] == (1. if accepted == 0 else 0.) for r in result.rounds))
                self.assertNotIn("stopping", result.support_summary)
                self.assertNotIn("constraint_support", result.support_summary)

    def test_missing_scores_and_unsupported_hook_combinations_fail_meaningfully(self):
        X, y = islands()
        scores = FixedRealScores(X, y, np.r_[np.full(16, -.8), np.arange(1, 9) / 10])
        prepared = prepare_protected_groups(X, y, 13, {0: 16, 1: 16}, PARAMETERS)
        trainer = self.logistic_trainer([])
        with self.assertRaisesRegex(ValueError, "(?i)score|OOF"):
            resample("ugo_combined", X, y, 13, model_fn=trainer)
        with self.assertRaisesRegex((TypeError, ValueError), "(?i)callable|score"):
            resample("ugo_combined", X, y, 13, model_fn=trainer, seed_score=np.ones(len(y)))
        for method in ("baseline", "ugo_corrected", "ugo_groups_only", "ugo_constraints_only", "ugo_stopping_only"):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "(?i)score|OOF"):
                resample(method, X, y, 13, model_fn=trainer, seed_score=scores)
        for method in ("baseline", "ugo_corrected", "ugo_oof_mma", "ugo_constraints_only", "ugo_stopping_only"):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "(?i)group|partition"):
                resample(method, X, y, 13, model_fn=trainer, group_partition=prepared)
        with self.assertRaisesRegex(ValueError, "(?i)constraint|geometry"):
            resample("ugo_combined", X, y, 13, model_fn=trainer,
                     seed_score=scores, constraint_geometry={})


if __name__ == "__main__":
    unittest.main()
