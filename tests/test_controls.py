import unittest
from unittest.mock import patch

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

from classifiers.factory import fit_estimator
from classifiers.mlp import train_classifier
from samplers.controls import resample
from samplers.ugo_original import OriginalUGOConfig, original_ugo


class ControlTests(unittest.TestCase):
    def test_mean_one_weights_preserve_duplication_objective_and_validation(self):
        X = torch.tensor([[0., 1.], [1., 0.], [2., 1.], [3., 2.]])
        y = torch.tensor([0, 0, 1, 1])
        weights = torch.tensor([1., 1., 3., 3.])
        validation = (X[[0, 2]] + .1, y[[0, 2]])
        # Independent dropout masks for duplicated rows differ stochastically;
        # disable dropout to compare the same deterministic loss objective.
        with patch("classifiers.mlp.nn.Dropout", lambda _: torch.nn.Identity()):
            weighted = train_classifier(X, y, epochs=3, device="cpu", sample_weight=weights, validation_data=validation)
            scaled = train_classifier(X, y, epochs=3, device="cpu", sample_weight=weights * 100, validation_data=validation)
            indices = torch.repeat_interleave(torch.arange(4), weights.long())
            repeated = train_classifier(X[indices], y[indices], epochs=3, device="cpu", validation_data=validation)
        for first, second, third in zip(weighted.parameters(), scaled.parameters(), repeated.parameters()):
            torch.testing.assert_close(first, second)
            torch.testing.assert_close(first, third, atol=2e-6, rtol=2e-5)
        self.assertEqual(weighted.validation_count_, 2)
        self.assertEqual(weighted.training_count_, 4)
        self.assertEqual(repeated.training_count_, 8)

    def test_unsupported_knn_weights_are_not_silently_ignored(self):
        with self.assertRaisesRegex(TypeError, "sample_weight"):
            fit_estimator(KNeighborsClassifier(), [[0.], [1.]], [0, 1], sample_weight=[1, 2])

    def test_smote_mixed_categories_are_nominal_and_real_rows_unchanged(self):
        rng = np.random.default_rng(9)
        X = np.column_stack((rng.normal(size=60), rng.choice([0., 7., 19.], size=60)))
        y = np.array([0] * 40 + [1] * 20)
        result = resample("smote", X, y, 9, categorical_columns=(1,), target_counts={0: 40, 1: 33})
        self.assertTrue(np.array_equal(result.X[:60], X))
        self.assertTrue(np.isin(result.X[:, 1], [0., 7., 19.]).all())
        self.assertEqual(int((result.y == 1).sum()), 33)
        self.assertEqual(result.support_summary["implementation"], "SMOTENC")

    def test_onehot_groups_never_generate_fractional_or_multiple_categories(self):
        rng = np.random.default_rng(14)
        categories = np.eye(3)[rng.integers(3, size=80)]
        X = np.column_stack((rng.normal(size=80), categories))
        y = np.array([0] * 50 + [1] * 30)
        for method in ("smote", "borderline_smote"):
            result = resample(method, X, y, 14, categorical_columns=(1, 2, 3),
                              categorical_groups=((1, 2, 3),), target_counts={0: 50, 1: 40})
            self.assertTrue(np.array_equal(result.X[:80], X))
            self.assertTrue(np.isin(result.X[:, 1:], [0, 1]).all())
            self.assertTrue(np.all(result.X[:, 1:].sum(axis=1) == 1))

    def test_smote_impossible_neighbors_fail_instead_of_ros_fallback(self):
        X = np.arange(16, dtype=float).reshape(8, 2)
        y = np.array([0] * 6 + [1] * 2)
        with self.assertRaises(ValueError):
            resample("smote", X, y, 2)

    def test_original_uses_augmented_current_observations_and_descendants(self):
        X = torch.cat((torch.linspace(-4, -2, 30), torch.linspace(2, 4, 20)))[:, None]
        y = torch.tensor([0] * 30 + [1] * 20)
        training_counts = []
        class Model:
            def predict_proba_tensor(self, rows):
                # Persistent unequal difficulty and nonzero within-class variance.
                positive = torch.where(rows[:, 0] < 0, .02 + .01 * rows[:, 0].abs(), .2 + .01 * rows[:, 0])
                return torch.stack((1 - positive, positive), dim=1)
        def train(rows, labels):
            training_counts.append(len(labels))
            return Model()
        config = OriginalUGOConfig(max_iterations=3, random_state=3, device="cpu", lambda_select=1., lambda_step=.8)
        result = original_ugo(X, y, train, config)
        self.assertEqual(training_counts, [50, 90, 130])
        self.assertGreater(result.rounds[1]["synthetic_support_count"], 0)
        self.assertTrue(any(row["seed_is_synthetic"] or row["partner_is_synthetic"] for row in result.provenance))
        self.assertIn("operational_iteration_cap", result.support_summary["protocol_deviations"])
        for row in result.provenance:
            self.assertLess(row["seed_index"], row["row_index"])
            self.assertLess(row["partner_index"], row["row_index"])

    def test_original_target_caps_never_synthetic_pad_or_remove_reals(self):
        X = torch.cat((torch.linspace(-4, -2, 30), torch.linspace(2, 4, 20)))[:, None]
        y = torch.tensor([0] * 30 + [1] * 20)
        class Model:
            def predict_proba_tensor(self, rows):
                positive = torch.where(rows[:, 0] < 0, .02 + .01 * rows[:, 0].abs(), .2 + .01 * rows[:, 0])
                return torch.stack((1 - positive, positive), dim=1)
        result = original_ugo(X, y, lambda *_: Model(), OriginalUGOConfig(max_iterations=3, device="cpu"), {0: 30, 1: 23})
        self.assertEqual(len(result.y), 53)
        self.assertTrue(torch.equal(result.X[:50], X))
        self.assertEqual(result.support_summary["actual_counts"], {0: 30, 1: 23})


if __name__ == "__main__":
    unittest.main()
