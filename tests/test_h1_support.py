"""Real-package tests for H1's frozen minority support and distance convention."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import warnings

import hdbscan
import numpy as np

from samplers.hdbscan_support import prepare_h1_support


def separated_islands(categorical=False):
    rng = np.random.default_rng(173)
    minority = np.concatenate((rng.normal((-4., 0.), .08, (30, 2)),
                               rng.normal((4., 0.), .08, (30, 2)),
                               [[0., 50.]]))
    majority = rng.normal((0., -8.), 1., (90, 2))
    X = np.concatenate((minority[:30], majority, minority[30:]))
    y = np.concatenate((np.full(30, 7), np.full(90, 2), np.full(31, 7)))
    if categorical:
        block = np.zeros((len(X), 3))
        block[:, 0] = 1
        block[120:150] = (0, 1, 0)
        X = np.column_stack((X, block))
    return X, y


class H1SupportTests(unittest.TestCase):
    def test_real_minority_islands_use_frozen_approximate_assignments(self):
        X, y = separated_islands()
        original_X, original_y = X.copy(), y.copy()
        row_ids = np.arange(len(y)) + 8000
        detector = prepare_h1_support(X, y, real_row_ids=row_ids)
        audit = detector.audit
        indices = np.flatnonzero(y == 7)
        self.assertEqual(audit["status"], "ready")
        self.assertEqual(audit["minority_class"], 7)
        self.assertEqual(audit["usable_clusters"], 2)
        self.assertEqual(audit["synthetic_training_count"], 0)
        self.assertEqual(audit["inference_count"], 0)
        np.testing.assert_array_equal(audit["fit_real_indices"], indices)
        np.testing.assert_array_equal(audit["fit_real_row_ids"], row_ids[indices])
        np.testing.assert_array_equal(detector._clusterer._raw_data, X[indices])
        direct_labels, direct_strengths = hdbscan.approximate_predict(detector._clusterer, X[indices])
        np.testing.assert_array_equal(audit["real_prediction_labels"], direct_labels)
        np.testing.assert_array_equal(audit["real_prediction_strengths"], direct_strengths)
        np.testing.assert_array_equal(audit["real_fit_cluster_labels"], detector._clusterer.labels_)
        self.assertEqual(audit["real_fit_cluster_labels"][-1], -1)
        # Real noise remains untouched in the caller's classifier training rows.
        np.testing.assert_array_equal(X, original_X)
        np.testing.assert_array_equal(y, original_y)
        self.assertEqual(len(y), 151)
        self.assertGreaterEqual(audit["training_seconds"], 0.)
        self.assertEqual(audit["configuration"]["h1_min_cluster_size"], 10)
        self.assertEqual(audit["configuration"]["h1_min_samples"], 5)
        self.assertTrue(detector._clusterer.prediction_data)
        scored = detector.predict(np.array([[-4., 0.], [4., 0.], [10000., 10000.]]))
        np.testing.assert_array_equal(scored["valid"], [True, True, True])
        np.testing.assert_array_equal(scored["labels"] != -1, [True, True, False])
        self.assertNotEqual(scored["labels"][0], scored["labels"][1])
        np.testing.assert_array_equal(scored["reasons"], ["accepted", "accepted", "hdbscan_noise"])
        self.assertTrue(np.all((scored["strengths"] >= 0) & (scored["strengths"] <= 1)))
        self.assertEqual(scored["strengths"][-1], 0.)

    def test_no_clusters_rejects_without_bypass_or_repeated_package_warnings(self):
        X = np.concatenate((np.zeros((24, 2)), np.full((30, 2), 5.)))
        y = np.concatenate((np.ones(24, dtype=int), np.zeros(30, dtype=int)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            detector = prepare_h1_support(X, y)
            for _ in range(3):
                result = detector.predict(np.array([[0., 0.], [5., 5.], [np.nan, 0.]]))
        self.assertEqual(caught, [])
        self.assertIsNotNone(detector._clusterer)  # Genuine degenerate package fit.
        self.assertEqual(detector.audit["status"], "no_usable_clusters")
        self.assertEqual(detector.audit["usable_clusters"], 0)
        np.testing.assert_array_equal(detector.audit["real_fit_cluster_labels"], -np.ones(24))
        np.testing.assert_array_equal(detector.audit["real_prediction_labels"], -np.ones(24))
        np.testing.assert_array_equal(detector.audit["real_prediction_strengths"], np.zeros(24))
        np.testing.assert_array_equal(result["labels"], [-1, -1, -1])
        np.testing.assert_array_equal(result["valid"], [True, True, False])
        np.testing.assert_array_equal(result["reasons"], ["no_usable_clusters", "no_usable_clusters", "nonfinite"])
        self.assertEqual(detector.audit["inference_count"], 9)

    def test_insufficient_real_rows_is_explicit_degeneracy_not_parameter_relaxation(self):
        detector = prepare_h1_support(np.arange(8.).reshape(4, 2), [1, 0, 0, 0])
        self.assertEqual(detector.audit["status"], "no_usable_clusters")
        self.assertEqual(detector.audit["degenerate_reason"], "insufficient_real_minority_rows")
        self.assertEqual(detector.audit["configuration"]["h1_min_samples"], 5)
        self.assertEqual(detector.predict([[0., 1.]])["reasons"][0], "no_usable_clusters")

    def test_categorical_distance_and_invalid_reasons_are_separate(self):
        X, y = separated_islands(categorical=True)
        before = X.copy()
        detector = prepare_h1_support(X, y, categorical_columns=(2, 3, 4))
        np.testing.assert_array_equal(X, before)
        represented = detector._represent(np.array([[2., 3., 1., 0., 0.],
                                                    [2., 3., 0., 1., 0.]]))
        np.testing.assert_array_equal(represented[:, :2], [[2., 3.], [2., 3.]])
        self.assertAlmostEqual(float(np.square(represented[0] - represented[1]).sum()), 1.)
        points = np.array([[-4., 0., 1., 0., 0.], [-4., 0., .5, .5, 0.],
                           [np.inf, 0., .5, .5, 0.], [4., 0., 0., 1., 0.]])
        scored = detector.predict(points)
        np.testing.assert_array_equal(scored["valid"], [True, False, False, True])
        np.testing.assert_array_equal(scored["labels"] != -1, [True, False, False, True])
        np.testing.assert_array_equal(scored["reasons"], ["accepted", "invalid_categorical", "nonfinite", "accepted"])
        np.testing.assert_array_equal(scored["strengths"][[1, 2]], [0., 0.])
        # Only flat columns are supplied: full-block sum validity is caller-owned.
        self.assertTrue(detector.predict([[-4., 0., 0., 0., 0.]])["valid"][0])
        self.assertIn("block_boundaries_not_supplied",
                      detector.audit["distance_representation"]["categorical_validation"])

    def test_synthetic_queries_and_caller_mutation_cannot_update_fit(self):
        X, y = separated_islands()
        ids = np.arange(len(y))
        detector = prepare_h1_support(X, y, real_row_ids=ids)
        clusterer = detector._clusterer
        arrays = {"raw": clusterer._raw_data.copy(), "labels": clusterer.labels_.copy(),
                  "tree": clusterer.condensed_tree_.to_numpy().copy(),
                  "linkage": clusterer.single_linkage_tree_.to_numpy().copy(),
                  "cores": clusterer.prediction_data_.core_distances.copy()}
        frozen_audit = deepcopy({k: v for k, v in detector.audit.items()
                                if k not in ("inference_count", "inference_seconds")})
        query = np.array([[-4., 0.], [4., 0.], [1000., 1000.]])
        baseline = detector.predict(query)
        X[:] = 9999.
        y[:] = 99
        ids[:] = -1
        with patch.object(hdbscan.HDBSCAN, "fit", side_effect=AssertionError("synthetic refit")):
            detector.predict(np.random.default_rng(49).normal(size=(200, 2)))
            repeated = detector.predict(query)
        np.testing.assert_array_equal(baseline["labels"], repeated["labels"])
        np.testing.assert_array_equal(baseline["strengths"], repeated["strengths"])
        for key, actual in (("raw", clusterer._raw_data), ("labels", clusterer.labels_),
                            ("tree", clusterer.condensed_tree_.to_numpy()),
                            ("linkage", clusterer.single_linkage_tree_.to_numpy()),
                            ("cores", clusterer.prediction_data_.core_distances)):
            np.testing.assert_array_equal(actual, arrays[key])
        for key, value in frozen_audit.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(detector.audit[key], value)
            else:
                self.assertEqual(detector.audit[key], value)
        self.assertEqual(detector.audit["inference_count"], 206)
        self.assertGreaterEqual(detector.audit["inference_seconds"], 0.)
        self.assertFalse(detector._fit_X.flags.writeable)

    def test_parameter_input_validation_and_explicit_fit_errors(self):
        X, y = separated_islands()
        for name, bad_values in (("h1_min_cluster_size", (0, 1, True, 2.5, np.nan, "10")),
                                 ("h1_min_samples", (0, -1, False, 1.5, np.inf, "5"))):
            for value in bad_values:
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    prepare_h1_support(X, y, parameters={name: value})
        for labels in (np.zeros(len(y)), np.arange(len(y)) % 3, y[:-1], y[:, None]):
            with self.subTest(labels_shape=labels.shape), self.assertRaises(ValueError):
                prepare_h1_support(X, labels)
        for columns in ((0, 0), (-1,), (2,), (True,), (.5,)):
            with self.subTest(columns=columns), self.assertRaisesRegex(ValueError, "categorical_columns"):
                prepare_h1_support(X, y, categorical_columns=columns)
        for bad_X in (np.full_like(X, np.nan), X.astype(complex), X[:, 0]):
            with self.assertRaises(ValueError):
                prepare_h1_support(bad_X, y)
        with self.assertRaisesRegex(ValueError, "real_row_ids"):
            prepare_h1_support(X, y, real_row_ids=np.arange(3))
        with self.assertRaisesRegex(ValueError, "categorical"):
            prepare_h1_support(X, y, categorical_columns=(0,))
        with patch.object(hdbscan.HDBSCAN, "fit", side_effect=ValueError("failed package fit")):
            with self.assertRaisesRegex(RuntimeError, "real-minority fit/prediction failed.*failed package fit"):
                prepare_h1_support(X, y)

    def test_empty_queries_width_validation_and_independent_settings(self):
        X, y = separated_islands()
        detector = prepare_h1_support(X, y, parameters={"h1_min_cluster_size": np.int64(5),
                                                       "h1_min_samples": np.int64(1),
                                                       "group_alpha": .7})
        self.assertEqual(detector.audit["configuration"]["h1_min_cluster_size"], 5)
        self.assertEqual(detector.audit["configuration"]["h1_min_samples"], 1)
        self.assertNotIn("group_alpha", detector.audit["configuration"])
        for array in detector.predict(np.empty((0, 2))).values():
            self.assertEqual(array.shape, (0,))
        self.assertEqual(detector.audit["inference_count"], 0)
        for points in ([[1.]], [1., 2.], [[1., 2j]], [["a", "b"]]):
            with self.subTest(points=points), self.assertRaisesRegex(ValueError, "fitted feature width"):
                detector.predict(points)


if __name__ == "__main__":
    unittest.main()
