import unittest

import numpy as np

from notebooks.portable_ugo.sampling import resample_control


class PortableSamplingTests(unittest.TestCase):
    def test_rwo_uses_class_standard_error_not_global_or_variance_scale(self):
        # Class 0 is minority. Its population variance is 4, so the RWO
        # perturbation variance is 4/2=2 and the generated mixture variance is 6.
        X = np.column_stack((np.r_[[-2., 2.], np.full(40000, 1000.)],
                             np.full(40002, 7.)))
        y = np.r_[np.zeros(2, dtype=int), np.ones(40000, dtype=int)]
        generated, labels, audit = resample_control('rwo', X, y, {0: 30002, 1: 40000}, 37)
        samples = generated[len(X):]
        self.assertLess(abs(samples[:, 0].mean()), .06)
        self.assertLess(abs(samples[:, 0].var() - 6.), .15)
        np.testing.assert_array_equal(samples[:, 1], 7.)
        np.testing.assert_array_equal(labels[len(y):], 0)
        self.assertEqual(audit['accepted_count'], 30000)

    def test_rwo_nominal_roulette_uses_minority_frequencies_and_independent_draws(self):
        n_majority = 20000
        numeric = np.r_[np.zeros(60), np.full(20, 10.), np.full(n_majority, -100.)]
        category = np.r_[np.zeros(60, dtype=int), np.ones(20 + n_majority, dtype=int)]
        X = np.column_stack((numeric, np.eye(2)[category]))
        y = np.r_[np.zeros(80, dtype=int), np.ones(n_majority, dtype=int)]
        generated, labels, _ = resample_control('rwo', X, y, {0: 18080, 1: n_majority}, 91,
                                               categorical_groups=((1, 2),))
        samples = generated[len(X):]
        self.assertTrue(np.isin(samples[:, 1:], (0., 1.)).all())
        np.testing.assert_array_equal(samples[:, 1:].sum(axis=1), 1.)
        self.assertLess(abs(samples[:, 1].mean() - .75), .02)
        self.assertLess(abs(np.corrcoef(samples[:, 0], samples[:, 2])[0, 1]), .04)
        np.testing.assert_array_equal(generated[:len(X)], X)
        np.testing.assert_array_equal(labels[:len(y)], y)

    def test_smote_keeps_class_geometry_and_projects_complete_nominal_blocks(self):
        rng = np.random.default_rng(17)
        y = np.r_[np.ones(40, dtype=int), np.zeros(12, dtype=int)]
        numeric = np.r_[rng.normal(100, .1, 40), np.linspace(-3., -1., 12)]
        X = np.column_stack((numeric, np.eye(3)[np.arange(len(y)) % 3]))
        generated, labels, audit = resample_control('smote', X, y, {0: 25, 1: 40}, 5,
                                                   categorical_groups=((1, 2, 3),))
        samples = generated[len(X):]
        self.assertTrue(((samples[:, 0] >= -3.) & (samples[:, 0] <= -1.)).all())
        self.assertTrue(np.isin(samples[:, 1:], (0., 1.)).all())
        np.testing.assert_array_equal(samples[:, 1:].sum(axis=1), 1.)
        self.assertTrue(any(not np.any(X[y == 0, 0] == value) for value in samples[:, 0]))
        self.assertEqual(int((labels == 0).sum()), 25)
        self.assertEqual(audit['accepted_count'], 13)
        np.testing.assert_array_equal(generated[:len(X)], X)

    def test_kmeans_smote_cluster_rounding_never_exceeds_declared_class_cap(self):
        rng = np.random.default_rng(13)
        X = np.vstack((rng.normal([-12., 0.], .15, size=(20, 2)),
                       rng.normal([12., 0.], .15, size=(20, 2)),
                       rng.normal([0., 12.], .15, size=(60, 2))))
        y = np.r_[np.zeros(40, dtype=int), np.ones(60, dtype=int)]
        generated, labels, audit = resample_control('kmeans_smote', X, y, {0: 43, 1: 60}, 4,
            parameters={'k_neighbors': 2, 'kmeans_estimator': 3,
                        'cluster_balance_threshold': 0., 'density_exponent': 'auto'})
        self.assertEqual(int((labels == 0).sum()), 43)
        self.assertEqual(int((labels == 1).sum()), 60)
        self.assertGreater(audit['raw_generated_count'], audit['accepted_count'])
        self.assertEqual(audit['trimmed_count'], audit['raw_generated_count'] - 3)
        self.assertEqual(audit['accepted_count'], 3)
        np.testing.assert_array_equal(generated[:len(X)], X)

    def test_impossible_neighbors_do_not_substitute_random_oversampling(self):
        X = np.arange(16, dtype=float).reshape(8, 2)
        y = np.array([0, 0, 1, 1, 1, 1, 1, 1])
        with self.assertRaises(ValueError):
            resample_control('smote', X, y, {0: 6, 1: 6}, 2)
        with self.assertRaises(RuntimeError):
            resample_control('kmeans_smote', X, y, {0: 6, 1: 6}, 2,
                parameters={'k_neighbors': 2, 'kmeans_estimator': 2,
                            'cluster_balance_threshold': 0., 'density_exponent': 'auto'})


if __name__ == '__main__':
    unittest.main()
