import json
import unittest

import numpy as np

from samplers.geometry import ConstraintGeometry, prepare_constraint_geometry


class ConstraintGeometryTests(unittest.TestCase):
    def test_exact_real_distances_match_independent_multiclass_exhaustive_oracle(self):
        rng = np.random.default_rng(43)
        X = rng.normal(size=(39, 5)).astype(np.float32)
        X[:, 1] = rng.integers(0, 3, len(X)) * 1000
        y = np.tile([0, 1, 2], 13)
        geometry = prepare_constraint_geometry(X, y, categorical_columns=(1,))
        seeds = np.array([0, 4, 8, 10, 12, 17, 25, 30, 38])
        points = X[seeds].copy()
        points[:, [0, 2, 3, 4]] += rng.normal(size=(len(seeds), 4)).astype(np.float32) * .3
        eta, rho, epsilon = .5, .7, 1e-12
        metrics = geometry.candidate_metrics(points, seeds, eta, rho, epsilon)
        numeric = [0, 2, 3, 4]
        real = X[:, numeric].astype(np.float64)
        emitted = points[:, numeric].astype(np.float64)
        same, opposing, seed_opposing = [], [], []
        for point, seed in zip(emitted, seeds):
            distances = np.linalg.norm(real - point, axis=1)
            same.append(distances[y == y[seed]].min())
            opposing.append(distances[y != y[seed]].min())
            seed_opposing.append(np.linalg.norm(real[y != y[seed]] - real[seed], axis=1).min())
        displacement = np.linalg.norm(emitted - real[seeds], axis=1)
        ratio = np.array(same) / (np.array(opposing) + epsilon)
        np.testing.assert_allclose(metrics["same_class_distance"], same, rtol=1e-14)
        np.testing.assert_allclose(metrics["opposing_class_distance"], opposing, rtol=1e-14)
        np.testing.assert_allclose(metrics["seed_opposing_distance"], seed_opposing, rtol=1e-14)
        np.testing.assert_allclose(metrics["displacement"], displacement, rtol=1e-14)
        np.testing.assert_allclose(metrics["ratio"], ratio, rtol=1e-14)
        np.testing.assert_array_equal(metrics["passes"],
                                      (displacement <= eta * np.array(seed_opposing)) & (ratio <= rho))
        oracle = [np.linalg.norm(real[y != label] - point, axis=1).min()
                  for point, label in zip(real, y)]
        np.testing.assert_allclose(geometry.seed_opposing_distances, oracle, rtol=1e-14)

    def test_radius_boundary_has_no_slack_and_uses_emitted_float32(self):
        geometry = prepare_constraint_geometry([[0.], [4.], [8.]], [1, 1, 0])
        boundary = np.float32(2)
        points = np.array([[np.nextafter(boundary, -np.inf, dtype=np.float32)],
                           [boundary], [np.nextafter(boundary, np.inf, dtype=np.float32)]])
        metrics = geometry.candidate_metrics(points, [0, 0, 0], eta=.25, rho=10, epsilon=1e-12)
        np.testing.assert_array_equal(metrics["displacement_pass"], [True, True, False])
        np.testing.assert_array_equal(metrics["passes"], [True, True, False])
        # Input doubles that round onto the boundary must be evaluated after
        # Float32 emission, not rejected using unrepresentable input precision.
        rounded = geometry.candidate_metrics([[2 + 2 ** -25]], [0], .25, 10, 1e-12)
        self.assertEqual(rounded["displacement"][0], 2)
        self.assertTrue(rounded["passes"][0])

    def test_ratio_exact_boundary_and_immediately_adjacent_float32_values(self):
        geometry = prepare_constraint_geometry([[0.], [4.]], [1, 0])
        points = np.array([[np.nextafter(np.float32(1), -np.inf, dtype=np.float32)],
                           [1.], [np.nextafter(np.float32(1), np.inf, dtype=np.float32)]])
        metrics = geometry.candidate_metrics(points, [0, 0, 0], eta=1, rho=.25, epsilon=1)
        self.assertEqual(metrics["ratio"][1], .25)
        np.testing.assert_array_equal(metrics["ratio_pass"], [True, True, False])
        np.testing.assert_array_equal(metrics["passes"], [True, True, False])

    def test_zero_opposing_distance_permits_no_displacement(self):
        geometry = prepare_constraint_geometry([[0.], [2.], [0.], [4.]], [1, 1, 0, 0])
        tiny = np.nextafter(np.float32(0), np.float32(1))
        metrics = geometry.candidate_metrics([[0.], [tiny], [1.]], [0, 0, 0], 100, 100, 1e-12)
        np.testing.assert_array_equal(metrics["seed_opposing_distance"], [0, 0, 0])
        np.testing.assert_array_equal(metrics["displacement_pass"], [True, False, False])
        np.testing.assert_array_equal(metrics["passes"], [True, False, False])
        self.assertEqual(metrics["ratio"][0], 0)

    def test_singleton_and_duplicate_groups_keep_multiplicity_not_fake_coordinates(self):
        geometry = prepare_constraint_geometry([[2.], [2.], [5.], [8.]], [1, 1, 2, 0],
                                               targets={1: 3, 2: 2})
        np.testing.assert_array_equal(geometry.row_group_ids, [0, 0, 1, -1])
        self.assertEqual([group["real_support"] for group in geometry.groups], [2, 1])
        self.assertEqual([group["distinct_coordinates"] for group in geometry.groups], [1, 1])
        np.testing.assert_array_equal(geometry.pools[0], [0, 1])
        np.testing.assert_array_equal(geometry.pools[1], [2])
        nearest = geometry.nearest_in_pool([[5.], [8.]], [2, 3], np.ones(4, dtype=bool))
        np.testing.assert_array_equal(nearest, [2, -1])
        untouched = prepare_constraint_geometry([[2.], [3.]], [0, 1], targets={0: 1, 1: 1})
        self.assertEqual(untouched.groups, [])
        np.testing.assert_array_equal(untouched.row_pool_ids, [-1, -1])
        # Native operation (targets=None) describes all classes, not just the
        # minority label or classes with an explicit matched budget.
        native = prepare_constraint_geometry([[2.], [3.]], [0, 1])
        np.testing.assert_array_equal(native.row_group_ids, [0, 1])

    def test_disconnected_islands_never_allow_all_class_relocation_fallback(self):
        X = np.array([[10.], [0.], [10.1], [.1], [10.2], [.2], [100.]])
        y = np.array([1, 1, 1, 1, 1, 1, 0])
        geometry = prepare_constraint_geometry(X, y, targets={1: 7},
                                               parameters={"graph_neighbors": 2, "graph_radius_quantile": 1})
        np.testing.assert_array_equal(geometry.row_group_ids, [0, 1, 0, 1, 0, 1, -1])
        eligible = np.array([False, True, False, True, True, True, True])
        nearest = geometry.nearest_in_pool([[0.], [10.]], [0, 1], eligible)
        np.testing.assert_array_equal(nearest, [4, 5])
        eligible[[0, 2, 4]] = False
        np.testing.assert_array_equal(geometry.nearest_in_pool([[0.]], [0], eligible), [-1])

    def test_nominal_tuple_pools_are_exact_and_graph_ignores_category_codes(self):
        X = np.array([[0., 11., 7.], [1., 11., 7.], [2., 99., 7.],
                      [3., 11., 9.], [20., 99., 7.]])
        geometry = prepare_constraint_geometry(X, [1, 1, 1, 1, 0], categorical_columns=(1, 2),
                                               targets={1: 5},
                                               parameters={"graph_neighbors": 3, "graph_radius_quantile": 1})
        np.testing.assert_array_equal(geometry.row_group_ids, [0, 0, 0, 0, -1])
        self.assertEqual(geometry.groups[0]["distinct_coordinates"], 4)
        np.testing.assert_array_equal(geometry.pools[0], [0, 1])
        np.testing.assert_array_equal(geometry.pools[1], [2])
        np.testing.assert_array_equal(geometry.pools[2], [3])
        points = np.array([[2., 11., 7.], [2., 11., 9.], [2., 99., 7.]])
        metrics = geometry.candidate_metrics(points, [0, 0, 2], 1, 1, 1e-12)
        np.testing.assert_array_equal(metrics["categorical_valid"], [True, False, True])
        np.testing.assert_array_equal(metrics["passes"], [True, False, True])
        nearest = geometry.nearest_in_pool(points, [0, 0, 2], np.ones(len(X), dtype=bool))
        np.testing.assert_array_equal(nearest, [1, -1, 2])

    def test_full_onehot_blocks_and_corrupted_candidates_are_rejected(self):
        X = [[0., 1., 0.], [1., 1., 0.], [2., 0., 1.], [10., 0., 1.]]
        geometry = prepare_constraint_geometry(X, [1, 1, 1, 0], categorical_groups=((1, 2),),
                                               parameters={"graph_neighbors": 2, "graph_radius_quantile": 1})
        self.assertEqual(geometry.numeric_columns, (0,))
        self.assertEqual(geometry.categorical_columns, (1, 2))
        points = [[.5, 1., 0.], [.5, .5, .5], [.5, 1., 1.], [.5, 0., 0.], [.5, 0., 1.]]
        metrics = geometry.candidate_metrics(points, [0] * 5, 1, 1, 1e-12)
        np.testing.assert_array_equal(metrics["categorical_valid"], [True, False, False, False, False])
        np.testing.assert_array_equal(metrics["passes"], [True, False, False, False, False])
        nearest = geometry.nearest_in_pool(points, [0] * 5, np.ones(4, dtype=bool))
        self.assertIn(nearest[0], [0, 1])
        np.testing.assert_array_equal(nearest[1:], [-1, -1, -1, -1])
        for bad in ([[0., .5, .5], [10., 0., 1.]],
                    [[0., 0., 0.], [10., 0., 1.]],
                    [[0., 1., 1.], [10., 0., 1.]]):
            with self.subTest(source=bad), self.assertRaises(ValueError):
                prepare_constraint_geometry(bad, [1, 0], categorical_groups=((1, 2),))

    def test_all_categorical_geometry_has_zero_norms_and_no_distinct_partner(self):
        X = [[1., 0.], [1., 0.], [0., 1.], [0., 1.]]
        geometry = prepare_constraint_geometry(X, [1, 1, 1, 0], categorical_groups=((0, 1),))
        self.assertEqual(geometry.numeric_columns, ())
        np.testing.assert_array_equal(geometry.seed_opposing_distances, np.zeros(4))
        self.assertEqual([group["distinct_coordinates"] for group in geometry.groups], [1, 1])
        for pool in geometry.pools:
            self.assertEqual(len(np.unique(geometry.X[pool], axis=0)), 1)
        metrics = geometry.candidate_metrics([[1., 0.], [0., 1.]], [0, 0], 0, 0, 1e-12)
        np.testing.assert_array_equal(metrics["displacement"], [0, 0])
        np.testing.assert_array_equal(metrics["same_class_distance"], [0, 0])
        np.testing.assert_array_equal(metrics["opposing_class_distance"], [0, 0])
        np.testing.assert_array_equal(metrics["ratio"], [0, 0])
        np.testing.assert_array_equal(metrics["passes"], [True, False])
        np.testing.assert_array_equal(geometry.nearest_in_pool([[1., 0.]], [0],
                                                              np.array([False, True, True, True])), [1])

    def test_queries_cannot_train_geometry_and_reference_includes_ineligible_real_rows(self):
        X = np.array([[0.], [2.], [10.], [12.]], dtype=np.float32)
        y = np.array([1, 1, 0, 0])
        geometry = ConstraintGeometry(X, y, parameters={"graph_neighbors": 1, "graph_radius_quantile": 1})
        before = geometry.snapshot()
        # Numeric nearest same-class distance is to row 1 even though row 1
        # is ineligible for relocation; no pre-noise filtering of references.
        eligible = np.array([True, False, False, False])
        self.assertEqual(geometry.nearest_in_pool([[9.]], [0], eligible)[0], 0)
        for _ in range(3):
            metrics = geometry.candidate_metrics([[9.]], [0], 1, 10, 1e-12)
            self.assertEqual(metrics["same_class_distance"][0], 7)
            self.assertEqual(metrics["opposing_class_distance"][0], 1)
            self.assertEqual(metrics["seed_opposing_distance"][0], 10)
        X[:] = 9
        y[:] = 3
        metrics = geometry.candidate_metrics([[9.]], [0], 1, 10, 1e-12)
        self.assertEqual(metrics["same_class_distance"][0], 7)
        self.assertEqual(geometry.snapshot(), before)
        self.assertEqual(geometry.X.dtype, np.float32)
        with self.assertRaises(ValueError):
            geometry.X[0, 0] = 9

    def test_snapshot_attaches_real_ids_without_geometry_cache_policy_contamination(self):
        source = [[0.], [1.], [5.]]
        first = prepare_constraint_geometry(source, [1, 1, 0], parameters={"eta": .25, "rho": .5, "repair": "shorten"})
        second = prepare_constraint_geometry(source, [1, 1, 0], parameters={"eta": 1, "rho": 1.5, "repair": "reject"})
        self.assertEqual(first.configuration, second.configuration)
        self.assertIsNone(first.real_row_ids)
        first.real_row_ids = np.array([21, 18, 4])
        snapshot = first.snapshot()
        self.assertEqual(snapshot["real_row_ids"], [21, 18, 4])
        self.assertEqual(snapshot["groups"][0]["row_ids"], [21, 18])
        self.assertEqual(snapshot["synthetic_training_count"], 0)
        self.assertEqual(snapshot["geometry_configuration"]["graph_space"], "numeric_training_preprocessed")
        json.dumps(snapshot)
        snapshot["groups"][0]["indices"].append(100)
        snapshot["geometry_configuration"]["numeric_columns"].clear()
        np.testing.assert_array_equal(first.groups[0]["indices"], [0, 1])
        self.assertEqual(first.configuration["numeric_columns"], [0])

    def test_invalid_source_layouts_graph_domains_and_query_parameters(self):
        for X, y in (([[0.], [np.nan]], [0, 1]), ([[0.], [np.inf]], [0, 1]),
                     ([[0.], [1.]], [0]), ([[0.], [1.]], [0, 0]),
                     ([[0.], [1.]], [0, np.nan]), ([[0.], [1e100]], [0, 1])):
            with self.subTest(X=X, y=y), self.assertRaises(ValueError):
                prepare_constraint_geometry(X, y)
        for columns, blocks in (((3,), ()), ((True,), ()), ((.5,), ()), ((0, 0), ()),
                                ((), ((0, 0),)), ((), ((),)), ((), ((0,), (0,))),
                                ((0,), ((1, 2),))):
            with self.subTest(columns=columns, blocks=blocks), self.assertRaises(ValueError):
                prepare_constraint_geometry([[0., 1., 0.], [1., 0., 1.]], [0, 1], columns, blocks)
        for parameters in ({"graph_neighbors": 0}, {"graph_neighbors": True}, {"graph_neighbors": 1.5},
                           {"graph_radius_quantile": -.1}, {"graph_radius_quantile": 1.1},
                           {"graph_radius_quantile": np.nan}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                prepare_constraint_geometry([[0.], [1.]], [0, 1], parameters=parameters)
        for targets in ({2: 3}, {0: 0}, {0: 1.5}, {0: True}):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                prepare_constraint_geometry([[0.], [1.]], [0, 1], targets=targets)
        geometry = prepare_constraint_geometry([[0.], [1.]], [0, 1])
        for eta, rho, epsilon in ((-1, 1, 1e-12), (1, np.nan, 1e-12), (1, 1, 0),
                                  (1, 1, np.inf), (True, 1, 1e-12)):
            with self.subTest(values=(eta, rho, epsilon)), self.assertRaises(ValueError):
                geometry.candidate_metrics([[0.]], [0], eta, rho, epsilon)
        for points, seeds in (([[0.]], [-1]), ([[0.]], [2]), ([[0.]], [.5]), ([[0., 1.]], [0])):
            with self.subTest(points=points, seeds=seeds), self.assertRaises(ValueError):
                geometry.candidate_metrics(points, seeds, 1, 1, 1e-12)
        metrics = geometry.candidate_metrics([[np.nan], [np.inf], [1e100]], [0, 0, 0], 1, 1, 1e-12)
        np.testing.assert_array_equal(metrics["passes"], [False, False, False])
        np.testing.assert_array_equal(geometry.nearest_in_pool([[np.nan]], [0], np.ones(2, dtype=bool)), [-1])
        with self.assertRaises(ValueError):
            geometry.nearest_in_pool([[0.]], [0], [0, 1])


if __name__ == "__main__":
    unittest.main()
