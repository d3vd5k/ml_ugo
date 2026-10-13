import unittest
from dataclasses import FrozenInstanceError, asdict
from unittest.mock import patch

import numpy as np

from samplers.groups import GroupConfig, _co_membership_jaccard, _fit_groups, prepare_groups


class GroupTests(unittest.TestCase):
    def test_config_is_frozen_and_rejects_nonfinite_or_invalid_ranges(self):
        config = GroupConfig()
        with self.assertRaises(FrozenInstanceError):
            config.repeats = 1
        invalid = [{"grouping": "unknown"}]
        for name in ("n_clusters", "n_neighbors", "repeats"):
            invalid.extend({name: value} for value in (0, -1, 1.5, True, np.inf, np.nan))
        for name in ("sample_fraction", "radius_quantile"):
            invalid.extend({name: value} for value in (-.1, 1.1, np.inf, -np.inf, np.nan, True))
        invalid.append({"sample_fraction": 0})
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                GroupConfig(**values)

    def test_both_groupings_find_islands_and_canonical_real_row_membership(self):
        X = np.array([[10.], [0.], [10.1], [.1], [10.2], [.2], [100.]])
        y = np.array([1, 1, 1, 1, 1, 1, 0])
        for grouping in ("kmeans", "graph"):
            with self.subTest(grouping=grouping):
                config = GroupConfig(grouping=grouping, n_clusters=2, n_neighbors=2,
                                     sample_fraction=1, radius_quantile=1)
                result = prepare_groups(X, y, [1], 13, config)
                np.testing.assert_array_equal(result["row_group_ids"], [0, 1, 0, 1, 0, 1, -1])
                for group, indices in zip(result["groups"], ([0, 2, 4], [1, 3, 5])):
                    np.testing.assert_array_equal(group["indices"], indices)
                    self.assertEqual(group["class"], 1)
                    self.assertEqual(group["real_support"], 3)
                    self.assertEqual(group["distinct_coordinates"], 3)
                    self.assertEqual(group["stability"], 1)
                    self.assertEqual(group["evaluable_repeats"], 5)

    def test_graph_requires_mutual_neighbors_and_empirical_radius(self):
        mutual = prepare_groups(np.array([[0.], [1.], [3.]]), np.ones(3, dtype=int), [1], 2,
                                GroupConfig(grouping="graph", n_neighbors=1,
                                            sample_fraction=1, radius_quantile=1))
        np.testing.assert_array_equal(mutual["row_group_ids"], [0, 0, 1])
        cutoff = prepare_groups(np.array([[0.], [.1], [10.], [11.]]), np.ones(4, dtype=int), [1], 2,
                                GroupConfig(grouping="graph", n_neighbors=1,
                                            sample_fraction=1, radius_quantile=0))
        np.testing.assert_array_equal(cutoff["row_group_ids"], [0, 0, 1, 2])
        self.assertEqual([group["distinct_coordinates"] for group in cutoff["groups"]], [2, 1, 1])

    def test_duplicate_coordinates_keep_real_support_but_not_interpolative_support(self):
        X = np.array([[2.], [2.], [5.], [8.]])
        y = np.array([1, 1, 2, 0])
        for grouping in ("kmeans", "graph"):
            with self.subTest(grouping=grouping):
                result = prepare_groups(X, y, [2, 1], 4, GroupConfig(grouping=grouping))
                np.testing.assert_array_equal(result["row_group_ids"], [0, 0, 1, -1])
                duplicate, singleton = result["groups"]
                self.assertEqual(duplicate["real_support"], 2)
                self.assertEqual(duplicate["distinct_coordinates"], 1)
                self.assertEqual(duplicate["stability"], 1)
                self.assertEqual(duplicate["evaluable_repeats"], 5)
                self.assertEqual(singleton["real_support"], 1)
                self.assertEqual(singleton["distinct_coordinates"], 1)
                self.assertEqual(singleton["stability"], 0)
                self.assertEqual(singleton["evaluable_repeats"], 0)
                # A one-coordinate group stays explicit, but an allocator must
                # not interpret repeated observations as interpolation partners.
                self.assertTrue(all(group["distinct_coordinates"] < 2 for group in result["groups"]))

    def test_kmeans_caps_cluster_count_at_unique_coordinates_and_expands_inverse(self):
        result = prepare_groups(np.array([[4.], [0.], [4.], [2.], [0.], [2.]]),
                                np.ones(6, dtype=int), [1], 11,
                                GroupConfig(n_clusters=8, sample_fraction=1))
        np.testing.assert_array_equal(result["row_group_ids"], [0, 1, 0, 2, 1, 2])
        self.assertEqual([group["real_support"] for group in result["groups"]], [2, 2, 2])
        self.assertEqual([group["distinct_coordinates"] for group in result["groups"]], [1, 1, 1])

    def test_jaccard_penalizes_mergers_with_outsiders_as_well_as_splits(self):
        merged, evaluable = _co_membership_jaccard(np.array([0, 0, 1, 1]), np.zeros(4, dtype=int), 2)
        np.testing.assert_allclose(merged, [.2, .2])
        np.testing.assert_array_equal(evaluable, [True, True])
        split, _ = _co_membership_jaccard(np.zeros(4, dtype=int), np.array([0, 0, 1, 1]), 1)
        np.testing.assert_allclose(split, [1 / 3])
        both, _ = _co_membership_jaccard(np.array([0, 0, 1, 1, 1]), np.array([0, 0, 0, 1, 1]), 2)
        np.testing.assert_allclose(both, [1 / 3, .2])
        empty, evaluable = _co_membership_jaccard(np.array([], dtype=int), np.array([], dtype=int), 2)
        np.testing.assert_array_equal(empty, [0, 0])
        np.testing.assert_array_equal(evaluable, [False, False])
        singleton, evaluable = _co_membership_jaccard(np.array([0]), np.array([0]), 2)
        np.testing.assert_array_equal(singleton, [0, 0])
        np.testing.assert_array_equal(evaluable, [False, False])


    def test_seeded_real_row_subsamples_refit_independently_and_match_pair_oracle(self):
        X = np.array([0., .05, .1, 1., 1.05, 1.1, 4., 4.05, 4.1, 10., 10.05, 10.1])[:, None]
        y = np.ones(len(X), dtype=int)
        original = X.copy()
        for grouping in ("kmeans", "graph"):
            with self.subTest(grouping=grouping):
                config = GroupConfig(grouping=grouping, n_clusters=3, n_neighbors=2)
                fits = []

                def record_fit(rows, seed, config):
                    labels, distinct = _fit_groups(rows, seed, config)
                    fits.append((rows.copy(), seed, labels.copy()))
                    return labels, distinct

                with patch("samplers.groups._fit_groups", side_effect=record_fit):
                    first = prepare_groups(X, y, [1], 31, config)
                first_fits = fits.copy()
                fits.clear()
                with patch("samplers.groups._fit_groups", side_effect=record_fit):
                    second = prepare_groups(X, y, [1], 31, config)
                np.testing.assert_array_equal(first["row_group_ids"], second["row_group_ids"])
                self.assertEqual(len(first_fits), 6)
                for before, after in zip(first_fits, fits):
                    np.testing.assert_array_equal(before[0], after[0])
                    self.assertEqual(before[1], after[1])
                    np.testing.assert_array_equal(before[2], after[2])
                for rows, _, _ in first_fits[1:]:
                    self.assertEqual(len(rows), 10)  # ceil(.8 * 12), not unique-coordinate fraction
                    self.assertEqual(len(np.unique(rows)), len(rows))
                    self.assertTrue(np.isin(rows, X).all())
                base_labels = first_fits[0][2]
                for group in first["groups"]:
                    scores = []
                    for sampled_rows, _, replica_labels in first_fits[1:]:
                        source = np.searchsorted(X[:, 0], sampled_rows[:, 0])
                        members = base_labels[source] == group["id"]
                        if members.sum() < 2:
                            continue
                        # Tiny independent oracle deliberately enumerates pairs
                        # here only; production uses count algebra and sparse kNN.
                        base_pairs, incident_pairs = set(), set()
                        for i in range(len(source)):
                            for j in range(i + 1, len(source)):
                                pair = (int(source[i]), int(source[j]))
                                if members[i] and members[j]:
                                    base_pairs.add(pair)
                                if replica_labels[i] == replica_labels[j] and (members[i] or members[j]):
                                    incident_pairs.add(pair)
                        scores.append(len(base_pairs & incident_pairs) / len(base_pairs | incident_pairs))
                    expected = float(np.mean(scores)) if scores else 0
                    self.assertAlmostEqual(group["stability"], expected)
                    self.assertEqual(group["evaluable_repeats"], len(scores))
                    self.assertTrue(np.isfinite(group["stability"]))
                    self.assertGreaterEqual(group["stability"], 0)
                    self.assertLessEqual(group["stability"], 1)
        np.testing.assert_array_equal(X, original)


    def test_non_target_rows_never_enter_fits_and_global_ids_follow_source_order(self):
        X = np.array([[9.], [100.], [1.], [9.1], [1.1], [1000.]])
        y = np.array([2, 0, 1, 2, 1, 0])
        with patch("samplers.groups._fit_groups", wraps=_fit_groups) as fit:
            result = prepare_groups(X, y, [1, 2], 7, GroupConfig(n_clusters=1, sample_fraction=1))
        np.testing.assert_array_equal(result["row_group_ids"], [0, -1, 1, 0, 1, -1])
        self.assertEqual([group["class"] for group in result["groups"]], [2, 1])
        self.assertEqual(fit.call_count, 12)
        self.assertTrue(all(not np.isin(call.args[0], [100., 1000.]).any() for call in fit.call_args_list))
        empty = prepare_groups(X, y, [], 7, GroupConfig())
        self.assertEqual(empty["groups"], [])
        np.testing.assert_array_equal(empty["row_group_ids"], np.full(6, -1))


if __name__ == "__main__":
    unittest.main()
