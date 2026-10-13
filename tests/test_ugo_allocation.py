import unittest

import numpy as np

from samplers.allocation import AllocationConfig, GroupAllocator, integer_quotas


def group(group_id, indices, label=1, stability=1., distinct=None):
    indices = np.asarray(indices, dtype=np.int64)
    return {"id": group_id, "class": label, "indices": indices,
            "real_support": len(indices),
            "distinct_coordinates": len(indices) if distinct is None else distinct,
            "stability": stability, "evaluable_repeats": 4}


class IntegerQuotaTests(unittest.TestCase):
    def test_proportions_and_deterministic_remainders(self):
        np.testing.assert_array_equal(integer_quotas([1, 2, 3], [100] * 3, 10), [2, 3, 5])
        np.testing.assert_array_equal(integer_quotas([1, 1, 1], [10] * 3, 2), [1, 1, 0])

    def test_saturation_iteratively_redistributes_without_zero_weight_fallback(self):
        np.testing.assert_array_equal(integer_quotas([100, 10, 1, 0], [1, 2, 20, 100], 15), [1, 2, 12, 0])
        np.testing.assert_array_equal(integer_quotas([100, 1, 1], [0, 9, 9], 7), [0, 4, 3])
        np.testing.assert_array_equal(integer_quotas([1, 0], [2, 100], 30), [2, 0])

    def test_zero_weights_zero_caps_zero_budget_and_empty(self):
        for weights, caps, budget in (([0, 0], [4, 9], 100), ([1, 9], [0, 0], 10),
                                       ([1, 9], [4, 9], 0), ([], [], 10)):
            with self.subTest(weights=weights, caps=caps, budget=budget):
                result = integer_quotas(weights, caps, budget)
                np.testing.assert_array_equal(result, np.zeros(len(weights), dtype=int))
                self.assertTrue(np.issubdtype(result.dtype, np.integer))

    def test_large_finite_weights_do_not_overflow_normalization(self):
        np.testing.assert_array_equal(integer_quotas([1e308, 1e308], [10, 10], 9), [5, 4])
        np.testing.assert_array_equal(integer_quotas([1e308, 1e-300], [1, 10], 5), [1, 4])

    def test_invalid_quota_inputs(self):
        cases = (([1], [1, 2], 1), ([-1], [1], 1), ([np.nan], [1], 1),
                 ([np.inf], [1], 1), ([1], [-1], 1), ([1], [1.5], 1),
                 ([1], [1], -1), ([1], [1], 1.5), ([1], [1], True))
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                integer_quotas(*args)


class AllocationConfigTests(unittest.TestCase):

    def test_finite_ranges(self):
        invalid = ({"allocation": "other"}, {"group_alpha": -1}, {"group_alpha": 1.1},
                   {"group_alpha": np.nan}, {"group_alpha": np.inf}, {"epsilon": 0},
                   {"epsilon": -1}, {"epsilon": np.inf}, {"epsilon": np.nan},
                   {"seed_cap": 0}, {"seed_cap": 1.5}, {"seed_cap": True},
                   {"group_cap_multiplier": 0}, {"group_cap_multiplier": -1},
                   {"group_cap_multiplier": np.nan}, {"group_cap_multiplier": np.inf})
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                AllocationConfig(**values)


class GroupAllocationTests(unittest.TestCase):
    def test_top_selection_ceil_stable_ties_and_uniform_seed_counts(self):
        allocator = GroupAllocator({"groups": [group(10, [0, 1, 2, 3, 4])]},
                                   np.ones(5, dtype=int), {1: 50},
                                   AllocationConfig(seed_cap=4, group_cap_multiplier=10))
        plan = allocator.allocate([3, 3, 2, 1, 0], 1, 5, .21)
        np.testing.assert_array_equal(plan["seed_indices"], [0, 0, 0, 1, 1])
        self.assertEqual(plan["count"], 5)
        self.assertEqual(plan["group_trace"][0]["capacity"], 8)
        np.testing.assert_array_equal(allocator.used_seed, [0] * 5)

    def test_scarce_groups_seed_caps_and_multiple_partial_commits(self):
        groups = [group(10, [0, 1]), group(20, [2, 3, 4, 5])]
        allocator = GroupAllocator({"groups": groups}, np.ones(6, dtype=int), {1: 30},
                                   AllocationConfig(allocation="uniform", seed_cap=2,
                                                    group_cap_multiplier=1))
        scores = np.array([9, 8, 7, 6, 5, 4])
        first = allocator.allocate(scores, 1, 10, .5)
        np.testing.assert_array_equal(first["seed_indices"], [0, 0, 2, 2, 3, 3])
        self.assertEqual(allocator.records[-1]["unspent"], 4)
        # Only two proposals were accepted: planning never consumes six.
        allocator.commit(np.array([0, 2]))
        np.testing.assert_array_equal(allocator.used_group, [1, 1])
        self.assertEqual(allocator.records[-1]["accepted"], 2)
        second = allocator.allocate(scores, 1, 10, .5)
        np.testing.assert_array_equal(second["seed_indices"], [0, 2, 3, 3])
        allocator.commit(second["seed_indices"])
        np.testing.assert_array_equal(allocator.used_group, [2, 4])
        self.assertIsNone(allocator.allocate(scores, 1, 10, 1))
        self.assertEqual(allocator.used_class[1], 6)
        self.assertEqual([g["real_support"] for g in groups], [2, 4])
        self.assertEqual(len(allocator.used_seed), 6)
        self.assertEqual(allocator.records[-1]["allocated"], 0)
        self.assertTrue(all("group_cap_exhausted" in g["exclusions"]
                            for g in allocator.records[-1]["group_trace"]))

    def test_seed_exhaustion_selects_next_real_seed_not_descendants(self):
        allocator = GroupAllocator({"groups": [group(0, [0, 1, 2])]}, [1, 1, 1], {1: 20},
                                   AllocationConfig(seed_cap=1, group_cap_multiplier=10))
        for expected in (0, 1, 2):
            plan = allocator.allocate([9, 8, 7], 1, 10, .1)
            np.testing.assert_array_equal(plan["seed_indices"], [expected])
            allocator.commit(plan["seed_indices"])
        self.assertIsNone(allocator.allocate([9, 8, 7], 1, 10, .1))
        np.testing.assert_array_equal(allocator.used_seed, [1, 1, 1])

    def test_singleton_identical_coordinates_and_unstable_groups_never_allocate(self):
        groups = [group(0, [0]), group(1, [1, 2], distinct=1), group(2, [3, 4], stability=0)]
        for policy in ("stability", "size", "uniform", "difficulty"):
            with self.subTest(policy=policy):
                allocator = GroupAllocator({"groups": groups}, [1] * 5, {1: 100},
                                           AllocationConfig(allocation=policy))
                self.assertIsNone(allocator.allocate([100, 99, 98, 97, 96], 1, 50, 1))
                for seed in range(5):
                    with self.assertRaises(ValueError):
                        allocator.commit([seed])
                np.testing.assert_array_equal(allocator.used_seed, [0] * 5)
                self.assertTrue(all(t["quota"] == t["weight"] == t["capacity"] == 0
                                    for t in allocator.records[-1]["group_trace"]))

    def test_all_policies_skew_as_specified_with_fixed_real_support(self):
        groups = [group(10, [0, 1]), group(20, range(2, 8), stability=.2),
                  group(30, range(8, 12))]
        scores = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected = {"stability": [16, 0, 8], "size": [4, 12, 8],
                    "uniform": [8, 8, 8], "difficulty": [12, 3, 9]}
        h = np.array([11 / 12, 3 / 12, 8 / 12])
        for policy, quotas in expected.items():
            with self.subTest(policy=policy):
                allocator = GroupAllocator({"groups": groups}, [1] * 12, {1: 100},
                                           AllocationConfig(allocation=policy, seed_cap=20,
                                                            group_cap_multiplier=20))
                plan = allocator.allocate(scores, 1, 24, 1)
                self.assertEqual([g["quota"] for g in plan["group_trace"]], quotas)
                np.testing.assert_allclose([g["h"] for g in plan["group_trace"]], h)
                if policy == "stability":
                    np.testing.assert_allclose([g["weight"] for g in plan["group_trace"]],
                                               np.array([1, .2, 1]) * (h + .001) /
                                               (np.array([2, 6, 4]) + .001) ** .5)
                allocator.commit(plan["seed_indices"])
                np.testing.assert_array_equal(allocator.used_group, quotas)
                self.assertEqual(sum(allocator.real_counts.values()), 12)
                self.assertEqual([g["real_support"] for g in groups], [2, 6, 4])
                again = allocator.allocate(scores, 1, 24, 1)
                self.assertEqual([g["n_real"] for g in again["group_trace"]], [2, 6, 4])
                np.testing.assert_allclose([g["h"] for g in again["group_trace"]], h)

    def test_percentiles_are_within_class_and_average_ties(self):
        allocator = GroupAllocator({"groups": [group(0, [0, 1]), group(1, [2, 3])]},
                                   [1, 1, 1, 1, 0, 0], {1: 20})
        plan = allocator.allocate([5, 5, 9, 9, -1000, 1000], 1, 4, 1)
        np.testing.assert_allclose([g["h"] for g in plan["group_trace"]], [.25, .75])

    def test_default_targets_final_counts_and_atomic_class_cap(self):
        allocator = GroupAllocator({"groups": [group(0, [0, 1]), group(1, range(2, 7), label=0)]},
                                   [1, 1, 0, 0, 0, 0, 0], None)
        self.assertEqual(allocator.target_counts, {0: 5, 1: 5})
        plan = allocator.allocate(np.arange(7), 1, 100, 1)
        self.assertEqual(plan["count"], 3)
        with self.assertRaisesRegex(ValueError, "class target"):
            allocator.commit([0, 0, 1, 1])
        np.testing.assert_array_equal(allocator.used_seed, [0] * 7)
        allocator.commit(plan["seed_indices"])
        self.assertIsNone(allocator.allocate(np.arange(7), 1, 100, 1))
        self.assertIsNone(allocator.allocate(np.arange(7), 0, 100, 1))

    def test_group_caps_are_floored_not_rounded_or_weakened(self):
        allocator = GroupAllocator({"groups": [group(0, [0, 1, 2])]}, [1] * 3, {1: 100},
                                   AllocationConfig(group_cap_multiplier=.5))
        plan = allocator.allocate([1, 2, 3], 1, 10, 1)
        self.assertEqual(plan["count"], 1)
        with self.assertRaisesRegex(ValueError, "group cap"):
            allocator.commit([0, 1])
        self.assertEqual(int(allocator.used_group.sum()), 0)
        allocator.commit(plan["seed_indices"])
        self.assertIsNone(allocator.allocate([1, 2, 3], 1, 10, 1))

    def test_commit_checks_source_membership_class_and_seed_caps_atomically(self):
        allocator = GroupAllocator({"groups": [group(0, [0, 1]), group(1, [2, 3], label=0)]},
                                   [1, 1, 0, 0, 1], {1: 20, 0: 20},
                                   AllocationConfig(seed_cap=1, group_cap_multiplier=10))
        for seeds in ([-1], [5], [4], [0, 2], [0, 0], [0.5]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                allocator.commit(seeds)
            self.assertEqual(int(allocator.used_seed.sum()), 0)
        allocator.commit([0])
        with self.assertRaisesRegex(ValueError, "per-seed"):
            allocator.commit([0, 1])
        np.testing.assert_array_equal(allocator.used_seed, [1, 0, 0, 0, 0])
        allocator.commit([])

    def test_empty_groups_and_zero_round_budget(self):
        allocator = GroupAllocator({"groups": []}, [1, 1], {1: 20})
        self.assertIsNone(allocator.allocate([1, 2], 1, 10, 1))
        self.assertEqual(allocator.records[-1]["unspent"], 10)
        allocator = GroupAllocator({"groups": [group(0, [0, 1])]}, [1, 1], {1: 20})
        plan = allocator.allocate([1, 2], 1, 0, 1)
        self.assertIsNotNone(plan)
        self.assertEqual(plan["count"], 0)
        self.assertEqual(plan["seed_indices"].shape, (0,))

    def test_invalid_targets_metadata_and_scores(self):
        prepared = {"groups": [group(0, [0, 1])]}
        for targets in ({1: 1}, {2: 10}, {1: 2.5}, {1: True}):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                GroupAllocator(prepared, [1, 1], targets)
        bad_groups = ([group(0, [0, 0])], [group(0, [0, 2])],
                      [group(0, [0, 1], label=0)], [group(0, [0, 1], stability=np.nan)],
                      [group(0, [0, 1]), group(1, [1])])
        for groups in bad_groups:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                GroupAllocator({"groups": groups}, [1, 1], {1: 20})
        allocator = GroupAllocator(prepared, [1, 1], {1: 20})
        for args in (([1], 1, 1, 1), ([np.nan, 1], 1, 1, 1), ([1, 2], 2, 1, 1),
                     ([1, 2], 1, -1, 1), ([1, 2], 1, 1, 0), ([1, 2], 1, 1, np.inf)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                allocator.allocate(*args)


if __name__ == "__main__":
    unittest.main()
