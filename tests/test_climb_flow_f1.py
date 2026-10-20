import unittest

from benchmarks.climb_flow_f1 import choose_f1_threshold


class MinorityF1ThresholdTests(unittest.TestCase):
    def test_equal_scores_are_never_split_to_inflate_f1(self):
        threshold, score = choose_f1_threshold([1, 0, 1], [.6, .6, .4])
        self.assertEqual(threshold, .4)
        self.assertAlmostEqual(score, .8)

    def test_exact_f1_tie_prefers_threshold_nearest_half_then_higher(self):
        threshold, score = choose_f1_threshold([1, 0, 0, 1], [.9, .8, .7, .4])
        self.assertEqual(threshold, .4)
        self.assertAlmostEqual(score, 2 / 3)
        threshold, score = choose_f1_threshold([1, 0, 0, 1], [.6, .55, .45, .4])
        self.assertEqual(threshold, .6)
        self.assertAlmostEqual(score, 2 / 3)

    def test_zero_probability_positives_can_require_all_positive_predictions(self):
        threshold, score = choose_f1_threshold([1, 0, 1], [0., 0., 0.])
        self.assertEqual(threshold, 0.)
        self.assertAlmostEqual(score, .8)


if __name__ == '__main__':
    unittest.main()
