import unittest

import numpy as np
import pandas as pd

from samplers.tabdiff_control import (COUNTS, NUMERIC, ShoppersSchema,
                                     feature_masks, match_group_counts)


def real_table():
    frame = pd.DataFrame({c: [1., 2.] for c in NUMERIC})
    for c in COUNTS:
        frame[c] = [1, 2]
        frame[c + "_Duration"] = [10., 20.]
    for c in ("BounceRates", "ExitRates", "SpecialDay"):
        frame[c] = [.1, .2]
    for c, values in {"Month": ["May", "Nov"], "OperatingSystems": [1, 2],
                      "Browser": [1, 2], "Region": [1, 2], "TrafficType": [1, 2],
                      "VisitorType": ["New_Visitor", "Returning_Visitor"],
                      "Weekend": [False, True]}.items():
        frame[c] = pd.Categorical(values)
    return frame


class CountAndConstraintTests(unittest.TestCase):
    def test_common_rounding_does_not_rescue_raw_invalidity(self):
        real = real_table()
        schema = ShoppersSchema(real)
        candidates = pd.concat([real.iloc[[0]]] * 5, ignore_index=True)
        candidates["Administrative"] = [1.4, -.2, .2, 2.5, 3.5]
        candidates["Administrative_Duration"] = [10., 0., 10., 10., 10.]
        decoded, valid, _, changes, failures = schema.decode(candidates)
        np.testing.assert_array_equal(decoded["Administrative"], [1, 0, 0, 2, 4])
        np.testing.assert_array_equal(valid, [True, False, False, True, True])
        self.assertTrue(failures["negative_raw_numeric"][1])
        self.assertTrue(failures["zero_count_nonzero_duration:Administrative"][2])
        self.assertTrue(changes[:, 0].all())

    def test_domain_checks_do_not_invent_empirical_constraints(self):
        real = real_table()
        schema = ShoppersSchema(real)
        candidates = pd.concat([real.iloc[[0]]] * 5, ignore_index=True)
        candidates["BounceRates"] = [.3, 1.01, .1, .1, .1]
        candidates["ExitRates"] = [.1] * 5
        candidates["PageValues"] = [100000., 1., -1., np.inf, 1.]
        candidates["Month"] = candidates["Month"].astype(object)
        candidates.loc[4, "Month"] = "unknown"
        _, valid, _, _, _ = schema.decode(candidates)
        np.testing.assert_array_equal(valid, [True, False, False, False, False])

    def test_exact_duplicate_diagnostics_distinguish_real_and_repeated_pool(self):
        real = real_table()
        schema = ShoppersSchema(real)
        generated = pd.concat([real.iloc[[0]]] * 3, ignore_index=True)
        generated.loc[2, "PageValues"] = 3.5
        valid = np.array([True, True, True])
        real_match, pool_match = schema.duplicates(generated, valid)
        np.testing.assert_array_equal(real_match, [True, True, False])
        np.testing.assert_array_equal(pool_match, [False, True, False])


class MatchedDoseTests(unittest.TestCase):
    def test_feasible_per_group_counts_match_without_cross_group_refill(self):
        groups = np.array([0, 0, 0, 1, 1, 1])
        eligibility = {"A": np.array([1, 1, 0, 1, 1, 1], dtype=bool),
                       "B": np.array([1, 1, 1, 0, 0, 0], dtype=bool),
                       "C": np.array([0, 1, 1, 1, 1, 0], dtype=bool)}
        quotas = [{"group_id": 0, "quota": 3}, {"group_id": 1, "quota": 3}]
        selected, report = match_group_counts(eligibility, groups, quotas, 7)
        for arm, mask in selected.items():
            self.assertEqual(int(mask.sum()), 2)
            self.assertTrue(np.all(~mask | eligibility[arm]))
            self.assertFalse(mask[groups == 1].any())
        self.assertEqual([r["quota_shortfall"] for r in report], [1, 3])

    def test_masks_keep_original_feature_blocks_and_label_out(self):
        frame = real_table()
        for seed in range(5):
            masks, metadata = feature_masks(frame, seed)
            self.assertTrue(set(masks["B"]) < set(masks["C"]))
            for arm, target in (("B", .25), ("C", .50)):
                self.assertLessEqual(abs(len(masks[arm]) / len(frame.columns) - target), 1 / len(frame.columns))
                for block in metadata["blocks"]:
                    self.assertTrue(not set(block) & set(masks[arm]) or set(block) <= set(masks[arm]))
                self.assertNotIn("Revenue", masks[arm])


if __name__ == "__main__":
    unittest.main()
