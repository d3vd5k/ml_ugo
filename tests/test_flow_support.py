"""F2's actual full-density calibration and unchanged F1 boundary behavior."""
from copy import deepcopy
import json
import unittest

import numpy as np
import pandas as pd
import torch

from datasets.scripts.climb import make_preprocessor
from samplers.flow_gmm import FlowGMMConfig, FlowGMMCore, FlowGate, SharedRealNVP, fit_flow_gmm
from samplers.flow_support import CalibratedFlowGate, calibrate_flow


def real_fixture():
    rng = np.random.default_rng(710)
    y = np.repeat([0, 1], [30, 18])
    frame = pd.DataFrame({"amount": rng.normal(3 * y - 1, .5),
                          "count": rng.integers(0, 8, len(y)).astype(float),
                          "kind": np.where(np.arange(len(y)) % 3 == 0, "B", "A")})
    return frame, y


def small_config():
    return FlowGMMConfig(epochs=4, layers=2, hidden=12, batch_size=16, device="cpu")


def analytic_fixture():
    # Standardization is exactly identity. The first coupling has variable
    # log-Jacobian 2*tanh(max(x0, 0)) and leaves x1=0 fixed.
    fit = pd.DataFrame({"x0": [-1., 1., -1., 1.], "x1": [-1., 1., 1., -1.], "kind": ["A"] * 4})
    preprocessor = make_preprocessor(fit).fit(fit)
    config = FlowGMMConfig(epochs=1, layers=1, hidden=2, batch_size=8, device="cpu")
    model = SharedRealNVP(3, config, [[0., 0., 1.5]] * 2, [[1., 1., 100.]] * 2)
    with torch.no_grad():
        for parameter in model.couplings[0].network.parameters():
            parameter.zero_()
        network = model.couplings[0].network
        network[0].weight[0, 0] = 1.
        network[2].weight[0, 0] = 1.
        network[4].weight[1, 0] = 1.
    core = FlowGMMCore(model, preprocessor, [0, 1], np.arange(4), config, 7, {})
    raw = pd.DataFrame({"x0": [0., .5, 1., 0., .25], "x1": np.zeros(5), "kind": ["A"] * 5})
    labels, ids = np.array([0, 0, 0, 1, 1]), np.arange(100, 105)
    base = FlowGate(core, preprocessor, calibration_row_ids=ids, validation_row_ids=[200, 201])
    calibration = calibrate_flow(core, raw, labels, ids)
    return core, raw, labels, ids, base, calibration


class FlowSupportTests(unittest.TestCase):
    def test_full_density_jacobian_counterexample_and_original_scores(self):
        core, raw, labels, _, base, calibration = analytic_fixture()
        gate = CalibratedFlowGate(base, calibration, q=0.)
        candidate = pd.DataFrame({"x0": [1.5], "x1": [0.], "kind": ["A"]})
        represented = core.represent(candidate)
        latent, determinant = core.forward(represented)
        gaussian = core.model.gaussian_log_density(latent).numpy()[:, 0]
        full = gate.predict(base.classifier_preprocessor.transform(candidate), [0], .5)
        np.testing.assert_allclose(full["log_likelihood_intended"], gaussian + determinant.numpy(), atol=0., rtol=0.)
        calibrated_latent = calibration.scores["log_likelihood_intended"] - calibration.scores["log_determinant"]
        latent_only_keep = gaussian >= np.min(calibrated_latent[labels == 0])
        self.assertFalse(latent_only_keep[0])
        self.assertTrue(full["accepted"][0])
        self.assertGreater(full["log_determinant"][0], 1.5)
        f1 = base.predict(base.classifier_preprocessor.transform(raw), labels, .5)
        f2 = gate.predict(base.classifier_preprocessor.transform(raw), labels, .5)
        for key in ("log_likelihood_intended", "log_likelihood_opposing", "log_ratio", "log_determinant", "compatibility_score"):
            np.testing.assert_array_equal(f1[key], f2[key])

    def test_linear_class_quantiles_inclusive_boundary_and_q0_not_off(self):
        core, raw, labels, _, base, calibration = analytic_fixture()
        self.assertEqual(calibration.class_counts, {"0": 3, "1": 2})
        expected = {}
        for label in (0, 1):
            scores = np.sort(calibration.scores["log_likelihood_intended"][labels == label])
            expected[str(label)] = scores[0] + (len(scores) - 1) * .25 * (scores[1] - scores[0])
        np.testing.assert_allclose(list(calibration.thresholds(.25).values()), list(expected.values()), atol=0., rtol=1e-15)
        q0 = CalibratedFlowGate(base, calibration, q=0)
        off = CalibratedFlowGate(base, calibration, q=None)
        candidates = pd.DataFrame({"x0": [0., 10.], "x1": [0., 0.], "kind": ["A", "A"]})
        X = core.preprocessor.transform(candidates)
        active, inactive = q0.predict(X, [0, 0], .5), off.predict(X, [0, 0], .5)
        self.assertEqual(active["accepted"].tolist(), [True, False])
        self.assertEqual(active["reasons"].tolist(), ["accepted", "flow_low_likelihood"])
        self.assertEqual(active["log_likelihood_intended"][0], active["likelihood_threshold"][0])
        self.assertEqual(inactive["accepted"].tolist(), [True, True])
        self.assertTrue(np.isnan(inactive["likelihood_threshold"]).all())
        self.assertTrue(q0.audit["support_filter"])
        self.assertTrue(q0.audit["calibration_used"])
        self.assertFalse(off.audit["support_filter"])
        self.assertFalse(off.audit["calibration_used"])
        self.assertEqual(q0.audit["original_calibration_class_counts"], {"0": 3, "1": 2})
        self.assertEqual(q0.audit["calibration_scores"], {key: value.tolist() for key, value in calibration.scores.items()})
        self.assertEqual(q0.audit["calibration_classes"], labels.tolist())
        self.assertTrue(q0.audit["roles_disjoint"])
        self.assertIn("0.25", q0.audit["thresholds_per_q"])
        self.assertIn("linear", q0.audit["threshold_convention"])
        self.assertIn("preprocessing_checksum", q0.audit["source_binding"])
        json.dumps(q0.audit, allow_nan=False)
        for q in (False, True, -1., 1.01, np.nan, np.inf, "off"):
            with self.subTest(q=q), self.assertRaises(ValueError):
                CalibratedFlowGate(base, calibration, q=q)
        self.assertEqual(set(calibration.thresholds(1.)), {"0", "1"})

    def test_exact_rows_missing_classes_overlap_and_unknown_ids_fail_closed(self):
        core, raw, labels, ids, base, _ = analytic_fixture()
        cases = [(raw.iloc[:3], labels[:3], ids[:3]),
                 (raw, labels, [100, 100, 102, 103, 104]),
                 (raw, labels, [0, 101, 102, 103, 104]),
                 (raw.iloc[:0], labels[:0], ids[:0]),
                 (raw, labels, [None, 101, 102, 103, 104]),
                 (raw, [0, 0, 0, 1, 2], ids)]
        for frame, y, row_ids in cases:
            with self.subTest(ids=row_ids), self.assertRaises(ValueError):
                calibrate_flow(core, frame, y, row_ids)
        for positions, identities in (([0, 3], [100, 103]), (list(range(5)), [100, 101, 102, 103, 999])):
            calibration = calibrate_flow(core, raw.iloc[positions], labels[positions], identities)
            with self.assertRaisesRegex(ValueError, "exactly"):
                CalibratedFlowGate(base, calibration, q=.05)
        # Ordering is immaterial; the entire exact reserved set is mandatory.
        reverse = calibrate_flow(core, raw.iloc[::-1], labels[::-1], ids[::-1])
        CalibratedFlowGate(base, reverse, q=.05)
        base.validation_row_ids = np.array([100])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            CalibratedFlowGate(base, reverse, q=.05)

    def test_nonfinite_features_scores_and_unknown_categories_never_dropped(self):
        core, raw, labels, ids, _, _ = analytic_fixture()
        for value in (np.nan, np.inf, -np.inf):
            bad = raw.copy()
            bad.loc[0, "x0"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                calibrate_flow(core, bad, labels, ids)
        with torch.no_grad():
            core.model.means[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            calibrate_flow(core, raw, labels, ids)
        frame, y = real_fixture()
        fitted = fit_flow_gmm(frame, y, np.arange(len(y)), 91, small_config())
        calibration_frame = frame.iloc[[0, 30]].copy()
        calibration_frame.loc[calibration_frame.index[0], "kind"] = "unknown"
        calibration = calibrate_flow(fitted, calibration_frame, [0, 1], [100, 101])
        self.assertEqual(calibration.class_counts, {"0": 1, "1": 1})
        self.assertEqual(calibration.unknown_class_counts, {"0": 1, "1": 0})
        self.assertTrue(np.isneginf(calibration.scores["log_likelihood_intended"][0]))

    def test_unknown_rows_keep_all_quantile_ranks_and_zero_density_support(self):
        core, _, _, _, _, _ = analytic_fixture()
        raw = pd.DataFrame({"x0": [0., .25, .5, .75, 1., 0., .25],
                            "x1": np.zeros(7), "kind": ["unknown", "A", "A", "A", "A", "unknown", "unknown"]})
        labels, ids = np.array([0] * 5 + [1] * 2), np.arange(100, 107)
        calibration = calibrate_flow(core, raw, labels, ids)
        self.assertEqual(calibration.class_counts, {"0": 5, "1": 2})
        self.assertEqual(calibration.finite_class_counts, {"0": 4, "1": 0})
        self.assertEqual(calibration.unknown_class_counts, {"0": 1, "1": 2})
        np.testing.assert_array_equal(calibration.row_ids, ids)
        self.assertTrue(np.isneginf(calibration.thresholds(0.)["0"]))
        self.assertTrue(np.isneginf(calibration.thresholds(.1)["0"]))
        self.assertTrue(np.isneginf(calibration.thresholds(1.)["1"]))
        finite = np.sort(calibration.scores["log_likelihood_intended"][:5][calibration.known[:5]])
        # Rank .625*(5-1)=2.5 includes the unknown zero-density row at rank 0.
        self.assertEqual(calibration.thresholds(.625)["0"], (finite[1] + finite[2]) / 2)
        self.assertNotEqual(calibration.thresholds(.625)["0"], np.quantile(finite, .625, method="linear"))
        self.assertEqual(calibration.thresholds(.25)["0"], finite[0])
        self.assertEqual(calibration.thresholds(1.)["0"], finite[-1])
        audit = calibration.audit
        self.assertEqual(audit["calibration_scores"]["log_likelihood_intended"][0], "-inf")
        self.assertIsNone(audit["calibration_scores"]["log_determinant"][0])
        self.assertEqual(audit["calibration_score_kinds"][0], "negative_infinity")
        self.assertEqual(audit["calibration_reasons"][0], "unknown_flow_category")
        json.dumps(audit, allow_nan=False)
        classifier = make_preprocessor(raw).fit(raw)
        base = FlowGate(core, classifier, calibration_row_ids=ids, validation_row_ids=[200])
        gate = CalibratedFlowGate(base, calibration, q=0.)
        result = gate.predict(classifier.transform(raw), labels, .5)
        self.assertTrue(np.isneginf(result["likelihood_threshold"]).all())
        self.assertEqual(result["reasons"][[0, 5, 6]].tolist(), ["unknown_flow_category"] * 3)
        self.assertFalse(result["accepted"][[0, 5, 6]].any())
        self.assertEqual(gate.audit["likelihood_thresholds"], {"0": "-inf", "1": "-inf"})
        json.dumps(gate.audit, allow_nan=False)
        bad = raw.copy()
        bad.loc[0, "x0"] = np.inf
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            calibrate_flow(core, bad, labels, ids)

    def test_stale_refitted_models_and_preprocessing_require_recalibration(self):
        for change in ("model_weights", "numeric_preprocessing", "preprocessor_replace", "model_replace", "classifier_refit"):
            with self.subTest(change=change):
                core, raw, labels, ids, base, calibration = analytic_fixture()
                classifier = deepcopy(core.preprocessor)
                base = FlowGate(core, classifier, calibration_row_ids=ids, validation_row_ids=[200])
                gate = CalibratedFlowGate(base, calibration, q=.05)
                if change == "model_weights":
                    with torch.no_grad():
                        core.model.means[0, 0] += .1
                elif change == "numeric_preprocessing":
                    core.preprocessor.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"].mean_[0] += .1
                elif change == "preprocessor_replace":
                    core.preprocessor = deepcopy(core.preprocessor)
                elif change == "model_replace":
                    core.model = deepcopy(core.model)
                else:
                    classifier.fit(raw)
                with self.assertRaisesRegex(ValueError, "Stale"):
                    gate.predict(classifier.transform(raw), labels, .5)
                with self.assertRaisesRegex(ValueError, "Stale"):
                    _ = gate.audit
                if change != "classifier_refit":
                    refreshed = calibrate_flow(core, raw, labels, ids)
                    new_base = FlowGate(core, classifier, calibration_row_ids=ids, validation_row_ids=[200])
                    CalibratedFlowGate(new_base, refreshed, q=.05)

    def test_f2_subset_at_fixed_model_tau_and_invalid_reason_independence(self):
        frame, labels = real_fixture()
        fit = np.r_[0:24, 30:44]
        reserved = np.r_[24:30, 44:48]
        core = fit_flow_gmm(frame.iloc[fit], labels[fit], fit, 91, small_config())
        classifier = make_preprocessor(frame).fit(frame)
        base = FlowGate(core, classifier, calibration_row_ids=reserved, validation_row_ids=[200])
        calibration = calibrate_flow(core, frame.iloc[reserved], labels[reserved], reserved)
        X = classifier.transform(frame)
        X[0, core.categorical_groups[0]] = .5
        X[1, 0] = np.nan
        f1 = base.predict(X, labels, .6)
        for q in (None, 0., .01, .05, .1, 1.):
            gate = CalibratedFlowGate(base, calibration, q=q)
            result = gate.predict(X, labels, .6)
            self.assertFalse(np.any(result["accepted"] & ~f1["accepted"]))
            self.assertEqual(result["reasons"][:2].tolist(), ["invalid_onehot", "nonfinite_domain"])
            self.assertFalse(result["compatibility_accepted"][:2].any())
            self.assertFalse(result["likelihood_accepted"][:2].any())
            if q is None:
                np.testing.assert_array_equal(result["accepted"], f1["accepted"])
                np.testing.assert_array_equal(result["reasons"], f1["reasons"])
