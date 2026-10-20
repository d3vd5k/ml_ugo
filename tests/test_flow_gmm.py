"""Behavioral numerical tests for the real, independently written FlowGMM core."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from datasets.scripts.climb import make_preprocessor
from samplers.flow_gmm import (FlowGMMConfig, FlowGMMCore, FlowGate, SharedRealNVP,
                               fit_flow_gmm)

ROOT = Path(__file__).resolve().parents[1]


def real_fixture():
    rng = np.random.default_rng(710)
    y = np.repeat([0, 1], [30, 18])
    frame = pd.DataFrame({"amount": rng.normal(3 * y - 1, .5),
                          "count": rng.integers(0, 8, len(y)).astype(float),
                          "kind": np.where(np.arange(len(y)) % 3 == 0, "B", "A")})
    return frame, y


def small_config():
    return FlowGMMConfig(epochs=4, layers=2, hidden=12, batch_size=16, device="cpu")


def new_core():
    frame, y = real_fixture()
    return fit_flow_gmm(frame, y, np.arange(len(y)), 91, small_config())


def new_gate(core, frame):
    classifier = make_preprocessor(frame).fit(frame)
    return FlowGate(core, classifier, calibration_row_ids=[100, 101], validation_row_ids=[200, 201])


class FlowGMMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.y = real_fixture()
        cls.core = new_core()

    def test_defaults_provenance_both_classes_and_strict_positive_covariance(self):
        default = FlowGMMConfig()
        self.assertEqual((default.layers, default.hidden, default.batch_size, default.epochs), (6, 64, 256, 60))
        self.assertEqual(default.learning_rate, .001)
        audit = self.core.audit
        self.assertEqual(audit["implementation_version"], "flowgmm_tabular_v1")
        self.assertEqual(audit["source_sha256"], hashlib.sha256((ROOT / "samplers/flow_gmm.py").read_bytes()).hexdigest())
        self.assertEqual(audit["torch_version"], str(torch.__version__))
        self.assertEqual(audit["reference_commit"], "071285be739defd548ed0c90eec9e56e1112b773")
        self.assertFalse(audit["vendor_code"])
        self.assertEqual(audit["fit_class_counts"], {"0": 30, "1": 18})
        self.assertEqual(audit["equal_priors"], [.5, .5])
        self.assertTrue(torch.isfinite(self.core.model.log_stds.exp()).all())
        self.assertTrue((self.core.model.log_stds.exp() > 0).all())
        self.assertEqual(len(audit["training_losses"]), 4)
        self.assertTrue(np.isfinite(audit["training_losses"]).all())
        self.assertFalse(audit["early_stopping"])
        self.assertEqual(audit["retries"], 0)
        self.assertFalse(self.core.model.training)
        self.assertTrue(all(not p.requires_grad for p in self.core.model.parameters()))
        self.assertFalse(audit["support_filter"])

    def test_forward_inverse_and_autograd_jacobian(self):
        points = torch.tensor(self.core.represent(self.frame.iloc[:3]), dtype=torch.float64)
        z, determinant = self.core.forward(points)
        recovered, inverse_determinant = self.core.inverse(z)
        torch.testing.assert_close(recovered, points, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(inverse_determinant, -determinant, rtol=1e-11, atol=1e-11)
        x = points[0].clone().requires_grad_(True)
        jacobian = torch.autograd.functional.jacobian(lambda v: self.core.forward(v[None])[0][0], x)
        actual = torch.linalg.slogdet(jacobian).logabsdet
        torch.testing.assert_close(actual, determinant[0], rtol=1e-10, atol=1e-10)
        for coupling in self.core.model.couplings:
            scale, _ = coupling._parameters_at(points * coupling.mask)
            self.assertTrue((scale.abs() <= 2).all())

    def test_likelihood_gaussian_normalization_plus_jacobian_and_ratio(self):
        points = self.core.represent(self.frame.iloc[:8])
        z, determinant = self.core.forward(points)
        gaussian = torch.distributions.Normal(self.core.model.means, self.core.model.log_stds.exp())
        expected = gaussian.log_prob(z[:, None, :]).sum(-1).detach().numpy()
        result = self.core.density(points, self.y[:8])
        np.testing.assert_allclose(result["log_likelihood_intended"], expected[:, 0] + determinant.numpy(), rtol=1e-12)
        np.testing.assert_allclose(result["log_likelihood_opposing"], expected[:, 1] + determinant.numpy(), rtol=1e-12)
        np.testing.assert_allclose(result["log_ratio"], expected[:, 0] - expected[:, 1], rtol=1e-12)

    def test_extreme_ratio_cancels_huge_likelihood_and_jacobian(self):
        config = FlowGMMConfig(epochs=1, layers=2, hidden=4, device="cpu")
        model = SharedRealNVP(1, config, [[0.], [1.]], [[1.], [1.]])
        z = torch.tensor([[1e10], [-1e10]], dtype=torch.float64)
        ratios = model.gaussian_log_ratio(z, torch.tensor([1, 1]))
        np.testing.assert_allclose(ratios.detach().numpy(), [1e10 - .5, -1e10 - .5], atol=0., rtol=0.)
        # Subtracting the full Gaussian likelihoods loses the .5 and other bits.
        naive = model.gaussian_log_density(z)[:, 1] - model.gaussian_log_density(z)[:, 0]
        self.assertFalse(torch.equal(ratios, naive))

    def test_logit_inclusive_boundary_and_strict_tau(self):
        gate = new_gate(self.core, self.frame)
        points = gate.classifier_preprocessor.transform(self.frame.iloc[:3])
        first = gate.predict(points, self.y[:3], .5)
        positive = np.flatnonzero(first["log_ratio"] > 0)[0]
        threshold = float(np.exp(-np.logaddexp(0., -first["log_ratio"][positive])))
        result = gate.predict(points, self.y[:3], threshold)
        expected = result["log_ratio"] >= np.log(threshold) - np.log1p(-threshold)
        np.testing.assert_array_equal(result["accepted"], expected)
        # Identical components give ratio exactly zero and acceptance at tau=.5.
        symmetric = new_core()
        with torch.no_grad():
            symmetric.model.means[1].copy_(symmetric.model.means[0])
            symmetric.model.log_stds[1].copy_(symmetric.model.log_stds[0])
        exact = new_gate(symmetric, self.frame).predict(points, self.y[:3], .5)
        np.testing.assert_array_equal(exact["log_ratio"], np.zeros(3))
        self.assertTrue(exact["accepted"].all())
        self.assertFalse(new_gate(symmetric, self.frame).predict(points, self.y[:3], np.nextafter(.5, 1.))["accepted"].any())
        for tau in (0., 1., -1., np.nan, np.inf, True):
            with self.assertRaises(ValueError):
                gate.predict(points, self.y[:3], tau)

    def test_numeric_only_classifier_bridge_preserves_raw_density_and_acceptance(self):
        frame = self.frame.drop(columns="kind")
        core = fit_flow_gmm(frame, self.y, np.arange(len(frame)), 91, small_config())
        classifier = make_preprocessor(frame).fit(frame.iloc[::2])
        gate = FlowGate(core, classifier, calibration_row_ids=[100, 101],
                        validation_row_ids=[200, 201])
        chosen = np.array([2, 31, 40])
        raw = frame.iloc[chosen]
        points = classifier.named_steps["columns"].transform(raw)
        actual = gate.predict(points, self.y[chosen], .6)
        expected = core.density(core.represent(raw), self.y[chosen])
        np.testing.assert_allclose(actual["log_ratio"], expected["log_ratio"], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(actual["accepted"],
                                      expected["log_ratio"] >= np.log(.6) - np.log1p(-.6))
        np.testing.assert_allclose(gate.inverse_classifier(points), raw, rtol=1e-12, atol=1e-12)

    def test_class_swap_is_symmetric_supervised_fit(self):
        swapped = fit_flow_gmm(self.frame, 1 - self.y, np.arange(len(self.y)), 91, small_config())
        original = self.core.density(self.core.represent(self.frame), self.y)
        inverse = swapped.density(swapped.represent(self.frame), 1 - self.y)
        for key in original:
            np.testing.assert_allclose(original[key], inverse[key], rtol=1e-12, atol=1e-12)
        for coupling_a, coupling_b in zip(self.core.model.couplings, swapped.model.couplings):
            for a, b in zip(coupling_a.parameters(), coupling_b.parameters()):
                torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_category_codes_are_not_continuous_coordinates(self):
        changed = self.frame.copy()
        changed["kind"] = pd.Categorical(changed["kind"], categories=["B", "A"])
        swapped_codes = fit_flow_gmm(changed, self.y, np.arange(len(self.y)), 91, small_config())
        np.testing.assert_array_equal(self.core.represent(self.frame), swapped_codes.represent(changed))
        self.assertEqual(self.core.audit["model_checksum"], swapped_codes.audit["model_checksum"])
        base = self.core.preprocessor.transform(self.frame)
        noisy = self.core.represent(self.frame)
        columns = self.core.categorical_columns
        self.assertEqual(len(columns), 2)
        self.assertTrue(np.isin(base[:, columns], [0, 1]).all())
        self.assertTrue(((noisy[:, columns] - base[:, columns] >= 0)
                         & (noisy[:, columns] - base[:, columns] < 1)).all())
        # Counts receive no jitter or rounding.
        np.testing.assert_array_equal(noisy[:, :2], base[:, :2])

    def test_content_keyed_jitter_duplicate_order_batch_and_class_invariance(self):
        frame = self.frame.iloc[[7, 2, 7, 10, 2]]
        full = self.core.represent(frame)
        np.testing.assert_array_equal(full, np.concatenate([self.core.represent(frame.iloc[:2]), self.core.represent(frame.iloc[2:])]))
        np.testing.assert_array_equal(full[0], full[2])
        np.testing.assert_array_equal(full[1], full[4])
        np.testing.assert_array_equal(full[::-1], self.core.represent(frame.iloc[::-1]))
        gate = new_gate(self.core, self.frame)
        points = gate.classifier_preprocessor.transform(frame)
        labels = np.zeros(len(frame), dtype=int)
        all_scores = gate.predict(points, labels, .6)
        chunk_scores = [gate.predict(points[:2], labels[:2], .6), gate.predict(points[2:], labels[2:], .6)]
        reordered = gate.predict(points[::-1], labels[::-1], .6)
        for key in all_scores:
            actual = np.concatenate([result[key] for result in chunk_scores])
            if actual.dtype.kind == "f":
                np.testing.assert_allclose(actual, all_scores[key], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(reordered[key], all_scores[key][::-1], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(all_scores[key][0], all_scores[key][2], rtol=1e-12, atol=1e-12)
            else:
                np.testing.assert_array_equal(actual, all_scores[key])
                np.testing.assert_array_equal(reordered[key], all_scores[key][::-1])
                np.testing.assert_array_equal(all_scores[key][0], all_scores[key][2])
        reverse_labels = gate.predict(points, 1 - labels, .6)
        np.testing.assert_allclose(reverse_labels["log_ratio"], -all_scores["log_ratio"], atol=1e-12)
        np.testing.assert_allclose(reverse_labels["log_determinant"], all_scores["log_determinant"], atol=1e-12)

    def test_unknown_detector_category_and_original_domain_validity_are_separate(self):
        classifier_frame = self.frame.copy()
        classifier_frame.loc[0, "kind"] = "new"
        gate = new_gate(self.core, classifier_frame)
        points = gate.classifier_preprocessor.transform(classifier_frame.iloc[:4])
        group = gate._groups[0]
        points[1, group] = 0
        points[2, group] = .5
        points[3, 0] = np.inf
        result = gate.predict(points, self.y[:4], .5)
        np.testing.assert_array_equal(result["valid"], [False] * 4)
        np.testing.assert_array_equal(result["accepted"], [False] * 4)
        np.testing.assert_array_equal(result["reasons"], ["unknown_flow_category", "invalid_onehot", "invalid_onehot", "nonfinite_domain"])
        with self.assertRaisesRegex(ValueError, "unknown_flow_category"):
            self.core.represent(classifier_frame.iloc[:1])

    def test_nonfinite_scores_reject_explicitly(self):
        gate = new_gate(self.core, self.frame)
        points = gate.classifier_preprocessor.transform(self.frame.iloc[:2]).astype(np.float64)
        points[0, 0] = 1e100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = gate.predict(points, self.y[:2], .5)
        self.assertEqual(result["reasons"][0], "nonfinite_score")
        self.assertFalse(result["valid"][0])
        self.assertFalse(result["accepted"][0])
        self.assertTrue(result["valid"][1])

    def test_own_preprocessing_never_uses_classifier_validation_statistics(self):
        other = self.frame.copy()
        other["amount"] += 1000
        other["count"] *= 3
        classifier_frame = pd.concat([self.frame, other], ignore_index=True)
        gate = new_gate(self.core, classifier_frame)
        columns = self.core.preprocessor.named_steps["columns"]
        scaler = columns.named_transformers_["numeric"].named_steps["scale"]
        np.testing.assert_allclose(scaler.mean_, self.frame[["amount", "count"]].mean().to_numpy())
        own_before = self.core.represent(self.frame.iloc[:5])
        classifier_points = gate.classifier_preprocessor.transform(self.frame.iloc[:5])
        decoded = gate.inverse_classifier(classifier_points.astype(np.float64))
        np.testing.assert_allclose(decoded[["amount", "count"]], self.frame.iloc[:5][["amount", "count"]], atol=1e-4)
        gate.predict(classifier_points, self.y[:5], .7)
        np.testing.assert_array_equal(own_before, self.core.represent(self.frame.iloc[:5]))
        self.assertTrue(gate.audit["roles_disjoint"])
        self.assertFalse(gate.audit["calibration_used"])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            FlowGate(self.core, gate.classifier_preprocessor, calibration_row_ids=[0], validation_row_ids=[200])

    def test_checkpoint_and_repeated_inference_freeze_every_model_value(self):
        gate = new_gate(self.core, self.frame)
        points = gate.classifier_preprocessor.transform(self.frame.iloc[:5])
        state = deepcopy(self.core.model.state_dict())
        first = gate.predict(points, self.y[:5], .6)
        second = gate.predict(points, self.y[:5], .6)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        for name, value in state.items():
            torch.testing.assert_close(value, self.core.model.state_dict()[name], rtol=0., atol=0.)
        self.assertEqual(gate.audit["inference_rows"], 10)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "core.pt"
            self.core.save(path)
            loaded = FlowGMMCore.load(path)
            loaded_gate = FlowGate(loaded, gate.classifier_preprocessor,
                                  calibration_row_ids=[100, 101], validation_row_ids=[200, 201])
            restored = loaded_gate.predict(points, self.y[:5], .6)
            for key in first:
                np.testing.assert_array_equal(first[key], restored[key])
            self.assertEqual(loaded.audit["model_checksum"], self.core.audit["model_checksum"])
            self.assertTrue(all(not p.requires_grad for p in loaded.model.parameters()))

    def test_fit_rng_is_local_and_fit_is_deterministic(self):
        torch.manual_seed(203)
        np.random.seed(203)
        torch_before = torch.random.get_rng_state().clone()
        numpy_before = np.random.get_state()
        fitted = new_core()
        torch.testing.assert_close(torch_before, torch.random.get_rng_state(), rtol=0., atol=0.)
        numpy_after = np.random.get_state()
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertEqual(fitted.audit["model_checksum"], self.core.audit["model_checksum"])
        self.assertEqual(fitted.audit["training_losses"], self.core.audit["training_losses"])

    def test_training_fresh_epoch_jitter_and_exact_class_balanced_objective(self):
        # Negligible learning rate keeps the identity flow and empirical latent
        # moments numerically unchanged, exposing the actual training objective.
        fitted = fit_flow_gmm(self.frame, self.y, np.arange(len(self.y)), 91,
                             FlowGMMConfig(epochs=2, layers=2, hidden=4, batch_size=128,
                                           learning_rate=1e-300, device="cpu"))
        base = fitted.preprocessor.transform(self.frame).astype(np.float64)
        columns = fitted.categorical_columns
        means_input = base.copy()
        means_input[:, columns] += .5
        means = np.stack([means_input[self.y == label].mean(0) for label in [0, 1]])
        variances = np.stack([base[self.y == label].var(0) for label in [0, 1]])
        variances[:, columns] += 1 / 12
        std = np.sqrt(np.maximum(variances, 1e-4))
        np.testing.assert_allclose(fitted.model.means.numpy(), means, rtol=0., atol=1e-15)
        np.testing.assert_allclose(fitted.model.log_stds.exp().numpy(), std, rtol=1e-15)
        rng = np.random.default_rng(91)
        expected = []
        for _ in range(2):
            points = base.copy()
            points[:, columns] += rng.random((len(base), len(columns)))
            rng.permutation(len(base))
            gaussian = (-.5 * ((points - means[self.y]) / std[self.y]) ** 2
                        - np.log(std[self.y]) - .5 * np.log(2 * np.pi)).sum(1)
            expected.append(-.5 * (gaussian[self.y == 0].mean() + gaussian[self.y == 1].mean()))
        np.testing.assert_allclose(fitted.audit["training_losses"], expected, rtol=1e-13)
        self.assertNotEqual(expected[0], expected[1])

    def test_missing_class_and_nonfinite_training_are_errors_not_retries(self):
        with self.assertRaisesRegex(ValueError, "both real binary classes"):
            fit_flow_gmm(self.frame, np.zeros(len(self.y)), np.arange(len(self.y)), 91, small_config())
        # Constant numeric coordinates initialize finite, strictly positive covariance.
        constant = pd.DataFrame({"value": np.zeros(8)})
        model = fit_flow_gmm(constant, np.repeat([0, 1], 4), np.arange(8), 11,
                             FlowGMMConfig(epochs=2, layers=2, hidden=4, device="cpu"))
        self.assertTrue((model.model.log_stds.exp() > 0).all())
        # An enormous but finite step produces nonfinite objective/parameters;
        # it must fail immediately without early stopping or a substitute fit.
        with self.assertRaisesRegex(RuntimeError, "Nonfinite FlowGMM training"):
            fit_flow_gmm(self.frame, self.y, np.arange(len(self.y)), 91,
                         FlowGMMConfig(epochs=1, layers=2, hidden=4, learning_rate=1e100, device="cpu"))

    def test_actual_shoppers_default_fit_and_gate_smoke(self):
        from benchmarks.protocols import load_dataset, load_protocol
        protocol = load_protocol(ROOT / "configs/benchmarks/shoppers_controlled.yaml")
        dataset = load_dataset(protocol, ROOT / ".cache/benchmarks", download=False)
        selected, _ = train_test_split(np.arange(len(dataset["y"])), train_size=240,
                                       stratify=dataset["y"], random_state=91)
        fit, reserved = train_test_split(selected, test_size=.25, stratify=dataset["y"][selected], random_state=91)
        calibration, validation = train_test_split(reserved, test_size=.5, stratify=dataset["y"][reserved], random_state=91)
        core = fit_flow_gmm(dataset["X"].iloc[fit], dataset["y"][fit], dataset["row_ids"][fit],
                            91, FlowGMMConfig(device="cpu"))
        classifier = make_preprocessor(dataset["X"].iloc[selected]).fit(dataset["X"].iloc[selected])
        gate = FlowGate(core, classifier, calibration_row_ids=dataset["row_ids"][calibration],
                        validation_row_ids=dataset["row_ids"][validation])
        result = gate.predict(classifier.transform(dataset["X"].iloc[validation]), dataset["y"][validation], .5)
        self.assertEqual(len(result["accepted"]), 30)
        self.assertTrue(np.isfinite(result["log_ratio"][result["valid"]]).all())
        self.assertEqual(core.audit["fit_rows"], 180)
        self.assertEqual(len(core.audit["training_losses"]), 60)
        self.assertEqual(sum(core.audit["fit_class_counts"].values()), 180)
        self.assertGreater(result["valid"].sum(), 0)
        print("Shoppers FlowGMM smoke:", {"fit_rows": 180, "validation_rows": 30,
              "valid": int(result["valid"].sum()), "accepted": int(result["accepted"].sum()),
              "reasons": dict(zip(*np.unique(result["reasons"], return_counts=True))),
              "training_seconds": core.audit["training_seconds"], "epochs": 60,
              "model_checksum": core.audit["model_checksum"]})


if __name__ == "__main__":
    unittest.main()
