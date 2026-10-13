"""Real Shoppers inverse geometry and native-pipeline regressions."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from samplers.flow_geometry import FlowParentCache, GroupGeometryProposer
from samplers.flow_gmm import _checksum, _layout
from samplers.pool_random import RandomPoolSelector
from samplers.ugo import GroupBudget, UGOConfig, build_support, accept_relocation
from tests import test_pool_random as pool_fixture


class FlowGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pool_fixture.ShoppersRandomPoolTests.setUpClass()
        cls.fixture = pool_fixture.ShoppersRandomPoolTests()
        f = cls.fixture
        cls.raw = f.data["X"].iloc[f.split["fit_indices"]].copy()
        cls.cache = FlowParentCache(f.gate.core, f.partitions["inner"], cls.raw, f.X,
                                   f.data["row_ids"][f.split["fit_indices"]])
        cls.cfg = UGOConfig(device="cpu", categorical_columns=f.categorical)
        cls.real = build_support(torch.from_numpy(f.X), torch.from_numpy(f.y), cls.cfg)
        cls.seeds = torch.as_tensor(np.repeat(np.flatnonzero(f.y == 1), 5))
        cls.budget = GroupBudget(1, cls.seeds, len(cls.seeds))

    def propose(self, mode, cache=None, rng=None):
        generator = GroupGeometryProposer(cache or self.cache, self.fixture.prepared["row_group_ids"], mode)
        candidates = generator.propose(self.real, self.budget, rng or np.random.default_rng(19),
                                       self.cfg, seed_indices=self.seeds)
        return generator, candidates

    def test_input_nearest_partner_draws_and_missing_group_do_not_fallback(self):
        generator, candidates = self.propose("input")
        rng = np.random.default_rng(19)
        choices = rng.random(len(self.seeds))
        coefficients = rng.random((len(self.seeds), 1)).astype(np.float32)
        groups = self.fixture.prepared["row_group_ids"]
        expected = []
        for anchor, choice in zip(self.seeds.numpy(), choices):
            eligible = np.flatnonzero((self.fixture.y == self.fixture.y[anchor]) & (groups == groups[anchor]))
            distances = np.sum((self.fixture.X[eligible].astype(np.float64) - self.fixture.X[anchor]) ** 2, axis=1)
            order = np.argsort(distances, kind="stable")
            options = eligible[order][distances[order] > 0][:self.cfg.k_pre]
            expected.append(options[int(choice * len(options))])
        np.testing.assert_array_equal(candidates.partner_indices, expected)
        np.testing.assert_array_equal(generator.last_proposal["interpolation_coefficients"], coefficients[:, 0])
        np.testing.assert_array_equal(generator.last_proposal["raw_numeric_decoded"],
            self.cache.raw_numeric[self.seeds.numpy()] + coefficients *
            (self.cache.raw_numeric[expected] - self.cache.raw_numeric[self.seeds.numpy()]))
        missing = GroupGeometryProposer(self.cache, np.full(len(groups), -1), "latent")
        pool = missing.propose(self.real, self.budget, np.random.default_rng(19), self.cfg, seed_indices=self.seeds)
        self.assertEqual(len(pool.X), len(self.seeds))
        self.assertTrue((pool.partner_indices == -1).all())
        self.assertFalse(missing.last_proposal["geometry_valid"].any())
        self.assertTrue(torch.isnan(pool.X).all())

    def test_actual_inverse_roundtrip_matched_tuples_projection_and_nonlinearity(self):
        before = _checksum(self.cache.core.model)
        valid = self.cache.parent_valid
        np.testing.assert_allclose(self.cache.roundtrip_coordinates[valid], self.cache.encoded[valid], atol=1e-10)
        linear, a = self.propose("input")
        flow, b = self.propose("latent")
        np.testing.assert_array_equal(a.partner_indices, b.partner_indices)
        np.testing.assert_array_equal(linear.last_proposal["interpolation_coefficients"], flow.last_proposal["interpolation_coefficients"])
        parents = b.partner_indices.numpy()
        seeds = b.seed_indices.numpy()
        representable = flow.last_proposal["parent_valid"]
        self.assertGreater(int(representable.sum()), 0)
        np.testing.assert_array_equal(self.fixture.y[parents[representable]], self.fixture.y[seeds[representable]])
        groups = self.fixture.prepared["row_group_ids"]
        np.testing.assert_array_equal(groups[parents[representable]], groups[seeds[representable]])
        np.testing.assert_array_equal(b.X.numpy()[representable][:, self.fixture.categorical],
                                      self.fixture.X[parents[representable]][:, self.fixture.categorical])
        coeff = flow.last_proposal["interpolation_coefficients"][representable, None]
        z = self.cache.latents[seeds[representable]] + coeff * (self.cache.latents[parents[representable]] - self.cache.latents[seeds[representable]])
        with torch.inference_mode():
            decoded = self.cache.core.inverse(z)[0].cpu().numpy()
        np.testing.assert_allclose(decoded, flow.last_proposal["inverse_detector_coordinates"][representable], atol=1e-12)
        self.assertGreater(np.nanmax(flow.last_proposal["nonlinearity_displacement"]), 1e-5)
        self.assertEqual(before, _checksum(self.cache.core.model))

    def test_numeric_endpoints_retain_actual_inverse_and_partner_categories(self):
        class EndpointRNG:
            def __init__(self, endpoint): self.endpoint = endpoint
            def random(self, shape):
                return np.full(shape, self.endpoint if isinstance(shape, tuple) else .5)
        for endpoint in (0., 1.):
            generator, candidates = self.propose("latent", rng=EndpointRNG(endpoint))
            valid = generator.last_proposal["parent_valid"]
            indices = (candidates.seed_indices if endpoint == 0 else candidates.partner_indices).numpy()[valid]
            expected = self.cache.detector_scaler.inverse_transform(self.cache.encoded[indices, :len(self.cache.numeric_columns)])
            np.testing.assert_allclose(generator.last_proposal["raw_numeric_decoded"][valid], expected, atol=1e-8)
            np.testing.assert_array_equal(candidates.X.numpy()[valid][:, self.fixture.categorical],
                self.fixture.X[candidates.partner_indices.numpy()[valid]][:, self.fixture.categorical])

    def test_unknown_and_invalid_raw_parents_are_row_isolated(self):
        raw = self.raw.copy()
        _, categorical, _, _ = _layout(self.cache.core.preprocessor)
        column = categorical[0]
        raw[column] = raw[column].astype(object)
        raw.iloc[0, raw.columns.get_loc(column)] = "unseen-geometry-category"
        raw.iloc[1, raw.columns.get_loc("PageValues")] = -1.
        raw.iloc[2, raw.columns.get_loc("SpecialDay")] = 2.
        raw.iloc[3, raw.columns.get_loc("PageValues")] = np.inf
        cache = FlowParentCache(self.cache.core, self.cache.classifier_preprocessor, raw,
                                self.fixture.X, self.cache.real_ids)
        self.assertFalse(cache.parent_valid[:4].any())
        np.testing.assert_array_equal(cache.parent_valid[4:], self.cache.parent_valid[4:])
        np.testing.assert_allclose(cache.latents[4:], self.cache.latents[4:], equal_nan=True)
        self.assertTrue(np.isnan(cache.encoded[:4]).all())

    def test_intrinsic_inverse_failures_never_reach_native_relocation(self):
        f = self.fixture
        original_inverse = self.cache.core.inverse
        for failure in ("negative", "rate", "nonfinite"):
            def invalid_inverse(z):
                values, determinant = original_inverse(z)
                values = values.clone()
                raw = self.cache.detector_scaler.inverse_transform(values.cpu().numpy()[:, :len(self.cache.numeric_columns)])
                raw[:, self.cache.numeric_columns.index("SpecialDay")] = {"negative": -1., "rate": 2., "nonfinite": np.nan}[failure]
                values[:, :len(self.cache.numeric_columns)] = torch.as_tensor((raw - self.cache.detector_scaler.mean_) / self.cache.detector_scaler.scale_, device=values.device)
                return values, determinant
            generator = GroupGeometryProposer(self.cache, f.prepared["row_group_ids"], "latent")
            selector = RandomPoolSelector(19, proposal_generator=generator)
            def audited_relocation(real, candidates, cfg):
                self.assertEqual(len(candidates.X), 0)
                return accept_relocation(real, candidates, cfg)
            with patch.object(self.cache.core, "inverse", side_effect=invalid_inverse), patch("samplers.protected.accept_relocation", side_effect=audited_relocation):
                result = f.run_pool(selector, rounds=1)
            self.assertEqual(len(result.y), len(f.y))
            row = selector.rounds[0]
            self.assertFalse(row["geometry_valid"].any())
            self.assertFalse(row["selected"].any())
            self.assertTrue((row["relocation_indices"] == -1).all())

    def test_native_lr_rf_lightgbm_real_retraining_and_frozen_core(self):
        f = self.fixture
        before = _checksum(self.cache.core.model)
        for algorithm in ("logistic_regression", "random_forest", "lightgbm"):
            for mode in ("input", "latent"):
                generator = GroupGeometryProposer(self.cache, f.prepared["row_group_ids"], mode)
                selector = RandomPoolSelector(19, proposal_generator=generator)
                with patch.object(self.cache.core, "represent", side_effect=AssertionError("cached parents only")), patch.object(self.cache.core, "forward", side_effect=AssertionError("cached latents only")):
                    result = f.run_pool(selector, algorithm=algorithm, rounds=2)
                self.assertGreater(len(result.y), len(f.y))
                fitted = f.learner(algorithm)(result.X, result.y)
                probabilities = fitted.predict_proba(f.partitions["X_test"])
                self.assertTrue(np.isfinite(probabilities).all())
                for row in selector.rounds:
                    self.assertEqual(row["state_before_hash"], row["state_after_hash"])
                    self.assertFalse(np.any(row["selected"] & ~row["geometry_valid"]))
        self.assertEqual(before, _checksum(self.cache.core.model))


if __name__ == "__main__":
    unittest.main()
