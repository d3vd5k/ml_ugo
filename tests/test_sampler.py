import unittest

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from samplers.ugo import (
    UGOConfig, UGOPipeline, _welch_anova_p_value, nearest_indices,
)


class SamplerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.X = torch.tensor(np.vstack((rng.normal(-3, .3, (24, 3)),
                                         rng.normal(3, .3, (8, 3)))), dtype=torch.float32)
        self.y = torch.tensor([0] * 24 + [1] * 8)

    @staticmethod
    def fit(X, y):
        return LogisticRegression().fit(X.cpu().numpy(), y.cpu().numpy())

    def test_constant_groups_stop_only_when_values_agree(self):
        self.assertEqual(_welch_anova_p_value([np.full(8, -.8), np.full(8, .8)]), 0)
        self.assertEqual(_welch_anova_p_value([np.full(8, .4), np.full(8, .4)]), 1)
        self.assertEqual(_welch_anova_p_value([np.full(8, .4), np.arange(8)]), 0)

    def test_fixed_cap_clips_batch_and_preserves_real_ancestry(self):
        config = UGOConfig(device="cpu", alpha=1, phi=1, lambda_step=.8, max_iterations=1)
        result = UGOPipeline(config).run(self.X, self.y, self.fit)
        self.assertEqual(len(result.y), 48)
        self.assertEqual(result.stop_reason, "generation_cap")
        torch.testing.assert_close(result.X[:32], self.X)
        for entry in result.provenance:
            seed, partner = entry["seed_index"], entry["partner_index"]
            self.assertLess(seed, 32)
            self.assertLess(partner, 32)
            self.assertNotEqual(seed, partner)
            self.assertEqual(self.y[seed], self.y[partner])

    def test_multiple_rounds_do_not_use_descendants_as_seeds(self):
        result = UGOPipeline(UGOConfig(device="cpu", alpha=1, lambda_step=.2,
                                      max_iterations=3)).run(self.X, self.y, self.fit)
        self.assertEqual({entry["round"] for entry in result.provenance}, {1, 2, 3})
        self.assertTrue(all(entry["seed_index"] < 32 and entry["partner_index"] < 32
                            for entry in result.provenance))

    def test_singletons_generate_nothing(self):
        result = UGOPipeline(UGOConfig(device="cpu", k_pre=1)).run(
            torch.tensor([[-1.], [1.]]), torch.tensor([0, 1]), self.fit)
        self.assertEqual(result.stop_reason, "no_eligible_group")
        self.assertEqual(len(result.y), 2)

    def test_categorical_blocks_remain_valid(self):
        categorical = torch.nn.functional.one_hot(torch.arange(32) % 3, 3).float()
        X = torch.cat((self.X, categorical), dim=1)
        result = UGOPipeline(UGOConfig(device="cpu", alpha=1, max_iterations=2,
                            categorical_columns=(3, 4, 5))).run(X, self.y, self.fit)
        synthetic = result.X[32:, 3:]
        self.assertTrue(torch.all((synthetic == 0) | (synthetic == 1)))
        torch.testing.assert_close(synthetic.sum(1), torch.ones(len(synthetic)))

    def test_same_seed_replays_generation(self):
        pipeline = UGOPipeline(UGOConfig(device="cpu", alpha=1, max_iterations=2))
        first = pipeline.run(self.X, self.y, self.fit)
        second = pipeline.run(self.X, self.y, self.fit)
        torch.testing.assert_close(first.X, second.X, rtol=0, atol=0)
        self.assertEqual(first.provenance, second.provenance)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_neighbors_and_pipeline(self):
        cpu = nearest_indices(self.X, self.X, 3)
        gpu = nearest_indices(self.X.cuda(), self.X.cuda(), 3)
        torch.testing.assert_close(cpu, gpu.cpu())
        result = UGOPipeline(UGOConfig(device="cuda", alpha=1, max_iterations=2)).run(
            self.X, self.y, self.fit)
        self.assertEqual(result.X.device.type, "cuda")
        self.assertEqual(len(result.y), 38)


if __name__ == "__main__":
    unittest.main()
