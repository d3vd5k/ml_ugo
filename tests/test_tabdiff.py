"""Short actual official TabDiff training/sampling, not a mocked generator."""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from samplers import tabdiff
from samplers.tabdiff import FrozenTabDiff, TabDiffConfig, fit_tabdiff


def fixture():
    rng = np.random.default_rng(812)
    frame = pd.DataFrame({"amount": rng.normal(size=20),
                          "count": rng.integers(0, 9, 20),
                          "kind": pd.Categorical(["A", "B"] * 10, categories=["A", "B", "HELDOUT_ONLY"]),
                          "flag": np.arange(20) % 3 == 0})
    y = np.repeat([0, 1], [12, 8])
    return frame, y


class TabDiffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.frame, cls.y = fixture()
        # Short development training only; the architecture is official 1024.
        cls.config = TabDiffConfig(epochs=2, batch_size=8, num_steps=4)
        cls.core = fit_tabdiff(cls.frame, cls.y, np.arange(20) + 100, 71, cls.folder.name, cls.config, "cpu")
        cls.anchors = cls.frame.iloc[12:].copy()
        cls.labels = np.ones(8, dtype=int)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_official_defaults_and_real_training(self):
        defaults = TabDiffConfig()
        self.assertEqual((defaults.epochs, defaults.batch_size, defaults.hidden_dim, defaults.num_steps), (8000, 4096, 1024, 50))
        self.assertEqual((defaults.learning_rate, defaults.ema_decay), (.001, .997))
        audit = self.core.training_audit
        self.assertEqual(audit["actual_epochs"], 2)
        self.assertEqual(audit["actual_updates"], 6)
        self.assertEqual(audit["ema_update_frequency"], "once_per_epoch")
        self.assertFalse(audit["active_validation"])
        self.assertNotEqual(audit["initial_model_checksum"], audit["model_checksum"])
        self.assertEqual(self.core._diffusion.__class__.__module__, "tabdiff.models.unified_ctime_diffusion")
        self.assertIsNone(self.core._diffusion.y_only_model)
        self.assertEqual(self.core._diffusion.scheduler, "power_mean_per_column")
        self.assertEqual(self.core._diffusion.cat_scheduler, "log_linear_per_column")
        self.assertTrue(all(not p.requires_grad for p in self.core._diffusion.parameters()))
        self.assertTrue(np.isfinite([entry["numeric_loss"] for entry in audit["loss_history"]]).all())

    def test_training_only_layout_and_immutable_stats(self):
        prep = self.core.preprocessing
        self.assertEqual(prep.numeric, ("amount", "count"))
        self.assertEqual(prep.categorical, ("kind", "flag"))
        self.assertNotIn("HELDOUT_ONLY", prep.dtypes["kind"].categories)
        self.assertNotIn("HELDOUT_ONLY", prep.categories[0])
        self.assertIsInstance(prep.categories[1][0], bool)
        self.assertEqual(len(prep.label_categories), 2)
        self.assertEqual(prep.quantile.quantiles_[0, 0], self.frame.amount.min())
        self.assertEqual(prep.quantile.quantiles_[-1, 0], self.frame.amount.max())
        prep.quantile.quantiles_[:] = -999
        self.assertNotEqual(self.core.preprocessing.quantile.quantiles_[0, 0], -999)
        audit = self.core.training_audit
        audit["training_row_ids"].append(-1)
        self.assertNotIn(-1, self.core.training_audit["training_row_ids"])

    def test_actual_sampling_load_parity_unmasked_and_boolean(self):
        before = tabdiff._checksum(self.core._diffusion)
        generated, audit = self.core.inpaint(self.anchors, ["count", "flag"], self.labels, 54)
        self.assertTrue(all(audit["valid_mask"]))
        self.assertTrue(all(audit["categorical_end_tokens_valid"]))
        pd.testing.assert_frame_equal(generated[["amount", "kind"]], self.anchors[["amount", "kind"]], check_exact=True)
        self.assertEqual(generated.flag.dtype, np.dtype(bool))
        self.assertTrue(all(isinstance(v, bool) for v in generated.flag))
        self.assertEqual(generated["count"].dtype, np.dtype(float))
        self.assertTrue(np.isfinite(generated["count"]).all())
        self.assertEqual(before, tabdiff._checksum(self.core._diffusion))
        loaded = FrozenTabDiff.load(self.folder.name, "cpu")
        repeated, loaded_audit = loaded.inpaint(self.anchors, ["count", "flag"], self.labels, 54)
        pd.testing.assert_frame_equal(generated, repeated, check_exact=True)
        self.assertEqual(audit["checkpoint_sha256"], loaded_audit["checkpoint_sha256"])
        self.assertEqual(before, tabdiff._checksum(loaded._diffusion))

    def test_known_categories_real_anchors_and_labels_reject_per_row(self):
        bad = self.anchors.copy()
        bad.loc[bad.index[0], "kind"] = "HELDOUT_ONLY"
        bad.loc[bad.index[1], "amount"] = 123456
        self.assertEqual(self.core.encode_valid_mask(bad).tolist(), [False, False] + [True] * 6)
        labels = self.labels.copy()
        labels[2] = 0
        generated, audit = self.core.inpaint(bad, ["count"], labels, 18)
        self.assertEqual(audit["valid_mask"], [False] * 3 + [True] * 5)
        self.assertEqual(audit["row_rejection_reasons"][:3], ["unknown_category:kind", "not_real_minority_training_anchor", "label_not_training_minority"])
        self.assertTrue(generated["count"].iloc[:3].isna().all())
        pd.testing.assert_series_equal(generated.amount, bad.amount, check_exact=True)
        self.assertEqual(audit["proposed_rows"], 8)
        self.assertEqual(audit["generated_rows"], 5)

    def test_masked_values_never_enter_prior_or_network(self):
        captured = []
        actual_sampler = tabdiff._sample_inpaint
        def capture(diffusion, numeric, categorical, num_missing, cat_missing):
            captured.append((numeric.clone(), categorical.clone(), num_missing.clone(), cat_missing.clone()))
            return actual_sampler(diffusion, numeric, categorical, num_missing, cat_missing)
        with patch.object(tabdiff, "_sample_inpaint", side_effect=capture):
            self.core.inpaint(self.anchors, ["amount", "kind"], self.labels, 49)
        numeric, categorical, num_missing, cat_missing = captured[0]
        self.assertTrue(torch.all(numeric[:, num_missing] == 0))
        self.assertTrue(torch.all(categorical[:, cat_missing] == 0))
        self.assertFalse(cat_missing[-1].item())
        # Internal sampler invariance even if masked raw coordinates are changed:
        # the independent prior and known-only torch.where make them irrelevant.
        num_other, cat_other = numeric.clone(), categorical.clone()
        num_other[:, num_missing] = 9999
        cat_other[:, cat_missing] = 1
        with tabdiff._rng(993, torch.device("cpu")):
            first = actual_sampler(self.core._diffusion, numeric, categorical, num_missing, cat_missing)
        with tabdiff._rng(993, torch.device("cpu")):
            second = actual_sampler(self.core._diffusion, num_other, cat_other, num_missing, cat_missing)
        torch.testing.assert_close(first[0][:, num_missing], second[0][:, num_missing], rtol=0, atol=0)
        torch.testing.assert_close(first[1][:, cat_missing], second[1][:, cat_missing], rtol=0, atol=0)

    def test_actual_time_known_forward_noise_spy(self):
        calls, evaluations = [], []
        diffusion = self.core._diffusion
        original_forward = tabdiff._forward_known
        original_denoise = diffusion._denoise_fn.forward
        def known_spy(model, num, cat, sigma_num, sigma_cat):
            state = torch.random.get_rng_state()
            with torch.random.fork_rng():
                torch.random.set_rng_state(state)
                expected_num = num + torch.randn_like(num) * sigma_num
                uniforms = torch.rand(*cat.shape, device=cat.device)
                expected_cat = torch.where(uniforms < -torch.expm1(-sigma_cat), model.mask_index, cat)
            result = original_forward(model, num, cat, sigma_num, sigma_cat)
            torch.testing.assert_close(result[0], expected_num, rtol=0, atol=0)
            torch.testing.assert_close(result[1], expected_cat, rtol=0, atol=0)
            calls.append((sigma_num.clone(), sigma_cat.clone(), result[0].clone(), result[1].clone()))
            return result
        def denoise_spy(num, cat, t, sigma=None):
            evaluations.append((num.clone(), cat.clone(), t.clone(), sigma.clone()))
            return original_denoise(num, cat, t, sigma=sigma)
        with patch.object(tabdiff, "_forward_known", side_effect=known_spy), patch.object(diffusion._denoise_fn, "forward", side_effect=denoise_spy):
            self.core.inpaint(self.anchors, ["count", "kind"], self.labels, 29)
        self.assertEqual(len(calls), 7)  # Four Euler + three Heun evaluations.
        self.assertEqual(len(evaluations), len(calls))
        for (sigma_num, sigma_cat, known_num, known_cat), (num, cat, time, sigma) in zip(calls, evaluations):
            torch.testing.assert_close(sigma[0], sigma_num, rtol=0, atol=0)
            expected_cat_sigma = diffusion.cat_schedule.total_noise(time[:1, None])[0]
            torch.testing.assert_close(sigma_cat, expected_cat_sigma, rtol=0, atol=0)
            torch.testing.assert_close(num[:, 0], known_num[:, 0], rtol=0, atol=0)
            observed_cat = torch.stack([part.argmax(1) for part in torch.split(cat, diffusion.num_classes_w_mask, dim=1)], dim=1)
            # Both flag and label are known, but noisy, at this actual time.
            torch.testing.assert_close(observed_cat[:, 1:], known_cat[:, 1:], rtol=0, atol=0)
        self.assertTrue((calls[0][3][:, -1] == diffusion.mask_index[-1]).any())
        self.assertTrue(any(not torch.equal(call[2][:, 0], calls[-1][2][:, 0]) for call in calls[:-1]))

    def test_checkpoint_full_state_and_rng_isolation(self):
        state = torch.random.get_rng_state().clone()
        self.core.inpaint(self.anchors, ["count"], self.labels, 88)
        torch.testing.assert_close(state, torch.random.get_rng_state(), rtol=0, atol=0)
        payload = torch.load(self.core.checkpoint_path, weights_only=False)
        self.assertTrue(payload["optimizer_state"]["state"])
        self.assertIn("scheduler_state", payload)
        self.assertIn("training_rng_state", payload)
        self.assertEqual(set(payload["final_ema_state"]), set(payload["last_training_state"]))
        state = torch.random.get_rng_state().clone()
        FrozenTabDiff.load(self.folder.name, "cpu")
        torch.testing.assert_close(state, torch.random.get_rng_state(), rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as folder:
            independent = fit_tabdiff(self.frame, self.y, np.arange(20) + 100, 71, folder, self.config, "cpu")
            self.assertEqual(independent.training_audit["model_checksum"], self.core.training_audit["model_checksum"])
            self.assertEqual(independent.training_audit["training_seed"], self.core.training_audit["training_seed"])
        torch.testing.assert_close(state, torch.random.get_rng_state(), rtol=0, atol=0)

    def test_numeric_decode_is_unrounded_and_has_no_bounds_repair(self):
        prep = self.core.preprocessing
        raw = prep.quantile.transform(self.frame.loc[:, prep.numeric].to_numpy(float))
        raw[:, 1] = .11
        cats = prep.encode(self.frame, self.y)[1]
        decoded = prep.decode(raw, cats, self.frame.index)
        self.assertTrue((decoded["count"] % 1 != 0).any())
        self.assertNotEqual(float(decoded["count"].iloc[0]), float(np.rint(decoded["count"].iloc[0])))

    def test_official_50_step_auto_device_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            core = fit_tabdiff(self.frame, self.y, np.arange(20), 82, folder,
                               TabDiffConfig(epochs=1, batch_size=20), device="auto")
            generated, audit = core.inpaint(self.anchors.iloc[:2], ["count", "flag"], self.labels[:2], 35)
            self.assertEqual(audit["sampling_steps"], 50)
            self.assertEqual(audit["denoiser_evaluations"], 99)
            self.assertTrue(all(audit["valid_mask"]))
            pd.testing.assert_frame_equal(generated[["amount", "kind"]], self.anchors.iloc[:2][["amount", "kind"]], check_exact=True)
            loaded = FrozenTabDiff.load(folder, device="auto")
            repeated, _ = loaded.inpaint(self.anchors.iloc[:2], ["count", "flag"], self.labels[:2], 35)
            pd.testing.assert_frame_equal(generated, repeated, check_exact=True)

    def test_training_singleton_tail_uses_all_rows(self):
        batches = list(tabdiff._epoch_batches(torch.arange(17), 8))
        self.assertEqual([len(v) for v in batches], [8, 9])
        torch.testing.assert_close(torch.cat(batches), torch.arange(17))


if __name__ == "__main__":
    unittest.main()
