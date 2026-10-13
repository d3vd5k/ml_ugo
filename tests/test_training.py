import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
import torch

from classifiers.mlp import SimpleMLP, train_classifier
from eval.evaluate import evaluate_and_plot


class ProbabilityModel(torch.nn.Module):
    def __init__(self, scores, positive_class=None):
        super().__init__()
        scores = torch.tensor(scores, dtype=torch.float32)
        self.logits = torch.nn.Parameter(torch.stack((1 - scores, scores), dim=1).log())
        if positive_class is not None:
            self.positive_class_ = positive_class

    def forward(self, X):
        return self.logits[:len(X)]


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.X = torch.arange(64, dtype=torch.float32).reshape(32, 2) / 64
        self.y = torch.tensor([0] * 24 + [1] * 8)
        self.validation_X = torch.tensor([[100., 101.], [102., 103.], [104., 105.]])
        self.validation_y = torch.tensor([0, 1, 0])

    def test_no_validation_uses_every_row_for_exact_epochs(self):
        calls = []
        original_forward = SimpleMLP.forward

        def record(model, X):
            calls.append((model.training, torch.is_grad_enabled(), X.detach().cpu().clone()))
            return original_forward(model, X)

        with patch.object(SimpleMLP, "forward", record):
            model = train_classifier(
                self.X, self.y, epochs=5, lr=0, warmup_epochs=0, patience=1,
                random_state=7, device="cpu",
            )
        self.assertEqual(len(calls), 5)
        for training, gradients, X in calls:
            self.assertTrue(training)
            self.assertTrue(gradients)
            self.assertTrue(torch.equal(X, self.X))
        self.assertEqual(model.training_count_, len(self.y))
        self.assertEqual(model.validation_count_, 0)
        self.assertEqual(model.best_epoch_, 5)
        self.assertFalse(model.training)

    def test_external_validation_stays_real_and_fixed_across_augmented_rounds(self):
        validation_X_before = self.validation_X.clone()
        validation_y_before = self.validation_y.clone()
        calls = []
        original_forward = SimpleMLP.forward

        def record(model, X):
            calls.append((model.training, torch.is_grad_enabled(), X.detach().cpu().clone()))
            return original_forward(model, X)

        training_rounds = [
            (self.X, self.y),
            (torch.cat((self.X, self.X[-4:] + 10)), torch.cat((self.y, self.y[-4:]))),
        ]
        for X, y in training_rounds:
            calls.clear()
            with patch.object(SimpleMLP, "forward", record):
                model = train_classifier(
                    X, y, epochs=8, lr=0, warmup_epochs=0, patience=1,
                    validation_data=(self.validation_X, self.validation_y), device="cpu",
                )
            train_calls = [rows for training, gradients, rows in calls if training and gradients]
            validation_calls = [rows for training, gradients, rows in calls if not training and not gradients]
            self.assertEqual(len(train_calls), 2)
            self.assertEqual(len(validation_calls), 2)
            for rows in train_calls:
                self.assertTrue(torch.equal(rows, X))
            for rows in validation_calls:
                self.assertTrue(torch.equal(rows, self.validation_X))
            self.assertEqual(model.training_count_, len(y))
            self.assertEqual(model.validation_count_, len(self.validation_y))
            self.assertEqual(model.best_epoch_, 1)
        self.assertTrue(torch.equal(self.validation_X, validation_X_before))
        self.assertTrue(torch.equal(self.validation_y, validation_y_before))

    def test_best_epoch_is_count_and_best_checkpoint_is_restored(self):
        original_criterion = torch.nn.CrossEntropyLoss
        validation_losses = iter([1., 2., 3.])

        class ControlledValidationLoss(original_criterion):
            def forward(self, inputs, targets):
                if not torch.is_grad_enabled():
                    return inputs.new_tensor(next(validation_losses))
                return super().forward(inputs, targets)

        with patch("classifiers.mlp.nn.CrossEntropyLoss", ControlledValidationLoss):
            selected = train_classifier(
                self.X, self.y, epochs=10, warmup_epochs=0, patience=2,
                validation_data=(self.validation_X, self.validation_y),
                random_state=17, device="cpu",
            )
        refit = train_classifier(
            self.X, self.y, epochs=selected.best_epoch_, random_state=17, device="cpu",
        )
        self.assertEqual(selected.best_epoch_, 1)
        for name, tensor in selected.state_dict().items():
            self.assertTrue(torch.equal(tensor, refit.state_dict()[name]), name)
        self.assertEqual(refit.validation_count_, 0)
        self.assertEqual(refit.training_count_, len(self.y))

    def test_cpu_seed_repeats_training_exactly(self):
        first = train_classifier(self.X, self.y, epochs=3, random_state=19, device="cpu")
        second = train_classifier(self.X, self.y, epochs=3, random_state=19, device="cpu")
        for name, tensor in first.state_dict().items():
            self.assertTrue(torch.equal(tensor, second.state_dict()[name]), name)
        self.assertEqual(first.positive_class_, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_training_validation_inference_and_determinism(self):
        kwargs = dict(
            epochs=3, random_state=23, device="cuda",
            validation_data=(self.validation_X, self.validation_y),
        )
        first = train_classifier(self.X.cuda(), self.y.cuda(), **kwargs)
        second = train_classifier(self.X, self.y, **kwargs)
        self.assertEqual(next(first.parameters()).device.type, "cuda")
        self.assertEqual(first.validation_count_, len(self.validation_y))
        for name, tensor in first.state_dict().items():
            self.assertTrue(torch.equal(tensor, second.state_dict()[name]), name)
        cpu_input_metrics = evaluate_and_plot(
            first, self.X, self.y.cuda(), show_plot=False, positive_class=1,
        )
        gpu_input_metrics = evaluate_and_plot(
            first, self.X.cuda(), self.y, show_plot=False, positive_class=1,
        )
        self.assertEqual(cpu_input_metrics, gpu_input_metrics)
        self.assertTrue(all(np.isfinite(value) for value in cpu_input_metrics.values()))
        cpu_model = train_classifier(self.X.cuda(), self.y.cuda(), epochs=2, device="cpu")
        evaluate_and_plot(cpu_model, self.X.cuda(), self.y.cuda(), show_plot=False, positive_class=1)


class EvaluationTests(unittest.TestCase):
    def test_pr_auc_is_average_precision_not_trapezoidal_area(self):
        scores = np.array([0.8, 0.9, 0.7, 0.1])
        labels = np.array([1, 0, 1, 0])
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        self.assertNotAlmostEqual(ap, auc(recall, precision))
        metrics = evaluate_and_plot(
            ProbabilityModel(scores), torch.zeros(4, 2), torch.tensor(labels),
            show_plot=False, positive_class=1,
        )
        self.assertAlmostEqual(metrics["pr_auc"], ap)

    def test_positive_class_is_fixed_by_training_not_test_prevalence(self):
        scores = np.array([0.2, 0.4, 0.9, 0.8])
        labels = np.array([1, 1, 1, 0])
        model = ProbabilityModel(scores, positive_class=1)
        default = evaluate_and_plot(model, torch.zeros(4, 2), torch.tensor(labels), show_plot=False)
        explicit = evaluate_and_plot(
            model, torch.zeros(4, 2), torch.tensor(labels), show_plot=False, positive_class=1,
        )
        self.assertEqual(default, explicit)
        self.assertAlmostEqual(default["pr_auc"], average_precision_score(labels, scores))



if __name__ == "__main__":
    unittest.main()
