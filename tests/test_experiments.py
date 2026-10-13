import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import experiments
from benchmarks.climb_benchmark import benchmark_dataset
from datasets.scripts.synthetic import create_synthetic_split
from experiments import MLPClassifierAdapter, RunArtifacts, real_inner_split, run_fold


def read_json(path):
    return json.loads(Path(path).read_text())


def small_dataset():
    split = create_synthetic_split(n_samples=120, dataset="linear", minority_weight=.25, random_state=19)
    X = torch.cat((split["X_train"], split["X_test"])).numpy()
    y = torch.cat((split["y_train"], split["y_test"])).numpy()
    return {"name": "isolated", "X": pd.DataFrame(X, columns=["x", "z"]), "y": y,
            "row_ids": np.concatenate((split["train_ids"], split["test_ids"]))}, len(split["y_train"])


class ExperimentTests(unittest.TestCase):
    def test_inner_split_retains_rare_classes_on_both_sides(self):
        labels = np.repeat(np.arange(3), [200, 2, 3])
        train = np.arange(len(labels))
        for fraction in (.001, .15, .999):
            fit, validation = real_inner_split(train, labels, 12, fraction)
            self.assertEqual(set(labels[fit]), {0, 1, 2})
            self.assertEqual(set(labels[validation]), {0, 1, 2})
            self.assertFalse(set(fit) & set(validation))
            self.assertEqual(set(fit) | set(validation), set(train))
        fit, validation = real_inner_split(train, labels, 12, .15)
        self.assertEqual(len(validation), round(len(labels) * .15))
        repeated = real_inner_split(train, labels, 12, .15)
        np.testing.assert_array_equal(fit, repeated[0])
        np.testing.assert_array_equal(validation, repeated[1])

    def test_inner_split_rejects_singleton_class_without_dropping_it(self):
        with self.assertRaises(ValueError):
            real_inner_split(np.arange(5), np.array([0, 0, 0, 0, 1]), 12)

    def test_real_holdout_is_excluded_from_preprocessing_and_provenance(self):
        dataset, n_train = small_dataset()
        train, test = np.arange(n_train), np.arange(n_train, len(dataset["y"]))
        fit, validation = real_inner_split(train, dataset["y"], 19, .25)
        dataset["X"].loc[validation, "x"] = 10000
        dataset["X"].loc[test, "x"] = 20000
        dataset["X"]["kind"] = "training"
        dataset["X"].loc[validation, "kind"] = "validation-only"
        dataset["X"].loc[test, "kind"] = "test-only"
        with tempfile.TemporaryDirectory() as root:
            artifacts = RunArtifacts(root, {"random_state": 19})
            run_fold(dataset, {"lr": LogisticRegression(max_iter=100)}, train, test,
                     1, 2, 19, "cpu", .25, artifacts, noise_ratio=.5)
            path = artifacts.fold_path(dataset["name"], 1)
            state = read_json(path / "preprocessing.json")
            expected = dataset["X"].iloc[fit][["x", "z"]].mean().to_numpy()
            np.testing.assert_allclose(state["inner"]["numeric"]["scale"]["mean_"], expected)
            self.assertEqual(state["inner"]["categorical"]["encode"]["categories_"], [["training"]])
            split = pd.read_csv(path / "splits.csv")
            heldout = split[split.role.isin(["inner_validation", "outer_test"])]
            np.testing.assert_array_equal(heldout.clean_label, heldout.training_label)
            self.assertTrue((split.clean_label != split.training_label).any())
            fit_ids = set(dataset["row_ids"][fit])
            for item in read_json(path / "lr" / "inner_provenance.json"):
                self.assertIn(item["seed_row_id"], fit_ids)
                self.assertIn(item["partner_row_id"], fit_ids)
                if item["relocation_row_id"] is not None:
                    self.assertIn(item["relocation_row_id"], fit_ids)
            support = read_json(path / "lr" / "inner_stop.json")["support_summary"]
            self.assertEqual(support["real_count"], len(fit))

    def test_all_algorithms_and_strategies_share_outer_test_rows(self):
        dataset, _ = small_dataset()
        with tempfile.TemporaryDirectory() as root:
            artifacts = RunArtifacts(root, {"random_state": 19})
            rows = benchmark_dataset(dataset, {"lr": LogisticRegression(max_iter=100),
                "knn": KNeighborsClassifier(n_neighbors=3)}, 2, 1, 19, "cpu", .25, artifacts)
            self.assertEqual(len(rows), 8)
            for fold in (1, 2):
                path = artifacts.fold_path(dataset["name"], fold)
                test_ids = pd.read_csv(path / "splits.csv").query("role == 'outer_test'").row_id.tolist()
                for algorithm in ("lr", "knn"):
                    for strategy in ("baseline", "ugo"):
                        predicted_ids = pd.read_csv(path / algorithm / f"{strategy}_predictions.csv").row_id.tolist()
                        self.assertEqual(predicted_ids, test_ids)

    def test_mlp_selection_uses_only_real_holdout_and_refits_fixed_epochs(self):
        dataset, n_train = small_dataset()
        train, test = np.arange(n_train), np.arange(n_train, len(dataset["y"]))
        fit, validation = real_inner_split(train, dataset["y"], 19, .25)
        calls = []
        original = experiments.train_classifier
        def traced(X, y, **kwargs):
            model = original(X, y, **kwargs)
            holdout = kwargs.get("validation_data")
            calls.append({"X": X.detach().cpu().numpy().copy(), "y": y.detach().cpu().numpy().copy(),
                          "validation": holdout, "epochs": kwargs["epochs"], "best": model.best_epoch_})
            return model
        with tempfile.TemporaryDirectory() as root, patch.object(experiments, "train_classifier", side_effect=traced):
            artifacts = RunArtifacts(root, {"random_state": 19})
            run_fold(dataset, {"mlp": MLPClassifierAdapter(epochs=3, device="cpu", random_state=19)},
                     train, test, 1, 2, 19, "cpu", .25, artifacts)
            config = read_json(artifacts.fold_path(dataset["name"], 1) / "mlp" / "configuration.json")
        selection = [call for call in calls if call["validation"] is not None]
        final = [call for call in calls if call["validation"] is None]
        self.assertGreaterEqual(len(selection), 2)
        for call in selection:
            X_validation, y_validation = call["validation"]
            np.testing.assert_array_equal(y_validation.numpy(), dataset["y"][validation])
            self.assertEqual(len(X_validation), len(validation))
            self.assertFalse(np.any(np.all(call["X"][:, None, :] == X_validation.numpy()[None, :, :], axis=2)))
        self.assertEqual(len(final[0]["y"]), n_train)
        self.assertEqual(final[0]["epochs"], config["baseline_epochs"])
        self.assertEqual(final[-1]["epochs"], config["final_epochs"])
        for call, epochs in zip(final[1:-1], config["replayed_round_epochs"]):
            self.assertEqual(call["epochs"], epochs)
        self.assertTrue(all(call["best"] == call["epochs"] for call in final))
        self.assertEqual(len(selection[0]["y"]), len(fit))

    def test_outer_labels_and_features_cannot_change_inner_selection(self):
        dataset, n_train = small_dataset()
        train, test = np.arange(n_train), np.arange(n_train, len(dataset["y"]))
        changed = {**dataset, "X": dataset["X"].copy(), "y": dataset["y"].copy()}
        changed["X"].iloc[test] += 1000
        changed["y"][test] = 1 - changed["y"][test]
        with tempfile.TemporaryDirectory() as root:
            configs, inner = [], []
            for data in (dataset, changed):
                artifacts = RunArtifacts(root, {"random_state": 19})
                run_fold(data, {"mlp": MLPClassifierAdapter(epochs=3, device="cpu", random_state=19)},
                         train, test, 1, 2, 19, "cpu", .25, artifacts)
                path = artifacts.fold_path(dataset["name"], 1) / "mlp"
                configs.append(read_json(path / "configuration.json"))
                inner.append(read_json(path / "inner_provenance.json"))
            self.assertEqual(configs[0], configs[1])
            self.assertEqual(inner[0], inner[1])

    def test_reproducible_predictions_and_recoverable_source_without_overwrite(self):
        dataset, n_train = small_dataset()
        train, test = np.arange(n_train), np.arange(n_train, len(dataset["y"]))
        with tempfile.TemporaryDirectory() as root:
            baseline = Path(root) / "historical.csv"
            baseline.write_text("unchanged baseline\n")
            runs = []
            for _ in range(2):
                artifacts = RunArtifacts(root, {"random_state": 19, "device": "cpu"})
                artifacts.dataset(dataset)
                rows = run_fold(dataset, {"lr": LogisticRegression(max_iter=100, random_state=19)},
                                train, test, 1, 2, 19, "cpu", .25, artifacts)
                artifacts.finish(rows)
                runs.append(artifacts)
            self.assertNotEqual(runs[0].path, runs[1].path)
            self.assertEqual(baseline.read_text(), "unchanged baseline\n")
            for strategy in ("baseline", "ugo"):
                first = runs[0].fold_path(dataset["name"], 1) / "lr" / f"{strategy}_predictions.csv"
                second = runs[1].fold_path(dataset["name"], 1) / "lr" / f"{strategy}_predictions.csv"
                self.assertEqual(first.read_bytes(), second.read_bytes())
            manifests = [read_json(run.path / "manifest.json") for run in runs]
            self.assertEqual(manifests[0]["datasets"], manifests[1]["datasets"])
            with zipfile.ZipFile(runs[0].path / "source.zip") as snapshot:
                for name, expected in manifests[0]["source_sha256"].items():
                    self.assertEqual(hashlib.sha256(snapshot.read(name)).hexdigest(), expected)
            registry = pd.read_csv(Path(root) / "experiment_registry.csv")
            self.assertEqual(len(registry), 4)
            self.assertEqual(set(registry.status), {"completed"})

    def test_multiclass_runner_does_not_invoke_binary_label_corruption(self):
        dataset, n_train = small_dataset()
        dataset["y"] = np.arange(len(dataset["y"]), dtype=np.int64) % 3
        rows = run_fold(dataset, {"lr": LogisticRegression(max_iter=100)},
                        np.arange(n_train), np.arange(n_train, len(dataset["y"])),
                        1, 1, 19, "cpu", .25)
        self.assertEqual({row["strategy"] for row in rows}, {"baseline", "ugo"})
        self.assertTrue(all(row["positive_class"] is None for row in rows))
        self.assertTrue(all(np.isfinite(row["pr_auc"]) for row in rows))


if __name__ == "__main__":
    unittest.main()
