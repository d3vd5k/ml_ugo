"""Real-data partitioning, fixed-epoch replay, and plain experiment artifacts."""
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4
import zipfile

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score)
from sklearn.preprocessing import label_binarize

from classifiers.mlp import train_classifier
from datasets.scripts.climb import categorical_columns, make_preprocessor
from datasets.scripts.synthetic import corrupt_labels
from samplers.ugo import UGOConfig, UGOPipeline


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _json_default(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                    default=_json_default) + "\n", encoding="utf-8")


def data_fingerprint(X, y, row_ids):
    frame = pd.DataFrame(X)
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update(str([(str(c), str(t)) for c, t in frame.dtypes.items()]).encode())
    digest.update(np.asarray(y).tobytes())
    digest.update(json.dumps(list(row_ids), default=_json_default).encode())
    return digest.hexdigest()


class RunArtifacts:
    """One unique directory per invocation; append-only registry at its parent."""
    def __init__(self, output_root, arguments):
        self.root = Path(output_root)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:12]
        self.path = self.root / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.arguments = dict(arguments)
        self.started = time.perf_counter()
        self.rows = []
        self.datasets = {}
        repo = Path(__file__).resolve().parent
        source = {}
        with zipfile.ZipFile(self.path / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as snapshot:
            for path in sorted(repo.rglob("*")):
                relative = path.relative_to(repo)
                if not path.is_file() or any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                    continue
                if "results" in relative.parts and relative.as_posix() != "results/published_targets.csv":
                    continue
                if path.suffix not in (".py", ".md", ".MD", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini") and relative.name not in ("requirements.txt", "climb_catalog.csv", "published_targets.csv"):
                    continue
                content = path.read_bytes()
                source[str(relative)] = hashlib.sha256(content).hexdigest()
                snapshot.writestr(str(relative), content)
        dependencies = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
        def git(*args):
            process = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
            return process.stdout if process.returncode == 0 else None
        self.manifest = {
            "run_id": self.run_id, "arguments": self.arguments,
            "command": sys.argv, "python": sys.version,
            "dependencies": dependencies, "source_sha256": source,
            "git_commit": git("rev-parse", "HEAD"), "git_diff": git("diff", "HEAD"),
            "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "datasets": self.datasets, "status": "running",
        }
        write_json(self.path / "manifest.json", self.manifest)

    def dataset(self, dataset):
        self.datasets[dataset["name"]] = {
            "fingerprint": data_fingerprint(dataset["X"], dataset["y"], dataset["row_ids"]),
            "metadata": dataset.get("metadata", {}),
            "rows": len(dataset["y"]),
        }
        write_json(self.path / "manifest.json", self.manifest)

    def fold_path(self, name, fold):
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        path = self.path / safe_name / f"fold-{fold}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def finish(self, rows, error=None):
        self.rows = rows
        runtime = time.perf_counter() - self.started
        status = "failed" if error is not None else "completed"
        self.manifest.update(status=status, runtime_seconds=runtime, error=error)
        write_json(self.path / "manifest.json", self.manifest)
        pd.DataFrame(rows).to_csv(self.path / "metrics.csv", index=False)
        registry = self.root / "experiment_registry.csv"
        fields = ["run_id", "variant", "dataset", "split", "seed", "configuration", "metrics", "runtime_seconds", "status"]
        with registry.open("a+", newline="", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0, os.SEEK_END)
            writer = csv.DictWriter(handle, fieldnames=fields)
            if handle.tell() == 0:
                writer.writeheader()
            for row in rows or [{}]:
                writer.writerow({
                    "run_id": self.run_id, "variant": row.get("strategy", ""),
                    "dataset": row.get("dataset", ""), "split": row.get("fold", ""),
                    "seed": row.get("seed", self.arguments.get("random_state", 42)),
                    "configuration": json.dumps({"arguments": self.arguments, "algorithm": row.get("algorithm")}, sort_keys=True),
                    "metrics": json.dumps(row, default=_json_default, sort_keys=True),
                    "runtime_seconds": row.get("runtime_seconds", runtime), "status": row.get("status", status),
                })
            fcntl.flock(handle, fcntl.LOCK_UN)


class MLPClassifierAdapter(ClassifierMixin, BaseEstimator):
    def __init__(self, epochs=160, warmup_epochs=100, patience=20, device="auto", random_state=42, lr=.001, hidden_dim=20):
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.lr = lr
        self.hidden_dim = hidden_dim

    def fit(self, X, y, validation_data=None, sample_weight=None):
        labels = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        self.classes_ = np.unique(labels)
        encoded_y = np.searchsorted(self.classes_, labels)
        self.n_features_in_ = X.shape[1]
        validation = None
        if validation_data is not None:
            validation_labels = validation_data[1].detach().cpu().numpy() if isinstance(validation_data[1], torch.Tensor) else np.asarray(validation_data[1])
            if not np.isin(validation_labels, self.classes_).all():
                raise ValueError("Validation labels must be present in training labels.")
            validation = (torch.as_tensor(validation_data[0], dtype=torch.float32),
                          torch.as_tensor(np.searchsorted(self.classes_, validation_labels), dtype=torch.long))
        self.model_ = train_classifier(
            torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(encoded_y, dtype=torch.long),
            epochs=self.epochs, warmup_epochs=self.warmup_epochs, patience=self.patience,
            random_state=self.random_state, device=self.device, validation_data=validation,
            sample_weight=sample_weight, lr=self.lr, hidden_dim=self.hidden_dim)
        self.best_epoch_ = self.model_.best_epoch_
        return self

    def predict_proba_tensor(self, X: torch.Tensor) -> torch.Tensor:
        device = next(self.model_.parameters()).device
        self.model_.eval()
        with torch.no_grad():
            return torch.softmax(self.model_(torch.as_tensor(X, dtype=torch.float32, device=device)), dim=1)

    def predict_proba(self, X):
        return self.predict_proba_tensor(X).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def metric_row(y_true, y_pred, probabilities, classes, positive_class, **metadata):
    binary = len(classes) == 2
    kwargs = {"average": "binary" if binary else "macro", "zero_division": 0}
    if binary:
        kwargs["pos_label"] = positive_class
        positive_index = np.flatnonzero(classes == positive_class)[0]
        ap = average_precision_score(np.asarray(y_true) == positive_class, probabilities[:, positive_index])
    else:
        ap = average_precision_score(label_binarize(y_true, classes=classes), probabilities, average="macro")
    return {**metadata, "positive_class": int(positive_class) if binary else None,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, **kwargs),
            "recall": recall_score(y_true, y_pred, **kwargs),
            "f1": f1_score(y_true, y_pred, **kwargs),
            "mcc": matthews_corrcoef(y_true, y_pred), "pr_auc": ap}


def real_inner_split(train_indices, y, random_state, validation_fraction=.15):
    """Real-only proportional holdout with at least one row/class on each side."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    train_indices = np.asarray(train_indices)
    labels = np.asarray(y)[train_indices]
    classes, counts = np.unique(labels, return_counts=True)
    if np.any(counts < 2):
        raise ValueError("Real inner validation requires at least two outer-training observations per class.")
    ideal = counts * validation_fraction
    quotas = np.clip(np.floor(ideal).astype(int), 1, counts - 1)
    target = int(np.clip(round(len(labels) * validation_fraction), len(classes), len(labels) - len(classes)))
    while quotas.sum() < target:
        index = np.argmax(np.where(quotas < counts - 1, ideal - quotas, -np.inf))
        quotas[index] += 1
    while quotas.sum() > target:
        index = np.argmax(np.where(quotas > 1, quotas - ideal, -np.inf))
        quotas[index] -= 1
    rng = np.random.default_rng(random_state)
    fit, validation = [], []
    for label, quota in zip(classes, quotas):
        indices = rng.permutation(train_indices[labels == label])
        validation.extend(indices[:quota])
        fit.extend(indices[quota:])
    return rng.permutation(fit), rng.permutation(validation)


def _array(tensor):
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)


def _fit(estimator, X, y, validation=None, epochs=None):
    model = clone(estimator)
    if isinstance(model, MLPClassifierAdapter):
        if epochs is not None:
            model.set_params(epochs=int(epochs))
        return model.fit(X, y, validation_data=validation)
    return model.fit(_array(X), _array(y))


def _preprocessing_state(preprocessor):
    state = {"features": preprocessor.get_feature_names_out().tolist()}
    for name, transformer, columns in preprocessor.named_steps["columns"].transformers_:
        if hasattr(transformer, "named_steps"):
            state[name] = {"columns": list(columns)}
            for step_name, step in transformer.named_steps.items():
                state[name][step_name] = {key: getattr(step, key) for key in
                    ("statistics_", "categories_", "mean_", "scale_", "var_") if hasattr(step, key)}
    return state


def run_fold(dataset, estimators, train_indices, test_indices, fold, max_iterations,
             random_state, device="auto", validation_fraction=.15, artifacts=None,
             noise_ratio=0., show_plots=False):
    """Tune on real inner holdout, then replay fixed epochs on full outer training."""
    seed_everything(random_state)
    X = pd.DataFrame(dataset["X"])
    y = np.asarray(dataset["y"], dtype=np.int64)
    row_ids = np.asarray(dataset.get("row_ids", np.arange(len(y))))
    fit_indices, validation_indices = real_inner_split(train_indices, y, random_state, validation_fraction)
    training_y = y.copy()
    if noise_ratio != 0:
        training_y[fit_indices] = corrupt_labels(y[fit_indices], noise_ratio, random_state)
    # Validation rejoins final outer refit clean, only after all epoch choices freeze.
    inner_preprocessor = make_preprocessor(X.iloc[fit_indices])
    X_fit = inner_preprocessor.fit_transform(X.iloc[fit_indices]).astype(np.float32)
    X_validation = inner_preprocessor.transform(X.iloc[validation_indices]).astype(np.float32)
    outer_preprocessor = make_preprocessor(X.iloc[train_indices])
    X_outer = outer_preprocessor.fit_transform(X.iloc[train_indices]).astype(np.float32)
    X_test = outer_preprocessor.transform(X.iloc[test_indices]).astype(np.float32)
    validation = (X_validation, y[validation_indices])
    classes, counts = np.unique(y[train_indices], return_counts=True)
    positive_class = classes[np.argmin(counts)]
    path = None if artifacts is None else artifacts.fold_path(dataset["name"], fold)
    if path is not None:
        roles = np.full(len(y), "unused", dtype=object)
        roles[fit_indices], roles[validation_indices], roles[test_indices] = "inner_fit", "inner_validation", "outer_test"
        pd.DataFrame({"row_id": row_ids, "role": roles, "clean_label": y,
                      "training_label": training_y}).to_csv(path / "splits.csv", index=False)
        write_json(path / "preprocessing.json", {
            "inner": _preprocessing_state(inner_preprocessor),
            "outer": _preprocessing_state(outer_preprocessor)})
    results = []
    for algorithm, estimator in estimators.items():
        method_path = None if path is None else path / algorithm
        if method_path is not None:
            method_path.mkdir(exist_ok=False)
        started = time.perf_counter()
        selected_baseline = _fit(estimator, X_fit, training_y[fit_indices], validation)
        baseline_epochs = getattr(selected_baseline, "best_epoch_", None)
        baseline = _fit(estimator, X_outer, training_y[train_indices], epochs=baseline_epochs)
        models = [("baseline", baseline, time.perf_counter() - started)]
        started = time.perf_counter()
        selected_epochs = []
        validation_metrics = []
        def select_round(X_current, y_current):
            model = _fit(estimator, X_current, y_current, validation)
            selected_epochs.append(getattr(model, "best_epoch_", None))
            validation_probabilities = model.predict_proba(X_validation)
            validation_predictions = model.classes_[validation_probabilities.argmax(axis=1)]
            validation_metrics.append(metric_row(y[validation_indices], validation_predictions, validation_probabilities, model.classes_, positive_class, round=len(selected_epochs) - 1, selected_epochs=selected_epochs[-1]))
            return model
        inner_config = UGOConfig(max_iterations=max_iterations, random_state=random_state,
                                 device=device, categorical_columns=categorical_columns(inner_preprocessor))
        inner_result = UGOPipeline(inner_config).run(torch.from_numpy(X_fit),
                            torch.from_numpy(training_y[fit_indices]), select_round)
        selected_final = _fit(estimator, inner_result.X, inner_result.y, validation)
        final_epochs = getattr(selected_final, "best_epoch_", None)
        replay_epochs = []
        def replay_round(X_current, y_current):
            # Original stopping can yield a different number of rounds on refit.
            # Reuse the last inner-selected epoch count; never retune on outer data.
            epoch = selected_epochs[min(len(replay_epochs), len(selected_epochs) - 1)] if selected_epochs else final_epochs
            replay_epochs.append(epoch)
            return _fit(estimator, X_current, y_current, epochs=epoch)
        outer_config = UGOConfig(max_iterations=max_iterations, random_state=random_state,
                                 device=device, categorical_columns=categorical_columns(outer_preprocessor))
        outer_result = UGOPipeline(outer_config).run(torch.from_numpy(X_outer),
                            torch.from_numpy(training_y[train_indices]), replay_round)
        ugo = _fit(estimator, outer_result.X, outer_result.y, epochs=final_epochs)
        models.append(("ugo", ugo, time.perf_counter() - started))
        if method_path is not None:
            write_json(method_path / "configuration.json", {
                "estimator": estimator.get_params(deep=True),
                "inner_sampler": vars(inner_config), "outer_sampler": vars(outer_config),
                "baseline_epochs": baseline_epochs, "selected_round_epochs": selected_epochs,
                "replayed_round_epochs": replay_epochs, "final_epochs": final_epochs,
                "positive_class": positive_class})
            for phase, result, reference in (("inner", inner_result, fit_indices), ("outer", outer_result, train_indices)):
                write_json(method_path / f"{phase}_rounds.json", result.rounds)
                write_json(method_path / f"{phase}_stop.json", {"stop_reason": result.stop_reason, "support_summary": result.support_summary, "selection_validation_metrics": validation_metrics if phase == "inner" else None})
                provenance = []
                for item in result.provenance:
                    item = dict(item)
                    for key in ("seed_index", "partner_index", "relocation_index"):
                        index = item.get(key)
                        item[key.replace("index", "row_id")] = None if index is None or index < 0 else row_ids[reference[index]]
                    provenance.append(item)
                write_json(method_path / f"{phase}_provenance.json", provenance)
        for strategy, model, runtime in models:
            probabilities = model.predict_proba(X_test)
            predicted = model.classes_[probabilities.argmax(axis=1)]
            row = metric_row(y[test_indices], predicted, probabilities, model.classes_, positive_class,
                             dataset=dataset["name"], algorithm=algorithm, strategy=strategy,
                             fold=fold, runtime_seconds=runtime)
            results.append(row)
            if method_path is not None:
                predictions = pd.DataFrame({"row_id": row_ids[test_indices], "y_true": y[test_indices], "y_pred": predicted})
                for i, label in enumerate(model.classes_):
                    predictions[f"probability_{label}"] = probabilities[:, i]
                predictions.to_csv(method_path / f"{strategy}_predictions.csv", index=False)
        if show_plots:
            from eval.plot import plot_datasets
            plot_datasets(torch.from_numpy(X_outer), torch.from_numpy(training_y[train_indices]),
                          outer_result.X.cpu(), outer_result.y.cpu())
    return results
