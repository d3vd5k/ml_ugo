"""Named benchmark setup and controlled evaluation; smoke is never a study result."""
import argparse
import hashlib
import json
from dataclasses import asdict
from contextlib import contextmanager
import threading
import time

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from benchmarks.protocols import load_dataset, load_protocol, make_splits
from benchmarks.results import (BenchmarkArtifacts, choose_threshold, enforce_stage_gate,
                                load_study, metric_results, require_variant)
from benchmarks.scoring import oof_mma, prepare_oof_partitions
from classifiers.factory import (candidate_parameters, fit_estimator, make_estimator,
                                 supports_sample_weight)
from datasets.scripts.climb import categorical_columns, make_preprocessor
from experiments import _preprocessing_state, data_fingerprint, real_inner_split, seed_everything, write_json
from samplers.controls import resample
from samplers.geometry import prepare_constraint_geometry
from samplers.protected import group_configuration, prepare_protected_groups
from samplers.stopping import stopping_budget, stopping_configuration
from samplers.hdbscan_support import prepare_h1_support


@contextmanager
def measure_memory(device):
    """Sample process RSS; CUDA counters cover Torch allocations, not other GPU libraries."""
    process = psutil.Process()
    stats = {"peak_rss_bytes": process.memory_info().rss, "rss_sample_interval_seconds": .01,
             "memory_scope": "cell process RSS sampled at 10ms; Torch CUDA allocator only"}
    stopped = threading.Event()
    cuda = device != "cpu" and torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def sample():
        while not stopped.wait(.01):
            stats["peak_rss_bytes"] = max(stats["peak_rss_bytes"], process.memory_info().rss)

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        yield stats
    finally:
        stopped.set()
        worker.join()
        stats["peak_rss_bytes"] = max(stats["peak_rss_bytes"], process.memory_info().rss)
        if cuda:
            torch.cuda.synchronize()
            stats["cuda_peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
            stats["cuda_peak_torch_reserved_bytes"] = torch.cuda.max_memory_reserved()


def as_array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def class_counts(labels):
    labels, counts = np.unique(as_array(labels), return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def augmentation_targets(labels, augmentation):
    """Prespecified total allocated by real class deficits; never based on test data."""
    if augmentation["mode"] == "native":
        return None
    counts = class_counts(labels)
    total = int(augmentation["total"])
    if total < 0:
        raise ValueError("Augmentation total must be nonnegative.")
    deficits = {label: max(counts.values()) - count for label, count in counts.items()}
    available = sum(deficits.values())
    budget = min(total, available)
    if available == 0:
        return counts
    raw = {label: budget * deficit / available for label, deficit in deficits.items()}
    allocations = {label: int(np.floor(value)) for label, value in raw.items()}
    remaining = budget - sum(allocations.values())
    order = sorted(counts, key=lambda label: (-(raw[label] - allocations[label]), label))
    for label in order[:remaining]:
        allocations[label] += 1
    return {label: count + allocations[label] for label, count in counts.items()}


def split_fingerprint(dataset, split):
    digest = hashlib.sha256()
    for role in ("train_indices", "fit_indices", "validation_indices", "test_indices"):
        identities = np.asarray(dataset["row_ids"])[split[role]].tolist()
        digest.update(json.dumps({role: identities}, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def smoke_split(dataset, split, seed, train_rows, test_rows, validation_fraction):
    """Every smoke observation comes from outer training; official test is untouched."""
    available = np.asarray(split["train_indices"])
    if train_rows + test_rows >= len(available):
        raise ValueError("Smoke sizes must leave unused outer-training rows.")
    chosen, _ = train_test_split(available, train_size=train_rows + test_rows,
                                stratify=dataset["y"][available], random_state=seed)
    train, test = train_test_split(chosen, train_size=train_rows,
                                  stratify=dataset["y"][chosen], random_state=seed)
    fit, validation = real_inner_split(train, dataset["y"], seed, validation_fraction)
    return {"fold": split["fold"], "train_indices": train, "test_indices": test,
            "fit_indices": fit, "validation_indices": validation}


def candidate_configs(protocol, algorithm, seed, budget):
    explicit = protocol.get("candidate_parameters", {}).get(algorithm)
    if explicit is not None:
        if len(explicit) < budget:
            raise ValueError(f"{algorithm}: requested {budget} candidates, only {len(explicit)} declared.")
        proposals = explicit[:budget]
    else:
        proposals = candidate_parameters(algorithm, protocol["tuning_budget"].get("sampler_seed", seed), budget,
                                        search_space=protocol.get("search_space", {}).get(algorithm))
    base = protocol.get("classifier_parameters", {}).get(algorithm, {})
    return [{**base, **parameters} for parameters in proposals]


def categorical_groups(preprocessor):
    columns = preprocessor.named_steps["columns"]
    encoder = columns.named_transformers_["categorical"].named_steps["encode"]
    position = columns.output_indices_["categorical"].start
    groups = []
    for categories in getattr(encoder, "categories_", ()):
        groups.append(tuple(range(position, position + len(categories))))
        position += len(categories)
    return tuple(groups)


def prepare_partitions(dataset, split):
    """Fit each preprocessing state once; no held-out features enter fitted statistics."""
    frame = dataset["X"]
    inner = make_preprocessor(frame.iloc[split["fit_indices"]])
    outer = make_preprocessor(frame.iloc[split["train_indices"]])
    return {"inner": inner, "outer": outer,
            "X_fit": inner.fit_transform(frame.iloc[split["fit_indices"]]).astype(np.float32),
            "X_validation": inner.transform(frame.iloc[split["validation_indices"]]).astype(np.float32),
            "X_train": outer.fit_transform(frame.iloc[split["train_indices"]]).astype(np.float32),
            "X_test": outer.transform(frame.iloc[split["test_indices"]]).astype(np.float32)}


def weights_for(method, labels):
    return compute_sample_weight("balanced", as_array(labels)) if method == "class_weight" else None


def select_score(metrics, primary_metric):
    value = metrics["primary_metric_value"]
    return -value if primary_metric == "aps_cost" else value


def _select_stopping(algorithm, parameters, X, y, categorical, seed, settings, validation,
                     groups, protocol, positive_class):
    """Select real-validation checkpoints; outer refits never enter this path."""
    method = "ugo_stopping_only"
    values = dict(settings["method_parameters"].get(method, {}))
    policy, fractions, _, _ = stopping_configuration(values)
    values["budget_fraction"] = 1.
    targets = augmentation_targets(y, settings["augmentation"])
    maximum, _, _, _ = stopping_budget(y, targets, values)
    # Small caps may map several fractions to the same integer budget.
    choices = {}
    for fraction in fractions:
        choices.setdefault(int(np.floor(maximum * fraction)), fraction)
    checkpoints, best, total_fits = [], None, 0

    def record(model, requested, actual, sequence):
        nonlocal best
        probabilities = model.predict_proba(validation[0])
        column = np.flatnonzero(model.classes_ == positive_class)[0]
        threshold = (choose_threshold(validation[1] == positive_class, probabilities[:, column],
                                      protocol["threshold"]) if len(model.classes_) == 2 else .5)
        metrics = metric_results(validation[1], probabilities, model.classes_, positive_class,
                                 threshold, protocol["primary_metric"], protocol["threshold"].get("fpr", .05))
        score = select_score(metrics, protocol["primary_metric"])
        item = {"fraction": choices[requested], "requested_count": requested, "actual_count": actual,
                "score": score, "threshold": threshold, "metrics": metrics,
                "epoch_sequence": list(sequence), "classes": model.classes_,
                "validation_probabilities": probabilities}
        checkpoints.append(item)
        key = (score, -actual, -requested)
        if best is None or key > best[0]:
            best = (key, model, item)
        return item

    def candidate_settings(fraction):
        return {**settings, "method_parameters": {**settings["method_parameters"],
                method: {**values, "budget_fraction": fraction}}}

    executions = []
    if policy == "fixed_grid":
        selected_audit = None
        for requested, fraction in choices.items():
            model, audit = train_method(algorithm, method, parameters, X, y, categorical, seed,
                                        candidate_settings(fraction), validation, groups=groups)
            total_fits += audit["classifier_fit_count"]
            previous = best
            record(model, requested, audit["augmentation"]["actual_count"], audit["epoch_sequence"])
            executions.append({"fraction": fraction, "audit": audit})
            if best is not previous:
                selected_audit = audit
        searched = sum(item["audit"]["augmentation"]["actual_count"] for item in executions)
    else:
        seen = set()
        def observe(model, actual, sequence, terminal):
            if actual in seen:
                return
            if actual in choices:
                requested = actual
            elif terminal:
                requested = next(count for count in choices if count >= actual)
            else:
                return
            seen.add(actual)
            record(model, requested, actual, sequence)
        _, trajectory = train_method(algorithm, method, parameters, X, y, categorical, seed,
                                      candidate_settings(1.), validation, groups=groups,
                                      checkpoint_callback=observe)
        total_fits = trajectory["classifier_fit_count"]
        searched = trajectory["augmentation"]["actual_count"]
        executions.append({"fraction": 1., "audit": trajectory})
        item = best[2]
        actual = item["actual_count"]
        selected_values = {**values, "budget_fraction": item["fraction"]}
        _, requested, counts, selected_targets = stopping_budget(y, targets, selected_values)
        provenance = [row for row in trajectory["provenance"] if row["row_index"] < len(y) + actual]
        actual_counts = class_counts(y)
        for row in provenance:
            actual_counts[row["class"]] += 1
        summary = {**trajectory["support_summary"], "generation_cap": min(
                   trajectory["support_summary"]["generation_cap"], requested)}
        summary["stopping"] = {**summary["stopping"], "budget_fraction": item["fraction"],
                               "requested_count": requested, "checkpoint_counts": counts,
                               "per_class_caps": selected_targets}
        if "checkpoint_class_caps" in summary["stopping"]:
            summary["stopping"]["checkpoint_class_caps"] = {
                count: caps for count, caps in summary["stopping"]["checkpoint_class_caps"].items()
                if int(count) <= requested}
        if selected_targets is not None:
            summary.update(target_counts=selected_targets, actual_counts=actual_counts)
        selected_audit = {**trajectory, "sampler_parameters": selected_values,
                          "support_summary": summary, "provenance": provenance,
                          "rounds": [row for row in trajectory["rounds"]
                                     if row.get("accepted", 0) and row.get("total_generated", 0) <= actual],
                          "epoch_sequence": item["epoch_sequence"],
                          "stop_reason": "selected_inner_checkpoint",
                          "augmentation": {**trajectory["augmentation"], "requested_count": requested,
                                           "actual_count": actual, "actual_counts": actual_counts,
                                           "underspend": requested - actual}}
    selected_audit = {**selected_audit}
    selected_audit.pop("stopping_replay", None)
    selected_audit["selected_procedure_fit_count"] = len(best[2]["epoch_sequence"])
    selected_audit["classifier_fit_count"] = total_fits
    selected_audit["stopping_selection"] = {
        "policy": policy, "objective": protocol["primary_metric"], "maximum_count": maximum,
        "selected_fraction": best[2]["fraction"], "selected_requested_count": best[2]["requested_count"],
        "selected_actual_count": best[2]["actual_count"], "generated_count_searched": searched,
        "fit_count": total_fits, "checkpoints": checkpoints, "executions": executions}
    return best[1], selected_audit



def train_method(algorithm, method, parameters, X, y, categorical, seed, settings,
                 validation=None, replay_epochs=None, groups=(), oof=None, group_partition=None,
                 constraint_geometry=None, stopping_objective=None, positive_class=None, checkpoint_callback=None,
                 h1_support=None, flow_support=None):
    if method == "ugo_stopping_only" and stopping_objective is not None:
        if validation is None:
            raise ValueError("U4 selection requires real inner validation")
        return _select_stopping(algorithm, parameters, X, y, categorical, seed, settings, validation,
                                groups, stopping_objective, positive_class)
    estimator = make_estimator(algorithm, seed, parameters=parameters,
                               device=settings["device"], mlp_epochs=settings["mlp_epochs"])
    epochs = []
    latest_model, latest_count = None, None

    def train_round(features, labels):
        nonlocal latest_model, latest_count
        epoch = None
        if replay_epochs:
            epoch = replay_epochs[min(len(epochs), len(replay_epochs) - 1)]
        model = fit_estimator(estimator, features, labels,
                              sample_weight=weights_for(method, labels),
                              validation_data=validation, epochs=epoch)
        epochs.append(getattr(model, "best_epoch_", None))
        latest_model, latest_count = model, len(labels)
        if checkpoint_callback is not None:
            checkpoint_callback(model, len(labels) - len(y), epochs, False)
        return model

    targets = augmentation_targets(y, settings["augmentation"])
    sampler_parameters = dict(settings["method_parameters"].get(method, {}))
    scoring_audit = None
    seed_score = None
    if method in ("ugo_oof_mma", "ugo_combined", "ugo_combined_f1"):
        if oof is None:
            raise ValueError("OOF MMA requires raw real-training cross-fit partitions")
        scores, scoring_audit = oof_mma(oof, as_array(y), algorithm, parameters, seed, settings)
        score_tensor = None

        def seed_score(model, features, labels):
            nonlocal score_tensor
            if score_tensor is None:
                score_tensor = torch.as_tensor(scores, device=features.device, dtype=features.dtype)
            return score_tensor

    if method == "ugo_original" and categorical:
        # Explicit controlled-track deviation; numeric original remains available independently.
        sampler_parameters["categorical_projection"] = True
    elif method not in ("baseline", "class_weight", "ugo_corrected", "ugo_oof_mma", "ugo_groups_only", "ugo_constraints_only", "ugo_stopping_only", "ugo_combined", "ugo_h1", "ugo_f1", "ugo_f2", "ugo_n_reject", "ugo_combined_f1") and groups:
        sampler_parameters["categorical_groups"] = groups
    sampled = resample(method, X, y, seed, categorical_columns=categorical,
                       max_iterations=settings["ugo_max_iterations"], device=settings["device"],
                       model_fn=train_round, target_counts=targets, parameters=sampler_parameters,
                       seed_score=seed_score, group_partition=group_partition,
                       categorical_groups=groups if method == "ugo_constraints_only" else (),
                       constraint_geometry=constraint_geometry, h1_support=h1_support, flow_support=flow_support)
    reuse_unchanged = method in ("ugo_h1", "ugo_f1", "ugo_f2", "ugo_n_reject", "ugo_combined_f1") and latest_count == len(sampled.y)
    model = latest_model if reuse_unchanged else train_round(sampled.X, sampled.y)
    if checkpoint_callback is not None:
        checkpoint_callback(model, len(sampled.y) - len(y), epochs, True)
    counts = class_counts(sampled.y)
    before = class_counts(y)
    actual = len(sampled.y) - len(y)
    requested = (0 if method in ("baseline", "class_weight") else
                 (sum(targets.values()) - len(y)) if targets is not None else None)
    if method == "ugo_stopping_only":
        requested = sampled.support_summary["stopping"]["requested_count"]
    audit = {"epoch_sequence": epochs, "parameters": parameters, "sampler_parameters": sampler_parameters,
             "stop_reason": sampled.stop_reason, "rounds": sampled.rounds,
             "provenance": sampled.provenance, "support_summary": sampled.support_summary,
             "augmentation": {"mode": settings["augmentation"]["mode"], "requested_count": requested,
                              "actual_count": actual, "real_counts": before, "actual_counts": counts,
                              "underspend": None if requested is None else requested - actual}}
    audit["classifier_fit_count"] = len(epochs) + (len(scoring_audit["folds"]) if scoring_audit is not None else 0)
    if method in ("ugo_h1", "ugo_f1", "ugo_f2", "ugo_n_reject", "ugo_combined_f1"):
        audit["unchanged_final_fit_skipped"] = reuse_unchanged
    if method == "ugo_stopping_only":
        audit["stopping_replay"] = sampled.support_summary["stopping"]
    if scoring_audit is not None:
        audit["seed_scoring"] = scoring_audit
    return model, audit


def evaluate_cell(dataset, split, algorithm, method, variant_id, candidates, settings, protocol, artifacts, metadata, partitions=None):
    if method == "ugo_f2":
        from benchmarks.f2 import evaluate_f2_cell
        return evaluate_f2_cell(dataset, split, algorithm, variant_id, candidates, settings,
                                protocol, artifacts, metadata, partitions)
    if method == "ugo_h1":
        from benchmarks.h1 import evaluate_h1_cell
        return evaluate_h1_cell(dataset, split, algorithm, variant_id, candidates, settings,
                                protocol, artifacts, metadata, partitions)
    if method in ("ugo_f1", "ugo_n_reject", "ugo_combined_f1"):
        from benchmarks.f1 import evaluate_f1_cell
        return evaluate_f1_cell(dataset, split, algorithm, method, variant_id, candidates,
                                settings, protocol, artifacts, metadata, partitions)
    started = time.perf_counter()
    partitions = partitions if partitions is not None else prepare_partitions(dataset, split)
    labels = dataset["y"]
    fit, validation, train, test = (split[key] for key in
                                   ("fit_indices", "validation_indices", "train_indices", "test_indices"))
    variant_parameters = metadata.get("sampler_parameters", {})
    search = metadata.get("sampler_search", {})
    if search:
        if method in ("ugo_groups_only", "ugo_combined") and isinstance(search, dict) and set(search) == {"group_alpha"}:
            if not search["group_alpha"] or any(not np.isfinite(value) or value < 0 for value in search["group_alpha"]):
                raise ValueError("U2/U5 group_alpha search requires finite nonnegative values")
        elif method == "ugo_constraints_only" and isinstance(search, dict) and set(search) == {"constraint_settings"}:
            pairs = search["constraint_settings"]
            if (not isinstance(pairs, list) or not pairs
                    or any(not isinstance(pair, dict) or set(pair) != {"eta", "rho"}
                           or any(isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not np.isfinite(value) or value <= 0 for value in pair.values())
                           for pair in pairs)):
                raise ValueError("U3 constraint_settings requires eta/rho dictionaries with finite positive values")
        else:
            raise ValueError("Component search supports only U2/U5 group_alpha or U3 constraint_settings")
    if variant_parameters or search:
        settings = {**settings, "method_parameters": {**settings["method_parameters"],
                    method: {**settings["method_parameters"].get(method, {}), **variant_parameters}}}

    def real_groups(role, indices, features):
        if method not in ("ugo_groups_only", "ugo_combined"):
            return None
        values = settings["method_parameters"].get(method, {})
        key = (role, json.dumps(asdict(group_configuration(values)), sort_keys=True))
        cache = partitions.setdefault("protected_groups", {})
        if key not in cache:
            cache[key] = prepare_protected_groups(features, labels[indices], metadata["seed"],
                                                 augmentation_targets(labels[indices], settings["augmentation"]), values)
            cache[key]["real_row_ids"] = dataset["row_ids"][indices]
        return cache[key]

    def real_geometry(role, indices, features, preprocessor):
        if method != "ugo_constraints_only":
            return None
        values = settings["method_parameters"].get(method, {})
        categorical = categorical_columns(preprocessor)
        blocks = categorical_groups(preprocessor)
        geometry_parameters = {"graph_neighbors": values.get("graph_neighbors", 5),
                               "graph_radius_quantile": values.get("graph_radius_quantile", .9)}
        targets = augmentation_targets(labels[indices], settings["augmentation"])
        target_classes = tuple(int(label) for label in np.unique(labels[indices])
                               if targets is None or targets[int(label)] > int((labels[indices] == label).sum()))
        key = (role, json.dumps(geometry_parameters, sort_keys=True), tuple(categorical), blocks, target_classes)
        cache = partitions.setdefault("constraint_geometry", {})
        if key not in cache:
            cache[key] = prepare_constraint_geometry(
                features, labels[indices], categorical_columns=categorical, categorical_groups=blocks,
                parameters=geometry_parameters,
                targets=targets)
            cache[key].real_row_ids = dataset["row_ids"][indices]
        return cache[key]

    if method in ("ugo_oof_mma", "ugo_combined", "ugo_combined_f1"):
        for role, indices in (("inner_oof", fit), ("outer_oof", train)):
            if role not in partitions:
                partitions[role] = prepare_oof_partitions(
                    dataset["X"].iloc[indices], labels[indices],
                    dataset["row_ids"][indices], metadata["seed"])
    classes = np.unique(labels[train])
    positive_class = int(protocol.get("positive_class", classes[np.argmin(np.unique(labels[train], return_counts=True)[1])]))
    inner = partitions["inner"]
    X_fit, X_validation = partitions["X_fit"], partitions["X_validation"]
    real_validation = (X_validation, labels[validation])
    folder = artifacts.fold_path(dataset["name"], split["fold"]) / f"seed-{metadata['seed']}" / algorithm / variant_id
    folder.mkdir(parents=True, exist_ok=False)
    inner_groups = real_groups("inner", fit, X_fit)
    inner_geometry = real_geometry("inner", fit, X_fit, inner)
    best = None
    if search:
        proposals = (search["constraint_settings"] if method == "ugo_constraints_only" else
                     [{"group_alpha": value} for value in search["group_alpha"]])
    for candidate_id, parameters in enumerate(candidates):
        candidate_settings = settings
        if search:
            values = {**settings["method_parameters"].get(method, {}),
                      **proposals[candidate_id % len(proposals)]}
            candidate_settings = {**settings, "method_parameters": {**settings["method_parameters"], method: values}}
        model, audit = train_method(algorithm, method, parameters, X_fit, labels[fit],
                                    categorical_columns(inner), metadata["seed"], candidate_settings, real_validation,
                                    groups=categorical_groups(inner), oof=partitions.get("inner_oof"),
                                    group_partition=inner_groups, constraint_geometry=inner_geometry,
                                    stopping_objective=protocol if method == "ugo_stopping_only" else None,
                                    positive_class=positive_class)
        probabilities = model.predict_proba(X_validation)
        threshold = choose_threshold(labels[validation] == positive_class,
                                     probabilities[:, np.flatnonzero(model.classes_ == positive_class)[0]],
                                     protocol["threshold"]) if len(classes) == 2 else .5
        metrics = metric_results(labels[validation], probabilities, model.classes_, positive_class,
                                 threshold, protocol["primary_metric"], protocol["threshold"].get("fpr", .05))
        score = select_score(metrics, protocol["primary_metric"])
        record = {"candidate_id": candidate_id, "threshold": threshold, "metrics": metrics, **audit}
        write_json(folder / f"candidate-{candidate_id}.json", record)
        key = (score, -audit["augmentation"]["actual_count"],
               -audit["augmentation"]["requested_count"]) if method == "ugo_stopping_only" else (score,)
        if best is None or key > best[0]:
            best = (key, candidate_id, parameters, threshold, audit)
    _, selected_id, parameters, threshold, selected_audit = best
    # Only final refit consumes full-training preprocessing; no test feedback selects parameters.
    outer = partitions["outer"]
    X_train, X_test = partitions["X_train"], partitions["X_test"]
    outer_settings = settings
    if method in ("ugo_groups_only", "ugo_constraints_only", "ugo_stopping_only", "ugo_combined"):
        outer_settings = {**settings, "method_parameters": {**settings["method_parameters"],
                         method: selected_audit["sampler_parameters"]}}
    model, outer_audit = train_method(algorithm, method, parameters, X_train, labels[train],
                                     categorical_columns(outer), metadata["seed"], outer_settings,
                                     replay_epochs=selected_audit["epoch_sequence"], groups=categorical_groups(outer),
                                     oof=partitions.get("outer_oof"),
                                     group_partition=real_groups("outer", train, X_train),
                                     constraint_geometry=real_geometry("outer", train, X_train, outer))
    probabilities = model.predict_proba(X_test)
    metrics = metric_results(labels[test], probabilities, model.classes_, positive_class, threshold,
                             protocol["primary_metric"], protocol["threshold"].get("fpr", .05))
    write_json(folder / "preprocessing.json", {"inner": _preprocessing_state(inner), "outer": _preprocessing_state(outer)})
    write_json(folder / "selection.json", {"selected_candidate": selected_id, "threshold": threshold,
                                           "candidate_configs": candidates, "inner": selected_audit})
    write_json(folder / "outer.json", outer_audit)
    # Original ancestry can reference descendants; preserve local indices with explicit partition mapping.
    write_json(folder / "reference_rows.json", {"inner_real_row_ids": dataset["row_ids"][fit],
                                               "outer_real_row_ids": dataset["row_ids"][train]})
    predicted = (np.where(probabilities[:, np.flatnonzero(model.classes_ == positive_class)[0]] >= threshold,
                          positive_class, classes[classes != positive_class][0]) if len(classes) == 2
                 else model.classes_[probabilities.argmax(axis=1)])
    predictions = pd.DataFrame({"row_id": dataset["row_ids"][test], "y_true": labels[test], "y_pred": predicted})
    for column, label in enumerate(model.classes_):
        predictions[f"probability_{label}"] = probabilities[:, column]
    predictions.to_csv(folder / "predictions.csv", index=False)
    return {**metadata, **metrics, "status": "completed", "strategy": method, "fold": split["fold"],
            "threshold": threshold, "positive_class": positive_class, "selected_candidate": selected_id,
            "candidate_configs": candidates, "tuning_budget": len(candidates),
            "predictions_path": str((folder / "predictions.csv").resolve()),
            "unique_candidate_count": len({json.dumps(parameters, sort_keys=True) for parameters in candidates}),
            "augmentation": outer_audit["augmentation"], "runtime_seconds": time.perf_counter() - started,
            "config": {"selected_parameters": parameters, "settings": outer_settings,
                       "selected_epoch_sequence": selected_audit["epoch_sequence"], "threshold": threshold}}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--protocol", default="configs/benchmarks/shoppers_controlled.yaml")
    result.add_argument("--study", default="configs/ablations/stage1_controls.yaml")
    result.add_argument("--data-home", default=".cache/benchmarks")
    result.add_argument("--local-data", help="Official local dataset file or directory (BAF Base requires this)")
    result.add_argument("--download", action="store_true")
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="Acquire, validate and snapshot only; fit no models")
    mode.add_argument("--smoke", action="store_true", help="Use outer-training-only small partitions; not an experiment")
    mode.add_argument("--run", action="store_true", help="Explicitly execute a study; never used by setup checks")
    result.add_argument("--experiment-type", choices=("development", "confirmatory", "reproduction"), default="development")
    result.add_argument("--study-id")
    result.add_argument("--classifiers", nargs="+")
    result.add_argument("--methods", nargs="+")
    result.add_argument("--variant", help="One declared component ablation; unavailable components are refused")
    result.add_argument("--seed", type=int, action="append", dest="seeds")
    result.add_argument("--tuning-budget", type=int)
    result.add_argument("--augmentation-count", type=int)
    result.add_argument("--budget-mode", choices=("matched", "native"))
    result.add_argument("--ugo-max-iterations", type=int)
    result.add_argument("--mlp-epochs", type=int)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--smoke-train-rows", type=int, default=400)
    result.add_argument("--smoke-test-rows", type=int, default=120)
    result.add_argument("--f1-source-run", help="Completed matching F1 source for F2; never refit its detector")
    result.add_argument("--output-dir")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    protocol = load_protocol(args.protocol)
    if protocol["status"] != "ready":
        raise ValueError("Unresolved published protocol: " + "; ".join(protocol["missing_prerequisites"]))
    study = load_study(args.study)
    if args.study_id:
        study = {**study, "study_id": args.study_id}
    experiment_type = "setup_smoke" if args.smoke else ("setup_prepare" if args.prepare else args.experiment_type)
    enforce_stage_gate(study, experiment_type)
    if args.variant and args.methods:
        raise ValueError("--variant and --methods are mutually exclusive.")
    variants = ([args.variant] if args.variant else args.methods or protocol["methods"])
    declarations = {variant: require_variant(study, variant) for variant in variants}
    algorithms = args.classifiers or protocol["classifiers"]
    seeds = args.seeds or ([protocol["seeds"][0]] if args.smoke else protocol["seeds"])
    budget = args.tuning_budget if args.tuning_budget is not None else (1 if args.smoke else protocol["tuning_budget"]["n_candidates"])
    if budget < 1:
        raise ValueError("Tuning budget must be positive.")
    augmentation = dict(protocol.get("augmentation", {"mode": "matched", "total": 1000}))
    if args.smoke:
        augmentation["total"] = 40
    if args.augmentation_count is not None:
        augmentation["total"] = args.augmentation_count
    if args.budget_mode:
        augmentation["mode"] = args.budget_mode
    if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique and nonnegative.")
    if augmentation["total"] < 0 or (args.mlp_epochs is not None and args.mlp_epochs < 1):
        raise ValueError("Augmentation count must be nonnegative and MLP epochs positive.")
    if args.ugo_max_iterations is not None and args.ugo_max_iterations < 1:
        raise ValueError("UGO iteration cap must be positive.")
    settings = {"device": args.device, "augmentation": augmentation,
                "mlp_epochs": args.mlp_epochs or (3 if args.smoke else protocol.get("mlp_epochs", 160)),
                "ugo_max_iterations": args.ugo_max_iterations if args.ugo_max_iterations is not None else (1 if args.smoke else 20),
                "method_parameters": {**protocol.get("method_parameters", {}),
                                      **(protocol.get("smoke_method_parameters", {}) if args.smoke else {})}}
    if any(declaration["method"] == "ugo_f2" for declaration in declarations.values()):
        source_key = "smoke_iterative_f1_source_run" if args.smoke else "iterative_f1_source_run"
        settings["f1_source_run"] = args.f1_source_run or study.get("f2_policy", {}).get(source_key)
        if not settings["f1_source_run"]:
            raise ValueError("F2 requires a matching completed F1 source run")
    # Resolve imports/parameters up front so a missing required dependency cannot silently remove a cell.
    for algorithm in algorithms:
        make_estimator(algorithm, seeds[0], parameters=protocol.get("classifier_parameters", {}).get(algorithm),
                       device=args.device, mlp_epochs=settings["mlp_epochs"])
    output = args.output_dir or ("results/smoke" if args.smoke else "results/runs")
    arguments = {**vars(args), "experiment_type": experiment_type, "effective_settings": settings,
                 "effective_seeds": seeds, "effective_classifiers": algorithms,
                 "effective_variants": variants, "effective_tuning_budget": budget}
    artifacts = BenchmarkArtifacts(output, arguments, protocol, study)
    rows = []
    try:
        dataset = load_dataset(protocol, args.data_home, local_data=args.local_data, download=args.download)
        artifacts.dataset(dataset)
        fingerprint = data_fingerprint(dataset["X"], dataset["y"], dataset["row_ids"])
        for seed in seeds:
            seed_everything(seed)
            for original_split in make_splits(dataset, protocol, seed):
                split = (smoke_split(dataset, original_split, seed, args.smoke_train_rows, args.smoke_test_rows,
                                     protocol["validation_fraction"]) if args.smoke else original_split)
                identity = split_fingerprint(dataset, split)
                split_path = artifacts.fold_path(dataset["name"], split["fold"]) / f"seed-{seed}"
                split_path.mkdir(parents=True, exist_ok=True)
                roles = np.full(len(dataset["y"]), "unused", dtype=object)
                for key, role in (("fit_indices", "inner_fit"), ("validation_indices", "inner_validation"), ("test_indices", "outer_test")):
                    roles[split[key]] = role
                pd.DataFrame({"row_id": dataset["row_ids"], "role": roles, "label": dataset["y"]}).to_csv(split_path / "splits.csv", index=False)
                write_json(split_path / "split.json", {"split_fingerprint": identity, "original_outer_test_untouched": args.smoke,
                                                       "outer_train_row_ids": dataset["row_ids"][split["train_indices"]]})
                partitions = None if args.prepare else prepare_partitions(dataset, split)
                for algorithm in algorithms:
                    candidates = candidate_configs(protocol, algorithm, seed, budget)
                    for variant_id, declaration in declarations.items():
                        method = declaration["method"]
                        metadata = {"dataset": dataset["name"], "dataset_fingerprint": fingerprint,
                                    "split_fingerprint": identity, "seed": seed, "algorithm": algorithm,
                                    "method": method, "variant_id": variant_id, "budget_mode": augmentation["mode"],
                                    "experiment_type": experiment_type, "tuning_budget": budget, "candidate_configs": candidates,
                                    "fold": split["fold"]}
                        if declaration.get("sampler_parameters") or declaration.get("sampler_search"):
                            metadata.update(sampler_parameters=declaration.get("sampler_parameters", {}),
                                            sampler_search=declaration.get("sampler_search", {}))
                        allowed = study.get("classifier_methods", {}).get(algorithm)
                        if allowed is not None and method not in allowed:
                            rows.append({**metadata, "status": "unsupported", "error": f"{algorithm} is an intrinsic ensemble control; declared methods: {allowed}."})
                            continue
                        if method == "class_weight" and not supports_sample_weight(algorithm):
                            rows.append({**metadata, "status": "unsupported", "error": f"{algorithm} does not support sample weighting."})
                            continue
                        if args.prepare:
                            rows.append({**metadata, "status": "prepared", "config": {"settings": settings}})
                            continue
                        print(f"{experiment_type}: {dataset['name']} seed={seed} {algorithm}/{variant_id}", flush=True)
                        try:
                            with measure_memory(args.device) as memory:
                                row = evaluate_cell(dataset, split, algorithm, method, variant_id, candidates,
                                                    settings, protocol, artifacts, metadata, partitions)
                            rows.append({**row, **memory})
                        except Exception as error:
                            rows.append({**metadata, "status": "failed", "error": f"{type(error).__name__}: {error}"})
                            raise
    except Exception as error:
        artifacts.finish(rows, error=f"{type(error).__name__}: {error}")
        print(f"Saved failed setup/run to {artifacts.path}", flush=True)
        raise
    artifacts.finish(rows, status="prepared" if args.prepare else None)
    print(f"Saved {'prepared setup' if args.prepare else experiment_type} to {artifacts.path}")
    return artifacts.path


if __name__ == "__main__":
    main()
