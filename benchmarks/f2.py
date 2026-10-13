"""Shoppers F2: lock completed F1 selection and calibrate its exact frozen core."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from benchmarks.h1 import group_recall
from benchmarks.results import choose_threshold, metric_results, snapshot_sha256
from datasets.scripts.climb import categorical_columns
from experiments import _preprocessing_state, write_json
from samplers.flow_gmm import FlowGMMCore, FlowGate
from samplers.flow_support import CalibratedFlowGate, calibrate_flow


def load_f1_source(dataset, split, algorithm, seed, candidates, settings, protocol, metadata, partitions):
    """Validate matching study/data/roles before loading trusted local artifacts."""
    source = Path(settings.get("f1_source_run", "")).resolve()
    if not settings.get("f1_source_run"):
        raise ValueError("F2 requires an explicit completed F1 source run")
    manifests = partitions.setdefault("f2_source_manifests", {})
    if str(source) not in manifests:
        manifest = json.loads((source / "manifest.json").read_text())
        records = json.loads((source / "benchmark_records.json").read_text())
        if manifest["status"] != "completed" or manifest["protocol_sha256"] != snapshot_sha256(protocol):
            raise ValueError("F2 source must be completed and use the identical protocol")
        manifests[str(source)] = manifest, records
    manifest, records = manifests[str(source)]
    wanted_smoke = metadata.get("experiment_type") == "setup_smoke"
    if (manifest["arguments"]["experiment_type"] == "setup_smoke") != wanted_smoke:
        raise ValueError("F2 smoke and full evaluation must not share source/test partitions")
    original_settings = manifest["arguments"]["effective_settings"]
    for field in ("augmentation", "ugo_max_iterations", "mlp_epochs", "device"):
        if settings[field] != original_settings[field]:
            raise ValueError(f"F2 must preserve F1 source {field}")
    matches = [row for row in records if row["algorithm"] == algorithm and row["seed"] == seed
               and row["split_fingerprint"] == metadata["split_fingerprint"] and row["variant_id"] == "F1"]
    if len(matches) != 1 or matches[0]["status"] != "completed":
        raise ValueError("F2 requires exactly one completed matching F1 cell")
    record = matches[0]
    for field in ("dataset_fingerprint", "split_fingerprint"):
        if record[field] != metadata[field]:
            raise ValueError(f"F2 source {field} differs")
    folder = source / dataset["name"] / f"fold-{split['fold']}" / f"seed-{seed}" / algorithm / "F1"
    selection = json.loads((folder / "selection.json").read_text())
    if selection["candidate_configs"] != candidates:
        raise ValueError("F2 must reuse the identical canonical U2 learner proposals")
    support = selection["inner"]["support_summary"]["flow_support"]
    identities = np.asarray(dataset["row_ids"])
    indices, calibration_indices = train_test_split(np.asarray(split["fit_indices"]), test_size=.2,
        stratify=np.asarray(dataset["y"])[split["fit_indices"]], random_state=int(seed) + 7919)
    for actual, expected, name in (
            (support["fit_row_ids"], identities[indices], "fit"),
            (support["calibration_row_ids"], identities[calibration_indices], "calibration"),
            (support["validation_row_ids"], identities[split["validation_indices"]], "validation")):
        if list(actual) != expected.tolist():
            raise ValueError(f"F2 source {name} roles differ")
    caches = partitions.setdefault("f2_frozen_sources", {})
    key = (str(source), int(seed), tuple(indices.tolist()), settings["device"])
    load_seconds = calibration_seconds = 0.
    checkpoint = folder / "flow-core.pt"
    if key not in caches:
        started = perf_counter()
        device = settings["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        core = FlowGMMCore.load(checkpoint, device=device)
        load_seconds = perf_counter() - started
        if (core.audit["model_checksum"] != support["model_checksum"]
                or core.fit_row_ids.tolist() != support["fit_row_ids"]):
            raise ValueError("F2 checkpoint does not match F1 fit/model provenance")
        calibration = calibrate_flow(core, dataset["X"].iloc[calibration_indices],
            np.asarray(dataset["y"])[calibration_indices], identities[calibration_indices])
        calibration_seconds = calibration.audit["calibration_seconds"]
        caches[key] = core, calibration
    core, calibration = caches[key]
    if (core.audit["model_checksum"] != support["model_checksum"]
            or core.fit_row_ids.tolist() != support["fit_row_ids"]
            or core.audit["model_config"] != support["model_config"]):
        raise ValueError("Cached F2 core differs from the requested F1 source")
    gates = {}
    for role in ("inner", "outer"):
        base = FlowGate(core, partitions[role], calibration_row_ids=identities[calibration_indices],
                        validation_row_ids=identities[split["validation_indices"]])
        base.role = role
        base.calibration_class_counts = dict(calibration.class_counts)
        gates[role] = base
    reference = {"source_run": str(source), "source_f1_folder": str(folder),
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "model_checksum": core.audit["model_checksum"], "new_flow_fit_count": 0,
        "checkpoint_load_seconds_actual": load_seconds, "calibration_seconds_actual": calibration_seconds,
        "original_f1_runtime_seconds_shared": record["runtime_seconds"],
        "original_flow_training_seconds_shared": core.audit["training_seconds"],
        "f1_selected_tau": selection["inner"]["sampler_parameters"]["f1_tau"],
        "f1_selected_candidate": selection["selected_candidate"],
        "manual_retention": "F1_kept_F2_discarded_by_user"}
    return record, selection, gates, calibration, reference


def evaluate_f2_cell(dataset, split, algorithm, variant_id, candidates, settings, protocol,
                     artifacts, metadata, partitions=None):
    from benchmarks.run import (augmentation_targets, categorical_groups, prepare_partitions,
                                prepare_protected_groups, select_score, train_method)
    if dataset["name"] != "shoppers" or algorithm not in ("logistic_regression", "random_forest", "lightgbm"):
        raise ValueError("F2 development evaluation requires Shoppers with LR, RF or LightGBM")
    search = metadata.get("sampler_search", {})
    if set(search) != {"f2_q"} or not search["f2_q"]:
        raise ValueError("F2 requires a separate quantile search with F1 tau already locked")
    quantiles = search["f2_q"]
    if (len(set(quantiles)) != len(quantiles) or any(q is not None and
            (isinstance(q, (bool, np.bool_)) or not np.isfinite(q) or not 0 <= q <= 1) for q in quantiles)):
        raise ValueError("F2 quantiles must be distinct in [0,1], or None for explicit off")
    quantiles = sorted(quantiles, key=lambda q: (-1. if q is None else q))
    started = perf_counter()
    partitions = partitions if partitions is not None else prepare_partitions(dataset, split)
    seed = metadata["seed"]
    record, selected, gates, calibration, reference = load_f1_source(
        dataset, split, algorithm, seed, candidates, settings, protocol, metadata, partitions)
    base_parameters = selected["inner"]["sampler_parameters"]
    learner = selected["inner"]["parameters"]
    tau = reference["f1_selected_tau"]
    y, ids = np.asarray(dataset["y"]), np.asarray(dataset["row_ids"])
    fit, validation, train, test = (split[key] for key in
        ("fit_indices", "validation_indices", "train_indices", "test_indices"))
    positive = record["positive_class"]
    root = artifacts.fold_path(dataset["name"], split["fold"]) / f"seed-{seed}" / algorithm
    folder = root / variant_id
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "calibration.json", calibration.audit)
    write_json(folder / "source_f1.json", reference)
    inner_groups = prepare_protected_groups(partitions["X_fit"], y[fit], seed,
        augmentation_targets(y[fit], settings["augmentation"]), base_parameters)
    inner_groups["real_row_ids"] = ids[fit]
    best, own_fits = None, 0
    inference_seconds = 0.
    candidates_audit = []
    for candidate_id, q in enumerate(quantiles):
        gate = CalibratedFlowGate(gates["inner"], calibration, q=q)
        sampler = {**base_parameters, "f2_q": q, "support_filter": q is not None}
        candidate_settings = {**settings, "method_parameters": {**settings["method_parameters"], "ugo_f2": sampler}}
        before = gate.audit["inference_seconds"]
        candidate_started = perf_counter()
        model, audit = train_method(algorithm, "ugo_f2", learner, partitions["X_fit"], y[fit],
            categorical_columns(partitions["inner"]), seed, candidate_settings,
            (partitions["X_validation"], y[validation]), groups=categorical_groups(partitions["inner"]),
            group_partition=inner_groups, flow_support=gate)
        probabilities = model.predict_proba(partitions["X_validation"])
        threshold = choose_threshold(y[validation] == positive,
            probabilities[:, np.flatnonzero(model.classes_ == positive)[0]], protocol["threshold"])
        metrics = metric_results(y[validation], probabilities, model.classes_, positive, threshold,
            protocol["primary_metric"], protocol["threshold"].get("fpr", .05))
        audit["group_recall"] = group_recall(partitions["X_validation"], y[validation], probabilities,
            model.classes_, threshold, positive, partitions["X_fit"], y[fit], inner_groups, ids[validation])
        audit["detector_inference_seconds"] = gate.audit["inference_seconds"] - before
        inference_seconds += audit["detector_inference_seconds"]
        own_fits += audit["classifier_fit_count"]
        item = {"detector_candidate_id": candidate_id, "q": q, "tau": tau,
                "support_filter": q is not None, "threshold": threshold, "metrics": metrics,
                "runtime_seconds": perf_counter() - candidate_started, **audit}
        write_json(folder / f"detector-candidate-{candidate_id}.json", item)
        candidates_audit.append({key: item[key] for key in ("detector_candidate_id", "q", "metrics", "runtime_seconds")})
        score = select_score(metrics, protocol["primary_metric"])
        if best is None or score > best[0]:
            best = score, candidate_id, q, threshold, deepcopy(audit), candidate_settings
    _, candidate_id, q, threshold, inner_audit, outer_settings = best
    outer_groups = prepare_protected_groups(partitions["X_train"], y[train], seed,
        augmentation_targets(y[train], settings["augmentation"]), base_parameters)
    outer_groups["real_row_ids"] = ids[train]
    gate = CalibratedFlowGate(gates["outer"], calibration, q=q)
    before = gate.audit["inference_seconds"]
    model, audit = train_method(algorithm, "ugo_f2", learner, partitions["X_train"], y[train],
        categorical_columns(partitions["outer"]), seed, outer_settings,
        replay_epochs=inner_audit["epoch_sequence"], groups=categorical_groups(partitions["outer"]),
        group_partition=outer_groups, flow_support=gate)
    own_fits += audit["classifier_fit_count"]
    audit["detector_inference_seconds"] = gate.audit["inference_seconds"] - before
    probabilities = model.predict_proba(partitions["X_test"])
    metrics = metric_results(y[test], probabilities, model.classes_, positive, threshold,
        protocol["primary_metric"], protocol["threshold"].get("fpr", .05))
    audit["group_recall"] = group_recall(partitions["X_test"], y[test], probabilities, model.classes_,
        threshold, positive, partitions["X_train"], y[train], outer_groups, ids[test])
    predicted = np.where(probabilities[:, np.flatnonzero(model.classes_ == positive)[0]] >= threshold,
                         positive, model.classes_[model.classes_ != positive][0])
    predictions = pd.DataFrame({"row_id": ids[test], "y_true": y[test], "y_pred": predicted})
    for index, label in enumerate(model.classes_):
        predictions[f"probability_{label}"] = probabilities[:, index]
    predictions.to_csv(folder / "predictions.csv", index=False)
    off_matches = None
    if q is None:
        expected = pd.read_csv(Path(reference["source_f1_folder"]) / "predictions.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(predictions.reset_index(drop=True), expected, check_exact=True)
        original = json.loads((Path(reference["source_f1_folder"]) / "outer.json").read_text())
        original_rows = original["support_summary"]["candidate_log"]
        current_rows = audit["support_summary"]["candidate_log"]
        keys = ("original_coordinates", "accepted", "seed_index", "partner_index", "group_id", "class", "round")
        if ([{key: row[key] for key in keys} for row in original_rows] !=
                [{key: row[key] for key in keys} for row in current_rows]):
            raise AssertionError("Explicit support-off must reproduce original F1 decisions and ancestry")
        off_matches = True
    write_json(folder / "outer.json", audit)
    write_json(folder / "selection.json", {**reference, "selected_candidate": selected["selected_candidate"],
        "selected_detector_candidate": candidate_id, "selected_q": q, "locked_tau": tau,
        "support_filter": q is not None, "threshold": threshold, "candidate_configs": candidates,
        "inner": inner_audit, "detector_search": search, "quantile_candidates": candidates_audit,
        "selection_policy": "lock completed F1 tau/learner/alpha; real-validation q; ties off then smaller q",
        "additional_classifier_fit_count": own_fits, "new_flow_fit_count": 0,
        "calibration_shared_across_roles_and_families": True, "support_off_matches_F1": off_matches})
    write_json(folder / "preprocessing.json", {"inner": _preprocessing_state(partitions["inner"]),
                                              "outer": _preprocessing_state(partitions["outer"])})
    write_json(folder / "reference_rows.json", {"inner_real_row_ids": ids[fit], "outer_real_row_ids": ids[train]})
    return {**metadata, **metrics, "status": "completed", "strategy": "ugo_f2",
        "threshold": threshold, "positive_class": positive, "selected_candidate": selected["selected_candidate"],
        "selected_detector_candidate": candidate_id, "selected_q": q, "locked_tau": tau,
        "support_filter": q is not None, "support_off_matches_F1": off_matches,
        "source_f1": reference, "candidate_configs": candidates, "detector_tuning_budget": len(quantiles),
        "additional_classifier_fit_count": own_fits, "new_flow_fit_count": 0,
        "calibration_seconds_actual": reference["calibration_seconds_actual"],
        "selection_detector_inference_seconds": inference_seconds,
        "outer_detector_inference_seconds": audit["detector_inference_seconds"],
        "predictions_path": str((folder / "predictions.csv").resolve()),
        "augmentation": audit["augmentation"], "runtime_seconds": perf_counter() - started,
        "config": {"selected_parameters": learner, "settings": outer_settings,
                   "selected_epoch_sequence": inner_audit["epoch_sequence"], "threshold": threshold}}
