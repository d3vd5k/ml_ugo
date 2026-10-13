"""F1 on locked canonical U2 or U5, plus the U2 neighbor-rejection control."""
from copy import deepcopy
import json
from time import perf_counter

import numpy as np
import pandas as pd

from benchmarks.flow_partitions import FLOW_PARAMETERS, flow_for_partition
from benchmarks.h1 import group_recall
from benchmarks.results import choose_threshold, metric_results
from datasets.scripts.climb import categorical_columns
from experiments import _preprocessing_state, write_json


def evaluate_f1_cell(dataset, split, algorithm, method, variant_id, candidates, settings,
                     protocol, artifacts, metadata, partitions=None):
    from benchmarks.run import (augmentation_targets, categorical_groups, evaluate_cell,
                                prepare_partitions, prepare_protected_groups, select_score, train_method)
    if dataset["name"] != "shoppers":
        raise ValueError("F1 development evaluation is restricted to Shoppers")
    started = perf_counter()
    combined = method == "ugo_combined_f1"
    is_flow = method in ("ugo_f1", "ugo_combined_f1")
    base_method, base_variant = ("ugo_combined", "U5") if combined else ("ugo_groups_only", "U2")
    partitions = partitions if partitions is not None else prepare_partitions(dataset, split)
    search = metadata.get("sampler_search", {})
    required = {"group_alpha", "f1_tau"} if is_flow else {"group_alpha"}
    if set(search) != required or any(not search[key] for key in required):
        raise ValueError("F1/N-reject require separate canonical group-alpha and F1-only tau grids")
    taus = search.get("f1_tau", [None])
    if is_flow and (len(set(taus)) != len(taus) or
            any(not np.isfinite(tau) or not 0 < tau < 1 for tau in taus)):
        raise ValueError("F1 tau candidates must be distinct and strictly between zero and one")
    values = metadata.get("sampler_parameters", {})
    base_values = {key: value for key, value in values.items()
                   if key not in FLOW_PARAMETERS and key != "f1_tau"}
    root = artifacts.fold_path(dataset["name"], split["fold"]) / f"seed-{metadata['seed']}" / algorithm
    canonical_cache = partitions.setdefault("f1_canonical", {})
    cache_key = (base_method, algorithm, metadata["seed"], json.dumps(candidates, sort_keys=True), json.dumps(base_values, sort_keys=True), tuple(search["group_alpha"]))
    if cache_key not in canonical_cache:
        # Reuse the declared canonical base; U5 and U2 selections are never mixed.
        canonical_variant = base_variant if (root / base_variant / "selection.json").exists() else f"F1_{base_variant}_selection"
        source_folder = root / canonical_variant
        if not (source_folder / "selection.json").exists():
            canonical_settings = {**settings, "method_parameters": {**settings["method_parameters"],
                base_method: {**settings["method_parameters"].get(base_method, {}), **base_values}}}
            canonical_metadata = {**metadata, "method": base_method, "variant_id": canonical_variant,
                "sampler_parameters": base_values, "sampler_search": {"group_alpha": search["group_alpha"]}}
            evaluate_cell(dataset, split, algorithm, base_method, canonical_variant, candidates, canonical_settings, protocol, artifacts, canonical_metadata, partitions)
        selected = json.loads((source_folder / "selection.json").read_text())
        if selected["candidate_configs"] != candidates:
            raise ValueError(f"Canonical {base_variant} selection used different learner proposals")
        canonical_fits = sum(json.loads(path.read_text())["classifier_fit_count"]
                             for path in source_folder.glob("candidate-*.json"))
        canonical_fits += json.loads((source_folder / "outer.json").read_text())["classifier_fit_count"]
        canonical_cache[cache_key] = selected, canonical_fits, canonical_variant
    selected, canonical_fits, canonical_variant = canonical_cache[cache_key]
    base_parameters = selected["inner"]["sampler_parameters"]
    learner_parameters = selected["inner"]["parameters"]
    labels = dataset["y"]
    fit, validation, train, test = (split[key] for key in
        ("fit_indices", "validation_indices", "train_indices", "test_indices"))
    if combined:
        from benchmarks.scoring import prepare_oof_partitions
        for role, indices in (("inner_oof", fit), ("outer_oof", train)):
            if role not in partitions:
                partitions[role] = prepare_oof_partitions(dataset["X"].iloc[indices], labels[indices],
                    dataset["row_ids"][indices], metadata["seed"])
    classes, counts = np.unique(labels[train], return_counts=True)
    positive_class = int(classes[np.argmin(counts)])
    folder = root / variant_id
    folder.mkdir(parents=True, exist_ok=False)
    real_validation = (partitions["X_validation"], labels[validation])
    inner_groups = prepare_protected_groups(partitions["X_fit"], labels[fit], metadata["seed"],
        augmentation_targets(labels[fit], settings["augmentation"]), base_parameters)
    inner_groups["real_row_ids"] = dataset["row_ids"][fit]
    flow_parameters = {key: value for key, value in values.items() if key in FLOW_PARAMETERS}
    inner_gate = (flow_for_partition(dataset, split, partitions, "inner", metadata["seed"],
                  flow_parameters, settings["device"]) if is_flow else None)
    if inner_gate is not None:
        inner_gate.core.save(folder / "flow-core.pt")
    best, own_fits = None, 0
    for detector_id, tau in enumerate(taus):
        sampler_parameters = {**base_parameters, **flow_parameters}
        if tau is not None:
            sampler_parameters["f1_tau"] = tau
        candidate_settings = {**settings, "method_parameters": {**settings["method_parameters"], method: sampler_parameters}}
        inference_before = inner_gate.audit["inference_seconds"] if inner_gate is not None else 0.
        model, audit = train_method(algorithm, method, learner_parameters, partitions["X_fit"], labels[fit],
            categorical_columns(partitions["inner"]), metadata["seed"], candidate_settings, real_validation,
            groups=categorical_groups(partitions["inner"]), group_partition=inner_groups, flow_support=inner_gate,
            oof=partitions.get("inner_oof") if combined else None)
        probabilities = model.predict_proba(partitions["X_validation"])
        threshold = choose_threshold(labels[validation] == positive_class,
            probabilities[:, np.flatnonzero(model.classes_ == positive_class)[0]], protocol["threshold"])
        metrics = metric_results(labels[validation], probabilities, model.classes_, positive_class,
            threshold, protocol["primary_metric"], protocol["threshold"].get("fpr", .05))
        audit["group_recall"] = group_recall(partitions["X_validation"], labels[validation], probabilities,
            model.classes_, threshold, positive_class, partitions["X_fit"], labels[fit], inner_groups,
            dataset["row_ids"][validation])
        audit["detector_inference_seconds"] = (inner_gate.audit["inference_seconds"] - inference_before
            if inner_gate is not None else audit["support_summary"]["neighbor_rejection"]["inference_seconds"])
        write_json(folder / f"detector-candidate-{detector_id}.json",
                   {"detector_candidate_id": detector_id, "tau": tau, "threshold": threshold, "metrics": metrics, **audit})
        own_fits += audit["classifier_fit_count"]
        score = select_score(metrics, protocol["primary_metric"])
        if best is None or score > best[0]:
            best = (score, detector_id, threshold, deepcopy(audit), candidate_settings)
    _, detector_id, threshold, selected_audit, outer_settings = best
    outer = partitions["outer"]
    X_train, X_test = partitions["X_train"], partitions["X_test"]
    outer_groups = prepare_protected_groups(X_train, labels[train], metadata["seed"],
        augmentation_targets(labels[train], settings["augmentation"]), selected_audit["sampler_parameters"])
    outer_groups["real_row_ids"] = dataset["row_ids"][train]
    outer_gate = (flow_for_partition(dataset, split, partitions, "outer", metadata["seed"],
                  flow_parameters, settings["device"]) if is_flow else None)
    model, audit = train_method(algorithm, method, learner_parameters, X_train, labels[train],
        categorical_columns(outer), metadata["seed"], outer_settings,
        replay_epochs=selected_audit["epoch_sequence"], groups=categorical_groups(outer),
        group_partition=outer_groups, flow_support=outer_gate,
        oof=partitions.get("outer_oof") if combined else None)
    own_fits += audit["classifier_fit_count"]
    probabilities = model.predict_proba(X_test)
    metrics = metric_results(labels[test], probabilities, model.classes_, positive_class, threshold,
        protocol["primary_metric"], protocol["threshold"].get("fpr", .05))
    audit["group_recall"] = group_recall(X_test, labels[test], probabilities, model.classes_, threshold,
        positive_class, X_train, labels[train], outer_groups, dataset["row_ids"][test])
    write_json(folder / "outer.json", audit)
    write_json(folder / "selection.json", {"selected_candidate": selected["selected_candidate"],
        "selected_detector_candidate": detector_id, "threshold": threshold, "candidate_configs": candidates,
        "inner": selected_audit, ("u5_selection" if combined else "u2_selection"): selected, "canonical_variant": canonical_variant,
        "detector_search": search, "selection_policy": f"lock canonical {base_variant}; inner-only post handling, F1 tunes tau and N-reject has no detector tuning",
        "flow_fit_count": 1 if is_flow else 0, "flow_fit_shared_across_roles_and_families": is_flow,
        "canonical_classifier_fit_count": canonical_fits, "additional_classifier_fit_count": own_fits,
        "total_classifier_fit_count": canonical_fits + own_fits})
    write_json(folder / "preprocessing.json", {"inner": _preprocessing_state(partitions["inner"]), "outer": _preprocessing_state(outer)})
    write_json(folder / "reference_rows.json", {"inner_real_row_ids": dataset["row_ids"][fit], "outer_real_row_ids": dataset["row_ids"][train]})
    predicted = np.where(probabilities[:, np.flatnonzero(model.classes_ == positive_class)[0]] >= threshold,
                         positive_class, model.classes_[model.classes_ != positive_class][0])
    predictions = pd.DataFrame({"row_id": dataset["row_ids"][test], "y_true": labels[test], "y_pred": predicted})
    for column, label in enumerate(model.classes_):
        predictions[f"probability_{label}"] = probabilities[:, column]
    predictions.to_csv(folder / "predictions.csv", index=False)
    return {**metadata, **metrics, "status": "completed", "strategy": method, "fold": split["fold"],
        "threshold": threshold, "positive_class": positive_class, "selected_candidate": selected["selected_candidate"],
        "selected_detector_candidate": detector_id, "candidate_configs": candidates, "tuning_budget": len(candidates),
        "detector_tuning_budget": len(taus) if is_flow else 0,
        "canonical_classifier_fit_count": canonical_fits, "additional_classifier_fit_count": own_fits,
        "total_classifier_fit_count": canonical_fits + own_fits,
        "predictions_path": str((folder / "predictions.csv").resolve()),
        "unique_candidate_count": len({json.dumps(value, sort_keys=True) for value in candidates}),
        "augmentation": audit["augmentation"], "runtime_seconds": perf_counter() - started,
        "config": {"selected_parameters": learner_parameters, "settings": outer_settings,
                   "selected_epoch_sequence": selected_audit["epoch_sequence"], "threshold": threshold}}
