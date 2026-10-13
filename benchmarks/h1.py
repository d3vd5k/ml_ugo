"""H1-only selection: lock canonical U2, then tune the independent support gate."""
from copy import deepcopy
from itertools import product
import json
from time import perf_counter


import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from benchmarks.results import choose_threshold, metric_results
from datasets.scripts.climb import categorical_columns
from experiments import _preprocessing_state, write_json
from samplers.hdbscan_support import prepare_h1_support


def group_recall(features, labels, probabilities, classes, threshold, positive_class,
                 real_features, real_labels, prepared, row_ids):
    """Frozen nearest-real-minority assignment; not a true subgroup guarantee."""
    real = np.flatnonzero(real_labels == positive_class)
    query = np.flatnonzero(labels == positive_class)
    result = []
    if not len(query):
        return result
    nearest = NearestNeighbors(n_neighbors=1).fit(real_features[real]).kneighbors(features[query], return_distance=False)[:, 0]
    assigned = prepared["row_group_ids"][real[nearest]]
    column = np.flatnonzero(classes == positive_class)[0]
    positive = probabilities[query, column] >= threshold
    for group in prepared["groups"]:
        if group["class"] != positive_class:
            continue
        mask = assigned == group["id"]
        support = int(mask.sum())
        result.append({"group_id": int(group["id"]), "real_training_support": group["real_support"],
                       "evaluation_support": support, "true_positives": int(positive[mask].sum()),
                       "recall": float(positive[mask].mean()) if support else None,
                       "row_ids": np.asarray(row_ids)[query[mask]].tolist(),
                       "assignment": "nearest real minority in frozen U2 representation"})
    return result


def evaluate_h1_cell(dataset, split, algorithm, variant_id, candidates, settings,
                     protocol, artifacts, metadata, partitions=None):
    from benchmarks.run import (augmentation_targets, categorical_groups, evaluate_cell,
                                prepare_partitions, prepare_protected_groups, select_score, train_method)
    if dataset["name"] != "shoppers":
        raise ValueError("H1 development evaluation is restricted to Shoppers")
    started = perf_counter()
    partitions = partitions if partitions is not None else prepare_partitions(dataset, split)
    search = metadata.get("sampler_search", {})
    required = {"group_alpha", "h1_min_cluster_size", "h1_min_samples"}
    if set(search) != required or any(not search[key] for key in required):
        raise ValueError("H1 requires independent group_alpha/min_cluster_size/min_samples grids")
    values = metadata.get("sampler_parameters", {})
    canonical_settings = {**settings, "method_parameters": {**settings["method_parameters"],
                          "ugo_groups_only": {**settings["method_parameters"].get("ugo_groups_only", {}), **values}}}
    canonical_metadata = {**metadata, "method": "ugo_groups_only", "variant_id": variant_id + "_U2_selection",
                          "sampler_search": {"group_alpha": search["group_alpha"]}}
    canonical = evaluate_cell(dataset, split, algorithm, "ugo_groups_only", variant_id + "_U2_selection",
                             candidates, canonical_settings, protocol, artifacts, canonical_metadata, partitions)
    root = artifacts.fold_path(dataset["name"], split["fold"]) / f"seed-{metadata['seed']}" / algorithm
    selected = json.loads((root / (variant_id + "_U2_selection") / "selection.json").read_text())
    base_parameters = selected["inner"]["sampler_parameters"]
    learner_parameters = selected["inner"]["parameters"]
    selected_id = selected["selected_candidate"]
    labels = dataset["y"]
    fit, validation, train, test = (split[key] for key in
                                  ("fit_indices", "validation_indices", "train_indices", "test_indices"))
    positive_class = canonical["positive_class"]
    folder = root / variant_id
    folder.mkdir(parents=True, exist_ok=False)
    inner = partitions["inner"]
    real_validation = (partitions["X_validation"], labels[validation])
    inner_groups = prepare_protected_groups(partitions["X_fit"], labels[fit], metadata["seed"],
        augmentation_targets(labels[fit], settings["augmentation"]), base_parameters)
    inner_groups["real_row_ids"] = dataset["row_ids"][fit]
    best = None
    detector_fits = 0
    total_classifier_fits = sum(json.loads(path.read_text())["classifier_fit_count"]
        for path in (root / (variant_id + "_U2_selection")).glob("candidate-*.json"))
    total_classifier_fits += json.loads((root / (variant_id + "_U2_selection") / "outer.json").read_text())["classifier_fit_count"]
    for detector_id, (cluster_size, min_samples) in enumerate(product(search["h1_min_cluster_size"], search["h1_min_samples"])):
        sampler_parameters = {**base_parameters, "h1_min_cluster_size": cluster_size, "h1_min_samples": min_samples}
        candidate_settings = {**settings, "method_parameters": {**settings["method_parameters"], "ugo_h1": sampler_parameters}}
        detector = prepare_h1_support(partitions["X_fit"], labels[fit], categorical_columns(inner), sampler_parameters,
                                      dataset["row_ids"][fit])
        detector_fits += 1
        model, audit = train_method(algorithm, "ugo_h1", learner_parameters, partitions["X_fit"], labels[fit],
            categorical_columns(inner), metadata["seed"], candidate_settings, real_validation,
            groups=categorical_groups(inner), group_partition=inner_groups, h1_support=detector)
        probabilities = model.predict_proba(partitions["X_validation"])
        threshold = choose_threshold(labels[validation] == positive_class,
            probabilities[:, np.flatnonzero(model.classes_ == positive_class)[0]], protocol["threshold"])
        metrics = metric_results(labels[validation], probabilities, model.classes_, positive_class,
                                 threshold, protocol["primary_metric"], protocol["threshold"].get("fpr", .05))
        audit["group_recall"] = group_recall(partitions["X_validation"], labels[validation], probabilities,
            model.classes_, threshold, positive_class, partitions["X_fit"], labels[fit], inner_groups,
            dataset["row_ids"][validation])
        record = {"detector_candidate_id": detector_id, "threshold": threshold, "metrics": metrics, **audit}
        write_json(folder / f"detector-candidate-{detector_id}.json", record)
        total_classifier_fits += audit["classifier_fit_count"]
        score = select_score(metrics, protocol["primary_metric"])
        if best is None or score > best[0]:
            best = (score, detector_id, threshold, deepcopy(audit), candidate_settings)
    _, detector_id, threshold, selected_audit, outer_settings = best
    outer = partitions["outer"]
    X_train, X_test = partitions["X_train"], partitions["X_test"]
    sampler_parameters = selected_audit["sampler_parameters"]
    outer_groups = prepare_protected_groups(X_train, labels[train], metadata["seed"],
        augmentation_targets(labels[train], settings["augmentation"]), sampler_parameters)
    outer_groups["real_row_ids"] = dataset["row_ids"][train]
    detector = prepare_h1_support(X_train, labels[train], categorical_columns(outer), sampler_parameters,
                                  dataset["row_ids"][train])
    model, audit = train_method(algorithm, "ugo_h1", learner_parameters, X_train, labels[train],
        categorical_columns(outer), metadata["seed"], outer_settings,
        replay_epochs=selected_audit["epoch_sequence"], groups=categorical_groups(outer),
        group_partition=outer_groups, h1_support=detector)
    total_classifier_fits += audit["classifier_fit_count"]
    probabilities = model.predict_proba(X_test)
    metrics = metric_results(labels[test], probabilities, model.classes_, positive_class, threshold,
                             protocol["primary_metric"], protocol["threshold"].get("fpr", .05))
    audit["group_recall"] = group_recall(X_test, labels[test], probabilities, model.classes_, threshold,
        positive_class, X_train, labels[train], outer_groups, dataset["row_ids"][test])
    write_json(folder / "outer.json", audit)
    write_json(folder / "selection.json", {"selected_candidate": selected_id, "selected_detector_candidate": detector_id,
        "threshold": threshold, "candidate_configs": candidates, "inner": selected_audit,
        "u2_selection": selected, "detector_search": search,
        "selection_policy": "lock canonical U2 learner/group_alpha; tune H1 on iterative real-validation trajectories",
        "detector_fit_count": detector_fits + 1, "total_classifier_fit_count": total_classifier_fits})
    write_json(folder / "preprocessing.json", {"inner": _preprocessing_state(inner), "outer": _preprocessing_state(outer)})
    write_json(folder / "reference_rows.json", {"inner_real_row_ids": dataset["row_ids"][fit],
                                              "outer_real_row_ids": dataset["row_ids"][train]})
    predicted = np.where(probabilities[:, np.flatnonzero(model.classes_ == positive_class)[0]] >= threshold,
                         positive_class, model.classes_[model.classes_ != positive_class][0])
    predictions = pd.DataFrame({"row_id": dataset["row_ids"][test], "y_true": labels[test], "y_pred": predicted})
    for column, label in enumerate(model.classes_):
        predictions[f"probability_{label}"] = probabilities[:, column]
    predictions.to_csv(folder / "predictions.csv", index=False)
    return {**metadata, **metrics, "status": "completed", "strategy": "ugo_h1", "fold": split["fold"],
        "threshold": threshold, "positive_class": positive_class, "selected_candidate": selected_id,
        "selected_detector_candidate": detector_id, "candidate_configs": candidates, "tuning_budget": len(candidates),
        "detector_tuning_budget": detector_fits, "total_classifier_fit_count": total_classifier_fits,
        "predictions_path": str((folder / "predictions.csv").resolve()),
        "unique_candidate_count": len({json.dumps(value, sort_keys=True) for value in candidates}),
        "augmentation": audit["augmentation"], "runtime_seconds": perf_counter() - started,
        "config": {"selected_parameters": learner_parameters, "settings": outer_settings,
                   "selected_epoch_sequence": selected_audit["epoch_sequence"], "threshold": threshold}}
