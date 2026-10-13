"""Shoppers-only immutable U2-pool H1 replay and source-group-matched thinning.

Run with ``python -m benchmarks.h1_pool --smoke`` or explicit ``--run``.
H1 selects support parameters on inner validation, independently of the locked
canonical U2 learner/group-alpha search. No replay generates or refills proposals.
"""
import argparse
from collections import Counter
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from benchmarks.protocols import load_dataset, load_protocol, make_splits
from benchmarks.results import (BenchmarkArtifacts, choose_threshold, load_study,
                                metric_results, require_variant, snapshot_sha256)
from benchmarks.run import (candidate_configs, evaluate_cell, prepare_partitions,
                            select_score, smoke_split, split_fingerprint)
from classifiers.factory import fit_estimator, make_estimator
from datasets.scripts.climb import categorical_columns
from experiments import data_fingerprint, seed_everything, write_json
from samplers.hdbscan_support import prepare_h1_support


METHOD = "ugo_groups_only"
H1_GRID = tuple({"h1_min_cluster_size": size, "h1_min_samples": samples}
                for size in (5, 10, 20, 40) for samples in (1, 5, 10))
THINNING_SEEDS = (0, 1, 2, 3, 4)
GROUP_CONVENTION = ("frozen nearest real minority training neighbor in training-fitted "
                    "U2 distance representation; original U2 group; model-based, "
                    "not a true subgroup or label-correctness guarantee")


def accepted_pool(audit, width):
    """Preserve all proposals, while replaying only canonical U2 accepted outputs."""
    rows = audit["support_summary"]["candidate_log"]
    accepted = [row for row in rows if row["accepted"]]
    coordinates = np.asarray([row["final_coordinates"] for row in accepted], dtype=np.float32)
    coordinates = coordinates.reshape(len(accepted), width)
    return {"rows": rows, "accepted_rows": accepted, "X": coordinates,
            "y": np.asarray([row["class"] for row in accepted], dtype=np.int64),
            "group_ids": np.asarray([row["group_id"] for row in accepted], dtype=np.int64),
            "proposal_ids": np.asarray([row["proposal_id"] for row in accepted], dtype=np.int64)}


def matched_thinning(group_ids, h1_mask, seed):
    """Uniform sampling without replacement, matching every original source group."""
    group_ids, h1_mask = np.asarray(group_ids), np.asarray(h1_mask, dtype=bool)
    if group_ids.shape != h1_mask.shape or group_ids.ndim != 1:
        raise ValueError("Expected aligned one-dimensional group IDs and acceptance mask")
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(group_ids), dtype=bool)
    for group in np.unique(group_ids):
        eligible = np.flatnonzero(group_ids == group)
        count = int(h1_mask[eligible].sum())
        if count:
            mask[rng.choice(eligible, size=count, replace=False)] = True
    return mask


def group_assignments(X_real, y_real, row_group_ids, real_row_ids,
                      X_evaluation, y_evaluation, evaluation_row_ids, minority_class):
    """Fit the assignment reference on real minority training rows only."""
    real_indices = np.flatnonzero(np.asarray(y_real) == minority_class)
    evaluation_indices = np.flatnonzero(np.asarray(y_evaluation) == minority_class)
    neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(X_real[real_indices])
    nearest = (neighbors.kneighbors(X_evaluation[evaluation_indices], return_distance=False)[:, 0]
               if len(evaluation_indices) else np.empty(0, dtype=int))
    source_indices = real_indices[nearest]
    return pd.DataFrame({"evaluation_index": evaluation_indices,
                         "row_id": np.asarray(evaluation_row_ids)[evaluation_indices],
                         "class": np.asarray(y_evaluation)[evaluation_indices],
                         "group_id": np.asarray(row_group_ids)[source_indices],
                         "nearest_real_index": source_indices,
                         "nearest_real_row_id": np.asarray(real_row_ids)[source_indices]})


def group_recall(assignments, prediction, group_support):
    groups = {int(group["id"]): group for group in group_support["groups"]}
    rows = []
    for group_id in sorted(set(groups) | set(assignments["group_id"].tolist())):
        assigned = assignments[assignments["group_id"] == group_id]
        indices = assigned["evaluation_index"].to_numpy(dtype=int)
        hits = int(np.sum(np.asarray(prediction)[indices] == assigned["class"].to_numpy()))
        group = groups.get(group_id, {})
        rows.append({"group_id": group_id, "real_training_support": group.get("real_support", 0),
                     "evaluation_minority_support": len(indices), "true_positives": hits,
                     "recall": hits / len(indices) if len(indices) else None,
                     "stability": group.get("stability"), "assignment_convention": GROUP_CONVENTION})
    return rows


def duplicate_counts(real, synthetic):
    """Exact float-coordinate duplicate counts; neither round nor deduplicate data."""
    real_set = {row.tobytes() for row in np.ascontiguousarray(real)}
    counts = Counter(row.tobytes() for row in np.ascontiguousarray(synthetic, dtype=real.dtype))
    return {"synthetic_count": len(synthetic), "unique_synthetic_count": len(counts),
            "repeated_synthetic_count": len(synthetic) - len(counts),
            "synthetic_matches_real_count": sum(count for key, count in counts.items() if key in real_set)}


def pool_accounting(pool, mask, audit, real):
    group_support = audit["support_summary"]["group_support"]
    quotas = Counter()
    for record in group_support["allocation_records"]:
        for trace in record["group_trace"]:
            quotas[int(trace["group_id"])] += int(trace["quota"])
    proposal_counts = Counter(int(row["group_id"]) for row in pool["rows"])
    groups = []
    for group in group_support["groups"]:
        group_id = int(group["id"])
        eligible = pool["group_ids"] == group_id
        accepted = int(np.sum(mask & eligible))
        u2_count = int(eligible.sum())
        groups.append({"group_id": group_id, "class": group["class"],
                       "real_support": group["real_support"], "stability": group["stability"],
                       "planned_proposals": quotas[group_id], "logged_proposals": proposal_counts[group_id],
                       "u2_accepted_count": u2_count, "accepted_count": accepted,
                       "acceptance_of_u2": accepted / u2_count if u2_count else None,
                       "acceptance_of_proposals": accepted / proposal_counts[group_id] if proposal_counts[group_id] else None,
                       "unspent_planned_quota": quotas[group_id] - accepted})
    augmentation = {**audit["augmentation"], "actual_count": int(mask.sum()),
                    "actual_counts": dict(Counter(np.concatenate((np.asarray(real[1]), pool["y"][mask])).tolist()))}
    requested = augmentation["requested_count"]
    augmentation["underspend"] = None if requested is None else requested - int(mask.sum())
    return {"augmentation": augmentation, "group_acceptance": groups,
            "duplicates": duplicate_counts(real[0], pool["X"][mask]),
            "unspent_quota": augmentation["underspend"],
            "u2_stop_reason": audit["stop_reason"], "no_quota_refill": True,
            "replay_status": "accepted_pool" if mask.any() else "no_accepted_candidates_real_only"}


def save_pool(folder, pool, audit, X, y, row_ids, link):
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "candidate_log.json", {**link, "rows": pool["rows"]})
    original = np.asarray([row["original_coordinates"] for row in pool["accepted_rows"]], dtype=np.float32)
    original = original.reshape(len(pool["X"]), X.shape[1])
    np.savez_compressed(folder / "pool.npz", X_real=X, y_real=y,
                        real_row_ids=np.asarray(row_ids, dtype=str),
                        row_group_ids=np.asarray(audit["support_summary"]["group_support"]["row_group_ids"]),
                        original_coordinates=original, final_coordinates=pool["X"],
                        classes=pool["y"], group_ids=pool["group_ids"], proposal_ids=pool["proposal_ids"])
    digest = hashlib.sha256((folder / "pool.npz").read_bytes()).hexdigest()
    write_json(folder / "pool.json", {**link, "pool_sha256": digest,
                                      "candidate_log_path": str((folder / "candidate_log.json").resolve()),
                                      "canonical_u2_accepted_count": len(pool["X"]),
                                      "proposal_count": len(pool["rows"]), "u2_audit": audit,
                                      "coordinates": "training-fitted transformed classifier features"})
    return {"pool_path": str((folder / "pool.npz").resolve()), "pool_sha256": digest}


def fit_replay(algorithm, parameters, settings, seed, X_real, y_real, pool, mask, epoch):
    estimator = make_estimator(algorithm, seed, parameters=parameters,
                               device=settings["device"], mlp_epochs=settings["mlp_epochs"])
    X = np.concatenate((X_real, pool["X"][mask])) if mask.any() else X_real
    y = np.concatenate((y_real, pool["y"][mask])) if mask.any() else y_real
    return fit_estimator(estimator, X, y, epochs=epoch)


def predictions(model, X, y, row_ids, positive_class, threshold, protocol, folder, role, link):
    probabilities = model.predict_proba(X)
    positive_index = int(np.flatnonzero(model.classes_ == positive_class)[0])
    prediction = np.where(probabilities[:, positive_index] >= threshold,
                          positive_class, model.classes_[1 - positive_index])
    metrics = metric_results(y, probabilities, model.classes_, positive_class, threshold,
                            protocol["primary_metric"], protocol["threshold"].get("fpr", .05))
    frame = pd.DataFrame({"row_id": row_ids, "y_true": y, "y_pred": prediction,
                          "split_fingerprint": link["split_fingerprint"],
                          "selected_config_sha256": link["selected_config_sha256"]})
    for index, label in enumerate(model.classes_):
        frame[f"probability_{label}"] = probabilities[:, index]
    path = folder / f"{role}_predictions.csv"
    frame.to_csv(path, index=False)
    write_json(folder / f"{role}_metrics.json", {**link, **metrics})
    return metrics, prediction, str(path.resolve())


def select_h1(algorithm, parameters, settings, seed, partitions, labels, split,
              pool, epoch, positive_class, row_ids, protocol, folder, base_link):
    """Only the immutable inner pool and real inner validation select H1 settings."""
    folder.mkdir(parents=True, exist_ok=False)
    best = None
    candidates = []
    for candidate_id, configuration in enumerate(H1_GRID):
        started = time.perf_counter()
        detector = prepare_h1_support(partitions["X_fit"], labels[split["fit_indices"]],
                                      categorical_columns=categorical_columns(partitions["inner"]),
                                      parameters=configuration, real_row_ids=row_ids[split["fit_indices"]])
        scores = detector.predict(pool["X"])
        mask = scores["valid"] & (scores["labels"] != -1)
        model = fit_replay(algorithm, parameters, settings, seed, partitions["X_fit"],
                           labels[split["fit_indices"]], pool, mask, epoch)
        probabilities = model.predict_proba(partitions["X_validation"])
        threshold = choose_threshold(labels[split["validation_indices"]] == positive_class,
                                     probabilities[:, np.flatnonzero(model.classes_ == positive_class)[0]],
                                     protocol["threshold"])
        metrics = metric_results(labels[split["validation_indices"]], probabilities,
                                 model.classes_, positive_class, threshold, protocol["primary_metric"],
                                 protocol["threshold"].get("fpr", .05))
        record = {**base_link, "candidate_id": candidate_id, "configuration": configuration,
                  "threshold": threshold, "metrics": metrics, "accepted_count": int(mask.sum()),
                  "detector": detector.audit, "runtime_seconds": time.perf_counter() - started,
                  "proposal_ids": pool["proposal_ids"], "accepted": mask,
                  "hdbscan_label": scores["labels"], "membership_strength": scores["strengths"],
                  "reason": scores["reasons"]}
        write_json(folder / f"candidate-{candidate_id}.json", record)
        candidates.append({key: record[key] for key in ("candidate_id", "configuration", "threshold", "metrics", "accepted_count")})
        score = select_score(metrics, protocol["primary_metric"])
        if best is None or score > best[0]:
            best = (score, configuration, threshold, mask, model, candidate_id,
                    scores, detector.audit)
    _, configuration, threshold, mask, model, candidate_id, scores, audit = best
    write_json(folder / "selection.json", {**base_link, "policy": "saved_pool_inner_validation",
                                            "search_space": H1_GRID, "candidates": candidates,
                                            "selected_candidate": candidate_id, "configuration": configuration,
                                            "threshold": threshold, "outer_test_used": False})
    return configuration, threshold, mask, model, scores, audit


def save_decisions(folder, pool, mask, link, scores=None):
    rows = []
    positions = {int(proposal): position for position, proposal in enumerate(pool["proposal_ids"])}
    for source in pool["rows"]:
        row = {**source, "u2_accepted": bool(source["accepted"])}
        position = positions.get(int(source["proposal_id"]))
        row["accepted"] = position is not None and bool(mask[position])
        if position is not None:
            row["reason"] = "accepted" if row["accepted"] else "random_thinning"
            if scores is not None:
                row.update(hdbscan_label=int(scores["labels"][position]),
                           membership_strength=float(scores["strengths"][position]),
                           reason=str(scores["reasons"][position]))
        rows.append(row)
    write_json(folder / "candidate_log.json", {**link, "rows": rows})


def evaluate_pool_cell(dataset, split, algorithm, candidates, settings, protocol,
                       artifacts, metadata, declaration, partitions, rows):
    started = time.perf_counter()
    u2_metadata = {**metadata, "method": METHOD, "variant_id": "U2",
                   "sampler_parameters": declaration.get("sampler_parameters", {}),
                   "sampler_search": declaration.get("sampler_search", {})}
    canonical = evaluate_cell(dataset, split, algorithm, METHOD, "U2", candidates,
                              settings, protocol, artifacts, u2_metadata, partitions)
    # The runner's selected inner/outer audits are the actual canonical pools,
    # not a regenerated approximation or a historical source of chosen settings.
    root = artifacts.fold_path(dataset["name"], split["fold"]) / f"seed-{metadata['seed']}" / algorithm
    with (root / "U2" / "selection.json").open(encoding="utf-8") as handle:
        selection = json.load(handle)
    with (root / "U2" / "outer.json").open(encoding="utf-8") as handle:
        outer_audit = json.load(handle)
    inner_audit = selection["inner"]
    parameters = canonical["config"]["selected_parameters"]
    epochs = inner_audit["epoch_sequence"]
    epoch = epochs[-1] if epochs else None
    labels, ids = dataset["y"], dataset["row_ids"]
    fit, validation, train, test = (split[key] for key in
                                   ("fit_indices", "validation_indices", "train_indices", "test_indices"))
    inner_pool = accepted_pool(inner_audit, partitions["X_fit"].shape[1])
    outer_pool = accepted_pool(outer_audit, partitions["X_train"].shape[1])
    positive_class = canonical["positive_class"]
    u2_config = {"selected_candidate": selection["selected_candidate"], "parameters": parameters,
                 "sampler_parameters": inner_audit["sampler_parameters"], "epoch": epoch,
                 "selection_policy": "canonical_u2_joint_inner_validation"}
    base_link = {**metadata, "selected_u2_config": u2_config,
                 "selected_u2_config_sha256": snapshot_sha256(u2_config)}
    pool_root = root / "saved_pool"
    inner_reference = save_pool(pool_root / "inner", inner_pool, inner_audit,
                                partitions["X_fit"], labels[fit], ids[fit], base_link)
    outer_reference = save_pool(pool_root / "outer", outer_pool, outer_audit,
                                partitions["X_train"], labels[train], ids[train], base_link)
    inner_assignments = group_assignments(partitions["X_fit"], labels[fit],
                                         inner_audit["support_summary"]["group_support"]["row_group_ids"],
                                         ids[fit], partitions["X_validation"], labels[validation], ids[validation], positive_class)
    outer_assignments = group_assignments(partitions["X_train"], labels[train],
                                         outer_audit["support_summary"]["group_support"]["row_group_ids"],
                                         ids[train], partitions["X_test"], labels[test], ids[test], positive_class)
    configuration, h1_threshold, inner_h1, inner_h1_model, inner_scores, inner_detector = select_h1(
        algorithm, parameters, settings, metadata["seed"], partitions, labels, split,
        inner_pool, epoch, positive_class, ids, protocol, root / "H1_pool_selection", base_link)
    shared_config = {**u2_config, "h1_parameters": configuration,
                     "h1_selection_policy": "saved_pool_inner_validation",
                     "h1_search_space": list(H1_GRID), "thinning_seeds": list(THINNING_SEEDS),
                     "replay_training": "one downstream fit; no iterative generation/retraining/refill"}
    common_link = {**base_link, "selected_config": shared_config,
                   "selected_config_sha256": snapshot_sha256(shared_config),
                   "inner_pool": inner_reference, "outer_pool": outer_reference,
                   "group_assignment_convention": GROUP_CONVENTION}
    detector = prepare_h1_support(partitions["X_train"], labels[train],
                                  categorical_columns=categorical_columns(partitions["outer"]),
                                  parameters=configuration, real_row_ids=ids[train])
    outer_scores = detector.predict(outer_pool["X"])
    outer_h1 = outer_scores["valid"] & (outer_scores["labels"] != -1)
    for assignments, role, name in ((inner_assignments, "inner", "minority_validation_groups.csv"),
                                    (outer_assignments, "outer", "minority_test_groups.csv")):
        assignments["split_fingerprint"] = metadata["split_fingerprint"]
        assignments["selected_config_sha256"] = common_link["selected_config_sha256"]
        assignments.to_csv(pool_root / role / name, index=False)
    canonical.update(common_link)
    canonical.update(pool_accounting(outer_pool, np.ones(len(outer_pool["X"]), dtype=bool),
                                     outer_audit, (partitions["X_train"], labels[train])))
    canonical_prediction = pd.read_csv(root / "U2" / "predictions.csv")["y_pred"].to_numpy()
    canonical["group_recall"] = group_recall(outer_assignments, canonical_prediction,
                                            outer_audit["support_summary"]["group_support"])
    canonical["record_scope"] = "canonical_u2_pool_source"
    rows.append(canonical)
    modes = [("U2_pool", np.ones(len(inner_pool["X"]), dtype=bool),
              np.ones(len(outer_pool["X"]), dtype=bool), None),
             ("H1_pool", inner_h1, outer_h1, None)]
    modes.extend((f"U2_thinning_{seed}", matched_thinning(inner_pool["group_ids"], inner_h1, seed),
                  matched_thinning(outer_pool["group_ids"], outer_h1, seed), seed) for seed in THINNING_SEEDS)
    replay_rows = []
    for variant, inner_mask, outer_mask, thinning_seed in modes:
        replay_started = time.perf_counter()
        folder = root / variant
        folder.mkdir(parents=True, exist_ok=False)
        link = {**common_link, "variant_id": variant, "thinning_seed": thinning_seed}
        if variant == "H1_pool":
            inner_model, threshold = inner_h1_model, h1_threshold
        else:
            inner_model = fit_replay(algorithm, parameters, settings, metadata["seed"],
                                     partitions["X_fit"], labels[fit], inner_pool, inner_mask, epoch)
            probabilities = inner_model.predict_proba(partitions["X_validation"])
            threshold = choose_threshold(labels[validation] == positive_class,
                                         probabilities[:, np.flatnonzero(inner_model.classes_ == positive_class)[0]],
                                         protocol["threshold"])
        inner_metrics, inner_prediction, _ = predictions(inner_model, partitions["X_validation"],
                                                         labels[validation], ids[validation], positive_class,
                                                         threshold, protocol, folder, "validation", link)
        outer_model = fit_replay(algorithm, parameters, settings, metadata["seed"],
                                 partitions["X_train"], labels[train], outer_pool, outer_mask, epoch)
        metrics, prediction, predictions_path = predictions(outer_model, partitions["X_test"],
                                                            labels[test], ids[test], positive_class,
                                                            threshold, protocol, folder, "test", link)
        accounting = pool_accounting(outer_pool, outer_mask, outer_audit, (partitions["X_train"], labels[train]))
        inner_accounting = pool_accounting(inner_pool, inner_mask, inner_audit, (partitions["X_fit"], labels[fit]))
        inner_folder = folder / "inner"
        inner_folder.mkdir()
        save_decisions(inner_folder, inner_pool, inner_mask,
                       {**link, "partition": "inner"}, inner_scores if variant == "H1_pool" else None)
        save_decisions(folder, outer_pool, outer_mask, link, outer_scores if variant == "H1_pool" else None)
        groups = group_recall(outer_assignments, prediction, outer_audit["support_summary"]["group_support"])
        validation_groups = group_recall(inner_assignments, inner_prediction, inner_audit["support_summary"]["group_support"])
        write_json(folder / "groups.json", {**link, "test": groups, "validation": validation_groups,
                                            "outer_acceptance": accounting["group_acceptance"],
                                            "inner_acceptance": inner_accounting["group_acceptance"]})
        config = {**shared_config, "threshold": threshold, "thinning_seed": thinning_seed}
        row = {**metadata, **link, **metrics, **accounting, "config": config,
               "method": "saved_u2_pool_replay", "strategy": variant, "status": "completed",
               "positive_class": positive_class, "threshold": threshold,
               "predictions_path": predictions_path, "validation_metrics": inner_metrics,
               "group_recall": groups, "validation_group_recall": validation_groups,
               "inner_accounting": inner_accounting, "runtime_seconds": time.perf_counter() - replay_started,
               "shared_canonical_u2_runtime_seconds": canonical["runtime_seconds"],
               "classifier_fit_count": 1 if variant == "H1_pool" else 2,
               "outer_classifier_fit_count": 1,
               "inner_model_reused_from_h1_selection": variant == "H1_pool",
               "detector": detector.audit if variant == "H1_pool" else None,
               "inner_detector": inner_detector if variant == "H1_pool" else None,
               "detector_training_seconds": detector.audit["training_seconds"] if variant == "H1_pool" else 0.,
               "detector_inference_seconds": detector.audit["inference_seconds"] if variant == "H1_pool" else 0.}
        write_json(folder / "replay.json", row)
        replay_rows.append(row)
        rows.append(row)
    reference = replay_rows[0]
    differences = []
    for row in replay_rows[1:]:
        differences.append({**common_link, "variant_id": row["variant_id"],
                            "reference_variant": "U2_pool", "thinning_seed": row["thinning_seed"],
                            "deltas": {key: row[key] - reference[key] for key in metrics
                                       if isinstance(row.get(key), (int, float)) and not isinstance(row[key], bool)
                                       and isinstance(reference.get(key), (int, float))}})
    h1_reference = replay_rows[1]
    for row in replay_rows[2:]:
        differences.append({**common_link, "variant_id": "H1_pool",
                            "reference_variant": row["variant_id"], "thinning_seed": row["thinning_seed"],
                            "deltas": {key: h1_reference[key] - row[key] for key in metrics
                                       if isinstance(h1_reference.get(key), (int, float))
                                       and not isinstance(h1_reference[key], bool)
                                       and isinstance(row.get(key), (int, float))}})
    write_json(pool_root / "paired_differences.json", differences)
    write_json(pool_root / "study_cell.json", {**common_link, "runtime_seconds": time.perf_counter() - started,
                                              "manual_retention": "user_only_pending",
                                              "iterative_results_equivalent": False,
                                              "selection_policy": "saved_pool_inner_validation"})


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    result.add_argument("--protocol", default="configs/benchmarks/shoppers_controlled.yaml")
    result.add_argument("--study", default="configs/ablations/stage1_h1.yaml")
    result.add_argument("--output-dir")
    result.add_argument("--data-home", default=".cache/benchmarks")
    result.add_argument("--local-data")
    result.add_argument("--download", action="store_true")
    result.add_argument("--classifiers", nargs="+")
    result.add_argument("--seed", type=int, action="append", dest="seeds")
    result.add_argument("--tuning-budget", type=int)
    result.add_argument("--augmentation-count", type=int)
    result.add_argument("--ugo-max-iterations", type=int)
    result.add_argument("--mlp-epochs", type=int)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--smoke-train-rows", type=int, default=400)
    result.add_argument("--smoke-test-rows", type=int, default=120)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    protocol = load_protocol(args.protocol)
    if protocol["dataset"] != "shoppers" or protocol["status"] != "ready":
        raise ValueError("Saved-pool H1 evaluation requires the ready Shoppers protocol")
    source_study = load_study(args.study)
    declaration = require_variant(source_study, "U2")
    if (declaration["method"] != METHOD or
            declaration.get("sampler_parameters") != {"grouping": "kmeans", "allocation": "stability"} or
            declaration.get("sampler_search") != {"group_alpha": [0., .5, 1.]}):
        raise ValueError("Saved-pool replay requires canonical U2 KMeans/stability with declared alpha search")
    experiment_type = "setup_smoke" if args.smoke else "development"
    algorithms = args.classifiers or [name for name in protocol["classifiers"]
                                      if name != "balanced_random_forest"]
    seeds = args.seeds or ([protocol["seeds"][0]] if args.smoke else protocol["seeds"])
    budget = args.tuning_budget if args.tuning_budget is not None else (1 if args.smoke else protocol["tuning_budget"]["n_candidates"])
    augmentation_count = args.augmentation_count if args.augmentation_count is not None else (40 if args.smoke else protocol["augmentation"]["total"])
    iterations = args.ugo_max_iterations if args.ugo_max_iterations is not None else (1 if args.smoke else 20)
    epochs = args.mlp_epochs if args.mlp_epochs is not None else (3 if args.smoke else protocol.get("mlp_epochs", 160))
    if not algorithms or len(set(algorithms)) != len(algorithms):
        raise ValueError("Classifiers must be nonempty and unique")
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique and nonnegative")
    if budget < 1 or augmentation_count < 0 or iterations < 1 or epochs < 1:
        raise ValueError("Positive tuning/iteration/epoch caps and nonnegative augmentation are required")
    if not args.smoke and budget != 10:
        raise ValueError("Study runs require the canonical ten joint U2 learner/group-alpha proposals")
    for algorithm in algorithms:
        allowed = source_study.get("classifier_methods", protocol.get("classifier_methods", {})).get(algorithm)
        if allowed is not None and METHOD not in allowed:
            raise ValueError(f"{algorithm} is not a declared U2 downstream classifier")
        make_estimator(algorithm, seeds[0], device=args.device, mlp_epochs=epochs)
    settings = {"device": args.device, "mlp_epochs": epochs, "ugo_max_iterations": iterations,
                "augmentation": {"mode": "matched", "total": augmentation_count},
                "method_parameters": {**protocol.get("method_parameters", {}),
                                      **(protocol.get("smoke_method_parameters", {}) if args.smoke else {})}}
    settings["method_parameters"][METHOD] = {**settings["method_parameters"].get(METHOD, {}), "capture_proposals": True}
    study = {**source_study, "study_id": "shoppers_h1_saved_u2_pool_v1",
             "saved_pool_policy": {"h1_search_space": list(H1_GRID), "thinning_seeds": list(THINNING_SEEDS),
                                   "selection": "inner_only_after_canonical_u2_selection",
                                   "manual_retention": "user_only", "real_only_frozen_support": True,
                                   "source_study_sha256": snapshot_sha256(source_study)}}
    arguments = {**vars(args), "experiment_type": experiment_type, "effective_settings": settings,
                 "effective_seeds": seeds, "effective_classifiers": algorithms,
                 "effective_tuning_budget": budget, "experiment_scope": "saved_u2_pool_replay"}
    output = args.output_dir or ("results/smoke" if args.smoke else "results/runs")
    artifacts = BenchmarkArtifacts(output, arguments, protocol, study)
    rows = []
    try:
        dataset = load_dataset(protocol, args.data_home, local_data=args.local_data, download=args.download)
        if len(np.unique(dataset["y"])) != 2:
            raise ValueError("H1 pool evaluation currently requires binary Shoppers labels")
        artifacts.dataset(dataset)
        fingerprint = data_fingerprint(dataset["X"], dataset["y"], dataset["row_ids"])
        for seed in seeds:
            seed_everything(seed)
            for original_split in make_splits(dataset, protocol, seed):
                split = (smoke_split(dataset, original_split, seed, args.smoke_train_rows,
                                     args.smoke_test_rows, protocol["validation_fraction"])
                         if args.smoke else original_split)
                identity = split_fingerprint(dataset, split)
                split_path = artifacts.fold_path(dataset["name"], split["fold"]) / f"seed-{seed}"
                split_path.mkdir(parents=True, exist_ok=True)
                roles = np.full(len(dataset["y"]), "unused", dtype=object)
                for key, role in (("fit_indices", "inner_fit"), ("validation_indices", "inner_validation"),
                                  ("test_indices", "outer_test")):
                    roles[split[key]] = role
                pd.DataFrame({"row_id": dataset["row_ids"], "role": roles, "label": dataset["y"]}).to_csv(split_path / "splits.csv", index=False)
                write_json(split_path / "split.json", {"split_fingerprint": identity,
                                                       "original_outer_test_untouched": args.smoke,
                                                       "outer_train_row_ids": dataset["row_ids"][split["train_indices"]]})
                partitions = prepare_partitions(dataset, split)
                for algorithm in algorithms:
                    candidates = candidate_configs(protocol, algorithm, seed, budget)
                    metadata = {"dataset": dataset["name"], "dataset_fingerprint": fingerprint,
                                "split_fingerprint": identity, "seed": seed, "algorithm": algorithm,
                                "budget_mode": "matched", "experiment_type": experiment_type,
                                "tuning_budget": budget, "candidate_configs": candidates, "fold": split["fold"]}
                    print(f"{experiment_type}: {dataset['name']} seed={seed} {algorithm}/saved_pool", flush=True)
                    evaluate_pool_cell(dataset, split, algorithm, candidates, settings, protocol,
                                       artifacts, metadata, declaration, partitions, rows)
    except Exception as error:
        artifacts.finish(rows, error=f"{type(error).__name__}: {error}")
        raise
    artifacts.finish(rows)
    print(f"Saved {experiment_type} pool replay to {artifacts.path}", flush=True)
    return artifacts.path


if __name__ == "__main__":
    main()
