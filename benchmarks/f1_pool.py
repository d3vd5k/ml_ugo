"""Shoppers immutable U2 proposal replay: U2, neighbor rejection, and F1 only.

python -m benchmarks.f1_pool --smoke
python -m benchmarks.f1_pool --run

F1 is user-retained; F2 is discarded. Smoke uses outer-training rows only;
full runs default to LR/RF/LightGBM, seeds 0..4, ten joint U2 proposals.
"""
from copy import deepcopy
import hashlib
import json
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from benchmarks.flow_partitions import flow_for_partition
from benchmarks.h1_pool import (METHOD, GROUP_CONVENTION, accepted_pool, fit_replay,
                                group_assignments, group_recall, pool_accounting,
                                predictions, save_pool, parser as pool_parser)
from benchmarks.protocols import load_dataset, load_protocol, make_splits
from benchmarks.results import (BenchmarkArtifacts, METRIC_KEYS, choose_threshold,
                                load_study, metric_results, require_variant, snapshot_sha256)
from benchmarks.run import (candidate_configs, evaluate_cell, prepare_partitions,
                            select_score, smoke_split, split_fingerprint)
from experiments import data_fingerprint, seed_everything, write_json

TAU_GRID = (.5, .6, .7, .8)
CLASSIFIERS = ("logistic_regression", "random_forest", "lightgbm")


def original_pool(audit, width):
    """All source proposals, never conditioned on the canonical post-detector."""
    rows = audit["support_summary"]["candidate_log"]
    return {"rows": rows, "accepted_rows": rows,
            "X": np.asarray([r["original_coordinates"] for r in rows], dtype=np.float32).reshape(len(rows), width),
            "y": np.asarray([r["class"] for r in rows], dtype=np.int64),
            "group_ids": np.asarray([r["group_id"] for r in rows], dtype=np.int64),
            "proposal_ids": np.asarray([r["proposal_id"] for r in rows], dtype=np.int64)}


def save_source(folder, audit, X, y, row_ids, link):
    canonical = accepted_pool(audit, X.shape[1])
    reference = save_pool(folder, canonical, audit, X, y, row_ids, link)
    original = original_pool(audit, X.shape[1])
    rows = original["rows"]
    path = folder / "proposals.npz"
    np.savez_compressed(path, original_coordinates=original["X"],
        canonical_u2_coordinates=np.asarray([r["final_coordinates"] for r in rows], dtype=np.float32).reshape(len(rows), X.shape[1]),
        canonical_u2_accepted=np.asarray([r["accepted"] for r in rows], dtype=bool),
        classes=original["y"], original_source_group_ids=original["group_ids"],
        proposal_ids=original["proposal_ids"], rounds=np.asarray([r["round"] for r in rows], dtype=np.int64),
        real_parent_indices=np.asarray([[r["seed_index"], r["partner_index"]] for r in rows], dtype=np.int64).reshape(len(rows), 2),
        real_parent_row_ids=np.asarray([[row_ids[r["seed_index"]], row_ids[r["partner_index"]]] for r in rows], dtype=str).reshape(len(rows), 2))
    reference.update(proposals_path=str(path.resolve()), proposals_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     candidate_log_sha256=hashlib.sha256((folder / "candidate_log.json").read_bytes()).hexdigest())
    write_json(folder / "immutable_source.json", {**link, **reference})
    return canonical, original, reference


def flow_decisions(scores, tau):
    """Threshold one frozen likelihood evaluation; equal priors, support off."""
    threshold = float(np.log(tau) - np.log1p(-tau))
    valid = np.asarray(scores["valid"], dtype=bool)
    ratio = np.asarray(scores["log_ratio"])
    accepted = valid & np.isfinite(ratio) & (ratio >= threshold)
    reasons = np.asarray(scores["reasons"], dtype=object).copy()
    reasons[valid & ~np.isfinite(ratio)] = "non_finite_flow_score"
    reasons[valid & np.isfinite(ratio) & ~accepted] = "flow_incompatible"
    reasons[accepted] = "accepted"
    return {**scores, "accepted": accepted, "reasons": reasons,
            "tau": tau, "log_ratio_threshold": threshold, "support_filter": "off"}


def neighbor_decisions(X, y, pool, parameters):
    from samplers.protected import neighbor_rejection_scores
    started = perf_counter()
    scores = neighbor_rejection_scores(X, y, pool["X"], pool["y"],
                                      k_after=parameters.get("k_after", 1),
                                      batch_size=parameters.get("distance_batch_size", 512))
    return scores, perf_counter() - started


def save_decisions(folder, pool, mask, link, scores=None, tau=None):
    records = []
    for index, source in enumerate(pool["accepted_rows"]):
        record = {key: source[key] for key in ("proposal_id", "round", "seed_index", "partner_index", "group_id", "class")}
        record.update(original_source_group_id=source["group_id"],
            real_parents=[source.get("seed_row_id", source["seed_index"]), source.get("partner_row_id", source["partner_index"])],
            original_coordinates=source["original_coordinates"], final_coordinates=pool["X"][index],
            canonical_u2_coordinates=source["final_coordinates"], canonical_u2_accepted=source["accepted"],
            accepted=bool(mask[index]), reason="accepted" if mask[index] else "rejected",
            tau=tau, threshold=None if tau is None else float(np.log(tau) - np.log1p(-tau)),
            support_filter="off", relocation_applied=link["variant_id"] == "U2_pool",
            relocation_index=source["relocation_index"] if link["variant_id"] == "U2_pool" else -1,
            canonical_u2_relocation_index=source["relocation_index"])
        if scores is not None:
            record["reason"] = str(scores["reasons"][index])
            for key in ("valid", "neighbor_label", "log_likelihood_intended", "log_likelihood_opposing",
                        "log_ratio", "compatibility_score", "log_determinant"):
                if key in scores:
                    value = np.asarray(scores[key])[index]
                    record[key] = value.item() if isinstance(value, np.generic) else value
            if "log_determinant" in record:
                for role in ("intended", "opposing"):
                    record["latent_log_likelihood_" + role] = record["log_likelihood_" + role] - record["log_determinant"]
        records.append(record)
    write_json(folder / "candidate_log.json", {**link, "rows": records})


def accounting(pool, mask, audit, X, y, scores=None):
    result = pool_accounting(pool, mask, audit, (X, y))
    canonical_counts = {}
    for row in pool["rows"]:
        if row["accepted"]:
            group = int(row["group_id"])
            canonical_counts[group] = canonical_counts.get(group, 0) + 1
    for group in result["group_acceptance"]:
        group["u2_accepted_count"] = canonical_counts.get(group["group_id"], 0)
        # F1 is not a subset of U2 outputs: acceptance_of_u2 is not meaningful.
        group.pop("acceptance_of_u2", None)
    result["proposal_count"] = len(pool["rows"])
    result["non_finite_candidate_count"] = int((~np.isfinite(pool["X"]).all(axis=1)).sum())
    result["feature_invalid_count"] = result["non_finite_candidate_count"]
    result["score_invalid_count"] = result["non_finite_candidate_count"]
    if scores is not None:
        reasons, counts = np.unique(np.asarray(scores["reasons"], dtype=str), return_counts=True)
        result["decision_reason_counts"] = dict(zip(reasons.tolist(), counts.tolist()))
        result["score_invalid_count"] = int((~np.asarray(scores["valid"], dtype=bool)).sum())
        result["feature_invalid_count"] = sum(result["decision_reason_counts"].get(reason, 0)
            for reason in ("nonfinite_domain", "invalid_onehot", "invalid_features"))
        result["unknown_detector_category_count"] = result["decision_reason_counts"].get("unknown_flow_category", 0)
        result["non_finite_score_count"] = sum(result["decision_reason_counts"].get(reason, 0)
            for reason in ("nonfinite_score", "non_finite_flow_score"))
    return result


def validation_fit(algorithm, parameters, settings, seed, partitions, y, fit, validation,
                   pool, mask, epoch, positive, protocol):
    model = fit_replay(algorithm, parameters, settings, seed, partitions["X_fit"], y[fit], pool, mask, epoch)
    probability = model.predict_proba(partitions["X_validation"])
    threshold = choose_threshold(y[validation] == positive,
        probability[:, np.flatnonzero(model.classes_ == positive)[0]], protocol["threshold"])
    metrics = metric_results(y[validation], probability, model.classes_, positive, threshold,
                             protocol["primary_metric"], protocol["threshold"].get("fpr", .05))
    return model, threshold, metrics


def evaluate_pool_cell(dataset, split, algorithm, candidates, settings, protocol,
                       artifacts, metadata, declaration, f1_declaration, partitions, rows):
    started = perf_counter()
    canonical = evaluate_cell(dataset, split, algorithm, METHOD, "U2", candidates, settings, protocol,
        artifacts, {**metadata, "method": METHOD, "variant_id": "U2",
                    "sampler_parameters": declaration["sampler_parameters"],
                    "sampler_search": declaration["sampler_search"]}, partitions)
    root = artifacts.fold_path(dataset["name"], split["fold"]) / f"seed-{metadata['seed']}" / algorithm
    selection = json.loads((root / "U2" / "selection.json").read_text())
    audits = {"inner": selection["inner"], "outer": json.loads((root / "U2" / "outer.json").read_text())}
    epoch_sequence = audits["inner"]["epoch_sequence"]
    epoch = epoch_sequence[-1] if epoch_sequence else None
    parameters = canonical["config"]["selected_parameters"]
    u2_config = {"selected_candidate": selection["selected_candidate"], "parameters": parameters,
                 "sampler_parameters": audits["inner"]["sampler_parameters"], "epoch": epoch,
                 "selection_policy": "canonical_u2_joint_inner_validation"}
    link = {**metadata, "selected_u2_config": u2_config, "selected_u2_config_sha256": snapshot_sha256(u2_config)}
    y, ids = dataset["y"], dataset["row_ids"]
    fit, validation, train, test = (split[key] for key in ("fit_indices", "validation_indices", "train_indices", "test_indices"))
    positive, seed = canonical["positive_class"], metadata["seed"]
    pool_root = root / "saved_pool"
    canonical_pools, originals, references, assignments, detectors, raw_scores, inference_times = {}, {}, {}, {}, {}, {}, {}
    flow_parameters = {**f1_declaration["sampler_parameters"], **settings.get("flow_parameters", {})}
    for role, real_indices, evaluation_indices, real_key, evaluation_key in (
            ("inner", fit, validation, "X_fit", "X_validation"), ("outer", train, test, "X_train", "X_test")):
        X = partitions[real_key]
        canonical_pools[role], originals[role], references[role] = save_source(
            pool_root / role, audits[role], X, y[real_indices], ids[real_indices], {**link, "partition": role})
        assignments[role] = group_assignments(X, y[real_indices],
            audits[role]["support_summary"]["group_support"]["row_group_ids"], ids[real_indices],
            partitions[evaluation_key], y[evaluation_indices], ids[evaluation_indices], positive)
        assignments[role].to_csv(pool_root / role / "minority_evaluation_groups.csv", index=False)
        detectors[role] = flow_for_partition(dataset, split, partitions, role, seed, flow_parameters, settings["device"])
        tick = perf_counter()
        raw_scores[role] = detectors[role].predict(originals[role]["X"], originals[role]["y"], .5)
        inference_times[role] = perf_counter() - tick
    model_path = pool_root / "flow.pt"
    detectors["inner"].core.save(model_path)
    flow_reference = {"path": str(model_path.resolve()), "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
    tuning_folder = root / "F1_pool_selection"
    tuning_folder.mkdir()
    best = None
    for tau in TAU_GRID:
        scores = flow_decisions(raw_scores["inner"], tau)
        model, threshold, metrics = validation_fit(algorithm, parameters, settings, seed, partitions, y, fit, validation,
            originals["inner"], scores["accepted"], epoch, positive, protocol)
        write_json(tuning_folder / f"tau-{tau}.json", {**link, "tau": tau, "threshold": threshold, "metrics": metrics,
            "accepted_count": int(scores["accepted"].sum()), "flow_model": flow_reference, "support_filter": "off"})
        score = select_score(metrics, "average_precision")
        if best is None or score > best[0]:
            best = score, tau, model, threshold
    _, tau, f1_inner_model, f1_threshold = best
    flow_scores = {role: flow_decisions(value, tau) for role, value in raw_scores.items()}
    neighbors, neighbor_times = {}, {}
    for role, key, indices in (("inner", "X_fit", fit), ("outer", "X_train", train)):
        neighbors[role], neighbor_times[role] = neighbor_decisions(partitions[key], y[indices], originals[role], u2_config["sampler_parameters"])
    config = {**u2_config, "f1_tau": tau, "f1_tau_search": list(TAU_GRID), "support_filter": "off",
              "f1_selection_policy": "real_inner_validation_average_precision_after_u2_lock",
              "flow_parameters": flow_parameters, "replay_training": "one fit; no iterative generation or quota refill"}
    common = {**link, "selected_config": config, "selected_config_sha256": snapshot_sha256(config),
              "inner_pool": references["inner"], "outer_pool": references["outer"], "flow_model": flow_reference,
              "group_assignment_convention": GROUP_CONVENTION, "manual_retention": "user_only_pending"}
    canonical.update(common)
    canonical.update(accounting(canonical_pools["outer"], np.ones(len(canonical_pools["outer"]["X"]), dtype=bool), audits["outer"], partitions["X_train"], y[train]))
    canonical["group_recall"] = group_recall(assignments["outer"], pd.read_csv(root / "U2" / "predictions.csv")["y_pred"].to_numpy(), audits["outer"]["support_summary"]["group_support"])
    canonical["record_scope"] = "canonical_u2_pool_source"
    rows.append(canonical)
    write_json(tuning_folder / "selection.json", {**common, "selected_tau": tau,
        "validation_average_precision": best[0], "threshold": f1_threshold,
        "model_and_representation_frozen_across_tau": True})
    replay_rows = []
    for variant in ("U2_pool", "N_reject_pool", "F1_pool"):
        tick = perf_counter()
        pools = canonical_pools if variant == "U2_pool" else originals
        scores = None if variant == "U2_pool" else neighbors if variant == "N_reject_pool" else flow_scores
        masks = {role: np.ones(len(pools[role]["X"]), dtype=bool) if scores is None else np.asarray(scores[role]["accepted"], dtype=bool) for role in ("inner", "outer")}
        folder = root / variant
        folder.mkdir()
        variant_link = {**common, "variant_id": variant}
        if variant == "F1_pool":
            inner_model, threshold = f1_inner_model, f1_threshold
        else:
            inner_model, threshold, _ = validation_fit(algorithm, parameters, settings, seed, partitions, y, fit, validation,
                pools["inner"], masks["inner"], epoch, positive, protocol)
        inner_metrics, inner_prediction, _ = predictions(inner_model, partitions["X_validation"], y[validation], ids[validation], positive,
            threshold, protocol, folder, "validation", variant_link)
        outer_model = fit_replay(algorithm, parameters, settings, seed, partitions["X_train"], y[train], pools["outer"], masks["outer"], epoch)
        metrics, prediction, prediction_path = predictions(outer_model, partitions["X_test"], y[test], ids[test], positive,
            threshold, protocol, folder, "test", variant_link)
        joblib.dump(inner_model, folder / "inner_classifier.joblib")
        joblib.dump(outer_model, folder / "outer_classifier.joblib")
        inner_folder = folder / "inner"
        inner_folder.mkdir()
        save_decisions(inner_folder, pools["inner"], masks["inner"], {**variant_link, "partition": "inner"},
            None if scores is None else scores["inner"], tau if variant == "F1_pool" else None)
        save_decisions(folder, pools["outer"], masks["outer"], {**variant_link, "partition": "outer"},
            None if scores is None else scores["outer"], tau if variant == "F1_pool" else None)
        outer_accounting = accounting(pools["outer"], masks["outer"], audits["outer"], partitions["X_train"], y[train],
                                      None if scores is None else scores["outer"])
        inner_accounting = accounting(pools["inner"], masks["inner"], audits["inner"], partitions["X_fit"], y[fit],
                                      None if scores is None else scores["inner"])
        groups = group_recall(assignments["outer"], prediction, audits["outer"]["support_summary"]["group_support"])
        validation_groups = group_recall(assignments["inner"], inner_prediction, audits["inner"]["support_summary"]["group_support"])
        duration = perf_counter() - tick
        row = {**metadata, **variant_link, **metrics, **outer_accounting, "config": {**config, "threshold": threshold},
            "method": "saved_u2_pool_replay", "strategy": variant, "status": "completed", "positive_class": positive,
            "threshold": threshold, "predictions_path": prediction_path, "validation_metrics": inner_metrics,
            "group_recall": groups, "validation_group_recall": validation_groups, "inner_accounting": inner_accounting,
            "runtime_seconds": duration, "shared_canonical_u2_runtime_seconds": canonical["runtime_seconds"],
            "classifier_fit_count": 5 if variant == "F1_pool" else 2, "outer_classifier_fit_count": 1,
            "inner_model_reused_from_f1_selection": variant == "F1_pool",
            "detector": deepcopy(detectors["outer"].audit) if variant == "F1_pool" else None,
            "inner_detector": deepcopy(detectors["inner"].audit) if variant == "F1_pool" else None,
            "detector_training_seconds": detectors["inner"].audit["training_seconds"] if variant == "F1_pool" else 0.,
            "detector_inference_seconds": sum(inference_times.values()) if variant == "F1_pool" else sum(neighbor_times.values()) if variant == "N_reject_pool" else 0.}
        write_json(folder / "groups.json", {**variant_link, "test": groups, "validation": validation_groups,
            "outer_acceptance": outer_accounting["group_acceptance"], "inner_acceptance": inner_accounting["group_acceptance"]})
        write_json(folder / "replay.json", row)
        rows.append(row)
        replay_rows.append(row)
    differences = [{**common, "variant_id": row["variant_id"], "reference_variant": reference["variant_id"],
                    "deltas": {key: row[key] - reference[key] for key in METRIC_KEYS
                               if isinstance(row.get(key), (int, float)) and isinstance(reference.get(key), (int, float))}}
                   for row in replay_rows[1:] for reference in (replay_rows[0], canonical)]
    differences.append({**common, "variant_id": "F1_pool", "reference_variant": "N_reject_pool",
        "deltas": {key: replay_rows[2][key] - replay_rows[1][key] for key in METRIC_KEYS
                   if isinstance(replay_rows[2].get(key), (int, float)) and isinstance(replay_rows[1].get(key), (int, float))}})
    write_json(pool_root / "paired_differences.json", differences)
    total_wall = perf_counter() - started
    write_json(pool_root / "study_cell.json", {**common, "runtime_seconds": total_wall,
        "total_wall_seconds": total_wall, "wall_time_scope": "canonical U2 selection, immutable pools, both detectors, tau selection, all replays and artifacts",
        "iterative_results_equivalent": False, "detector_core_count": 1, "classifier_real_access_unchanged": True,
        "H1": "discarded_not_evaluated", "F2": "not_implemented_or_evaluated"})


def parser():
    result = pool_parser()
    result.description = __doc__
    result.set_defaults(study="configs/ablations/stage1_f1.yaml")
    result.add_argument("--validate", action="store_true", help="Validate declaration and run settings without loading data or fitting")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    protocol, study = load_protocol(args.protocol), load_study(args.study)
    if protocol["dataset"] != "shoppers" or protocol["status"] != "ready" or protocol["primary_metric"] != "average_precision":
        raise ValueError("F1 pool requires ready Shoppers average-precision protocol")
    declaration, f1 = require_variant(study, "U2"), require_variant(study, "F1")
    if (declaration["method"] != METHOD or declaration.get("sampler_parameters") != {"grouping": "kmeans", "allocation": "stability"}
            or declaration.get("sampler_search") != {"group_alpha": [0., .5, 1.]}):
        raise ValueError("Canonical U2 KMeans/stability and alpha grid must remain unchanged")
    if f1.get("sampler_search") != {"group_alpha": [0., .5, 1.], "f1_tau": list(TAU_GRID)}:
        raise ValueError("F1 requires the declared independent alpha and tau grids")
    if "H1" in study.get("variants", {}) and study["variants"]["H1"].get("available"):
        raise ValueError("H1 was discarded and cannot enter active F1 selection")
    algorithms = args.classifiers or list(CLASSIFIERS)
    seeds = args.seeds or ([0] if args.smoke else list(range(5)))
    budget = args.tuning_budget if args.tuning_budget is not None else (1 if args.smoke else 10)
    count = args.augmentation_count if args.augmentation_count is not None else (40 if args.smoke else protocol["augmentation"]["total"])
    iterations = args.ugo_max_iterations if args.ugo_max_iterations is not None else (1 if args.smoke else 20)
    epochs = args.mlp_epochs if args.mlp_epochs is not None else (3 if args.smoke else protocol.get("mlp_epochs", 160))
    if not algorithms or len(set(algorithms)) != len(algorithms) or set(algorithms) - set(CLASSIFIERS):
        raise ValueError("Only unique LR, RF, LightGBM classifiers are supported")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be unique and nonnegative")
    if budget < 1 or count < 0 or iterations < 1 or epochs < 1 or (not args.smoke and budget != 10):
        raise ValueError("Positive caps and full-run canonical ten joint proposals required")
    settings = {"device": args.device, "mlp_epochs": epochs, "ugo_max_iterations": iterations,
        "augmentation": {"mode": "matched", "total": count},
        "method_parameters": {**protocol.get("method_parameters", {}), **(protocol.get("smoke_method_parameters", {}) if args.smoke else {})}}
    settings["method_parameters"][METHOD] = {**settings["method_parameters"].get(METHOD, {}), "capture_proposals": True}
    if args.smoke:
        settings["flow_parameters"] = {"flow_epochs": 2}
    if args.validate:
        print(json.dumps({"status": "valid", "classifiers": algorithms, "seeds": seeds, "joint_u2_proposals": budget,
                          "tau_grid": TAU_GRID, "support_filter": "off", "retention": "user_only_pending"}))
        return None
    experiment_type = "setup_smoke" if args.smoke else "development"
    saved_study = {**study, "study_id": "shoppers_f1_saved_u2_pool_v1", "saved_pool_policy": {
        "selection": "lock_canonical_u2_then_tau_on_real_inner_validation", "tau_grid": list(TAU_GRID),
        "support_filter": "off", "manual_retention": "user_only_pending", "source_study_sha256": snapshot_sha256(study)}}
    arguments = {**vars(args), "experiment_type": experiment_type, "effective_settings": settings,
        "effective_classifiers": algorithms, "effective_seeds": seeds, "effective_tuning_budget": budget,
        "experiment_scope": "saved_u2_pool_replay"}
    artifacts = BenchmarkArtifacts(args.output_dir or ("results/smoke" if args.smoke else "results/runs"), arguments, protocol, saved_study)
    rows = []
    try:
        dataset = load_dataset(protocol, args.data_home, local_data=args.local_data, download=args.download)
        if len(np.unique(dataset["y"])) != 2:
            raise ValueError("F1 pool requires binary Shoppers")
        artifacts.dataset(dataset)
        fingerprint = data_fingerprint(dataset["X"], dataset["y"], dataset["row_ids"])
        for seed in seeds:
            seed_everything(seed)
            for original_split in make_splits(dataset, protocol, seed):
                split = smoke_split(dataset, original_split, seed, args.smoke_train_rows, args.smoke_test_rows, protocol["validation_fraction"]) if args.smoke else original_split
                identity = split_fingerprint(dataset, split)
                split_folder = artifacts.fold_path(dataset["name"], split["fold"]) / f"seed-{seed}"
                split_folder.mkdir(parents=True, exist_ok=True)
                roles = np.full(len(dataset["y"]), "unused", dtype=object)
                for key, role in (("fit_indices", "inner_fit"), ("validation_indices", "inner_validation"), ("test_indices", "outer_test")):
                    roles[split[key]] = role
                pd.DataFrame({"row_id": dataset["row_ids"], "role": roles, "label": dataset["y"]}).to_csv(split_folder / "splits.csv", index=False)
                write_json(split_folder / "split.json", {"split_fingerprint": identity, "original_outer_test_untouched": args.smoke,
                    "outer_train_row_ids": dataset["row_ids"][split["train_indices"]]})
                partitions = prepare_partitions(dataset, split)
                for algorithm in algorithms:
                    candidates = candidate_configs(protocol, algorithm, seed, budget)
                    metadata = {"dataset": dataset["name"], "dataset_fingerprint": fingerprint, "split_fingerprint": identity,
                        "seed": seed, "algorithm": algorithm, "fold": split["fold"], "budget_mode": "matched",
                        "experiment_type": experiment_type, "tuning_budget": budget, "candidate_configs": candidates}
                    print(f"{experiment_type}: shoppers seed={seed} {algorithm}/F1_saved_pool", flush=True)
                    evaluate_pool_cell(dataset, split, algorithm, candidates, settings, protocol, artifacts,
                                       metadata, declaration, f1, partitions, rows)
    except Exception as error:
        artifacts.finish(rows, error=f"{type(error).__name__}: {error}")
        raise
    artifacts.finish(rows)
    print(f"Saved {experiment_type} F1 pool replay to {artifacts.path}", flush=True)
    return artifacts.path


if __name__ == "__main__":
    main()
