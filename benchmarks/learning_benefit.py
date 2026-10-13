"""Shoppers-only, actual logistic-regression learning-benefit experiment on U2.

GUIDANCE is training information for ranking only. Validation and test never
enter fitting, preprocessing, grouping, interpolation or relocation references.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from benchmarks.protocols import load_dataset, load_protocol, make_splits
from benchmarks.results import BenchmarkArtifacts, choose_threshold, load_study, metric_results
from benchmarks.run import augmentation_targets, as_array, candidate_configs, measure_memory, train_method
from classifiers.factory import fit_estimator, make_estimator
from datasets.scripts.climb import categorical_columns, make_preprocessor
from experiments import _preprocessing_state, data_fingerprint, seed_everything, write_json
from samplers.protected import prepare_protected_groups, protected_ugo

GUIDANCE_SEED_OFFSET = 104729
ALPHAS = (0., .5, 1.)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def learning_split(dataset, protocol, seed, smoke=False, fit_rows=400, evaluation_rows=120):
    """Keep the canonical outer/validation roles, then reserve GUIDANCE from inner fit."""
    original = make_splits(dataset, protocol, seed)[0]
    split = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in original.items()}
    if smoke:
        # Smoke evaluation is drawn exclusively from outer training, never official test.
        available = original["fit_indices"]
        if fit_rows + evaluation_rows >= len(available):
            raise ValueError("Smoke fitting and evaluation sizes must leave unused training rows")
        chosen, _ = train_test_split(available, train_size=fit_rows + evaluation_rows,
                                    random_state=seed + 17, stratify=dataset["y"][available])
        split["fit_indices"], split["test_indices"] = train_test_split(
            chosen, train_size=fit_rows, random_state=seed + 19, stratify=dataset["y"][chosen])
        validation = original["validation_indices"]
        if evaluation_rows < len(validation):
            split["validation_indices"], _ = train_test_split(
                validation, train_size=evaluation_rows, random_state=seed + 17,
                stratify=dataset["y"][validation])
    inner = split["fit_indices"]
    split["fit_indices"], split["guidance_indices"] = train_test_split(
        inner, test_size=.2, random_state=seed + GUIDANCE_SEED_OFFSET, stratify=dataset["y"][inner])
    split["inner_fitting_indices"] = inner
    split["canonical_validation_indices"] = original["validation_indices"]
    split["canonical_test_indices"] = original["test_indices"]
    return split


def split_identity(dataset, split):
    digest = hashlib.sha256()
    for role in ("fit_indices", "guidance_indices", "validation_indices", "test_indices"):
        digest.update(role.encode())
        digest.update(json.dumps(dataset["row_ids"][split[role]].tolist()).encode())
    return digest.hexdigest()


def prepare_learning_partitions(dataset, split):
    preprocessor = make_preprocessor(dataset["X"].iloc[split["fit_indices"]])
    X_fit = preprocessor.fit_transform(dataset["X"].iloc[split["fit_indices"]])
    return {"preprocessor": preprocessor, "X_fit": X_fit,
            **{f"X_{role}": preprocessor.transform(dataset["X"].iloc[split[f"{role}_indices"]])
               for role in ("guidance", "validation", "test")}}


def metrics_for(model, X, y, protocol, threshold=None):
    probabilities = model.predict_proba(as_array(X))
    positive = int(protocol.get("positive_class", 1))
    column = np.flatnonzero(model.classes_ == positive)[0]
    if threshold is None:
        threshold = choose_threshold(y == positive, probabilities[:, column], protocol["threshold"])
    metrics = metric_results(y, probabilities, model.classes_, positive, threshold,
                            protocol["primary_metric"], protocol["threshold"].get("fpr", .05))
    predicted = np.where(probabilities[:, column] >= threshold, positive,
                         model.classes_[np.flatnonzero(model.classes_ != positive)[0]])
    metrics.update(accuracy=float(accuracy_score(y, predicted)),
                   precision=float(precision_score(y == positive, predicted == positive, zero_division=0)),
                   ordinary_cross_entropy=float(log_loss(y, probabilities, labels=model.classes_)),
                   balanced_cross_entropy=float(log_loss(y, probabilities, labels=model.classes_,
                                                          sample_weight=compute_sample_weight("balanced", y))))
    return metrics, probabilities


def parent_training(prefix_X, prefix_y, real_X, real_y, anchors, partners):
    """Each selected proposal contributes two real parents of total weight one."""
    indices = np.column_stack((anchors, partners)).reshape(-1).astype(np.int64)
    return (np.concatenate((as_array(prefix_X), real_X[indices])),
            np.concatenate((as_array(prefix_y), real_y[indices])),
            np.r_[np.ones(len(prefix_y)), np.full(len(indices), .5)])


class Learner:
    """One actual sklearn procedure; diagnostic models never become live models."""
    def __init__(self, parameters, seed, validation):
        self.estimator = make_estimator("logistic_regression", seed, parameters=parameters, device="cpu")
        self.validation = validation
        self.counts = {"operational": 0, "parentdup": 0, "paired": 0, "initial": 0}
        self.seconds = {key: 0. for key in self.counts}

    def fit(self, X, y, weights=None, kind="operational"):
        start = perf_counter()
        model = fit_estimator(self.estimator, X, y, sample_weight=weights, validation_data=self.validation)
        self.seconds[kind] += perf_counter() - start
        self.counts[kind] += 1
        return model


def tune_u2(dataset, split, partitions, seed, settings, protocol, candidates, folder):
    """Canonical joint ten-proposal search, no full-outer refit and no guide access."""
    X, y = partitions["X_fit"], dataset["y"][split["fit_indices"]]
    categorical = categorical_columns(partitions["preprocessor"])
    base = {**settings["method_parameters"].get("ugo_groups_only", {}),
            "grouping": "kmeans", "allocation": "stability"}
    targets = augmentation_targets(y, settings["augmentation"])
    prepared = prepare_protected_groups(X, y, seed, targets, base)
    prepared["real_row_ids"] = dataset["row_ids"][split["fit_indices"]]
    validation = (partitions["X_validation"], dataset["y"][split["validation_indices"]])
    records, best, fits = [], None, 0
    for index, parameters in enumerate(candidates):
        values = {**base, "group_alpha": ALPHAS[index % len(ALPHAS)]}
        candidate_settings = {**settings, "method_parameters": {"ugo_groups_only": values}}
        model, audit = train_method("logistic_regression", "ugo_groups_only", parameters, X, y,
                                    categorical, seed, candidate_settings, validation, group_partition=prepared)
        metrics, _ = metrics_for(model, *validation, protocol)
        fits += audit["classifier_fit_count"]
        record = {"candidate_id": index, "parameters": parameters, "metrics": metrics, **audit}
        write_json(folder / f"candidate-{index}.json", record)
        records.append(record)
        if best is None or metrics["average_precision"] > best[0]:
            best = metrics["average_precision"], index, parameters, values
    selected = {"candidate_id": best[1], "parameters": best[2], "sampler_parameters": best[3],
                "candidate_configs": candidates, "classifier_fit_count": fits,
                "selection_metric": "validation_average_precision", "outer_refit": False}
    write_json(folder / "selection.json", selected)
    return selected, prepared


def save_model(folder, model, dataset, split, partitions, protocol):
    joblib.dump(model, folder / "model.joblib")
    validation_metrics, probabilities = metrics_for(model, partitions["X_validation"],
                                                    dataset["y"][split["validation_indices"]], protocol)
    threshold = validation_metrics["threshold"]
    output = {"validation": validation_metrics}
    for role in ("validation", "test"):
        indices = split[f"{role}_indices"]
        metrics, probabilities = metrics_for(model, partitions[f"X_{role}"], dataset["y"][indices],
                                            protocol, threshold)
        output[role] = metrics
        frame = pd.DataFrame({"row_id": dataset["row_ids"][indices], "label": dataset["y"][indices]})
        for column, label in enumerate(model.classes_):
            frame[f"probability_{label}"] = probabilities[:, column]
        frame.to_csv(folder / f"{role}_predictions.csv", index=False)
    write_json(folder / "metrics.json", output)
    return output


def save_round(folder, index, record):
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in record.items() if isinstance(value, np.ndarray)}
    arrays["vG_bias"] = np.asarray(record["vG_bias"])
    path = folder / f"round-{index:02d}-pool.npz"
    np.savez_compressed(path, **arrays)
    pool_sha = sha256(path)
    write_json(folder / f"round-{index:02d}.json",
               {**{key: value for key, value in record.items() if key not in arrays}, "pool_sha256": pool_sha,
                "array_fields": sorted(arrays)})
    return pool_sha


def paired_differences(rows):
    """Descriptive per-seed pairs, sample SD (not independent-fold inference)."""
    result = []
    for phase in ("initial", "iterative"):
        random = {row["seed"]: row for row in rows if row.get("phase") == phase and row["variant_id"] == f"{phase}_pool_random"}
        utility = {row["seed"]: row for row in rows if row.get("phase") == phase and row["variant_id"] == f"{phase}_pool_utility"}
        keys = sorted(set(random) & set(utility))
        if not keys:
            continue
        for role, field in (("test", "metrics"), ("validation", "validation_metrics")):
            for metric in sorted(random[keys[0]][field]):
                if not isinstance(random[keys[0]][field][metric], (int, float)):
                    continue
                values = [utility[seed][field][metric] - random[seed][field][metric] for seed in keys]
                result.append({"phase": phase, "evaluation_role": role, "metric": metric, "seeds": keys, "utility_minus_random": values,
                               "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                               "matched_counts": [random[s]["augmentation"]["actual_count"] == utility[s]["augmentation"]["actual_count"] for s in keys],
                               "shared_pool": phase == "initial"})
    return result


def run_trajectory(X, y, seed, categorical, settings, parameters, prepared, learner,
                   initial_model=None, selector=None, rounds=None):
    """Use the existing protected pipeline, not a second U2 implementation."""
    calls = []

    def model_fn(features, labels):
        model = initial_model if initial_model is not None and not calls else learner.fit(features, labels)
        calls.append(len(labels))
        return model

    start = perf_counter()
    result = protected_ugo(X, y, seed, categorical,
                           settings["ugo_max_iterations"] if rounds is None else rounds, "cpu",
                           model_fn, augmentation_targets(y, settings["augmentation"]),
                           {**parameters, **({"capture_proposals": True} if selector is None else {})}, prepared, candidate_selector=selector)
    # Unchanged U2 train_method always performs a final downstream fit.
    model = learner.fit(result.X, result.y)
    return model, result, perf_counter() - start, calls


def save_batch_diagnostics(folder, rounds, real_X, real_y, learner, protocol, validation):
    """Replay each shared checkpoint/pool; neither diagnostic fit becomes live."""
    prefix_X, prefix_y = real_X, real_y
    diagnostics = []
    for index, record in enumerate(rounds, 1):
        comparisons = {}
        masks = {branch: record[f"{branch}_selected"] for branch in ("random", "utility")}
        if masks["random"].sum() != masks["utility"].sum():
            raise ValueError("Paired same-pool diagnostic counts must be matched")
        for branch, selected in masks.items():
            X = np.concatenate((prefix_X, record["final_coordinates"][selected]))
            y = np.concatenate((prefix_y, record["labels"][selected]))
            model = learner.fit(X, y, kind="paired")
            parent_X, parent_y, weights = parent_training(prefix_X, prefix_y, real_X, real_y,
                record["anchor_indices"][selected], record["partner_indices"][selected])
            parent_model = learner.fit(parent_X, parent_y, weights, kind="parentdup")
            comparison = {"actual": metrics_for(model, *validation, protocol)[0],
                          "parentdup": metrics_for(parent_model, *validation, protocol)[0],
                          "batch_count": int(selected.sum()), "parent_added_weight": float(weights[len(prefix_y):].sum()),
                          "parent_rows": int(2 * selected.sum()), "prefix_count": len(prefix_y)}
            comparisons[branch] = comparison
            destination = folder / "paired_checkpoints" / f"round-{index:02d}" / branch
            destination.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, destination / "actual-model.joblib")
            joblib.dump(parent_model, destination / "parent-model.joblib")
            np.savez_compressed(destination / "parent-training.npz", X=parent_X, y=parent_y, sample_weight=weights)
            write_json(destination / "metrics.json", comparison)
        diagnostics.append({"round": record["round"], "checkpoint_state": record["state_before_hash"],
                            "matched_counts": True, "comparisons": comparisons})
        selected = record["selected"]
        prefix_X = np.concatenate((prefix_X, record["final_coordinates"][selected]))
        prefix_y = np.concatenate((prefix_y, record["labels"][selected]))
    write_json(folder / "paired_checkpoints.json", diagnostics)


def evaluate_seed(dataset, split, partitions, seed, settings, protocol, candidates, folder, metadata):
    from samplers.learning_benefit import CandidatePoolSelector

    folder.mkdir(parents=True, exist_ok=True)
    X, y = partitions["X_fit"], dataset["y"][split["fit_indices"]]
    validation = (partitions["X_validation"], dataset["y"][split["validation_indices"]])
    guidance_y = dataset["y"][split["guidance_indices"]]
    categorical = categorical_columns(partitions["preprocessor"])
    joblib.dump(partitions["preprocessor"], folder / "preprocessor.joblib")
    write_json(folder / "preprocessing.json", _preprocessing_state(partitions["preprocessor"]))
    np.savez_compressed(folder / "references.npz", X_fit=X, y_fit=y,
                        X_guidance=partitions["X_guidance"], y_guidance=guidance_y,
                        **{key: value for key, value in split.items() if key.endswith("indices")})
    roles = np.full(len(dataset["y"]), "unused", dtype=object)
    for role in ("fit", "guidance", "validation", "test"):
        roles[split[f"{role}_indices"]] = role
    pd.DataFrame({"row_id": dataset["row_ids"], "role": roles, "label": dataset["y"]}).to_csv(folder / "splits.csv", index=False)
    write_json(folder / "split.json", {"split_fingerprint": split_identity(dataset, split),
        "guidance_seed": seed + GUIDANCE_SEED_OFFSET, "outer_refit": False,
        "reference_row_ids": dataset["row_ids"][split["fit_indices"]],
        "roles": {role: dataset["row_ids"][split[f"{role}_indices"]] for role in ("fit", "guidance", "validation", "test")}})
    selected, prepared = tune_u2(dataset, split, partitions, seed, settings, protocol, candidates, folder)
    write_json(folder / "groups.json", prepared)
    learner = Learner(selected["parameters"], seed, validation)
    initial_model = learner.fit(X, y, kind="initial")
    joblib.dump(initial_model, folder / "initial-model.joblib")
    write_json(folder / "initial-checkpoint.json", {"coefficient": initial_model.coef_, "intercept": initial_model.intercept_,
        "validation": metrics_for(initial_model, *validation, protocol)[0], "real_training_count": len(y)})
    parameters = selected["sampler_parameters"]
    rows, initial = [], {}

    def record_variant(phase, variant, model, result, runtime, extra=None):
        destination = folder / f"{phase}_{variant}"
        destination.mkdir(parents=True, exist_ok=True)
        evaluated = save_model(destination, model, dataset, split, partitions, protocol)
        actual = len(result.y) - len(y)
        selector_audit = (extra or {}).get("selector", {})
        selector_rounds = selector_audit.get("rounds", [])
        guidance_seconds = selector_audit.get("guidance_scoring_seconds_total", 0.)
        terminal_guidance = guidance_seconds - sum(row["guidance_scoring_seconds"] for row in selector_rounds)
        diagnostic_seconds = sum(row["diagnostic_seconds"] for row in selector_rounds)
        random_diagnostic = sum(row["diagnostic_only_random_seconds"] for row in selector_rounds)
        if "random" in variant:
            diagnostic_seconds += terminal_guidance
            random_diagnostic += terminal_guidance
        costs = {"observed_trajectory_seconds": runtime,
                 "operational_trajectory_seconds": runtime - diagnostic_seconds,
                 "diagnostic_seconds": diagnostic_seconds,
                 "random_diagnostic_scoring_seconds": random_diagnostic,
                 "verification_seconds": sum(row["verification_seconds"] for row in selector_rounds),
                 "generation_seconds": sum(row.get("generation_seconds", 0.) for row in result.rounds),
                 "candidate_scoring_seconds": sum(row["scoring_seconds"] for row in selector_rounds),
                 "guidance_scoring_seconds": guidance_seconds}
        extra = {**costs, **(extra or {})}
        audit = {"rounds": result.rounds, "provenance": result.provenance,
                 "support_summary": result.support_summary, "stop_reason": result.stop_reason,
                 "synthetic_actual_training_weight": actual, "learner_fit_counts_cumulative": learner.counts.copy(),
                 "learner_fit_seconds_cumulative": learner.seconds.copy(), **(extra or {})}
        np.savez_compressed(destination / "training.npz", X=as_array(result.X), y=as_array(result.y),
                            sample_weight=np.ones(len(result.y)))
        write_json(destination / "audit.json", audit)
        rows.append({**metadata, **evaluated["test"], "metrics": evaluated["test"],
            "validation_metrics": evaluated["validation"], "variant_id": f"{phase}_{variant}", "phase": phase,
            "method": "ugo_groups_only", "status": "completed", "runtime_seconds": runtime,
            "predictions_path": str((destination / "test_predictions.csv").resolve()),
            "augmentation": {"mode": "matched", "requested_count": settings["augmentation"]["total"],
                             "actual_count": actual, "underspend": settings["augmentation"]["total"] - actual},
            "config": {"selected_parameters": selected["parameters"], "sampler_parameters": parameters,
                       "threshold": evaluated["validation"]["threshold"], "outer_refit": False}, **(extra or {})})
        return destination

    model, result, elapsed, _ = run_trajectory(X, y, seed, categorical, settings, parameters, prepared,
                                              learner, initial_model=initial_model, rounds=1)
    record_variant("initial", "U2", model, result, elapsed)
    baseline_initial_count = len(result.y) - len(y)
    for branch in ("random", "utility"):
        selector = CandidatePoolSelector(branch, partitions["X_guidance"], guidance_y, seed)
        model, result, elapsed, _ = run_trajectory(X, y, seed, categorical, settings, parameters, prepared,
                                                  learner, initial_model, selector, rounds=1)
        initial[branch] = (selector, model, result, elapsed)
    random_selector, _, random_result, _ = initial["random"]
    utility_selector, _, utility_result, _ = initial["utility"]
    if len(random_result.y) != len(utility_result.y):
        raise ValueError("Initial random/utility selected counts must be matched")
    if len(random_selector.rounds) != len(utility_selector.rounds):
        raise ValueError("Initial controls did not generate identical rounds")
    for index, (random, utility) in enumerate(zip(random_selector.rounds, utility_selector.rounds), 1):
        for key in ("original_coordinates", "final_coordinates", "anchor_indices", "partner_indices",
                    "relocation_indices", "group_ids", "scores", "coefficient", "intercept"):
            np.testing.assert_array_equal(random[key], utility[key], err_msg=f"Initial pool/checkpoint differs: {key}")
        shared = {key: value for key, value in random.items() if key not in ("selected", "random_selected", "utility_selected")}
        pool_sha = save_round(folder / "initial_shared_pool", index, shared)
        np.savez_compressed(folder / "initial_shared_pool" / f"round-{index:02d}-branch-masks.npz",
                            random_selected=random["selected"], utility_selected=utility["selected"])
        write_json(folder / "initial_shared_pool" / f"round-{index:02d}-link.json",
                   {"pool_sha256": pool_sha, "random_count": int(random["selected"].sum()),
                    "utility_count": int(utility["selected"].sum()), "same_frozen_checkpoint": True})
    for branch, (selector, model, result, elapsed) in initial.items():
        record_variant("initial", f"pool_{branch}", model, result, elapsed,
            {"matched_pool_counts": True, "baseline_equal_count": len(result.y) - len(y) == baseline_initial_count,
             "selector": selector.summary()})
    (folder / "initial_shared_pool").mkdir(exist_ok=True)
    save_batch_diagnostics(folder / "initial_shared_pool", random_selector.rounds, X, y, learner, protocol, validation)
    # Only now begin full independent live trajectories.
    model, result, elapsed, _ = run_trajectory(X, y, seed, categorical, settings, parameters, prepared, learner)
    record_variant("iterative", "U2", model, result, elapsed)
    for branch in ("random", "utility"):
        selector = CandidatePoolSelector(branch, partitions["X_guidance"], guidance_y, seed)
        counts_before, times_before = learner.counts.copy(), learner.seconds.copy()
        model, result, elapsed, calls = run_trajectory(X, y, seed, categorical, settings, parameters, prepared, learner, selector=selector)
        destination = record_variant("iterative", f"pool_{branch}", model, result, elapsed,
            {"matched_iterative_pool": False, "selector": selector.summary(), "live_training_counts": calls,
             "operational_classifier_fits": learner.counts["operational"] - counts_before["operational"],
             "operational_fit_seconds": learner.seconds["operational"] - times_before["operational"]})
        for index, record in enumerate(selector.rounds, 1):
            save_round(destination / "pools", index, record)
        save_batch_diagnostics(destination, selector.rounds, X, y, learner, protocol, validation)
    write_json(folder / "costs.json", {"tuning_classifier_fits": selected["classifier_fit_count"],
        "classifier_fits": learner.counts, "classifier_fit_seconds": learner.seconds,
        "true_total_classifier_fits": selected["classifier_fit_count"] + sum(learner.counts.values()),
        "observed_total_fit_seconds": sum(learner.seconds.values()), "diagnostics_never_live": True})
    return rows


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    result.add_argument("--protocol", default="configs/benchmarks/shoppers_controlled.yaml")
    result.add_argument("--study", default="configs/ablations/stage1_learning_benefit.yaml")
    result.add_argument("--study-id")
    result.add_argument("--data-home", default=".cache/benchmarks")
    result.add_argument("--local-data")
    result.add_argument("--download", action="store_true")
    result.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    result.add_argument("--output-dir")
    result.add_argument("--tuning-budget", type=int, default=10)
    result.add_argument("--augmentation-count", type=int)
    result.add_argument("--ugo-max-iterations", type=int)
    result.add_argument("--smoke-fit-rows", type=int, default=400)
    result.add_argument("--smoke-evaluation-rows", type=int, default=120)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    protocol, study = load_protocol(args.protocol), load_study(args.study)
    if protocol["dataset"] != "shoppers" or protocol["status"] != "ready":
        raise ValueError("Learning-benefit requires the resolved Shoppers development protocol")
    if protocol["split"] != {"type": "stratified_holdout", "test_fraction": .2, "outer_seed": 2026} or protocol["validation_fraction"] != .2:
        raise ValueError("Learning-benefit preserves the canonical outer/validation split")
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique and nonnegative")
    if args.tuning_budget < 1:
        raise ValueError("Tuning budget must be positive")
    if args.study_id:
        study = {**study, "study_id": args.study_id}
    settings = {"device": "cpu", "mlp_epochs": 160,
                "augmentation": {"mode": "matched", "total": args.augmentation_count if args.augmentation_count is not None else (40 if args.smoke else 1000)},
                "ugo_max_iterations": args.ugo_max_iterations if args.ugo_max_iterations is not None else (2 if args.smoke else 20),
                "method_parameters": {"ugo_groups_only": {"grouping": "kmeans", "allocation": "stability"}}}
    if settings["augmentation"]["total"] < 0 or settings["ugo_max_iterations"] < 1:
        raise ValueError("Target count must be nonnegative and round cap positive")
    arguments = {**vars(args), "experiment_type": "setup_smoke" if args.smoke else "development",
                 "effective_settings": settings, "effective_seeds": args.seeds,
                 "effective_classifiers": ["logistic_regression"], "effective_tuning_budget": args.tuning_budget}
    artifacts = BenchmarkArtifacts(args.output_dir or ("results/smoke" if args.smoke else "results/runs"), arguments, protocol, study)
    rows = []
    try:
        dataset = load_dataset(protocol, args.data_home, local_data=args.local_data, download=args.download)
        artifacts.dataset(dataset)
        fingerprint = data_fingerprint(dataset["X"], dataset["y"], dataset["row_ids"])
        for seed in args.seeds:
            seed_everything(seed)
            split = learning_split(dataset, protocol, seed, args.smoke, args.smoke_fit_rows, args.smoke_evaluation_rows)
            partitions = prepare_learning_partitions(dataset, split)
            candidates = candidate_configs(protocol, "logistic_regression", seed, args.tuning_budget)
            folder = artifacts.fold_path("shoppers", split["fold"]) / f"seed-{seed}" / "logistic_regression"
            metadata = {"dataset": "shoppers", "dataset_fingerprint": fingerprint,
                        "split_fingerprint": split_identity(dataset, split), "seed": seed,
                        "algorithm": "logistic_regression", "budget_mode": "matched",
                        "tuning_budget": args.tuning_budget, "candidate_configs": candidates,
                        "experiment_type": arguments["experiment_type"], "fold": split["fold"]}
            start = perf_counter()
            with measure_memory("cpu") as memory:
                seed_rows = evaluate_seed(dataset, split, partitions, seed, settings, protocol, candidates, folder, metadata)
            for row in seed_rows:
                row.update(memory, observed_seed_total_seconds=perf_counter() - start)
            rows.extend(seed_rows)
        write_json(artifacts.path / "paired_differences.json", paired_differences(rows))
        artifacts.finish(rows)
    except Exception as error:
        artifacts.finish(rows, error=str(error), status="failed")
        raise
    print(json.dumps({"path": str(artifacts.path), "completed_records": len(rows), "experiment_type": arguments["experiment_type"]}))
    return artifacts.path


if __name__ == "__main__":
    main()

