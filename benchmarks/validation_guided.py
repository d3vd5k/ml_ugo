"""Shoppers validation-guided U2 with original inner/outer refitting access.

LR uses its actual full-parameter CE gradients. RF/LightGBM explicitly use a
prefix-fitted LR surrogate, while anchors and retraining use the native model.
Original validation is training information for selection; outer test is never
used to choose settings. Outer guidance overlaps fitting and generation by design.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from benchmarks.learning_benefit import metrics_for, parent_training, run_trajectory, save_round
from benchmarks.protocols import load_dataset, load_protocol, make_splits
from benchmarks.results import BenchmarkArtifacts, enforce_stage_gate, load_study, metric_results
from benchmarks.run import (augmentation_targets, candidate_configs, evaluate_cell,
                            measure_memory, prepare_partitions, smoke_split, split_fingerprint)
from classifiers.factory import make_estimator
from datasets.scripts.climb import categorical_columns
from experiments import _preprocessing_state, data_fingerprint, seed_everything, write_json
from samplers.protected import prepare_protected_groups
from samplers.validation_guided import ValidationGuidedLearner, ValidationGuidedSelector

ALGORITHMS = ("logistic_regression", "random_forest", "lightgbm")
REFERENCES = ("U0", "U2", "U5", "U5_F1")


def validation_guided_split(dataset, protocol, seed, smoke=False, train_rows=400, test_rows=120):
    original = make_splits(dataset, protocol, seed)[0]
    split = (smoke_split(dataset, original, seed, train_rows, test_rows, protocol["validation_fraction"])
             if smoke else {key: value.copy() if isinstance(value, np.ndarray) else value
                            for key, value in original.items()})
    split["guidance_indices"] = split["validation_indices"].copy()
    split["canonical_test_indices"] = original["test_indices"].copy()
    return split


def prepare_validation_partitions(dataset, split):
    partitions = prepare_partitions(dataset, split)
    partitions["X_inner_guidance"] = partitions["X_validation"]
    partitions["X_outer_guidance"] = partitions["outer"].transform(
        dataset["X"].iloc[split["guidance_indices"]]).astype(np.float32)
    return partitions


def numeric_metrics(metrics):
    return {key: value for key, value in metrics.items()
            if isinstance(value, (int, float, np.number)) and key != "primary_metric_value"}


def prediction_metrics(frame, protocol):
    from sklearn.metrics import accuracy_score, log_loss, precision_score
    from sklearn.utils.class_weight import compute_sample_weight
    label = "y_true" if "y_true" in frame else "label"
    y = frame[label].to_numpy()
    probabilities = frame[["probability_0", "probability_1"]].to_numpy()
    positive = int(protocol.get("positive_class", 1))
    threshold = .5
    metrics = metric_results(y, probabilities, np.array([0, 1]), positive, threshold,
                            protocol["primary_metric"], .05)
    predicted = probabilities[:, positive] >= threshold
    metrics.update(accuracy=float(accuracy_score(y == positive, predicted)),
                   precision=float(precision_score(y == positive, predicted, zero_division=0)),
                   ordinary_cross_entropy=float(log_loss(y, probabilities, labels=[0, 1])),
                   balanced_cross_entropy=float(log_loss(y, probabilities, labels=[0, 1],
                                                sample_weight=compute_sample_weight("balanced", y))))
    return metrics


def save_predictions(folder, name, model, X, y, row_ids, protocol):
    metrics, probabilities = metrics_for(model, X, y, protocol, .5)
    frame = pd.DataFrame({"row_id": row_ids, "label": y})
    for column, label in enumerate(model.classes_):
        frame[f"probability_{label}"] = probabilities[:, column]
    frame.to_csv(folder / f"{name}_predictions.csv", index=False)
    return metrics


def batch_diagnostics(folder, rounds, X, y, learner, protocol, guidance, test=None):
    """Actual native batch/parent refits from the same prefix; never become live."""
    prefix_X, prefix_y = X, y
    records = []
    for index, pool in enumerate(rounds, 1):
        comparisons = {}
        counts = [int(pool[f"{branch}_selected"].sum()) for branch in ("random", "utility")]
        if counts[0] != counts[1]:
            raise ValueError("Same-pool selectors must have matched accepted counts")
        for branch in ("random", "utility"):
            mask = pool[f"{branch}_selected"]
            actual_X = np.concatenate((prefix_X, pool["final_coordinates"][mask]))
            actual_y = np.concatenate((prefix_y, pool["labels"][mask]))
            actual = learner.fit(actual_X, actual_y, kind="paired")
            parent_X, parent_y, weights = parent_training(prefix_X, prefix_y, X, y,
                pool["anchor_indices"][mask], pool["partner_indices"][mask])
            parent = learner.fit(parent_X, parent_y, weights, kind="parentdup")
            comparison = {"actual": metrics_for(actual, *guidance, protocol, .5)[0],
                "parentdup": metrics_for(parent, *guidance, protocol, .5)[0],
                "batch_count": int(mask.sum()), "parent_added_weight": float(weights[len(prefix_y):].sum()),
                "parent_rows": int(2 * mask.sum()), "prefix_count": len(prefix_y),
                "evaluation_role": "reused_guidance_not_independent_validation"}
            if test is not None:
                comparison["test_actual"] = metrics_for(actual, *test, protocol, .5)[0]
                comparison["test_parentdup"] = metrics_for(parent, *test, protocol, .5)[0]
            destination = folder / "paired_checkpoints" / f"round-{index:02d}" / branch
            destination.mkdir(parents=True, exist_ok=True)
            joblib.dump(actual, destination / "actual-model.joblib")
            joblib.dump(parent, destination / "parent-model.joblib")
            np.savez_compressed(destination / "actual-training.npz", X=actual_X, y=actual_y,
                                sample_weight=np.ones(len(actual_y)))
            np.savez_compressed(destination / "parent-training.npz", X=parent_X, y=parent_y,
                                sample_weight=weights)
            write_json(destination / "metrics.json", comparison)
            comparisons[branch] = comparison
        records.append({"round": pool["round"], "checkpoint_state": pool["state_before_hash"],
                        "matched_counts": True, "comparisons": comparisons})
        live = pool["selected"]
        prefix_X = np.concatenate((prefix_X, pool["final_coordinates"][live]))
        prefix_y = np.concatenate((prefix_y, pool["labels"][live]))
    write_json(folder / "paired_checkpoints.json", records)


def evaluate_role(dataset, split, partitions, seed, algorithm, selected, surrogate_parameters,
                  settings, protocol, folder, metadata, role):
    folder.mkdir(parents=True, exist_ok=True)
    indices = split["fit_indices" if role == "inner" else "train_indices"]
    X = partitions["X_fit" if role == "inner" else "X_train"]
    y = dataset["y"][indices]
    G = partitions[f"X_{role}_guidance"]
    gy = dataset["y"][split["guidance_indices"]]
    preprocessor = partitions[role]
    parameters = selected["inner"]["sampler_parameters"]
    native_parameters = selected["inner"]["parameters"]
    prepared = prepare_protected_groups(X, y, seed, augmentation_targets(y, settings["augmentation"]), parameters)
    prepared["real_row_ids"] = dataset["row_ids"][indices]
    write_json(folder / "groups.json", prepared)
    joblib.dump(preprocessor, folder / "preprocessor.joblib")
    write_json(folder / "preprocessing.json", _preprocessing_state(preprocessor))
    np.savez_compressed(folder / "real.npz", X=X, y=y, guidance_X=G, guidance_y=gy,
        row_ids=dataset["row_ids"][indices], guidance_row_ids=dataset["row_ids"][split["guidance_indices"]],
        test_row_ids=dataset["row_ids"][split["test_indices"]], real_indices=indices,
        guidance_indices=split["guidance_indices"], test_indices=split["test_indices"])
    write_json(folder / "role.json", {"role": role, "real_count": len(y), "guidance_count": len(gy),
        "guidance_fitting_overlap": int(np.isin(split["guidance_indices"], indices).sum()),
        "guidance_generation_overlap": role == "outer", "test_never_fits_or_generates": True,
        "native_parameters": native_parameters, "surrogate_parameters": surrogate_parameters,
        "guidance_evaluation_is_training_information": True})
    learner = ValidationGuidedLearner(algorithm, native_parameters, seed, surrogate_parameters,
                                     (G, gy) if role == "inner" else None)
    initial = learner.fit(X, y, kind="initial")
    joblib.dump(initial, folder / "initial-model.joblib")
    categorical = categorical_columns(preprocessor)
    test = ((partitions["X_test"], dataset["y"][split["test_indices"]]) if role == "outer" else None)
    rows, initial_branches = [], {}

    def make_selector(branch, destination):
        def capture(native, scorer, checkpoint):
            checkpoint_folder = destination / "checkpoints"
            checkpoint_folder.mkdir(parents=True, exist_ok=True)
            stem = f"round-{checkpoint['round']:02d}"
            joblib.dump(native, checkpoint_folder / f"{stem}-native.joblib")
            joblib.dump(scorer, checkpoint_folder / f"{stem}-scorer.joblib")
            px, py, weights = learner._prefixes[native]
            np.savez_compressed(checkpoint_folder / f"{stem}-prefix.npz", X=px, y=py,
                                sample_weight=np.ones(len(py)) if weights is None else weights)
            write_json(checkpoint_folder / f"{stem}.json", checkpoint)
        return ValidationGuidedSelector(branch, G, gy, seed, algorithm=algorithm,
            scorer_for_model=learner.scorer_for_model, checkpoint_callback=capture)

    def record(phase, variant, model, result, elapsed, selector, before):
        destination = folder / f"{phase}_{variant}"
        destination.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, destination / "model.joblib")
        np.savez_compressed(destination / "training.npz", X=np.asarray(result.X), y=np.asarray(result.y),
                            sample_weight=np.ones(len(result.y)))
        guidance_metrics = save_predictions(destination, "guidance", model, G, gy,
                                            dataset["row_ids"][split["guidance_indices"]], protocol)
        evaluated = {"guidance": guidance_metrics}
        if test is not None:
            evaluated["test"] = save_predictions(destination, "test", model, *test,
                                                  dataset["row_ids"][split["test_indices"]], protocol)
        write_json(destination / "metrics.json", evaluated)
        audit = selector.summary() if selector is not None else {}
        generated = len(result.y) - len(y)
        diagnostic = sum(pool["diagnostic_seconds"] for pool in (selector.rounds if selector else []))
        if selector is not None and branch_name(variant) == "random":
            terminal = audit["guidance_scoring_seconds_total"] - sum(pool["guidance_scoring_seconds"] for pool in selector.rounds)
            diagnostic += terminal
            # LR fits exist in random solely to audit the same scored pools.
            diagnostic += learner.surrogate_seconds - before["surrogate_seconds"]
        fits = {key: learner.counts[key] - before["counts"][key] for key in learner.counts}
        costs = {"observed_trajectory_seconds": elapsed,
                 "operational_trajectory_seconds": max(0., elapsed - diagnostic),
                 "diagnostic_seconds": diagnostic,
                 "native_classifier_fits": fits,
                 "native_fit_seconds": learner.seconds["operational"] - before["seconds"]["operational"],
                 "surrogate_fits": learner.surrogate_fits - before["surrogate_fits"],
                 "surrogate_fit_seconds": learner.surrogate_seconds - before["surrogate_seconds"],
                 "scorer_kind": "native_logistic" if algorithm == "logistic_regression" else "explicit_lr_surrogate"}
        write_json(destination / "audit.json", {"rounds": result.rounds, "provenance": result.provenance,
            "support_summary": result.support_summary, "stop_reason": result.stop_reason,
            "selector": audit, **costs})
        if selector is not None:
            for number, pool in enumerate(selector.rounds, 1):
                save_round(destination / "pools", number, pool)
            write_json(destination / "checkpoints.json", selector.checkpoints)
        row = {**metadata, "variant_id": f"{role}_{phase}_{variant}", "role": role, "phase": phase,
            "display_variant": variant, "method": "ugo_groups_only", "status": "completed",
            "metrics": evaluated.get("test", guidance_metrics), "guidance_metrics": guidance_metrics,
            "runtime_seconds": elapsed, **costs,
            "evaluation_role": "outer_test" if role == "outer" else "reused_guidance",
            "predictions_path": str((destination / ("test_predictions.csv" if test else "guidance_predictions.csv")).resolve()),
            "augmentation": {"mode": "matched", "requested_count": settings["augmentation"]["total"],
                             "actual_count": generated, "underspend": settings["augmentation"]["total"] - generated},
            "config": {"selected_parameters": native_parameters, "sampler_parameters": parameters,
                       "surrogate_parameters": surrogate_parameters, "threshold": .5,
                       "outer_refit": role == "outer", "guidance_is_validation": True}}
        rows.append(row)
        return destination

    def snapshot():
        return {"counts": learner.counts.copy(), "seconds": learner.seconds.copy(),
                "surrogate_fits": learner.surrogate_fits, "surrogate_seconds": learner.surrogate_seconds}

    for variant in ("U2", "pool_random", "pool_utility"):
        destination = folder / f"initial_{variant}"
        selector = None if variant == "U2" else make_selector(branch_name(variant), destination)
        before = snapshot()
        model, result, elapsed, _ = run_trajectory(X, y, seed, categorical, settings, parameters, prepared,
                                                  learner, initial, selector, rounds=1)
        record("initial", variant, model, result, elapsed, selector, before)
        if selector is not None:
            initial_branches[branch_name(variant)] = selector
    random = initial_branches["random"]
    utility = initial_branches["utility"]
    if len(random.rounds) != len(utility.rounds):
        raise ValueError("Initial selector round counts differ")
    for index, (r, u) in enumerate(zip(random.rounds, utility.rounds), 1):
        for key in ("original_coordinates", "final_coordinates", "anchor_indices", "partner_indices",
                    "relocation_indices", "group_ids", "scores", "coefficient", "intercept"):
            np.testing.assert_array_equal(r[key], u[key], err_msg=f"Initial pool/checkpoint differs: {key}")
        if r["accepted"] != u["accepted"]:
            raise ValueError("Initial random and utility counts differ")
        shared = {key: value for key, value in r.items()
                  if key not in ("selected", "random_selected", "utility_selected")}
        sha = save_round(folder / "initial_shared_pool", index, shared)
        np.savez_compressed(folder / "initial_shared_pool" / f"round-{index:02d}-branch-masks.npz",
                            random_selected=r["selected"], utility_selected=u["selected"])
        write_json(folder / "initial_shared_pool" / f"round-{index:02d}-link.json",
                   {"pool_sha256": sha, "random_count": r["accepted"], "utility_count": u["accepted"],
                    "same_frozen_checkpoint": True})
    batch_diagnostics(folder / "initial_shared_pool", random.rounds, X, y, learner, protocol, (G, gy), test)
    for variant in ("U2", "pool_random", "pool_utility"):
        destination = folder / f"iterative_{variant}"
        selector = None if variant == "U2" else make_selector(branch_name(variant), destination)
        before = snapshot()
        model, result, elapsed, calls = run_trajectory(X, y, seed, categorical, settings, parameters,
                                                      prepared, learner, selector=selector)
        record("iterative", variant, model, result, elapsed, selector, before)
        if selector is not None:
            batch_diagnostics(destination, selector.rounds, X, y, learner, protocol, (G, gy), test)
        if role == "outer" and variant == "U2":
            source = pd.read_csv(folder.parent / "U2" / "predictions.csv", float_precision="round_trip")
            actual = model.predict_proba(partitions["X_test"])
            np.testing.assert_array_equal(source[["probability_0", "probability_1"]].to_numpy(), actual)
            write_json(destination / "canonical_replay.json", {"exact_source_predictions": True})
    write_json(folder / "costs.json", {"native_classifier_fits": learner.counts,
        "native_fit_seconds": learner.seconds, "surrogate_fits": learner.surrogate_fits,
        "surrogate_fit_seconds": learner.surrogate_seconds, "diagnostics_never_live": True})
    return rows


def branch_name(variant):
    return variant.removeprefix("pool_")


def export_comparisons(path, rows):
    flattened = []
    for row in rows:
        flat = {key: row.get(key) for key in ("algorithm", "seed", "variant_id", "role", "phase", "display_variant",
                "scorer_kind", "runtime_seconds", "operational_trajectory_seconds", "surrogate_fit_seconds")}
        flat["accepted"] = row["augmentation"]["actual_count"]
        if row["evaluation_role"] == "outer_test":
            flat.update({f"test_{key}": value for key, value in numeric_metrics(row["metrics"]).items()})
        flat.update({f"guidance_{key}": value for key, value in numeric_metrics(row.get("guidance_metrics", {})).items()})
        flattened.append(flat)
    pd.DataFrame(flattened).to_csv(path / "all_metrics.csv", index=False)
    summaries = []
    for (algorithm, variant), group in pd.DataFrame(flattened).groupby(["algorithm", "variant_id"]):
        for key in [column for column in group if column.startswith(("test_", "guidance_"))]:
            values = group[key].dropna()
            if len(values):
                summaries.append({"algorithm": algorithm, "variant_id": variant, "metric": key,
                    "mean": values.mean(), "sample_sd": values.std(ddof=1), "seeds": len(values)})
    pd.DataFrame(summaries).to_csv(path / "metric_summary.csv", index=False)
    deltas = []
    for algorithm in ALGORITHMS:
        selected = [row for row in rows if row["algorithm"] == algorithm]
        index = {(row["variant_id"], row["seed"]): row for row in selected}
        for row in selected:
            if row["display_variant"] != "pool_utility":
                continue
            reference_ids = [f"{row['role']}_{row['phase']}_pool_random", f"{row['role']}_{row['phase']}_U2"]
            if row["role"] == "outer" and row["phase"] == "iterative":
                reference_ids += ["U0", "U5", "U5_F1"]
            for reference_id in reference_ids:
                reference = index[(reference_id, row["seed"])]
                for name, field in (("test", "metrics"), ("guidance", "guidance_metrics")):
                    if name == "test" and row["role"] != "outer":
                        continue
                    a, b = row.get(field, {}), reference.get(field, {})
                    for key in numeric_metrics(a).keys() & numeric_metrics(b).keys():
                        deltas.append({"algorithm": algorithm, "seed": row["seed"], "role": row["role"],
                            "phase": row["phase"], "reference_variant": reference_id, "evaluation_role": name,
                            "metric": key, "utility_minus_reference": a[key] - b[key],
                            "matched_accepted_counts": row["augmentation"]["actual_count"] == reference["augmentation"]["actual_count"],
                            "shared_frozen_pool": row["phase"] == "initial" and "pool_random" in reference_id})
    pd.DataFrame(deltas).to_csv(path / "paired_metric_deltas.csv", index=False)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    result.add_argument("--protocol", default="configs/benchmarks/shoppers_controlled.yaml")
    result.add_argument("--study", default="configs/ablations/stage1_validation_guided.yaml")
    result.add_argument("--data-home", default=".cache/benchmarks")
    result.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    result.add_argument("--tuning-budget", type=int, default=10)
    result.add_argument("--output-dir")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    protocol, study = load_protocol(args.protocol), load_study(args.study)
    if protocol["dataset"] != "shoppers" or protocol["status"] != "ready":
        raise ValueError("Validation guidance requires the resolved Shoppers protocol")
    if protocol["split"] != {"type": "stratified_holdout", "test_fraction": .2, "outer_seed": 2026} or protocol["validation_fraction"] != .2:
        raise ValueError("Preserve canonical historical Shoppers splits")
    if args.tuning_budget < 1 or any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Require positive tuning budget and unique nonnegative seeds")
    experiment_type = "setup_smoke" if args.smoke else "development"
    enforce_stage_gate(study, experiment_type)
    settings = {"device": "cpu", "mlp_epochs": 160,
        "augmentation": {"mode": "matched", "total": 40 if args.smoke else 1000},
        "ugo_max_iterations": 2 if args.smoke else 20,
        "method_parameters": {**protocol.get("method_parameters", {}),
                              **(protocol.get("smoke_method_parameters", {}) if args.smoke else {})}}
    for algorithm in ALGORITHMS:
        make_estimator(algorithm, args.seeds[0], device="cpu")
    arguments = {**vars(args), "experiment_type": experiment_type, "effective_settings": settings,
        "effective_classifiers": ALGORITHMS, "effective_seeds": args.seeds,
        "effective_tuning_budget": args.tuning_budget, "tree_scoring": "explicit_lr_surrogate_user_authorized"}
    artifacts = BenchmarkArtifacts(args.output_dir or ("results/smoke" if args.smoke else "results/runs"), arguments, protocol, study)
    rows = []
    try:
        dataset = load_dataset(protocol, args.data_home)
        artifacts.dataset(dataset)
        fingerprint = data_fingerprint(dataset["X"], dataset["y"], dataset["row_ids"])
        for seed in args.seeds:
            seed_everything(seed)
            split = validation_guided_split(dataset, protocol, seed, args.smoke)
            partitions = prepare_validation_partitions(dataset, split)
            root = artifacts.fold_path("shoppers", split["fold"]) / f"seed-{seed}"
            root.mkdir(parents=True, exist_ok=True)
            write_json(root / "split.json", {key: value for key, value in split.items()})
            surrogate_parameters = None
            for algorithm in ALGORITHMS:
                candidates = candidate_configs(protocol, algorithm, seed, args.tuning_budget)
                metadata = {"dataset": "shoppers", "dataset_fingerprint": fingerprint,
                    "split_fingerprint": split_fingerprint(dataset, split), "seed": seed, "algorithm": algorithm,
                    "budget_mode": "matched", "tuning_budget": args.tuning_budget, "candidate_configs": candidates,
                    "experiment_type": experiment_type, "fold": split["fold"]}
                started = perf_counter()
                with measure_memory(study["validation_guided_policy"]["retained_F1_reference_device"]) as memory:
                    for variant in REFERENCES:
                        declaration = study["variants"][variant]
                        refmeta = {**metadata, **declaration, "variant_id": variant, "role": "outer",
                            "phase": "reference", "display_variant": variant}
                        reference_settings = ({**settings, "device": study["validation_guided_policy"]["retained_F1_reference_device"]}
                                              if variant == "U5_F1" else settings)
                        reference = evaluate_cell(dataset, split, algorithm, declaration["method"], variant,
                            candidates, reference_settings, protocol, artifacts, refmeta, partitions)
                        frame = pd.read_csv(reference["predictions_path"], float_precision="round_trip")
                        reference["metrics"] = prediction_metrics(frame, protocol)
                        reference["evaluation_role"] = "outer_test"
                        rows.append(reference)
                    folder = root / algorithm
                    selected = json.loads((folder / "U2" / "selection.json").read_text())
                    if algorithm == "logistic_regression":
                        surrogate_parameters = selected["inner"]["parameters"]
                    if surrogate_parameters is None:
                        raise RuntimeError("Canonical LR-U2 selection must precede tree surrogate fitting")
                    for role in ("inner", "outer"):
                        rows.extend(evaluate_role(dataset, split, partitions, seed, algorithm, selected,
                            surrogate_parameters, settings, protocol, folder / role, metadata, role))
                for row in rows:
                    if row["seed"] == seed and row["algorithm"] == algorithm:
                        row.update(memory, observed_algorithm_total_seconds=perf_counter() - started)
                export_comparisons(artifacts.path, rows)
        artifacts.finish(rows)
    except Exception as error:
        artifacts.finish(rows, error=str(error), status="failed")
        raise
    print(json.dumps({"path": str(artifacts.path), "completed_records": len(rows), "experiment_type": experiment_type}))
    return artifacts.path


if __name__ == "__main__":
    main()
