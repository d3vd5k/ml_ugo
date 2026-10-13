"""Shoppers native U2 random pools with optional retained F1 rejection.

Choose randomly within unchanged U2 quotas before F1; never rescue a rejection.
Canonical U2 settings are locked, then four F1 thresholds use inner validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from benchmarks.flow_partitions import flow_for_partition
from benchmarks.protocols import load_dataset, load_protocol, make_splits
from benchmarks.results import BenchmarkArtifacts, enforce_stage_gate, load_study
from benchmarks.run import (augmentation_targets, candidate_configs, evaluate_cell,
                            prepare_partitions, smoke_split, split_fingerprint)
from benchmarks.validation_guided import ALGORITHMS, REFERENCES, numeric_metrics, prediction_metrics, save_predictions
from classifiers.factory import fit_estimator, make_estimator
from datasets.scripts.climb import categorical_columns
from experiments import _preprocessing_state, data_fingerprint, seed_everything, write_json
from samplers.flow_gmm import FlowGMMCore, _checksum
from samplers.pool_random import RandomPoolSelector
from samplers.protected import prepare_protected_groups, protected_ugo


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_pool(folder, number, record):
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in record.items() if isinstance(value, np.ndarray)}
    path = folder / f"round-{number:02d}-pool.npz"
    np.savez_compressed(path, **arrays)
    write_json(folder / f"round-{number:02d}.json", {
        **{key: value for key, value in record.items() if key not in arrays},
        "pool_sha256": sha256(path), "array_fields": sorted(arrays)})


def frozen_flow(dataset, split, partitions, root, seed, source=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = root / "flow-core.pt"
    if source is not None:
        original = source / "shoppers" / "fold-0" / f"seed-{seed}" / "logistic_regression" / "U5_F1" / "flow-core.pt"
        core = FlowGMMCore.load(original, device=device)
        fit, _ = train_test_split(split["fit_indices"], test_size=.2,
            stratify=dataset["y"][split["fit_indices"]], random_state=seed + 7919)
        np.testing.assert_array_equal(core.fit_row_ids, dataset["row_ids"][fit])
        key = (seed, tuple(fit.tolist()), json.dumps({}, sort_keys=True), "auto")
        partitions.setdefault("flow_cores", {})[key] = core
        checkpoint.symlink_to(original.resolve())
    gate = flow_for_partition(dataset, split, partitions, "inner", seed, {}, "auto")
    if source is None:
        gate.core.save(checkpoint)
    write_json(root / "flow.json", {**gate.audit,
        "calibration_row_ids": gate.calibration_row_ids,
        "validation_row_ids": gate.validation_row_ids,
        "source_checkpoint": str(original.resolve()) if source is not None else None,
        "source_sha256": sha256(checkpoint), "imported_frozen_checkpoint": source is not None,
        "inference_only": True, "fit_count_this_run": 0 if source is not None else 1,
        "effective_device": device})
    return gate.core


def prepare_role(dataset, split, partitions, role, seed, selected, settings, folder):
    folder.mkdir(parents=True, exist_ok=True)
    indices = split["fit_indices" if role == "inner" else "train_indices"]
    X = partitions["X_fit" if role == "inner" else "X_train"]
    y = dataset["y"][indices]
    validation = split["validation_indices"]
    V = partitions[role].transform(dataset["X"].iloc[validation]).astype(np.float32)
    targets = augmentation_targets(y, settings["augmentation"])
    groups = prepare_protected_groups(X, y, seed, targets, selected["inner"]["sampler_parameters"])
    groups["real_row_ids"] = dataset["row_ids"][indices]
    write_json(folder / "groups.json", groups)
    joblib.dump(partitions[role], folder / "preprocessor.joblib")
    write_json(folder / "preprocessing.json", _preprocessing_state(partitions[role]))
    np.savez_compressed(folder / "real.npz", X=X, y=y, validation_X=V,
        validation_y=dataset["y"][validation], real_indices=indices, validation_indices=validation,
        test_indices=split["test_indices"], row_ids=dataset["row_ids"][indices],
        validation_row_ids=dataset["row_ids"][validation], test_row_ids=dataset["row_ids"][split["test_indices"]])
    write_json(folder / "role.json", {"role": role, "real_count": len(y),
        "validation_count": len(validation), "validation_fitting_overlap": int(np.isin(validation, indices).sum()),
        "validation_generation_overlap": role == "outer", "test_never_fits_or_generates": True,
        "native_parameters": selected["inner"]["parameters"], "no_guidance_selection": True})
    return X, y, V, groups, targets


def trajectory(dataset, split, partitions, role, seed, algorithm, selected, settings,
               protocol, folder, prepared, metadata, tau=None, *, proposal_generator=None):
    folder.mkdir(parents=True, exist_ok=True)
    X, y, V, groups, targets = prepared
    estimator = make_estimator(algorithm, seed, parameters=selected["inner"]["parameters"], device="cpu")
    native_fits, native_seconds = 0, 0.
    prefix = {}

    def fit(px, py):
        nonlocal native_fits, native_seconds
        start = perf_counter()
        model = fit_estimator(estimator, px, py)
        native_seconds += perf_counter() - start
        native_fits += 1
        prefix.update(X=np.asarray(px), y=np.asarray(py))
        return model

    def capture(native, checkpoint):
        destination = folder / "checkpoints"
        destination.mkdir(parents=True, exist_ok=True)
        stem = f"round-{checkpoint['round']:02d}"
        joblib.dump(native, destination / f"{stem}-native.joblib")
        np.savez_compressed(destination / f"{stem}-prefix.npz", **prefix,
                            sample_weight=np.ones(len(prefix["y"])))
        write_json(destination / f"{stem}.json", checkpoint)

    if proposal_generator is not None and tau is not None:
        raise ValueError("Flow geometry must not be combined with F1 rejection")
    selector = RandomPoolSelector(seed, k=5, checkpoint_callback=capture,
                                  proposal_generator=proposal_generator)
    gate = flow_for_partition(dataset, split, partitions, role, seed, {}, "auto") if tau is not None else None
    tracked_core = (gate.core if gate is not None else
                    proposal_generator.cache.core if proposal_generator is not None else None)
    flow_before = _checksum(tracked_core.model) if tracked_core is not None else None
    parameters = {**selected["inner"]["sampler_parameters"], "capture_proposals": True}
    if tau is not None:
        parameters["f1_tau"] = tau
    start = perf_counter()
    result = protected_ugo(X, y, seed, categorical_columns(partitions[role]),
        settings["ugo_max_iterations"], "cpu", fit, targets, parameters, groups,
        flow_support=gate, candidate_selector=selector)
    model = fit(result.X, result.y)
    elapsed = perf_counter() - start
    if tracked_core is not None and _checksum(tracked_core.model) != flow_before:
        raise RuntimeError("F1 scoring mutated the frozen detector")
    joblib.dump(model, folder / "model.joblib")
    np.savez_compressed(folder / "training.npz", X=np.asarray(result.X), y=np.asarray(result.y),
                        sample_weight=np.ones(len(result.y)))
    validation_metrics = save_predictions(folder, "validation", model, V,
        dataset["y"][split["validation_indices"]], dataset["row_ids"][split["validation_indices"]], protocol)
    metrics = {"validation": validation_metrics}
    if role == "outer":
        metrics["test"] = save_predictions(folder, "test", model, partitions["X_test"],
            dataset["y"][split["test_indices"]], dataset["row_ids"][split["test_indices"]], protocol)
    write_json(folder / "metrics.json", metrics)
    costs = {"native_fit_count": native_fits, "native_fit_seconds": native_seconds,
        "runtime_seconds": elapsed, "flow_inference_seconds": gate.inference_seconds if gate is not None else 0.,
        "flow_inference_rows": gate.inference_rows if gate is not None else 0,
        "surrogate_fit_count": 0, "utility_gradient_count": 0}
    write_json(folder / "audit.json", {"rounds": result.rounds, "provenance": result.provenance,
        "support_summary": result.support_summary, "stop_reason": result.stop_reason,
        "selector": selector.summary(), "role": role, "tau": tau,
        "native_parameters": selected["inner"]["parameters"], "sampler_parameters": parameters,
        "geometry": proposal_generator.summary() if proposal_generator is not None else None,
        "flow_state_before": flow_before, "flow_state_after": _checksum(tracked_core.model) if tracked_core is not None else None,
        **costs})
    for number, record in enumerate(selector.rounds, 1):
        save_pool(folder / "pools", number, record)
    write_json(folder / "checkpoints.json", selector.checkpoints)
    name = (proposal_generator.variant_name if proposal_generator is not None
            else "pool_random" if tau is None else "pool_random_f1")
    generated = len(result.y) - len(y)
    return {**metadata, "variant_id": f"{role}_iterative_{name}", "display_variant": name,
        "role": role, "phase": "iterative", "method": "ugo_groups_only" if tau is None else "ugo_f1",
        "status": "completed", "metrics": metrics.get("test", validation_metrics),
        "validation_metrics": validation_metrics, "evaluation_role": "outer_test" if role == "outer" else "inner_validation",
        "predictions_path": str((folder / ("test_predictions.csv" if role == "outer" else "validation_predictions.csv")).resolve()),
        "trajectory_path": str(folder.resolve()), "augmentation": {"mode": "matched",
            "requested_count": settings["augmentation"]["total"], "actual_count": generated,
            "underspend": settings["augmentation"]["total"] - generated},
        "config": {"selected_parameters": selected["inner"]["parameters"], "sampler_parameters": parameters,
                   "threshold": .5, "outer_refit": role == "outer", "f1_tau": tau}, **costs}


def export_comparisons(path, rows):
    flattened = []
    for row in rows:
        flat = {key: row.get(key) for key in ("algorithm", "seed", "variant_id", "role", "phase", "runtime_seconds", "native_fit_count", "native_fit_seconds", "flow_inference_seconds", "source_run_id")}
        flat["accepted"] = row["augmentation"]["actual_count"]
        if row["evaluation_role"] == "outer_test":
            flat.update({f"test_{key}": value for key, value in numeric_metrics(row["metrics"]).items()})
        flat.update({f"validation_{key}": value for key, value in numeric_metrics(row.get("validation_metrics", {})).items()})
        flattened.append(flat)
    frame = pd.DataFrame(flattened)
    frame.to_csv(path / "all_metrics.csv", index=False)
    summaries = []
    for (algorithm, variant), group in frame.groupby(["algorithm", "variant_id"]):
        for key in [c for c in group if c.startswith(("test_", "validation_"))]:
            values = group[key].dropna()
            if len(values):
                summaries.append({"algorithm": algorithm, "variant_id": variant, "metric": key,
                    "mean": values.mean(), "sample_sd": values.std(ddof=1), "seeds": len(values)})
    pd.DataFrame(summaries).to_csv(path / "metric_summary.csv", index=False)
    deltas = []
    index = {(row["algorithm"], row["seed"], row["variant_id"]): row for row in rows}
    for row in rows:
        if row["variant_id"] != "outer_iterative_pool_random_f1":
            continue
        for name in (*REFERENCES, "outer_iterative_pool_random"):
            reference = index[(row["algorithm"], row["seed"], name)]
            for metric, value in numeric_metrics(row["metrics"]).items():
                deltas.append({"algorithm": row["algorithm"], "seed": row["seed"], "variant_id": row["variant_id"],
                    "reference_variant": name, "metric": metric, "delta": value - reference["metrics"][metric],
                    "selected_value": value, "reference_value": reference["metrics"][metric],
                    "selected_count": row["augmentation"]["actual_count"], "reference_count": reference["augmentation"]["actual_count"]})
    pd.DataFrame(deltas).to_csv(path / "paired_metric_deltas.csv", index=False)
    if deltas:
        pd.DataFrame(deltas).groupby(["algorithm", "reference_variant", "metric"])["delta"].agg(
            mean="mean", sample_sd="std", seeds="count").reset_index().to_csv(path / "paired_metric_summary.csv", index=False)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    result.add_argument("--protocol", default="configs/benchmarks/shoppers_controlled.yaml")
    result.add_argument("--study", default="configs/ablations/stage1_pool_random_f1.yaml")
    result.add_argument("--reference-run", default="results/runs/20261009T153348-af844bdecf5a")
    result.add_argument("--data-home", default=".cache/benchmarks")
    result.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    result.add_argument("--output-dir")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    protocol, study = load_protocol(args.protocol), load_study(args.study)
    if protocol["dataset"] != "shoppers" or protocol["status"] != "ready":
        raise ValueError("This experiment supports the resolved Shoppers protocol only")
    if protocol["split"] != {"type": "stratified_holdout", "test_fraction": .2, "outer_seed": 2026} or protocol["validation_fraction"] != .2:
        raise ValueError("Preserve the original Shoppers data splits")
    if not args.seeds or any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Require unique nonnegative matched seeds")
    experiment_type = "setup_smoke" if args.smoke else "development"
    enforce_stage_gate(study, experiment_type)
    settings = {"device": "cpu", "mlp_epochs": 160,
        "augmentation": {"mode": "matched", "total": 40 if args.smoke else 1000},
        "ugo_max_iterations": 2 if args.smoke else 20,
        "method_parameters": {**protocol.get("method_parameters", {}),
                              **(protocol.get("smoke_method_parameters", {}) if args.smoke else {})}}
    source = None if args.smoke else Path(args.reference_run)
    source_rows = {}
    if source is not None:
        proof = json.loads((source / "independent_verification.json").read_text())
        if not proof["passed"]:
            raise ValueError("Reference bundle must have passed independent verification")
        source_rows = {(row["seed"], row["algorithm"], row["variant_id"]): row
                       for row in json.loads((source / "benchmark_records.json").read_text())}
    arguments = {**vars(args), "experiment_type": experiment_type, "effective_settings": settings,
        "effective_classifiers": ALGORITHMS, "effective_seeds": args.seeds,
        "effective_tuning_budget": 1 if args.smoke else 10,
        "native_anchor_and_final_classifier": True, "no_utility_or_surrogate": True,
        "reference_import_policy": "explicit_previously_verified_predictions" if source else "fresh_development_smoke"}
    artifacts = BenchmarkArtifacts(args.output_dir or ("results/smoke" if args.smoke else "results/runs"), arguments, protocol, study)
    rows = []
    try:
        dataset = load_dataset(protocol, args.data_home)
        artifacts.dataset(dataset)
        fingerprint = data_fingerprint(dataset["X"], dataset["y"], dataset["row_ids"])
        for seed in args.seeds:
            seed_everything(seed)
            canonical = make_splits(dataset, protocol, seed)[0]
            split = smoke_split(dataset, canonical, seed, 400, 120, .2) if args.smoke else canonical
            partitions = prepare_partitions(dataset, split)
            root = artifacts.fold_path("shoppers", split["fold"]) / f"seed-{seed}"
            root.mkdir(parents=True, exist_ok=True)
            write_json(root / "split.json", {**split, "canonical_test_indices": canonical["test_indices"]})
            frozen_flow(dataset, split, partitions, root, seed, source)
            for algorithm in ALGORITHMS:
                folder = root / algorithm
                folder.mkdir(parents=True, exist_ok=True)
                candidates = candidate_configs(protocol, algorithm, seed, 1 if args.smoke else 10)
                metadata = {"dataset": "shoppers", "dataset_fingerprint": fingerprint,
                    "split_fingerprint": split_fingerprint(dataset, split), "seed": seed, "algorithm": algorithm,
                    "budget_mode": "matched", "tuning_budget": len(candidates), "candidate_configs": candidates,
                    "experiment_type": experiment_type, "fold": split["fold"]}
                for name in REFERENCES:
                    declaration = study["variants"][name]
                    if source is not None:
                        original = source_rows[(seed, algorithm, name)]
                        if (original["dataset_fingerprint"] != fingerprint or original["split_fingerprint"] != metadata["split_fingerprint"]
                                or original["candidate_configs"] != candidates):
                            raise ValueError("Imported reference does not match data, splits or canonical search")
                        origin_folder = source / "shoppers" / "fold-0" / f"seed-{seed}" / algorithm / name
                        (folder / name).symlink_to(origin_folder.resolve(), target_is_directory=True)
                        reference = {**original, **metadata, "source_run_id": original["run_id"],
                            "source_record_id": original["record_id"], "source_artifact_path": str(origin_folder.resolve()),
                            "imported_reference": True, "reference_runtime_reused_not_rerun": True}
                    else:
                        refmeta = {**metadata, **declaration, "variant_id": name, "role": "outer", "phase": "reference", "display_variant": name}
                        reference = evaluate_cell(dataset, split, algorithm, declaration["method"], name,
                            candidates, {**settings, "device": "auto"} if name == "U5_F1" else settings,
                            protocol, artifacts, refmeta, partitions)
                    reference["metrics"] = prediction_metrics(pd.read_csv(reference["predictions_path"], float_precision="round_trip"), protocol)
                    reference.update(evaluation_role="outer_test", role="outer", phase="reference", display_variant=name)
                    rows.append(reference)
                selected = json.loads((folder / "U2" / "selection.json").read_text())
                inner = prepare_role(dataset, split, partitions, "inner", seed, selected, settings, folder / "inner")
                random_inner = trajectory(dataset, split, partitions, "inner", seed, algorithm, selected,
                    settings, protocol, folder / "inner" / "pool_random", inner, metadata)
                rows.append(random_inner)
                trials = []
                for tau in study["composition_policy"]["threshold_grid"]:
                    destination = folder / "inner" / f"pool_random_f1_tau-{tau:g}"
                    trial = trajectory(dataset, split, partitions, "inner", seed, algorithm, selected,
                        settings, protocol, destination, inner, metadata, tau)
                    trials.append({"tau": tau, "validation_average_precision": trial["metrics"]["average_precision"],
                                   "artifact_path": str(destination.resolve()), "row": trial})
                best = max(trials, key=lambda trial: trial["validation_average_precision"])
                (folder / "inner" / "pool_random_f1").symlink_to(Path(best["artifact_path"]).name, target_is_directory=True)
                rows.append(best["row"])
                write_json(folder / "selection.json", {"canonical_U2_selection": selected["inner"],
                    "canonical_U2_source": str((folder / "U2" / "selection.json").resolve()),
                    "tau_candidates": [{key: value for key, value in trial.items() if key != "row"} for trial in trials],
                    "selected_tau": best["tau"], "objective": "inner_validation_average_precision",
                    "native_parameters_locked": True, "group_alpha_locked": True, "no_test_selection": True})
                outer = prepare_role(dataset, split, partitions, "outer", seed, selected, settings, folder / "outer")
                for name, tau in (("pool_random", None), ("pool_random_f1", best["tau"])):
                    row = trajectory(dataset, split, partitions, "outer", seed, algorithm, selected,
                        settings, protocol, folder / "outer" / name, outer, metadata, tau)
                    if name == "pool_random" and source is not None:
                        original = source_rows[(seed, algorithm, "outer_iterative_pool_random")]
                        actual = pd.read_csv(row["predictions_path"], float_precision="round_trip")
                        old = pd.read_csv(original["predictions_path"], float_precision="round_trip")
                        np.testing.assert_array_equal(actual.to_numpy(), old.to_numpy())
                        write_json(folder / "outer" / name / "historical_replay.json", {
                            "exact_source_predictions": True, "source_run_id": original["run_id"],
                            "source_record_id": original["record_id"], "gradients_and_surrogates_removed": True})
                    rows.append(row)
                export_comparisons(artifacts.path, rows)
                print(f"seed={seed} classifier={algorithm}: complete; tau={best['tau']}", flush=True)
        artifacts.finish(rows)
    except Exception as error:
        artifacts.finish(rows, error=str(error), status="failed")
        raise
    print(f"Artifacts: {artifacts.path}", flush=True)
    return artifacts.path


if __name__ == "__main__":
    main()
