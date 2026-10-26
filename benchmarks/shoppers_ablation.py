"""Execute the weight-one Shoppers factorial/knockout and matched-proposal study.

Prepare once, run each native classifier, then collect/verify. All experiment
state is local and immutable native source hashes guard classifier workers.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from threadpoolctl import threadpool_limits
import torch

from benchmarks.protocols import load_dataset, load_protocol
from datasets.scripts.climb import make_preprocessor, categorical_columns
from reports.synthetic_six_methods import notebook_runtime
from samplers.flow_gmm import FlowGMMConfig, FlowGMMCore, FlowGate, fit_flow_gmm, _checksum
from samplers.groups import GroupConfig, prepare_groups

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/ablations/shoppers_u2_flow_weight1.yaml"
SOURCE_PATHS = ["benchmarks/shoppers_ablation.py", "benchmarks/shoppers_ablation_sampling.py",
    "samplers/ugo.py", "samplers/groups.py", "samplers/allocation.py", "samplers/flow_gmm.py",
    "datasets/scripts/climb.py", "UGO_30_Datasets.ipynb", "configs/ablations/shoppers_u2_flow_weight1.yaml"]


def json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_PATHS}


def grouped_validation(indices, dataset, groups, seed):
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    fit, validation = next(splitter.split(dataset["X"].iloc[indices], dataset["y"][indices], groups[indices]))
    return indices[fit], indices[validation]


def prepare(run_dir, smoke, benchmark):
    if (run_dir / "manifest.json").exists():
        raise FileExistsError("Choose a new run directory; prepared splits are immutable")
    config = json.loads(CONFIG_PATH.read_text())
    config["device"] = benchmark.resolve_device("cuda")
    if smoke:
        config.update(seeds=[0], n_candidates=2, mlp_epochs=3, flow_epochs=3,
            proposal_cap=64, max_rounds=3, replay_mask_seeds=[0, 1])
    protocol = load_protocol(ROOT / "configs/benchmarks/shoppers_controlled.yaml")
    dataset = load_dataset(protocol, ROOT / ".cache/benchmarks", download=False)
    if smoke:
        training, _ = train_test_split(np.arange(len(dataset["y"])), test_size=.2,
            stratify=dataset["y"], random_state=2026)
        selected, _ = train_test_split(training, train_size=800,
            stratify=dataset["y"][training], random_state=101)
        dataset = {**dataset, "X": dataset["X"].iloc[selected].reset_index(drop=True),
                   "y": dataset["y"][selected], "row_ids": dataset["row_ids"][selected]}
    groups = pd.factorize(pd.MultiIndex.from_frame(dataset["X"]), sort=False)[0]
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(dataset, run_dir / "dataset.joblib")
    save_json(run_dir / "dataset.json", {"row_ids": dataset["row_ids"], "y": dataset["y"],
        "duplicate_groups": groups, "metadata": dataset["metadata"], "evaluated_rows": len(groups)})
    hashes = source_hashes()
    units = len(config["classifiers"]) * config["outer_folds"] * len(config["seeds"])
    manifest = {"study_id": config["study_id"], "mode": "smoke" if smoke else "scientific",
        "status": "preparing", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "source_hashes": hashes, "dataset_rows": len(groups),
        "expected_counts": {"main": units * 8, "replay": units * (1 + 2 * len(config["replay_mask_seeds"]))},
        "versions": {name: importlib.metadata.version(name) for name in
            ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "torch", "xgboost")},
        "platform": platform.platform(), "gpu": torch.cuda.get_device_name(0),
        "aggregation": "five-fold OOF within training seed; random-mask metric means within seed; sample SD over three training seeds",
        "claims": "development only; previously examined Shoppers; no independent-fold or random-mask significance",
        "classifier_checkpoints_saved": False, "weights": {"real": 1., "synthetic": 1.}}
    save_json(run_dir / "manifest.json", manifest)
    outer = StratifiedGroupKFold(config["outer_folds"], shuffle=True, random_state=config["split_seed"])
    for fold, (train, test) in enumerate(outer.split(dataset["X"], dataset["y"], groups)):
        for seed in config["seeds"]:
            folder = run_dir / "shared" / f"fold-{fold}_seed-{seed}"
            folder.mkdir(parents=True)
            fit, validation = grouped_validation(train, dataset, groups, 10000 + fold * 101 + seed)
            flow_fit, calibration = grouped_validation(fit, dataset, groups, 20000 + fold * 101 + seed)
            split = dict(train=train, test=test, fit=fit, validation=validation,
                         flow_fit=flow_fit, flow_calibration=calibration)
            np.savez_compressed(folder / "split.npz", **split)
            preprocessors = {"inner": make_preprocessor(dataset["X"].iloc[fit]),
                             "outer": make_preprocessor(dataset["X"].iloc[train])}
            features = {"fit": preprocessors["inner"].fit_transform(dataset["X"].iloc[fit]).astype(np.float32),
                "validation": preprocessors["inner"].transform(dataset["X"].iloc[validation]).astype(np.float32),
                "train": preprocessors["outer"].fit_transform(dataset["X"].iloc[train]).astype(np.float32),
                "test": preprocessors["outer"].transform(dataset["X"].iloc[test]).astype(np.float32)}
            np.savez_compressed(folder / "role_X.npz", **features)
            joblib.dump(preprocessors, folder / "preprocessors.joblib")
            prepared = {}
            for role, key in [("inner", "fit"), ("outer", "train")]:
                y = dataset["y"][split[key]]
                prepared[role] = prepare_groups(features[key], y, [1], seed, GroupConfig())
                prepared[role]["real_row_ids"] = dataset["row_ids"][split[key]]
            joblib.dump(prepared, folder / "groups.joblib")
            core = fit_flow_gmm(dataset["X"].iloc[flow_fit], dataset["y"][flow_fit],
                dataset["row_ids"][flow_fit], seed, FlowGMMConfig(epochs=config["flow_epochs"], device="cuda"))
            core.save(folder / "flow.pt")
            save_json(folder / "metadata.json", {"fold": fold, "seed": seed, "source_hashes": hashes,
                "flow": {**core.audit, "backend": "pytorch_cuda", "fit_indices": flow_fit,
                    "fit_count": len(flow_fit), "fit_call_count": 1, "unused_calibration": True},
                "flow_fit_row_ids": dataset["row_ids"][flow_fit],
                "calibration_row_ids": dataset["row_ids"][calibration],
                "validation_row_ids": dataset["row_ids"][validation],
                "flow_core_sha256": _checksum(core.model), "calibration_used": False,
                "duplicate_group_partitioning": True, "device": str(core.device)})
            print(f"Prepared fold {fold}, seed {seed}: train={len(train)} test={len(test)} flow_fit={len(flow_fit)}", flush=True)
            del core
    manifest["status"] = "prepared"
    save_json(run_dir / "manifest.json", manifest)


def probabilities(benchmark, model, X):
    return benchmark._positive_probability(model, X, 1)


def fit_native(benchmark, classifier, seed, parameters, config, X, y):
    model = benchmark.make_estimator(classifier, seed, parameters, device="cuda",
        epochs=config["mlp_epochs"], n_jobs=config["n_jobs"])
    model = benchmark.fit_estimator(model, X, y)
    if classifier == "mlp" and str(next(model.model_.parameters()).device) != "cuda:0":
        raise ValueError("MLP did not train on CUDA")
    if classifier == "random_forest":
        # Parallel tree fitting is seeded, but parallel probability accumulation
        # changes floating-point tie order. Keep every MMA/final prediction in
        # deterministic tree order; training still uses config["n_jobs"].
        model.n_jobs = config["random_forest_prediction_n_jobs"]
    return model


def choose_parameters(benchmark, classifier, seed, config, X, y, V, truth):
    candidates = benchmark.candidate_parameters(classifier, seed, config["n_candidates"], smoke=config.get("smoke", False))
    entries, best = [], None
    for candidate_id, parameters in enumerate(candidates):
        tick = time.perf_counter()
        model = fit_native(benchmark, classifier, seed, parameters, config, X, y)
        p = probabilities(benchmark, model, V)
        threshold, score = benchmark.choose_f1_threshold(truth, p)
        m = benchmark._metrics(truth, p, threshold)
        entry = {"candidate_id": candidate_id, "parameters": parameters, "threshold": threshold,
            "validation_f1": score, "validation_ap": m["average_precision"], "backend": model.backend_,
            "f1_numerator": 2 * m["tp"], "f1_denominator": 2 * m["tp"] + m["fp"] + m["fn"],
            "fit_seconds": time.perf_counter() - tick}
        entries.append(entry)
        if benchmark._better(entry, best): best = entry
    return {"candidates": entries, "selected_parameters": best["parameters"], "selected_candidate_id": best["candidate_id"],
        "selection_objective": "real_only_validation_minority_F1_AP_secondary", "validation_used_in_fit": False}


def save_predictions(path, ids, y, p, threshold):
    pd.DataFrame({"row_id": ids, "y_true": y, "probability": p, "threshold": threshold,
        "y_pred": (p >= threshold).astype(np.int64), "y_pred_05": (p >= .5).astype(np.int64)}).to_csv(
            path, index=False, float_format="%.17g")


def freeze_threshold(benchmark, folder, truth, probability):
    threshold, score = benchmark.choose_f1_threshold(truth, probability)
    save_json(folder / "threshold.json", {"threshold": threshold, "validation_f1": score,
        "selection": "real_validation_only", "selected_before_outer_fit": True})


def run_role(benchmark, sampler, dataset, split, features, preprocessors, groups, core,
             role, variant, classifier, seed, parameters, config, folder, shared_metadata):
    folder.mkdir(parents=True, exist_ok=True)
    key = "fit" if role == "inner" else "train"
    indices = split[key]
    X, y = features[key], dataset["y"][indices]
    n = len(y)
    targets = {0: int(np.count_nonzero(y == 0)), 1: int(np.count_nonzero(y == 1)) + config["proposal_cap"]}
    gate = FlowGate(core, preprocessors[role], calibration_row_ids=dataset["row_ids"][split["flow_calibration"]],
        validation_row_ids=dataset["row_ids"][split["validation"]])
    gate.role = role
    before = _checksum(core.model)
    fits, fit_seconds, latest = 0, 0., {}
    tick = time.perf_counter()
    def train(X_round, y_round):
        nonlocal fits, fit_seconds
        torch.cuda.synchronize()
        start = time.perf_counter()
        model = fit_native(benchmark, classifier, seed, parameters, config, X_round, y_round)
        torch.cuda.synchronize()
        fit_seconds += time.perf_counter() - start
        fits += 1
        latest.update(model=model, count=len(y_round))
        return model
    trace, result = None, None
    if variant == "A0":
        sampled_X, sampled_y = X, y
        model = train(X, y)
    else:
        result, trace = sampler.run_ablation(X, y, train, variant=variant, seed=seed,
            targets=targets, device="cuda", prepared=groups[role],
            flow_support=gate if variant in ("A2", "A4", "A5", "A6") else None,
            categorical_columns=categorical_columns(preprocessors[role]),
            categorical_groups=benchmark._categorical_groups(preprocessors[role]),
            max_iterations=config["max_rounds"], capture=variant == "A4")
        sampled_X, sampled_y = benchmark.as_array(result.X), benchmark.as_array(result.y)
        model = latest["model"] if latest.get("count") == len(sampled_y) else train(sampled_X, sampled_y)
    np.testing.assert_array_equal(sampled_X[:n], X)
    np.testing.assert_array_equal(sampled_y[:n], y)
    assert np.all(sampled_y[n:] == 1)
    if _checksum(core.model) != before or core.model.training or any(p.requires_grad for p in core.model.parameters()):
        raise ValueError("Frozen Flow core changed during generation")
    accepted = len(sampled_y) - n
    if trace is not None:
        np.testing.assert_array_equal(trace["X"][trace["accepted"]], sampled_X[n:])
        np.savez_compressed(folder / "proposal_trace.npz", **trace)
    np.savez_compressed(folder / "final_training.npz", X=sampled_X, y=sampled_y)
    V = features["validation"] if role == "inner" else features["test"]
    probability = probabilities(benchmark, model, V)
    summary = {} if result is None else result.support_summary
    proposal = summary.get("proposal_budget", {})
    attempted = int(proposal.get("attempted", accepted))
    audit = {"role": role, "variant": variant, "real_count": n, "real_prefix_count": n,
        "accepted_count": accepted, "requested_count": 0 if variant == "A0" else config["proposal_cap"],
        "proposal_count": attempted, "rejected_count": attempted - accepted,
        "real_weight": 1., "synthetic_weight": 1., "sample_weight": None, "cold_refit": True,
        "original_parents": True, "gate_no_relocation": variant != "A7",
        "stop_reason": "no_oversampling" if result is None else result.stop_reason,
        "round_count": 0 if result is None else len(result.rounds),
        "rounds": [] if result is None else result.rounds,
        "provenance": [] if result is None else result.provenance,
        "support_summary": benchmark._compact(summary), "native_parameters": parameters,
        "classifier_fit_count": fits, "native_fit_seconds": fit_seconds,
        "backend": model.backend_, "flow_state_before": before, "flow_state_after": _checksum(core.model),
        "flow_fit_shared": True, "flow_training_seconds_reference": core.audit["training_seconds"],
        "runtime_seconds": time.perf_counter() - tick}
    save_json(folder / "audit.json", audit)
    return {"probability": probability, "audit": audit, "trace": trace,
            "training_X": sampled_X, "training_y": sampled_y}


def random_mask(trace, strategy, seed):
    mask = np.zeros(len(trace["y"]), dtype=bool)
    rng = np.random.default_rng(seed)
    if strategy == "flow": return trace["accepted"].astype(bool).copy()
    keys = np.column_stack([trace["round"], trace["group_id"]]) if strategy == "random_round_group" else trace["round"][:, None]
    for value in np.unique(keys, axis=0):
        membership = np.all(keys == value, axis=1)
        valid = np.flatnonzero(membership & trace["valid"])
        count = np.count_nonzero(membership & trace["accepted"])
        if count > len(valid): raise ValueError("Replay count exceeds valid source candidates")
        mask[rng.choice(valid, size=count, replace=False)] = True
    return mask


def replay_role(benchmark, dataset, split, features, role, classifier, seed, parameters,
                config, trace, strategy, mask_seed, folder, source_trace, run_dir):
    folder.mkdir(parents=True, exist_ok=True)
    key = "fit" if role == "inner" else "train"
    X, y = features[key], dataset["y"][split[key]]
    mask = random_mask(trace, strategy, mask_seed if mask_seed >= 0 else 0)
    sampled_X = np.concatenate((X, trace["X"][mask]))
    sampled_y = np.concatenate((y, trace["y"][mask]))
    tick = time.perf_counter()
    model = fit_native(benchmark, classifier, seed, parameters, config, sampled_X, sampled_y)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - tick
    probability = probabilities(benchmark, model, features["validation" if role == "inner" else "test"])
    np.save(folder / "replay_mask.npy", mask)
    np.savez_compressed(folder / "final_training.npz", X=sampled_X, y=sampled_y)
    audit = {"role": role, "real_count": len(y), "real_prefix_count": len(y),
        "accepted_count": int(mask.sum()), "requested_count": config["proposal_cap"],
        "proposal_count": len(mask), "rejected_count": len(mask) - int(mask.sum()),
        "real_weight": 1., "synthetic_weight": 1., "sample_weight": None,
        "original_parents": True, "gate_no_relocation": True, "source_trace": str(source_trace.relative_to(run_dir)),
        "replay_strategy": strategy, "mask_seed": mask_seed, "native_parameters": parameters,
        "round_count": 0, "stop_reason": "fixed_proposal_final_fit_intervention",
        "classifier_fit_count": 1, "native_fit_seconds": elapsed, "runtime_seconds": elapsed,
        "backend": model.backend_, "cold_refit": True}
    save_json(folder / "audit.json", audit)
    return {"probability": probability, "audit": audit}


def finish_record(benchmark, run_dir, unit, dataset, split, classifier, fold, seed,
                  variant, inner, outer, inner_folder, outer_folder,
                  result_type="main", replay_strategy="", mask_seed=-1):
    threshold = json.loads((inner_folder / "threshold.json").read_text())["threshold"]
    save_predictions(inner_folder / "validation_predictions.csv", dataset["row_ids"][split["validation"]],
        dataset["y"][split["validation"]], inner["probability"], threshold)
    save_predictions(outer_folder / "test_predictions.csv", dataset["row_ids"][split["test"]],
        dataset["y"][split["test"]], outer["probability"], threshold)
    audit = outer["audit"]
    metrics = benchmark._metrics(dataset["y"][split["test"]], outer["probability"], threshold)
    metrics["precision"] = metrics.pop("minority_precision")
    record = {"classifier": classifier, "fold": fold, "seed": seed, "variant": variant,
        "result_type": result_type, "replay_strategy": replay_strategy, "mask_seed": mask_seed,
        "threshold": threshold, **metrics,
        **{name: audit[name] for name in ["accepted_count", "requested_count", "proposal_count", "rejected_count",
            "round_count", "classifier_fit_count", "native_fit_seconds", "runtime_seconds", "backend"]},
        "predictions_path": str((outer_folder / "test_predictions.csv").relative_to(run_dir)),
        "validation_predictions_path": str((inner_folder / "validation_predictions.csv").relative_to(run_dir)),
        "artifact_dir": str(outer_folder.relative_to(run_dir)), "status": "complete"}
    name = variant if result_type == "main" else f"replay_{replay_strategy}_{mask_seed}"
    save_json(unit / "records" / f"{name}.json", record)
    return record


def run_unit(run_dir, manifest, dataset, classifier, fold, seed, benchmark, sampler):
    config = {**manifest["config"], "smoke": manifest["mode"] == "smoke"}
    shared = run_dir / "shared" / f"fold-{fold}_seed-{seed}"
    unit = run_dir / "cells" / classifier / f"fold-{fold}_seed-{seed}"
    if (unit / "completion.json").exists():
        print(f"Already complete: {classifier} fold {fold} seed {seed}", flush=True)
        return
    unit.mkdir(parents=True, exist_ok=True)
    with np.load(shared / "split.npz") as data: split = {key: data[key] for key in data.files}
    with np.load(shared / "role_X.npz") as data: features = {key: data[key] for key in data.files}
    preprocessors = joblib.load(shared / "preprocessors.joblib")
    groups = joblib.load(shared / "groups.joblib")
    metadata = json.loads((shared / "metadata.json").read_text())
    core = FlowGMMCore.load(shared / "flow.pt", device="cuda")
    if _checksum(core.model) != metadata["flow_core_sha256"]: raise ValueError("Shared Flow checkpoint checksum mismatch")
    selection = choose_parameters(benchmark, classifier, seed, config, features["fit"], dataset["y"][split["fit"]],
        features["validation"], dataset["y"][split["validation"]])
    save_json(unit / "selection.json", selection)
    parameters = selection["selected_parameters"]
    full = {}
    for variant in sampler.VARIANTS:
        roles, folders = {}, {}
        for role in ("inner", "outer"):
            folders[role] = unit / role / variant
            roles[role] = run_role(benchmark, sampler, dataset, split, features, preprocessors, groups, core,
                role, variant, classifier, seed, parameters, config, folders[role], metadata)
            if role == "inner":
                freeze_threshold(benchmark, folders[role], dataset["y"][split["validation"]], roles[role]["probability"])
        record = finish_record(benchmark, run_dir, unit, dataset, split, classifier, fold, seed,
            variant, roles["inner"], roles["outer"], folders["inner"], folders["outer"])
        if variant == "A4": full = roles
        print(f"{classifier} fold {fold} seed {seed} {variant}: F1={record['minority_f1']:.4f} accepted={record['accepted_count']}", flush=True)
    strategies = [("flow", -1)] + [(strategy, mask_seed) for strategy in
        ("random_round", "random_round_group") for mask_seed in config["replay_mask_seeds"]]
    for strategy, mask_seed in strategies:
        roles, folders = {}, {}
        for role in ("inner", "outer"):
            folders[role] = unit / role / "replay" / strategy / f"mask-{mask_seed}"
            roles[role] = replay_role(benchmark, dataset, split, features, role, classifier, seed, parameters,
                config, full[role]["trace"], strategy, mask_seed, folders[role],
                unit / role / "A4" / "proposal_trace.npz", run_dir)
            if role == "inner":
                freeze_threshold(benchmark, folders[role], dataset["y"][split["validation"]], roles[role]["probability"])
            if strategy == "flow":
                np.testing.assert_array_equal(roles[role]["probability"], full[role]["probability"])
        finish_record(benchmark, run_dir, unit, dataset, split, classifier, fold, seed, "A4",
            roles["inner"], roles["outer"], folders["inner"], folders["outer"], "replay", strategy, mask_seed)
    save_json(unit / "completion.json", {"status": "complete", "classifier": classifier, "fold": fold,
        "seed": seed, "main_count": 8, "replay_count": len(strategies), "source_hashes": manifest["source_hashes"]})
    print(f"Completed {classifier} fold {fold} seed {seed}: 8 variants + {len(strategies)} replays", flush=True)


def run_classifier(run_dir, classifier, benchmark):
    from benchmarks import shoppers_ablation_sampling as sampler
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if source_hashes() != manifest["source_hashes"]: raise ValueError("Native source changed since preparation")
    dataset = joblib.load(run_dir / "dataset.joblib")
    started = time.perf_counter()
    try:
        for fold in range(manifest["config"]["outer_folds"]):
            for seed in manifest["config"]["seeds"]:
                run_unit(run_dir, manifest, dataset, classifier, fold, seed, benchmark, sampler)
    except Exception as error:
        save_json(run_dir / "workers" / f"{classifier}.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()})
        raise
    save_json(run_dir / "workers" / f"{classifier}.json", {"status": "complete", "elapsed_seconds": time.perf_counter() - started})


def collect(run_dir):
    rows = [json.loads(path.read_text()) for path in sorted((run_dir / "cells").glob("*/fold-*_seed-*/records/*.json"))]
    frame = pd.DataFrame(rows)
    frame.to_csv(run_dir / "metrics.csv", index=False, float_format="%.17g")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    actual = frame.result_type.value_counts().to_dict()
    manifest["completed_counts"] = actual
    manifest["status"] = "complete" if actual == manifest["expected_counts"] else "incomplete"
    save_json(run_dir / "manifest.json", manifest)
    if manifest["status"] != "complete": raise ValueError(f"Incomplete study: {actual} vs {manifest['expected_counts']}")
    from benchmarks.shoppers_ablation_report import summarize, verify
    summarize(run_dir)
    print(json.dumps(verify(run_dir), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--collect", action="store_true")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--classifier", choices=("mlp", "random_forest", "xgboost"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.run and args.classifier is None: parser.error("--run requires --classifier")
    with notebook_runtime(ROOT / "UGO_30_Datasets.ipynb") as benchmark, threadpool_limits(limits=4):
        torch.set_num_threads(4)
        if args.prepare: prepare(args.run_dir, args.smoke, benchmark)
        elif args.run: run_classifier(args.run_dir, args.classifier, benchmark)
        else: collect(args.run_dir)


if __name__ == "__main__":
    main()
