"""F1-first CLIMB compatibility study: original UGO versus locked U2 + FlowGMM.

Smoke observations are drawn exclusively from outer training and are not study
results. No historical selection protocol or stored historical result is used.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from time import perf_counter
import traceback
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import torch

from benchmarks.flow_partitions import flow_for_partition
from benchmarks.run import (augmentation_targets, categorical_groups,
                            prepare_partitions, prepare_protected_groups,
                            split_fingerprint, train_method)
from classifiers.factory import candidate_parameters
from datasets.scripts.climb import categorical_columns, load_climb_datasets


METHODS = ("original_ugo", "u2_flow_gate")
DATASETS = ("ozone-level-8hr", "wilt", "mammography", "dis", "Satellite")
CLASSIFIERS = ("mlp", "logistic_regression", "xgboost", "random_forest")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _dump(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    default=_json_default, allow_nan=False) + "\n")
    temporary.replace(path)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    default=_json_default).encode()).hexdigest()


def choose_f1_threshold(y_true, probability):
    """Evaluate whole tied-score groups; exact F1 ties prefer near .5, then high."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Threshold selection requires aligned nonempty vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Threshold selection requires binary labels and finite probabilities")
    if not y.sum():
        raise ValueError("Validation must contain the positive class")
    order = np.argsort(-p, kind="stable")
    ordered = p[order]
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(y) - 1]
    cumulative = np.cumsum(y[order])
    positives = int(y.sum())
    best = None
    for end in ends:
        tp = int(cumulative[end])
        denominator = int(end) + 1 + positives
        threshold = float(ordered[end])
        if best is None:
            best = (tp, denominator, threshold)
            continue
        comparison = tp * best[1] - best[0] * denominator
        if comparison > 0 or (comparison == 0 and
                (-abs(threshold - .5), threshold) > (-abs(best[2] - .5), best[2])):
            best = (tp, denominator, threshold)
    return best[2], 2 * best[0] / best[1]


def _metrics(y, probability, threshold):
    truth = np.asarray(y, dtype=np.int64)
    predicted = np.asarray(probability) >= threshold
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    return {"minority_f1": float(f1_score(truth, predicted, zero_division=0)),
            "minority_precision": float(precision_score(truth, predicted, zero_division=0)),
            "minority_recall": float(recall_score(truth, predicted, zero_division=0)),
            "average_precision": float(average_precision_score(truth, probability)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
            "mcc": float(matthews_corrcoef(truth, predicted)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "f1_at_0_5": float(f1_score(truth, np.asarray(probability) >= .5, zero_division=0))}


def _positive_probability(model, features, positive):
    columns = np.flatnonzero(np.asarray(model.classes_) == positive)
    if len(columns) != 1:
        raise ValueError("Fitted learner has no unique minority-class probability column")
    probability = np.asarray(model.predict_proba(features), dtype=np.float64)[:, columns[0]]
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Learner produced invalid probabilities")
    return probability


def _compact_audit(audit, role, row_ids, real_labels, maximum, iterations, real_only):
    """Check full ancestry/proposals in memory, retain only compact verifiable proof."""
    ids = np.asarray(row_ids)
    labels = list(np.asarray(real_labels, dtype=np.int64))
    n = len(ids)
    provenance = audit["provenance"]
    accepted = int(audit["augmentation"]["actual_count"])
    requested = int(audit["augmentation"]["requested_count"])
    if len(provenance) != accepted or not 0 <= accepted <= requested <= maximum:
        raise ValueError("Augmentation counts do not match ancestry or proposal cap")
    roots = [{index} for index in range(n)]
    used_roots = set()
    direct, synthetic = 0, 0
    for offset, entry in enumerate(provenance):
        row = n + offset
        if int(entry["row_index"]) != row:
            raise ValueError("Synthetic ancestry is not contiguous/topological")
        parents = [int(entry["seed_index"]), int(entry["partner_index"])]
        relocation = int(entry.get("relocation_index", -1))
        if relocation >= 0:
            parents.append(relocation)
        row_roots = set()
        for parent in parents:
            if not 0 <= parent < row or (real_only and parent >= n):
                raise ValueError("Synthetic parent falls outside allowed real/training ancestry")
            if labels[parent] != int(entry["class"]):
                raise ValueError("Synthetic parent does not share intended class")
            direct += parent < n
            synthetic += parent >= n
            row_roots.update(roots[parent])
        if not row_roots:
            raise ValueError("Synthetic observation is not rooted in real training")
        roots.append(row_roots)
        labels.append(int(entry["class"]))
        used_roots.update(row_roots)
    summary = audit["support_summary"]
    candidate_log = summary.get("candidate_log")
    if candidate_log is not None:
        if len(candidate_log) > requested or sum(bool(x["accepted"]) for x in candidate_log) != accepted:
            raise ValueError("Proposal log does not reconcile accepted ancestry")
        accepted_log = [x for x in candidate_log if x["accepted"]]
        for entry, proposed in zip(provenance, accepted_log):
            for key in ("seed_index", "partner_index", "relocation_index", "class", "proposal_id"):
                if entry[key] != proposed[key]:
                    raise ValueError("Accepted proposal and final ancestry disagree")
        for proposal in candidate_log:
            for key in ("seed_index", "partner_index"):
                if not 0 <= int(proposal[key]) < n:
                    raise ValueError("Flow proposal parent is not real role training")
                if labels[int(proposal[key])] != int(proposal["class"]):
                    raise ValueError("Flow proposal parent does not share intended class")
            if int(proposal["relocation_index"]) != -1:
                raise ValueError("Flow handling must reject without relocating proposals")
            tau = float(audit["sampler_parameters"]["f1_tau"])
            valid_score = proposal["reason"] in ("accepted", "flow_incompatible")
            expected = (valid_score and proposal.get("log_ratio") is not None and
                        proposal["log_ratio"] >= math.log(tau) - math.log1p(-tau))
            if bool(proposal["accepted"]) != expected:
                raise ValueError("Flow acceptance violates the fixed log-ratio threshold")
    budget = summary.get("proposal_budget")
    if budget is not None:
        budget = {key: budget[key] for key in ("cap", "attempted", "accepted", "rejected")}
        if (not 0 <= budget["attempted"] <= requested or budget["cap"] != requested or
                budget["accepted"] != accepted or
                budget["rejected"] != budget["attempted"] - accepted):
            raise ValueError("Flow proposal budget accounting is inconsistent")
        if candidate_log is None or budget["attempted"] != len(candidate_log):
            raise ValueError("Flow attempts are not fully represented in the in-memory audit")
    else:
        budget = {"cap": requested, "attempted": accepted, "accepted": accepted, "rejected": 0}
    rounds = []
    for row in audit["rounds"]:
        proposed = int(row.get("proposed", row.get("generated", 0)))
        count = int(row.get("accepted", row.get("generated", 0)))
        rounds.append({"proposed_count": proposed, "accepted_count": count,
                       "rejected_count": proposed - count})
    if len(rounds) > iterations or sum(x["accepted_count"] for x in rounds) != accepted:
        raise ValueError("Round counts violate the fixed iteration budget")
    if sum(x["proposed_count"] for x in rounds) != budget["attempted"]:
        raise ValueError("Round proposals do not reconcile the proposal budget")
    return {"role": role, "real_row_ids_sha256": _hash(ids.tolist()), "real_count": n,
            "classifier_fit_count": int(audit["classifier_fit_count"]),
            "accepted_count": accepted, "requested_count": requested,
            "stop_reason": audit["stop_reason"], "round_count": len(rounds), "rounds": rounds,
            "proposal_budget": budget, "synthetic_weight": 1.0,
            "protocol_deviations": summary.get("protocol_deviations", []),
            "ancestry": {"validated": True, "entry_count": len(provenance),
                "direct_real_parent_count": int(direct), "synthetic_parent_count": int(synthetic),
                "all_roots_in_real_role": True, "real_only_direct_parents": synthetic == 0,
                "root_row_ids": ids[sorted(used_roots)].tolist(),
                "provenance_sha256": _hash(provenance)}}


def _settings(config, method, sampler_parameters):
    return {"device": config["device"], "mlp_epochs": config["mlp_epochs"],
            "ugo_max_iterations": config["max_iterations"],
            "augmentation": {"mode": "matched", "total": config["augmentation_total"]},
            "method_parameters": {method: sampler_parameters}}


def _fit(dataset, split, partitions, algorithm, method, parameters, sampler_parameters,
         config, seed, role, gate=None):
    settings = _settings(config, method, sampler_parameters)
    key = "fit_indices" if role == "inner" else "train_indices"
    indices = split[key]
    features = partitions["X_fit" if role == "inner" else "X_train"]
    preprocessor = partitions[role]
    groups = None
    if method != "ugo_original":
        groups = prepare_protected_groups(features, dataset["y"][indices], seed,
            augmentation_targets(dataset["y"][indices], settings["augmentation"]), sampler_parameters)
        groups["real_row_ids"] = np.asarray(dataset["row_ids"])[indices]
    model, full_audit = train_method(algorithm, method, parameters, features,
        dataset["y"][indices], categorical_columns(preprocessor), seed, settings,
        validation=None, groups=categorical_groups(preprocessor),
        group_partition=groups, flow_support=gate)
    audit = _compact_audit(full_audit, role, np.asarray(dataset["row_ids"])[indices],
        dataset["y"][indices], config["augmentation_total"], config["max_iterations"],
        real_only=method != "ugo_original")
    return model, audit


def _validation_candidate(dataset, split, partitions, algorithm, method, parameters,
                          sampler_parameters, config, seed, positive, candidate_id, gate=None):
    model, audit = _fit(dataset, split, partitions, algorithm, method, parameters,
                        sampler_parameters, config, seed, "inner", gate)
    probability = _positive_probability(model, partitions["X_validation"], positive)
    threshold, score = choose_f1_threshold(dataset["y"][split["validation_indices"]] == positive, probability)
    return ({"candidate_id": candidate_id, "parameters": parameters,
             "sampler_parameters": sampler_parameters, "threshold": threshold,
             "validation_score": score, "audit": audit}, probability)


def _predictions(path, ids, truth, probability, threshold):
    pd.DataFrame({"row_id": ids, "y_true": np.asarray(truth, dtype=np.int64),
                  "probability": probability, "y_pred": (probability >= threshold).astype(np.int64),
                  "threshold": threshold}).to_csv(path, index=False, float_format="%.17g")


def _flow_evidence(inner, outer, dataset, split):
    core = inner.core
    if outer.core is not core or core.model.training or any(p.requires_grad for p in core.model.parameters()):
        raise ValueError("Flow core must remain frozen and identical across roles")
    audit = core.audit
    fit = core.fit_row_ids.tolist()
    calibration = inner.calibration_row_ids.tolist()
    validation = inner.validation_row_ids.tolist()
    identities = np.asarray(dataset["row_ids"])
    role_fit = set(identities[split["fit_indices"]].tolist())
    fit_set, calibration_set = set(fit), set(calibration)
    heldout = set(identities[split["validation_indices"]].tolist()) | set(identities[split["test_indices"]].tolist())
    if (fit_set & calibration_set or (fit_set | calibration_set) != role_fit or
            (fit_set | calibration_set) & heldout or
            validation != identities[split["validation_indices"]].tolist() or
            outer.calibration_row_ids.tolist() != calibration or
            audit["synthetic_training_count"] != 0 or audit["calibration_used"] or not audit["frozen"]):
        raise ValueError("Flow roles violate real-inner-fit-only frozen support")
    return {"core_sha256": audit["model_checksum"], "fit_row_ids": fit,
            "calibration_row_ids": calibration, "validation_row_ids": validation,
            "roles_disjoint": True, "fit_seconds": float(audit["training_seconds"]),
            "fit_count_shared": 1, "frozen_core_shared_across_roles": True,
            "calibration_used": False, "synthetic_training_count": 0,
            "model_config": audit["model_config"], "source_sha256": audit["source_sha256"]}


def _cell(dataset, split, partitions, algorithm, seed, method, config, candidates,
          run_dir, gates, flow_evidence, shared_flow_seconds):
    started = perf_counter()
    labels = dataset["y"]
    classes, counts = np.unique(labels[split["train_indices"]], return_counts=True)
    if len(classes) != 2:
        raise ValueError("The F1 compatibility gate requires binary datasets")
    positive = int(classes[np.argmin(counts)])
    selection = {"dataset": dataset["name"], "algorithm": algorithm, "seed": seed,
                 "method": method, "candidate_configs": candidates, "canonical_candidates": [],
                 "selected_tau_id": None}
    items, probabilities = [], []
    if method == "original_ugo":
        for index, parameters in enumerate(candidates):
            item, probability = _validation_candidate(dataset, split, partitions, algorithm,
                "ugo_original", parameters, {}, config, seed, positive, index)
            items.append(item)
            probabilities.append(probability)
        selected_index = max(range(len(items)), key=lambda i: items[i]["validation_score"])
        selected = items[selected_index]
        selected_probability = probabilities[selected_index]
        training_method = "ugo_original"
        selection["selected_candidate_id"] = selected_index
    else:
        canonical = []
        for index, parameters in enumerate(candidates):
            sampler = {"group_alpha": config["group_alphas"][index % len(config["group_alphas"]) ]}
            item, _ = _validation_candidate(dataset, split, partitions, algorithm,
                "ugo_groups_only", parameters, sampler, config, seed, positive, index)
            canonical.append(item)
        canonical_index = max(range(len(canonical)), key=lambda i: canonical[i]["validation_score"])
        locked = canonical[canonical_index]
        selection["canonical_candidates"] = canonical
        selection["selected_candidate_id"] = canonical_index
        for index, tau in enumerate(config["taus"]):
            sampler = {**locked["sampler_parameters"], "f1_tau": tau}
            item, probability = _validation_candidate(dataset, split, partitions, algorithm,
                "ugo_f1", locked["parameters"], sampler, config, seed, positive, index, gates[0])
            item["tau"] = tau
            items.append(item)
            probabilities.append(probability)
        selected_index = max(range(len(items)), key=lambda i: items[i]["validation_score"])
        selected = items[selected_index]
        selected_probability = probabilities[selected_index]
        training_method = "ugo_f1"
        selection["selected_tau_id"] = selected_index
        selection["flow_evidence"] = flow_evidence
    threshold = selected["threshold"]
    basename = f"{dataset['name']}__{algorithm}__{seed}__{method}"
    prediction_path = Path("predictions") / f"{basename}.csv"
    validation_path = Path("predictions") / f"{basename}__validation.csv"
    # This file is written before outer refitting: it is the actual selected inner learner.
    _predictions(run_dir / validation_path,
        np.asarray(dataset["row_ids"])[split["validation_indices"]],
        labels[split["validation_indices"]] == positive, selected_probability, threshold)
    model, outer_audit = _fit(dataset, split, partitions, algorithm, training_method,
        selected["parameters"], selected["sampler_parameters"], config, seed, "outer",
        gates[1] if method == "u2_flow_gate" else None)
    probability = _positive_probability(model, partitions["X_test"], positive)
    _predictions(run_dir / prediction_path, np.asarray(dataset["row_ids"])[split["test_indices"]],
        labels[split["test_indices"]] == positive, probability, threshold)
    selection.update(candidates=items, threshold=threshold,
        selected_validation_score=selected["validation_score"], selected_inner_audit=selected["audit"],
        outer_audit=outer_audit)
    fits = sum(item["audit"]["classifier_fit_count"] for item in
               selection["canonical_candidates"] + items) + outer_audit["classifier_fit_count"]
    selection["classifier_fit_count"] = fits
    if method == "u2_flow_gate":
        after = _flow_evidence(gates[0], gates[1], dataset, split)
        # Hash state tensors as well as checking the original immutable audit checksum.
        if after != flow_evidence or _core_hash(gates[0].core) != partitions["flow_tensor_sha256"]:
            raise ValueError("Flow core or role evidence changed during classifier fitting")
    record = {"dataset": dataset["name"], "algorithm": algorithm, "seed": seed, "method": method,
        "status": "completed", "positive_class": positive, "threshold": threshold,
        **_metrics(labels[split["test_indices"]] == positive, probability, threshold),
        "runtime_seconds": perf_counter() - started, "classifier_fit_count": fits,
        "accepted_count": outer_audit["accepted_count"], "requested_count": outer_audit["requested_count"],
        "flow_fit_seconds_shared": shared_flow_seconds,
        "split_fingerprint": split_fingerprint(dataset, split),
        "data_fingerprint": dataset["metadata"]["raw_fingerprint"],
        "predictions_path": str(prediction_path), "validation_predictions_path": str(validation_path)}
    return record, selection


def _core_hash(core):
    digest = hashlib.sha256()
    for name, value in sorted(core.model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _split(dataset, seed, config, smoke):
    rows = np.arange(len(dataset["y"]))
    train, test = train_test_split(rows, test_size=config["test_fraction"],
                                  stratify=dataset["y"], random_state=seed)
    if smoke:
        # Never inspect or consume the scientific outer test in training-only smoke.
        limit = min(int(config["smoke_rows"]), len(train) - 1)
        subset, _ = train_test_split(train, train_size=limit,
            stratify=dataset["y"][train], random_state=seed)
        train, test = train_test_split(subset, test_size=config["test_fraction"],
            stratify=dataset["y"][subset], random_state=seed)
    fit, validation = train_test_split(train, test_size=config["validation_fraction"],
        stratify=dataset["y"][train], random_state=seed)
    return {"fold": 0, "train_indices": train, "fit_indices": fit,
            "validation_indices": validation, "test_indices": test}


def _source_hashes(config_path):
    paths = {Path(__file__).resolve(), config_path.resolve(),
             Path(__file__).with_name("climb_flow_f1_report.py").resolve()}
    for name, module in tuple(sys.modules.items()):
        if name == "experiments" or name.startswith(("benchmarks.", "classifiers.", "samplers.", "datasets.scripts.")):
            filename = getattr(module, "__file__", None)
            if filename and filename.endswith(".py"):
                paths.add(Path(filename).resolve())
    root = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(root)) if path.is_relative_to(root) else str(path):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def _parse_subset(values, allowed, label):
    if values is None:
        return list(allowed)
    chosen = [item for value in values for item in value.split(",")]
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(allowed):
        raise ValueError(f"Unknown or duplicate {label}: {chosen}")
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Training-only reduced integration run, never scientific evidence")
    mode.add_argument("--run", action="store_true", help="Execute the configured scientific study")
    parser.add_argument("--config", type=Path, default=Path("configs/ablations/climb_flow_f1.json"))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--classifiers", nargs="+")
    parser.add_argument("--seeds", nargs="+")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    config["datasets"] = _parse_subset(args.datasets, config["datasets"], "datasets")
    config["classifiers"] = _parse_subset(args.classifiers, config["classifiers"], "classifiers")
    seeds = _parse_subset(args.seeds, [str(seed) for seed in config["seeds"]], "seeds")
    config["seeds"] = [int(seed) for seed in seeds]
    if set(config["datasets"]) - set(DATASETS) or set(config["classifiers"]) - set(CLASSIFIERS):
        raise ValueError("Only the five CLIMB datasets and four native learner families are supported")
    if args.smoke:
        config.update(tuning_candidates=2, max_iterations=2, augmentation_total=40,
                      mlp_epochs=2, flow_epochs=2, smoke_rows=1200)
    for key in ("tuning_candidates", "max_iterations", "augmentation_total", "mlp_epochs", "flow_epochs", "threads"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not config["group_alphas"] or not config["taus"]:
        raise ValueError("Canonical alpha and Flow tau grids must be nonempty")
    if len(set(config["taus"])) != len(config["taus"]) or any(not 0 < tau < 1 for tau in config["taus"]):
        raise ValueError("Flow taus must be distinct probabilities strictly inside (0, 1)")
    if any(not 0 <= alpha <= 1 for alpha in config["group_alphas"]):
        raise ValueError("Canonical group alphas must lie inside [0, 1]")
    output_root = args.output_root or Path("results/smoke" if args.smoke else "results/runs")
    run_dir = output_root / ("climb-flow-f1-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "predictions").mkdir()
    started = perf_counter()
    manifest = {"study": "climb_flow_f1", "mode": "smoke" if args.smoke else "run",
        "status": "running", "config": config, "config_path": str(args.config),
        "primary_metric": "minority_f1", "threshold_policy": "validation_unique_scores_max_f1_tie_nearest_half_then_higher",
        "methods": list(METHODS), "datasets": [], "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke_is_scientific_evidence": False, "source_hashes": _source_hashes(args.config),
        "runtime_seconds": 0., "shared_flow_fits": [],
        "python": sys.version, "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pandas", "torch", "scikit-learn", "xgboost",
                          "scipy", "openml", "imbalanced-learn")},
        "selection_policy": f"{config['tuning_candidates']} common learner proposals; alpha cycles by candidate; lock canonical U2 learner/alpha before tau search",
        "learner_training": "fixed native epochs; validation=None; unit synthetic weights",
        "split_policy": "repeated stratified holdouts; smoke re-splits only outer training"}
    records, selections, splits = [], [], []

    def flush():
        manifest["runtime_seconds"] = perf_counter() - started
        manifest["source_hashes"] = _source_hashes(args.config)
        _dump(run_dir / "manifest.json", manifest)
        _dump(run_dir / "records.json", records)
        _dump(run_dir / "selections.json", selections)
        _dump(run_dir / "splits.json", splits)
        pd.DataFrame(records).to_csv(run_dir / "metrics.csv", index=False, float_format="%.17g")

    flush()
    print(f"Run directory: {run_dir}", flush=True)
    current = {}
    try:
        torch.set_num_threads(config["threads"])
        with threadpool_limits(limits=config["threads"]):
            for name in config["datasets"]:
                current = {"dataset": name, "phase": "dataset_load"}
                loaded = load_climb_datasets(dataset_names=[name], data_home=config.get("data_home", ".cache/climb"))
                if len(loaded) != 1 or loaded[0]["name"] != name:
                    raise ValueError("Raw CLIMB loader did not return exactly the requested dataset")
                dataset = loaded[0]
                manifest["datasets"].append({"name": name, **dataset["metadata"]})
                for seed in config["seeds"]:
                    current = {"dataset": name, "seed": seed, "phase": "partitions_and_flow"}
                    split = _split(dataset, seed, config, args.smoke)
                    partitions = prepare_partitions(dataset, split)
                    flow_options = {"flow_epochs": config["flow_epochs"]}
                    inner_gate = flow_for_partition(dataset, split, partitions, "inner", seed, flow_options, config["device"])
                    outer_gate = flow_for_partition(dataset, split, partitions, "outer", seed, flow_options, config["device"])
                    evidence = _flow_evidence(inner_gate, outer_gate, dataset, split)
                    partitions["flow_tensor_sha256"] = _core_hash(inner_gate.core)
                    row_ids = np.asarray(dataset["row_ids"])
                    split_record = {"dataset": name, "seed": seed,
                        "split_fingerprint": split_fingerprint(dataset, split),
                        # Arrays are source row identities, not positions in preprocessed frames.
                        **{key: row_ids[split[key]].tolist() for key in
                           ("train_indices", "fit_indices", "validation_indices", "test_indices")},
                        "row_ids": {role: row_ids[split[role + "_indices"]].tolist()
                                    for role in ("train", "fit", "validation", "test")},
                        "flow_fit_row_ids": evidence["fit_row_ids"],
                        "flow_calibration_row_ids": evidence["calibration_row_ids"]}
                    splits.append(split_record)
                    manifest["shared_flow_fits"].append({"dataset": name, "seed": seed,
                        "fit_seconds": evidence["fit_seconds"], "core_sha256": evidence["core_sha256"],
                        "fit_count": 1})
                    flush()
                    for algorithm_index, algorithm in enumerate(config["classifiers"]):
                        candidates = candidate_parameters(algorithm, seed, config["tuning_candidates"])
                        if args.smoke and algorithm in ("random_forest", "xgboost"):
                            candidates = [{**parameters, "n_estimators": 12} for parameters in candidates]
                        for method in METHODS:
                            current = {"dataset": name, "algorithm": algorithm, "seed": seed, "method": method}
                            shared = evidence["fit_seconds"] if algorithm_index == 0 and method == "u2_flow_gate" else 0.
                            record, selection = _cell(dataset, split, partitions, algorithm, seed, method,
                                config, candidates, run_dir, (inner_gate, outer_gate), evidence, shared)
                            records.append(record)
                            selections.append(selection)
                            flush()
                            print(f"Completed {name} / {algorithm} / seed {seed} / {method}: "
                                  f"minority F1={record['minority_f1']:.6f}, "
                                  f"accepted={record['accepted_count']}/{record['requested_count']}, "
                                  f"fits={record['classifier_fit_count']}", flush=True)
        expected = len(config["datasets"]) * len(config["classifiers"]) * len(config["seeds"]) * len(METHODS)
        if len(records) != expected:
            raise RuntimeError(f"Incomplete run: {len(records)} completed cells, expected {expected}")
        manifest["status"] = "completed"
        flush()
        from benchmarks.climb_flow_f1_report import build_report
        build_report(run_dir)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failure"] = {**current, "exception_type": type(error).__name__,
                               "message": str(error), "traceback": traceback.format_exc()}
        flush()
        print(manifest["failure"]["traceback"], file=sys.stderr, flush=True)
        raise
    return run_dir


if __name__ == "__main__":
    main()
