"""Shoppers: synthetic-only weights in native iterative U2, F1 and random pools."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import torch

from benchmarks.flow_partitions import flow_for_partition
from benchmarks.pool_random_f1 import frozen_flow, prepare_role, save_pool, sha256
from benchmarks.protocols import load_dataset, load_protocol, make_splits
from benchmarks.results import BenchmarkArtifacts, enforce_stage_gate, load_study
from benchmarks.run import candidate_configs, prepare_partitions, smoke_split, split_fingerprint
from benchmarks.validation_guided import numeric_metrics, save_predictions
from classifiers.factory import fit_estimator, make_estimator
from datasets.scripts.climb import categorical_columns
from experiments import data_fingerprint, seed_everything, write_json
from samplers.flow_gmm import _checksum
from samplers.learning_benefit import model_state_hash
from samplers.pool_random import RandomPoolSelector
from samplers.protected import protected_ugo

ALGORITHMS = ("logistic_regression", "random_forest", "lightgbm", "mlp")
FAMILIES = ("U2_F1", "U2", "pool_random")
WEIGHTS = (.25, .5)
SOURCE = "results/runs/20261009T153348-af844bdecf5a"


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def native_estimator(algorithm, seed, selected, settings):
    return make_estimator(algorithm, seed, parameters=selected["inner"]["parameters"],
                          device="auto" if algorithm == "mlp" else "cpu",
                          mlp_epochs=settings["mlp_epochs"])


def run_trajectory(dataset, split, partitions, role, seed, algorithm, selected, settings,
                   protocol, folder, prepared, metadata, family, weight, tau=None):
    folder.mkdir(parents=True, exist_ok=True)
    X, y, V, groups, targets = prepared
    estimator = native_estimator(algorithm, seed, selected, settings)
    fit_count, fit_seconds = 0, 0.
    latest = {}
    fit_folder = folder / "fits"
    fit_folder.mkdir()
    tick = perf_counter()

    def fit(px, py):
        nonlocal fit_count, fit_seconds
        features, labels = array(px), array(py)
        np.testing.assert_array_equal(features[:len(y)], X)
        np.testing.assert_array_equal(labels[:len(y)], y)
        weights = np.ones(len(labels), dtype=np.float64)
        weights[len(y):] = weight
        start = perf_counter()
        model = fit_estimator(estimator, features, labels,
                              sample_weight=None if weight == 1 or len(labels) == len(y) else weights)
        elapsed = perf_counter() - start
        fit_seconds += elapsed
        fit_count += 1
        stem = f"fit-{fit_count:02d}"
        np.savez_compressed(fit_folder / f"{stem}.npz", X=features, y=labels,
            sample_weight=weights, real_probabilities=model.predict_proba(X),
            validation_probabilities=model.predict_proba(V))
        write_json(fit_folder / f"{stem}.json", {"fit_number": fit_count,
            "training_count": len(labels), "real_count": len(y), "synthetic_count": len(labels)-len(y),
            "synthetic_weight": weight, "real_weight": 1., "added_weight": weight*(len(labels)-len(y)),
            "parameters": model.get_params(deep=False), "state_hash": model_state_hash(model),
            "native_fit_seconds": elapsed, "actual_epochs": getattr(model, "best_epoch_", None),
            "device": str(next(model.model_.parameters()).device) if algorithm == "mlp" else "cpu",
            "architecture": str(model.model_) if algorithm == "mlp" else type(model).__name__,
            "MLP_loss": "sum(w*CE)/sum(w)" if algorithm == "mlp" else None,
            "validation_used_in_fit": False, "cold_refit": True})
        latest.update(model=model, count=len(labels), X=features, y=labels, weights=weights)
        return model

    gate = flow_for_partition(dataset, split, partitions, role, seed, {}, "auto") if family == "U2_F1" else None
    before = _checksum(gate.core.model) if gate is not None else None
    selector = RandomPoolSelector(seed, k=5) if family == "pool_random" else None
    parameters = {**selected["inner"]["sampler_parameters"], "capture_proposals": True}
    if gate is not None:
        parameters["f1_tau"] = tau
    result = None
    if family == "real_only":
        model = fit(X, y)
    else:
        result = protected_ugo(X, y, seed, categorical_columns(partitions[role]),
            settings["ugo_max_iterations"], "cpu", fit, targets, parameters, groups,
            flow_support=gate, candidate_selector=selector)
        # Retained F1 already skips an identical final fit; plain U2 keeps its final refit.
        model = latest["model"] if gate is not None and latest.get("count") == len(result.y) else fit(result.X, result.y)
    elapsed = perf_counter() - tick
    after = _checksum(gate.core.model) if gate is not None else None
    if before != after:
        raise RuntimeError("F1 scoring changed its frozen model")
    features, labels, weights = latest["X"], latest["y"], latest["weights"]
    joblib.dump(model, folder / "model.joblib")
    np.savez_compressed(folder / "training.npz", X=features, y=labels, sample_weight=weights)
    validation = save_predictions(folder, "validation", model, V,
        dataset["y"][split["validation_indices"]], dataset["row_ids"][split["validation_indices"]], protocol)
    metrics = validation
    if role == "outer":
        metrics = save_predictions(folder, "test", model, partitions["X_test"],
            dataset["y"][split["test_indices"]], dataset["row_ids"][split["test_indices"]], protocol)
    generated = len(labels)-len(y)
    audit = {"role": role, "family": family, "synthetic_weight": weight, "real_weight": 1.,
        "real_count": len(y), "accepted_count": generated, "added_weight": float(weights[len(y):].sum()),
        "total_weight": float(weights.sum()), "native_parameters": estimator.get_params(deep=False),
        "sampler_parameters": parameters, "tau": tau if gate is not None else None,
        "epochs": settings["mlp_epochs"] if algorithm == "mlp" else None,
        "rounds": [] if result is None else result.rounds,
        "provenance": [] if result is None else result.provenance,
        "support_summary": {} if result is None else result.support_summary,
        "stop_reason": "real_only" if result is None else result.stop_reason,
        "selector": None if selector is None else selector.summary(),
        "flow_state_before": before, "flow_state_after": after,
        "flow_inference_seconds": 0. if gate is None else gate.inference_seconds,
        "flow_inference_rows": 0 if gate is None else gate.inference_rows,
        "native_fit_count": fit_count, "native_fit_seconds": fit_seconds,
        "runtime_seconds": elapsed, "bagging_experiment": False,
        "final_fit_reused": gate is not None and fit_count == len(result.rounds),
        "no_refill": True, "no_utility_or_surrogate": True, "validation_used_in_fit": False}
    write_json(folder / "audit.json", audit)
    write_json(folder / "metrics.json", {"validation": validation, **({"test": metrics} if role == "outer" else {})})
    if selector is not None:
        for number, pool in enumerate(selector.rounds, 1):
            save_pool(folder / "pools", number, pool)
        write_json(folder / "checkpoints.json", selector.checkpoints)
    return {**metadata, "variant_id": f"{role}_{family}_w{weight:g}", "display_variant": family,
        "family": family, "synthetic_weight": weight, "role": role,
        "phase": "real_only" if result is None else "iterative", "status": "completed",
        "method": "baseline" if result is None else "ugo_f1" if gate is not None else "ugo_groups_only",
        "metrics": metrics, "validation_metrics": validation,
        "evaluation_role": "outer_test" if role == "outer" else "inner_validation",
        "predictions_path": str((folder / ("test_predictions.csv" if role == "outer" else "validation_predictions.csv")).resolve()),
        "trajectory_path": str(folder.resolve()),
        "augmentation": {"requested_count": 0 if result is None else settings["augmentation"]["total"],
            "actual_count": generated, "underspend": 0 if result is None else settings["augmentation"]["total"]-generated,
            "added_weight": audit["added_weight"]},
        "config": {"selected_parameters": selected["inner"]["parameters"],
            "sampler_parameters": parameters, "synthetic_weight": weight, "f1_tau": tau,
            "threshold": .5, "epochs": audit["epochs"]},
        **{key: audit[key] for key in ("runtime_seconds", "native_fit_count", "native_fit_seconds", "flow_inference_seconds")}}


def compare_trajectories(unit, weighted):
    a = json.loads((Path(unit["trajectory_path"]) / "audit.json").read_text())
    b = json.loads((Path(weighted["trajectory_path"]) / "audit.json").read_text())
    ca, cb = a["support_summary"].get("candidate_log", []), b["support_summary"].get("candidate_log", [])
    initial_a = [r for r in ca if r["round"] == 1]
    initial_b = [r for r in cb if r["round"] == 1]
    fields = ("seed_index", "partner_index", "group_id", "original_coordinates", "final_coordinates", "accepted")
    strip = lambda rows: [{key: row[key] for key in fields} for row in rows]
    identical_initial = strip(initial_a) == strip(initial_b)
    if not identical_initial:
        raise RuntimeError("Synthetic weight changed the initial real-only anchor/proposal/acceptance stream")
    if a["family"] == "pool_random":
        paths = [Path(r["trajectory_path"]) / "pools" / "round-01-pool.npz" for r in (unit,weighted)]
        if paths[0].exists() != paths[1].exists():
            raise RuntimeError("Weight changed initial pool availability")
        if paths[0].exists():
            with np.load(paths[0],allow_pickle=True) as pa, np.load(paths[1],allow_pickle=True) as pb:
                for field in ("original_coordinates","final_coordinates","anchor_indices","partner_indices",
                              "labels","group_ids","proposal_ids","attempted_selected","selected"):
                    np.testing.assert_array_equal(pa[field],pb[field])
    return {"weight": weighted["synthetic_weight"], "initial_selected_candidates_identical": True,
        "initial_pool_identity_checked": True,
        "later_candidate_trajectory_diverged": strip(ca) != strip(cb),
        "unit_accepted": unit["augmentation"]["actual_count"],
        "weighted_accepted": weighted["augmentation"]["actual_count"],
        "unit_rounds": len(a["rounds"]), "weighted_rounds": len(b["rounds"])}


def parent_control(dataset, split, partitions, seed, algorithm, selected, settings, protocol,
                   prepared, source, folder):
    folder.mkdir(parents=True, exist_ok=True)
    X, y, V, _, _ = prepared
    audit = json.loads((Path(source["trajectory_path"]) / "audit.json").read_text())
    anchors = np.asarray([p["seed_index"] for p in audit["provenance"]], dtype=int)
    features = np.concatenate((X, X[anchors]))
    labels = np.concatenate((y, y[anchors]))
    weights = np.ones(len(labels), dtype=np.float64)
    weights[len(y):] = source["synthetic_weight"]
    start = perf_counter()
    model = fit_estimator(native_estimator(algorithm, seed, selected, settings), features, labels,
                          sample_weight=None if not len(anchors) else weights)
    seconds = perf_counter()-start
    joblib.dump(model, folder / "model.joblib")
    np.savez_compressed(folder / "training.npz", X=features, y=labels, sample_weight=weights, anchor_indices=anchors)
    validation = save_predictions(folder, "validation", model, V, dataset["y"][split["validation_indices"]],
                                  dataset["row_ids"][split["validation_indices"]], protocol)
    metrics = save_predictions(folder, "test", model, partitions["X_test"], dataset["y"][split["test_indices"]],
                              dataset["row_ids"][split["test_indices"]], protocol)
    write_json(folder / "audit.json", {"source_trajectory": source["trajectory_path"],
        "family": source["family"], "real_count": len(y), "accepted_count": len(anchors),
        "synthetic_weight": source["synthetic_weight"], "added_weight": float(weights[len(y):].sum()),
        "native_parameters": model.get_params(deep=False), "actual_epochs": getattr(model, "best_epoch_", None),
        "device": str(next(model.model_.parameters()).device) if algorithm == "mlp" else "cpu",
        "architecture": str(model.model_) if algorithm == "mlp" else type(model).__name__,
        "native_fit_count": 1, "native_fit_seconds": seconds, "parent_only": True})
    return {**source, "variant_id": f"outer_{source['family']}_parent", "phase": "parent_duplication",
        "method": "matched_parent_duplication", "metrics": metrics, "validation_metrics": validation,
        "trajectory_path": str(folder.resolve()), "predictions_path": str((folder / "test_predictions.csv").resolve()),
        "native_fit_count": 1, "native_fit_seconds": seconds, "runtime_seconds": seconds,
        "flow_inference_seconds": 0., "source_trajectory": source["trajectory_path"]}


def export_results(path, rows):
    flat = []
    for row in rows:
        values = {key: row.get(key) for key in ("algorithm", "seed", "variant_id", "family", "role", "phase",
            "synthetic_weight", "runtime_seconds", "native_fit_count", "native_fit_seconds")}
        values.update(accepted=row["augmentation"]["actual_count"], added_weight=row["augmentation"]["added_weight"],
                      **numeric_metrics(row["metrics"]))
        flat.append(values)
    pd.DataFrame(flat).to_csv(path / "all_metrics.csv", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--protocol", default="configs/benchmarks/shoppers_controlled.yaml")
    parser.add_argument("--study", default="configs/ablations/stage1_downweight.yaml")
    parser.add_argument("--controls", default="configs/ablations/tabdiff_controls.json")
    parser.add_argument("--data-home", default=".cache/benchmarks")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    protocol, study = load_protocol(args.protocol), load_study(args.study)
    if protocol["dataset"] != "shoppers" or protocol["split"] != {"type":"stratified_holdout", "test_fraction":.2,"outer_seed":2026}:
        raise ValueError("Keep the active Shoppers data access and outer split")
    if args.run and args.seeds != list(range(5)):
        raise ValueError("Scientific comparison requires all five matched seeds")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("Seeds must be unique nonnegative integers")
    if tuple(study["classifiers"]) != ALGORITHMS or tuple(study["weights"]) != WEIGHTS:
        raise ValueError("Preserve the declared four classifiers and weighting grid")
    experiment_type = "setup_smoke" if args.smoke else "development"
    enforce_stage_gate(study, experiment_type)
    settings = {"mlp_epochs": study["mlp"]["smoke_epochs" if args.smoke else "epochs"],
        "ugo_max_iterations": 2 if args.smoke else 20,
        "augmentation": {"mode":"matched", "total":40 if args.smoke else 1000}}
    audit = json.loads(Path(args.controls).read_text())
    controls = {(c["seed"],c["algorithm"]): c["control"] for c in audit["cells"]}
    arguments = {**vars(args), "experiment_type": experiment_type, "effective_classifiers": ALGORITHMS,
        "effective_seeds": args.seeds, "effective_settings": settings, "weights": WEIGHTS,
        "control_weight":1., "bagging_deferred":True, "no_automatic_retention":True,
        "selection":"inner_validation_AP_tie_larger_weight", "bootstrap_draws":2000,"bootstrap_seed":20261012}
    artifacts = BenchmarkArtifacts(args.output_dir or ("results/smoke" if args.smoke else "results/runs"), arguments, protocol, study)
    rows = []
    try:
        dataset = load_dataset(protocol,args.data_home)
        artifacts.dataset(dataset)
        write_json(artifacts.path / "control_audit.json",audit)
        fingerprint = data_fingerprint(dataset["X"],dataset["y"],dataset["row_ids"])
        if not args.smoke:
            source_manifest = json.loads((Path(SOURCE) / "manifest.json").read_text())
            if source_manifest["status"] != "completed" or source_manifest["datasets"]["shoppers"]["fingerprint"] != fingerprint:
                raise ValueError("Frozen source must match the complete real dataset")
        for seed in args.seeds:
            seed_everything(seed)
            canonical = make_splits(dataset,protocol,seed)[0]
            split = smoke_split(dataset,canonical,seed,400,120,.2) if args.smoke else canonical
            partitions = prepare_partitions(dataset,split)
            root = artifacts.fold_path("shoppers",split["fold"]) / f"seed-{seed}"
            root.mkdir(parents=True,exist_ok=True)
            write_json(root / "split.json",{**split,"canonical_test_indices":canonical["test_indices"]})
            frozen_flow(dataset,split,partitions,root,seed,None if args.smoke else Path(SOURCE))
            for algorithm in ALGORITHMS:
                folder = root / algorithm
                folder.mkdir()
                metadata = {"algorithm":algorithm,"seed":seed,"dataset":"shoppers","fold":split["fold"],
                    "experiment_type":experiment_type,"dataset_fingerprint":fingerprint,
                    "split_fingerprint":split_fingerprint(dataset,split),"budget_mode":"matched",
                    "tuning_budget":1 if args.smoke else 10}
                if algorithm != "mlp":
                    control = controls[(seed,algorithm)]
                    for key, hash_key in (("source_selection","source_selection_sha256"),
                                          ("f1_checkpoint","f1_checkpoint_sha256")):
                        if sha256(control[key]) != control[hash_key]:
                            raise ValueError("Retained control artifact changed: "+key)
                    selected = json.loads(Path(control["source_selection"]).read_text())
                    tau = control["f1_tau"]
                    inner = prepare_role(dataset,split,partitions,"inner",seed,selected,settings,folder / "inner")
                else:
                    tuning = []
                    for index, params in enumerate(candidate_configs(protocol,algorithm,seed,1 if args.smoke else 10)):
                        candidate = {"inner":{"parameters":params,"sampler_parameters":{
                            "grouping":"kmeans","allocation":"stability","group_alpha":(0.,.5,1.)[index%3]}}}
                        prepared = prepare_role(dataset,split,partitions,"inner",seed,candidate,settings,
                                                folder / "canonical" / f"candidate-{index}" / "real")
                        trial = run_trajectory(dataset,split,partitions,"inner",seed,algorithm,candidate,settings,
                            protocol,folder / "canonical" / f"candidate-{index}" / "trajectory",prepared,metadata,"U2",1.)
                        tuning.append({"candidate":index,"selected":candidate,"AP":trial["metrics"]["average_precision"],
                                       "trajectory_path":trial["trajectory_path"]})
                    best = max(tuning,key=lambda row:row["AP"])
                    selected = best["selected"]
                    write_json(folder / "canonical" / "selection.json",{"trials":tuning,"selected_candidate":best["candidate"],
                        "objective":"inner_validation_AP","fixed_epochs":settings["mlp_epochs"],"no_test_selection":True})
                    inner = prepare_role(dataset,split,partitions,"inner",seed,selected,settings,folder / "inner")
                    tau_trials = []
                    for threshold in study["F1"]["new_MLP_threshold_grid"]:
                        trial = run_trajectory(dataset,split,partitions,"inner",seed,algorithm,selected,settings,
                            protocol,folder / "tau" / f"tau-{threshold:g}",inner,metadata,"U2_F1",1.,threshold)
                        tau_trials.append({"tau":threshold,"AP":trial["metrics"]["average_precision"],"row":trial})
                    tau_best = max(tau_trials,key=lambda row:row["AP"])
                    tau = tau_best["tau"]
                    write_json(folder / "tau" / "selection.json",{"trials":[{k:v for k,v in r.items() if k!="row"} for r in tau_trials],
                        "selected_tau":tau,"no_test_selection":True,"unit_weight_only":True})
                    control = {"parameters":selected["inner"]["parameters"],"group_alpha":selected["inner"]["sampler_parameters"]["group_alpha"],
                               "f1_tau":tau,"source":"new_native_MLP_inner_only_search","fixed_epochs":settings["mlp_epochs"]}
                write_json(folder / "control.json",{**control,"selected":selected["inner"],
                    "native_parameters_locked":True,"F1_tau_locked_across_weights":True})
                real_inner = run_trajectory(dataset,split,partitions,"inner",seed,algorithm,selected,settings,
                    protocol,folder / "inner" / "real_only",inner,metadata,"real_only",0.)
                real_inner["variant_id"] = "inner_real_only"
                rows.append(real_inner)
                selections, comparisons = {}, {"inner":{},"outer":{}}
                for family in FAMILIES:
                    trials = []
                    for weight in (1.,*WEIGHTS):
                        if algorithm == "mlp" and family == "U2_F1" and weight == 1:
                            row = tau_best["row"]
                        else:
                            row = run_trajectory(dataset,split,partitions,"inner",seed,algorithm,selected,settings,
                                protocol,folder / "inner" / family / f"weight-{weight:g}",inner,metadata,family,weight,
                                tau if family == "U2_F1" else None)
                        rows.append(row)
                        trials.append(row)
                    unit = trials[0]
                    comparisons["inner"][family] = [compare_trajectories(unit,r) for r in trials[1:]]
                    winner = max(trials[1:],key=lambda row:(row["metrics"]["average_precision"],row["synthetic_weight"]))
                    selections[family] = {"selected_weight":winner["synthetic_weight"],"selected_inner_path":winner["trajectory_path"],
                        "trials":[{"weight":r["synthetic_weight"],"AP":r["metrics"]["average_precision"],"path":r["trajectory_path"]} for r in trials],
                        "objective":"inner_validation_AP","tie_break":"larger_weight","no_test_selection":True}
                write_json(folder / "selection.json",{"families":selections,"no_test_selection":True,
                    "all_inner_selections_before_outer":True,"canonical_control":control})
                outer = prepare_role(dataset,split,partitions,"outer",seed,selected,settings,folder / "outer")
                real_outer = run_trajectory(dataset,split,partitions,"outer",seed,algorithm,selected,settings,
                    protocol,folder / "outer" / "real_only",outer,metadata,"real_only",0.)
                real_outer["variant_id"] = "outer_real_only"
                rows.append(real_outer)
                for family in FAMILIES:
                    pair = []
                    for suffix,weight in (("unit",1.),("selected",selections[family]["selected_weight"])):
                        row = run_trajectory(dataset,split,partitions,"outer",seed,algorithm,selected,settings,
                            protocol,folder / "outer" / family / f"weight-{weight:g}",outer,metadata,family,weight,
                            tau if family == "U2_F1" else None)
                        row["variant_id"] = f"outer_{family}_{suffix}"
                        pair.append(row)
                        rows.append(row)
                    comparisons["outer"][family] = compare_trajectories(*pair)
                    rows.append(parent_control(dataset,split,partitions,seed,algorithm,selected,settings,protocol,outer,
                                               pair[1],folder / "outer" / family / "parent"))
                write_json(folder / "comparison.json",comparisons)
                export_results(artifacts.path,rows)
                print(f"seed={seed} classifier={algorithm}: complete; weights="+
                      str({f:s["selected_weight"] for f,s in selections.items()}),flush=True)
        artifacts.finish(rows)
    except Exception as error:
        artifacts.finish(rows,error=str(error),status="failed")
        raise
    print(f"Artifacts: {artifacts.path}",flush=True)
    return artifacts.path


if __name__ == "__main__":
    main()
