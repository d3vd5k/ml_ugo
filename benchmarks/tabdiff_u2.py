"""Replay verified Shoppers U2/TabDiff proposal pools with F1 disabled."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from benchmarks.pool_random_f1 import sha256
from benchmarks.protocols import load_dataset, load_protocol
from benchmarks.results import BenchmarkArtifacts, enforce_stage_gate, load_study
from benchmarks.tabdiff_experiment import ARMS, export_tables, import_context
from benchmarks.validation_guided import ALGORITHMS, prediction_metrics, save_predictions
from classifiers.factory import fit_estimator, make_estimator
from experiments import data_fingerprint, write_json
from samplers.tabdiff_control import match_group_counts


SOURCE_RUN = "results/runs/20261011T154542-1b40c4b555c4"
SOURCE_SMOKE = "results/smoke/20261011T154408-cb42834863f1"
POOL_FIELDS = ("coordinates", "labels", "anchor_indices", "anchor_row_ids", "group_ids",
               "proposal_ids", "schema_valid", "count_changed", "schema_reasons",
               "exact_real_duplicate", "exact_pool_duplicate", "nearest_real_index",
               "nearest_real_distance", "near_duplicate")


def bind(paths):
    return {str(Path(p).resolve()): sha256(p) for p in paths}


def replay_cell(dataset, split, source, folder, algorithm, seed, role, protocol, metadata):
    folder.mkdir(parents=True, exist_ok=True)
    parent_seed = source.parents[1]
    control = json.loads((source.parent / "control.json").read_text())
    plan = json.loads((source / "schedule.json").read_text())
    with np.load(source / "real.npz", allow_pickle=True) as data:
        X, y = data["X"], data["y"]
        real_indices, row_ids = data["real_indices"], data["row_ids"]
    preprocessor = joblib.load(source / "preprocessor.joblib")
    evaluation_indices = np.asarray(split["validation_indices" if role == "inner" else "test_indices"])
    evaluation_X = preprocessor.transform(dataset["X"].iloc[evaluation_indices]).astype(np.float32)
    paths = [source / name for name in ("real.npz", "real_raw.joblib", "preprocessor.joblib",
        "preprocessing.json", "groups.json", "role.json", "schedule.npz", "schedule.json")]
    paths += [source.parent / "control.json", parent_seed / "split.json", parent_seed / "masks.json",
              parent_seed / role / "generator" / "generator.pt",
              parent_seed / role / "generator" / "training_audit.json"]
    start = perf_counter()
    pools, audits = {}, {}
    for arm in ARMS:
        original = source / arm
        paths += [original / name for name in ("proposals.npz", "generated_raw.joblib", "decoded.joblib", "generation.json", "audit.json")]
        with np.load(original / "proposals.npz", allow_pickle=True) as data:
            pools[arm] = {key: data[key] for key in POOL_FIELDS}
        pools[arm]["eligible"] = pools[arm]["schema_valid"].copy()
        audits[arm] = json.loads((original / "audit.json").read_text())
    load_seconds = perf_counter() - start
    paths_sha = bind(paths)
    write_json(folder / "source.json", {"parent_cell": str(source.resolve()),
        "parent_seed": str(parent_seed.resolve()), "paths_sha256": paths_sha,
        "source_pool_complete_before_F1": True, "F1_enabled": False,
        "real_count": len(y), "real_indices": real_indices, "row_ids": row_ids,
        "evaluation_indices": evaluation_indices, "schema_only_eligibility": True})
    group_ids = pools["A"]["group_ids"]
    start = perf_counter()
    selected, quotas = match_group_counts({arm:pools[arm]["eligible"] for arm in ARMS},
        group_ids, plan["allocation_record"]["group_trace"], seed)
    matching_seconds = perf_counter() - start
    write_json(folder / "matched_quotas.json", {"groups": quotas,
        "matching_seed": seed + 23021, "F1_enabled": False,
        "policy": "per_group_minimum_then_uniform_arm_specific_thinning",
        "no_refill_no_other_generator_no_relaxation": True})
    estimator = make_estimator(algorithm, seed, parameters=control["parameters"], device="cpu")
    rows = []
    for arm in ARMS:
        destination = folder / arm
        destination.mkdir(parents=True, exist_ok=True)
        pool, old = pools[arm], audits[arm]
        pool["selected"] = selected[arm]
        np.savez_compressed(destination / "proposals.npz", **pool)
        augmented_X = np.concatenate([X, pool["coordinates"][selected[arm]]])
        augmented_y = np.concatenate([y, pool["labels"][selected[arm]]])
        start = perf_counter()
        native = fit_estimator(estimator, augmented_X, augmented_y)
        fit_seconds = perf_counter() - start
        joblib.dump(native, destination / "model.joblib")
        np.savez_compressed(destination / "training.npz", X=augmented_X, y=augmented_y,
                            sample_weight=np.ones(len(augmented_y)))
        metrics = save_predictions(destination, "validation" if role == "inner" else "test", native,
            evaluation_X, dataset["y"][evaluation_indices], dataset["row_ids"][evaluation_indices], protocol)
        count = int(selected[arm].sum())
        checkpoint = old["generator_checkpoint"]
        audit = {"F1_enabled": False, "F1_scoring_calls": 0, "relocation": False,
            "utility_ranking": False, "TAP_policy": False, "schema": old["schema"],
            "schema_failure_counts": old["schema_failure_counts"],
            "count_decode_changed_cells": old["count_decode_changed_cells"],
            "proposal_count": len(group_ids), "schema_valid_count": int(pool["schema_valid"].sum()),
            "eligible_count": int(pool["eligible"].sum()), "accepted_count": count,
            "quota_shortfall": len(group_ids)-count, "native_parameters": control["parameters"],
            "native_fit_seconds": fit_seconds, "native_fit_count": 1,
            "sampling_seconds": 0., "filtering_seconds": 0., "diagnostic_seconds": 0.,
            "generator_training_seconds": 0.,
            "source_sampling_seconds": old["sampling_seconds"],
            "source_schema_filtering_includes_F1_not_new_cost": old["filtering_seconds"],
            "exact_real_duplicates": old["exact_real_duplicates"], "exact_pool_duplicates": old["exact_pool_duplicates"],
            "near_duplicates": old["near_duplicates"], "masked_columns": old["masked_columns"],
            "mask_fraction": old["mask_fraction"], "synthetic_label": old["synthetic_label"],
            "unmasked_exactly_preserved": old["unmasked_exactly_preserved"],
            "count_rounding_common_user_approved": True,
            "duplicate_policy": "diagnostics_only_preserve_original_no_new_dedup_gate",
            "generator_checkpoint": {k:checkpoint[k] for k in ("checkpoint_path", "checkpoint_sha256", "model_checksum", "config", "training_seconds")},
            "generator_retrained": False, "pool_reused": True,
            "source_artifacts": str((source / arm).resolve())}
        write_json(destination / "audit.json", audit)
        write_json(destination / "metrics.json", metrics)
        rows.append({**metadata, "variant_id":role+"_"+arm,"arm":arm,"role":role,
            "phase":"fixed_round","method":"ugo_schema_control" if arm=="A" else "tabdiff_inpaint",
            "status":"completed","metrics":metrics,
            "evaluation_role":"inner_validation" if role=="inner" else "outer_test",
            "predictions_path":str((destination / ("validation_predictions.csv" if role=="inner" else "test_predictions.csv")).resolve()),
            "trajectory_path":str(destination.resolve()),
            "augmentation":{"mode":"matched","requested_count":len(group_ids),"actual_count":count,"underspend":len(group_ids)-count,"nominal_final_target":1000},
            "config":{"native_parameters":control["parameters"],"group_alpha":control["group_alpha"],"F1_enabled":False,"relocation":False,
                      "mask_columns":old["masked_columns"],"mask_fraction":old["mask_fraction"],"threshold":.5,"count_decoder":"common_np_rint_ties_even"},
            "sampling_seconds":0.,"filtering_seconds":0.,"native_fit_seconds":fit_seconds,
            "runtime_seconds":fit_seconds,"native_fit_count":1,
            "shared_generator_checkpoint":checkpoint["checkpoint_path"],"generator_checkpoint_sha256":checkpoint["checkpoint_sha256"]})
    return rows, {"seed":seed,"algorithm":algorithm,"role":role,"pool_load_seconds":load_seconds,
                  "matching_seconds":matching_seconds,"generator_training_seconds":0.,"generator_retrained":False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke",action="store_true");mode.add_argument("--run",action="store_true")
    parser.add_argument("--source");parser.add_argument("--seeds",nargs="+",type=int,default=list(range(5)))
    parser.add_argument("--output-dir");parser.add_argument("--data-home",default=".cache/benchmarks")
    args=parser.parse_args(argv)
    if args.run and args.seeds != list(range(5)): raise ValueError("Use all five scientific matched seeds")
    source=Path(args.source or (SOURCE_SMOKE if args.smoke else SOURCE_RUN)).resolve()
    manifest=json.loads((source/"manifest.json").read_text())
    proof=json.loads((source/"independent_verification.json").read_text())
    if manifest["status"]!="completed" or proof["status"]!="verified" or bool(manifest["arguments"]["smoke"])!=args.smoke:
        raise ValueError("Require a completed verified source of the same smoke/scientific role")
    for path,digest in proof["actual_file_sha256"].items():
        if sha256(path)!=digest: raise ValueError("Verified source changed: "+path)
    protocol=load_protocol("configs/benchmarks/shoppers_controlled.yaml")
    study=load_study("configs/ablations/stage1_tabdiff_u2.yaml")
    experiment_type="setup_smoke" if args.smoke else "development"
    enforce_stage_gate(study,experiment_type)
    arguments={**vars(args),"source_root":str(source),"experiment_type":experiment_type,
        "effective_classifiers":ALGORITHMS,"effective_seeds":args.seeds,"F1_enabled":False,
        "effective_settings":{"augmentation_rounds":1,"generator_retrained":False},
        "source_verification_sha256":sha256(source/"independent_verification.json"),
        "mask_selection":"new_inner_validation_AP_only_B_tiebreak","bootstrap_draws":2000,"bootstrap_seed":20261011}
    artifacts=BenchmarkArtifacts(args.output_dir or ("results/smoke" if args.smoke else "results/runs"),arguments,protocol,study)
    dataset=load_dataset(protocol,args.data_home);artifacts.dataset(dataset)
    fingerprint=data_fingerprint(dataset["X"],dataset["y"],dataset["row_ids"])
    audit=json.loads((source/"control_audit.json").read_text())
    write_json(artifacts.path/"control_audit.json",audit)
    old_records=json.loads((source/"benchmark_records.json").read_text())
    rows,resources=[],[];start=perf_counter()
    try:
        for seed in args.seeds:
            parent=source/"shoppers"/"fold-0"/f"seed-{seed}"
            split=json.loads((parent/"split.json").read_text())
            root=artifacts.fold_path("shoppers",0)/f"seed-{seed}";root.mkdir(parents=True,exist_ok=True)
            write_json(root/"split.json",split);write_json(root/"masks.json",json.loads((parent/"masks.json").read_text()))
            selections={}
            for role in ("inner","outer"):
                for algorithm in ALGORITHMS:
                    previous=next(r for r in old_records if r["seed"]==seed and r["algorithm"]==algorithm and r["variant_id"]==role+"_A")
                    metadata={key:previous[key] for key in ("dataset","dataset_fingerprint","split_fingerprint","seed","algorithm","budget_mode","tuning_budget","candidate_configs","fold")}
                    if metadata["dataset_fingerprint"]!=fingerprint: raise ValueError("Source dataset changed")
                    metadata["experiment_type"]=experiment_type
                    folder=root/algorithm/role
                    current,resource=replay_cell(dataset,split,parent/algorithm/role,folder,algorithm,seed,role,protocol,metadata)
                    rows.extend(current);resources.append(resource)
                    if role=="inner":
                        by_arm={r["arm"]:r for r in current}
                        choice=max(("B","C"),key=lambda a:(by_arm[a]["metrics"]["average_precision"],a=="B"))
                        selections[algorithm]={"selected_arm":choice,"inner_AP":{a:by_arm[a]["metrics"]["average_precision"] for a in ARMS},
                            "source":"new_no_F1_inner_validation_only","outer_test_seen":False}
                        write_json(root/algorithm/"mask_selection.json",selections[algorithm])
                    else:
                        chosen=next(r for r in current if r["arm"]==selections[algorithm]["selected_arm"])
                        rows.append({**chosen,"variant_id":"outer_selected_TabDiff","selected_arm":chosen["arm"],"additional_training":False,
                            "native_fit_count":0,"config":{**chosen["config"],"selection":selections[algorithm]}})
                        if not args.smoke: rows.extend(import_context(audit,metadata,seed,algorithm,protocol))
                        for prior in old_records:
                            if prior["seed"]==seed and prior["algorithm"]==algorithm and prior["variant_id"] in ("outer_A","outer_B","outer_C","outer_selected_TabDiff"):
                                suffix=prior["variant_id"][6:]
                                rows.append({**prior,"variant_id":"prior_f1_"+suffix,"phase":"prior_F1_context","native_fit_count":0,
                                    "imported_context":True,"source_root":str(source),"prior_F1_counts_not_matched":True,
                                    "metrics":prediction_metrics(pd.read_csv(prior["predictions_path"],float_precision="round_trip"),protocol)})
                    print(f"seed={seed} role={role} classifier={algorithm} noF1 matched={current[0]['augmentation']['actual_count']}",flush=True)
            export_tables(artifacts.path,rows)
        write_json(artifacts.path/"resources.json",resources)
        write_json(artifacts.path/"invocation_costs.json",{"wall_seconds":perf_counter()-start,"generator_fits":0,
            "F1_scoring_calls":0,"source_training_and_sampling_reused":True,"source_root":str(source),
            "source_invocation":json.loads((source/"invocation_costs.json").read_text()),"native_arm_fits":18*len(args.seeds)})
        artifacts.finish(rows)
    except Exception as error:
        artifacts.finish(rows,error=str(error),status="failed");raise
    print("Artifacts: "+str(artifacts.path),flush=True)
    return artifacts.path


if __name__=="__main__":main()
