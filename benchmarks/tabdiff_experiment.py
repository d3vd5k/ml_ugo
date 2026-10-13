"""Shoppers fixed-round, count-matched native U2/F1 A/B/C TabDiff experiment.

Mask selection is inner-validation-only. No iterative pipeline or utility policy
is changed. Historical U5 OOF rows are context, not the user's new U5=U2 control.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import torch

from benchmarks.flow_partitions import flow_for_partition
from benchmarks.pool_random_f1 import frozen_flow, sha256
from benchmarks.protocols import load_dataset, load_protocol, make_splits
from benchmarks.results import BenchmarkArtifacts, enforce_stage_gate, load_study
from benchmarks.run import augmentation_targets, prepare_partitions, smoke_split, split_fingerprint
from benchmarks.validation_guided import ALGORITHMS, numeric_metrics, prediction_metrics, save_predictions
from classifiers.factory import fit_estimator, make_estimator
from datasets.scripts.climb import categorical_columns
from experiments import _preprocessing_state, data_fingerprint, seed_everything, write_json
from samplers.flow_gmm import _checksum as flow_checksum
from samplers.protected import prepare_protected_groups
from samplers.tabdiff import TabDiffConfig, fit_tabdiff, _checksum as tabdiff_checksum
from samplers.tabdiff_control import (ShoppersSchema, feature_masks, fixed_u2_schedule,
                                     match_group_counts, original_raw_proposals)
from samplers.ugo import nearest_indices

ARMS = ("A", "B", "C")


def prepare_real(dataset, split, partitions, role, folder, parameters, target_counts, seed):
    folder.mkdir(parents=True, exist_ok=True)
    indices = split["fit_indices" if role == "inner" else "train_indices"]
    X = partitions["X_fit" if role == "inner" else "X_train"]
    y = dataset["y"][indices]
    real = dataset["X"].iloc[indices].reset_index(drop=True)
    groups = prepare_protected_groups(X, y, seed, target_counts, parameters)
    groups["real_row_ids"] = dataset["row_ids"][indices]
    joblib.dump(partitions[role], folder / "preprocessor.joblib")
    joblib.dump(real, folder / "real_raw.joblib")
    np.savez_compressed(folder / "real.npz", X=X, y=y, real_indices=indices,
        row_ids=dataset["row_ids"][indices], test_indices=split["test_indices"],
        validation_indices=split["validation_indices"] if role == "inner" else np.empty(0, dtype=int),
        former_inner_validation_indices=split["validation_indices"])
    write_json(folder / "preprocessing.json", _preprocessing_state(partitions[role]))
    write_json(folder / "groups.json", groups)
    write_json(folder / "role.json", {"role": role, "real_count": len(y),
        "active_evaluation_role": "inner_validation" if role == "inner" else "outer_test",
        "former_inner_validation_now_training": role == "outer",
        "test_never_fits_generates_or_selects": True, "synthetics_never_fit_generator_or_preprocessor": True})
    return real, X, y, groups


def run_round(dataset, split, partitions, role, algorithm, seed, control, generator,
              masks, schema, cap, nominal_total, protocol, folder, core, metadata):
    parameters = {"grouping": control["grouping"], "allocation": control["allocation"],
                  "group_alpha": control["group_alpha"], "capture_proposals": True}
    indices = split["fit_indices" if role == "inner" else "train_indices"]
    y = dataset["y"][indices]
    targets = augmentation_targets(y, {"mode": "matched", "total": nominal_total})
    real, X, y, groups = prepare_real(dataset, split, partitions, role, folder, parameters, targets, seed)
    estimator = make_estimator(algorithm, seed, parameters=control["parameters"], device="cpu")
    start = perf_counter()
    initial = fit_estimator(estimator, X, y)
    initial_seconds = perf_counter() - start
    joblib.dump(initial, folder / "initial_native.joblib")
    start = perf_counter()
    current, plan = fixed_u2_schedule(X, y, initial, groups, parameters, seed,
        categorical_columns(partitions[role]), targets, cap)
    schedule_seconds = perf_counter() - start - plan["original_proposal_seconds"]
    anchors = current.seed_indices.numpy()
    partners = current.partner_indices.numpy()
    group_ids = np.asarray(groups["row_group_ids"])[anchors]
    labels = current.y.numpy()
    anchor_frame = real.iloc[anchors].reset_index(drop=True)
    coefficients = plan.pop("interpolation_coefficients")
    scores = plan.pop("scores")
    np.savez_compressed(folder / "schedule.npz", anchor_indices=anchors,
        anchor_row_ids=dataset["row_ids"][indices][anchors], group_ids=group_ids,
        partner_indices=partners, partner_row_ids=dataset["row_ids"][indices][partners],
        labels=labels, proposal_ids=np.arange(len(anchors)), interpolation_coefficients=coefficients,
        original_classifier_coordinates=current.X.numpy(), initial_real_scores=scores)
    write_json(folder / "schedule.json", {**plan, "initial_fit_seconds": initial_seconds,
        "schedule_seconds": schedule_seconds, "initial_native_sha256": sha256(folder / "initial_native.joblib"),
        "same_schedule_all_arms": True, "one_proposal_per_scheduled_slot": True,
        "sampling_stream_B_C": seed + 70001 + ALGORITHMS.index(algorithm) * 1009 + (role == "outer") * 3001})
    gate = flow_for_partition(dataset, split, partitions, role, seed, {}, "auto")
    tau = control["f1_tau"]
    flow_before, generator_before = flow_checksum(core.model), tabdiff_checksum(generator._diffusion)
    pending, eligible = {}, {}
    for arm in ARMS:
        destination = folder / arm
        destination.mkdir(parents=True, exist_ok=True)
        start = perf_counter()
        if arm == "A":
            raw = original_raw_proposals(real, current, coefficients)
            generation = {"generator": "original_U2_input_interpolation", "masked_columns": None,
                "physical_raw_affine_decode": True, "label_unmasked": True,
                "valid_mask": np.ones(len(anchors), dtype=bool)}
            sampling_seconds = perf_counter() - start + plan["original_proposal_seconds"]
        else:
            sampling_seed = seed + 70001 + ALGORITHMS.index(algorithm) * 1009 + (role == "outer") * 3001
            raw, generation = generator.inpaint(anchor_frame, masks[arm], labels, sampling_seed)
            sampling_seconds = perf_counter() - start
        start = perf_counter()
        decoded, valid, reasons, changes, failures = schema.decode(raw)
        valid &= np.asarray(generation["valid_mask"], dtype=bool)
        if arm != "A":
            fixed = [column for column in real if column not in masks[arm]]
            for column in fixed:
                if not decoded.loc[valid, column].reset_index(drop=True).equals(
                    anchor_frame.loc[valid, column].reset_index(drop=True)):
                    # Numeric/category dtype restoration must not change values.
                    if not np.array_equal(decoded.loc[valid, column].to_numpy(), anchor_frame.loc[valid, column].to_numpy()):
                        raise RuntimeError("Inpainting changed an unmasked original field: " + column)
        coordinates = np.full((len(raw), X.shape[1]), np.nan, dtype=np.float32)
        positions = np.flatnonzero(valid)
        if len(positions):
            coordinates[positions] = partitions[role].transform(decoded.iloc[positions]).astype(np.float32)
        decision = {name: np.full(len(raw), np.nan) for name in
                    ("log_ratio", "log_likelihood_intended", "log_likelihood_opposing", "compatibility_score", "log_determinant")}
        f1_valid = np.zeros(len(raw), dtype=bool)
        f1_accepted = np.zeros(len(raw), dtype=bool)
        f1_reasons = np.full(len(raw), "schema_or_generation_invalid", dtype=object)
        if len(positions):
            scored = gate.predict(coordinates[positions], labels[positions], tau)
            f1_valid[positions], f1_accepted[positions] = scored["valid"], scored["accepted"]
            f1_reasons[positions] = scored["reasons"]
            for name in decision:
                decision[name][positions] = scored[name]
        filtering_seconds = perf_counter() - start
        diagnostic_start = perf_counter()
        real_duplicates, pool_duplicates = schema.duplicates(decoded, valid)
        nearest_distance = np.full(len(raw), np.nan)
        nearest_real = np.full(len(raw), -1, dtype=int)
        if len(positions):
            query = torch.as_tensor(coordinates[positions])
            neighbors = nearest_indices(query, torch.as_tensor(X), 1).squeeze(1)
            nearest_real[positions] = neighbors.numpy()
            nearest_distance[positions] = torch.linalg.vector_norm(query - torch.as_tensor(X)[neighbors], dim=1).numpy()
        diagnostic_seconds = perf_counter() - diagnostic_start
        joblib.dump(raw, destination / "generated_raw.joblib")
        joblib.dump(decoded, destination / "decoded.joblib")
        write_json(destination / "generation.json", generation)
        arrays = {"coordinates": coordinates, "labels": labels, "anchor_indices": anchors,
            "anchor_row_ids": dataset["row_ids"][indices][anchors], "group_ids": group_ids,
            "proposal_ids": np.arange(len(anchors)), "schema_valid": valid,
            "f1_valid": f1_valid, "f1_accepted": f1_accepted, "eligible": valid & f1_accepted,
            "count_changed": changes, "schema_reasons": np.asarray(reasons, dtype=str),
            "f1_reasons": np.asarray(f1_reasons, dtype=str), "exact_real_duplicate": real_duplicates,
            "exact_pool_duplicate": pool_duplicates, "nearest_real_index": nearest_real,
            "nearest_real_distance": nearest_distance, "near_duplicate": nearest_distance <= .001, **decision}
        eligible[arm] = arrays["eligible"]
        pending[arm] = (arrays, {"schema_failure_counts": {name: int(mask.sum()) for name, mask in failures.items()},
            "sampling_seconds": sampling_seconds, "filtering_seconds": filtering_seconds,
            "diagnostic_seconds": diagnostic_seconds, "F1_tau": tau,
            "generation_valid_count": int(np.asarray(generation["valid_mask"]).sum()),
            "schema_valid_count": int(valid.sum()), "F1_accepted_count": int(f1_accepted.sum()),
            "count_decode_changed_cells": int(changes.sum()), "exact_real_duplicates": int(real_duplicates.sum()),
            "exact_pool_duplicates": int(pool_duplicates.sum()), "near_duplicates": int(arrays["near_duplicate"].sum()),
            "masked_columns": masks.get(arm), "mask_fraction": len(masks[arm]) / len(real.columns) if arm != "A" else None})
    start = perf_counter()
    selected, quotas = match_group_counts(eligible, group_ids, plan["allocation_record"]["group_trace"], seed)
    matching_seconds = perf_counter() - start
    write_json(folder / "matched_quotas.json", {"groups": quotas, "matching_seed": seed + 23021,
        "policy": "per_group_minimum_then_uniform_arm_specific_thinning", "matching_seconds": matching_seconds,
        "no_refill_no_other_generator_no_relaxation": True})
    rows = []
    for arm in ARMS:
        destination = folder / arm
        arrays, audit = pending[arm]
        arrays["selected"] = selected[arm]
        np.savez_compressed(destination / "proposals.npz", **arrays)
        augmented_X = np.concatenate([X, arrays["coordinates"][selected[arm]]])
        augmented_y = np.concatenate([y, labels[selected[arm]]])
        start = perf_counter()
        native = fit_estimator(estimator, augmented_X, augmented_y)
        fit_seconds = perf_counter() - start
        joblib.dump(native, destination / "model.joblib")
        np.savez_compressed(destination / "training.npz", X=augmented_X, y=augmented_y,
                            sample_weight=np.ones(len(augmented_y)))
        evaluation_indices = split["validation_indices" if role == "inner" else "test_indices"]
        evaluation_X = partitions["X_validation" if role == "inner" else "X_test"]
        metrics = save_predictions(destination, "validation" if role == "inner" else "test", native,
            evaluation_X, dataset["y"][evaluation_indices], dataset["row_ids"][evaluation_indices], protocol)
        count = int(selected[arm].sum())
        audit.update(accepted_count=count, proposal_count=len(anchors), quota_shortfall=len(anchors)-count,
            native_fit_seconds=fit_seconds, native_fit_count=1, matching_seconds_shared=matching_seconds,
            native_parameters=control["parameters"], seed=seed, role=role,
            flow_state_before=flow_before, flow_state_after=flow_checksum(core.model),
            generator_state_before=generator_before, generator_state_after=tabdiff_checksum(generator._diffusion),
            generator_checkpoint=generator.training_audit, schema=schema.audit(),
            synthetic_label=int(plan["minority_label"]), unmasked_exactly_preserved=arm != "A",
            relocation=False, count_rounding_common_user_approved=True, utility_ranking=False,
            duplicate_policy="diagnostics_only_preserve_original_no_new_dedup_gate")
        write_json(destination / "audit.json", audit)
        write_json(destination / "metrics.json", metrics)
        rows.append({**metadata, "variant_id": role + "_" + arm, "arm": arm, "role": role,
            "phase": "fixed_round", "method": "ugo_f1" if arm == "A" else "tabdiff_inpaint_f1",
            "status": "completed", "metrics": metrics,
            "evaluation_role": "inner_validation" if role == "inner" else "outer_test",
            "predictions_path": str((destination / ("validation_predictions.csv" if role == "inner" else "test_predictions.csv")).resolve()),
            "trajectory_path": str(destination.resolve()),
            "augmentation": {"mode": "matched", "requested_count": len(anchors), "actual_count": count,
                             "underspend": len(anchors)-count, "nominal_final_target": nominal_total},
            "config": {"native_parameters": control["parameters"], "sampler_parameters": parameters,
                       "f1_tau": tau, "count_decoder": "common_np_rint_ties_even",
                       "mask_columns": masks.get(arm), "mask_fraction": audit["mask_fraction"],
                       "threshold": .5, "proposal_cap": cap},
            "sampling_seconds": audit["sampling_seconds"], "filtering_seconds": audit["filtering_seconds"],
            "native_fit_seconds": fit_seconds, "native_fit_count": 1,
            "runtime_seconds": audit["sampling_seconds"] + audit["filtering_seconds"] + fit_seconds,
            "shared_generator_checkpoint": str(generator.checkpoint_path.resolve()),
            "generator_checkpoint_sha256": generator.training_audit["checkpoint_sha256"]})
    if flow_checksum(core.model) != flow_before or tabdiff_checksum(generator._diffusion) != generator_before:
        raise RuntimeError("Frozen generator or F1 state changed during controlled sampling/filtering")
    return rows, {"initial_native_fit_seconds": initial_seconds, "initial_native_fit_count": 1,
                  "schedule_seconds": schedule_seconds, "matching_seconds": matching_seconds}


def import_context(audit, metadata, seed, algorithm, protocol):
    result = []
    for reference in audit["historical_context"]:
        if reference["seed"] != seed or reference["algorithm"] != algorithm:
            continue
        predictions = Path(reference["predictions_path"])
        if sha256(predictions) != reference["predictions_sha256"]:
            raise ValueError("Historical reference predictions changed")
        if reference["split_fingerprint"] != metadata["split_fingerprint"]:
            raise ValueError("Historical references use a different split")
        metrics = prediction_metrics(pd.read_csv(predictions, float_precision="round_trip"), protocol)
        result.append({**metadata, "variant_id": "historical_" + reference["variant_id"],
            "method": reference["method"], "role": "outer", "phase": "historical_context",
            "status": "completed", "metrics": metrics, "evaluation_role": "outer_test",
            "predictions_path": str(predictions.resolve()), "source_run_id": Path(reference["source_run"]).name,
            "source_record_id": reference["source_record_id"], "historical_label": reference["label"],
            "source_provenance": reference, "historical_not_fixed_round_control": True,
            "augmentation": {"mode": "matched", "requested_count": reference["requested_count"],
                             "actual_count": reference["actual_count"], "underspend": reference["underspend"]},
            "config": {"source_native_parameters": reference["parameters"], "source_sampler_parameters": reference["sampler_parameters"]}})
    return result


def export_tables(path, rows):
    flat = [{**{key: row.get(key) for key in ("algorithm","seed","variant_id","role","phase","sampling_seconds","filtering_seconds","native_fit_seconds","runtime_seconds")},
             "accepted": row["augmentation"]["actual_count"], **numeric_metrics(row["metrics"])} for row in rows]
    pd.DataFrame(flat).to_csv(path / "all_metrics.csv", index=False)
    index = {(r["algorithm"],r["seed"],r["variant_id"]): r for r in rows}
    deltas = []
    for row in rows:
        if row["variant_id"] not in ("outer_B", "outer_C", "outer_selected_TabDiff"):
            continue
        a = index[(row["algorithm"], row["seed"], "outer_A")]
        for metric, value in numeric_metrics(row["metrics"]).items():
            deltas.append({"algorithm":row["algorithm"],"seed":row["seed"],"variant_id":row["variant_id"],
                           "reference_variant":"outer_A","metric":metric,"delta":value-a["metrics"][metric]})
    pd.DataFrame(deltas).to_csv(path / "paired_metric_deltas.csv", index=False)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    result.add_argument("--protocol", default="configs/benchmarks/shoppers_controlled.yaml")
    result.add_argument("--study", default="configs/ablations/stage1_tabdiff.yaml")
    result.add_argument("--controls", default="configs/ablations/tabdiff_controls.json")
    result.add_argument("--reference-run", default="results/runs/20261009T170013-d5be50bc2107")
    result.add_argument("--data-home", default=".cache/benchmarks")
    result.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    result.add_argument("--output-dir")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    protocol, study = load_protocol(args.protocol), load_study(args.study)
    if protocol["dataset"] != "shoppers" or protocol["split"] != {"type":"stratified_holdout","test_fraction":.2,"outer_seed":2026}:
        raise ValueError("Preserve the active Shoppers real-data protocol")
    if args.run and args.seeds != list(range(5)):
        raise ValueError("The scientific experiment requires all five matched locked seeds 0..4")
    experiment_type = "setup_smoke" if args.smoke else "development"
    enforce_stage_gate(study, experiment_type)
    audit = json.loads(Path(args.controls).read_text())
    controls = {(c["seed"],c["algorithm"]): c["control"] for c in audit["cells"]}
    cap, nominal = (40,40) if args.smoke else (study["protocol"]["proposal_cap"],1000)
    generator_config = TabDiffConfig(**{key:value for key,value in study["generator"].items()
                                       if key in TabDiffConfig.__dataclass_fields__})
    if args.smoke:
        generator_config = TabDiffConfig(**{**asdict(generator_config), "epochs": 32, "batch_size": 128})
    arguments = {**vars(args), "experiment_type":experiment_type, "effective_classifiers":ALGORITHMS,
        "effective_seeds":args.seeds,"effective_settings":{"augmentation":{"mode":"matched","total":nominal},
        "proposal_cap":cap,"augmentation_rounds":1,"generator":asdict(generator_config)},
        "mask_selection":"inner_validation_AP_only_B_tiebreak", "no_utility_no_TAP_policy":True,
        "no_iterative_change":True,"user_approved_count_rounding_all_arms":True,
        "bootstrap_draws":2000,"bootstrap_seed":20261011}
    artifacts = BenchmarkArtifacts(args.output_dir or ("results/smoke" if args.smoke else "results/runs"), arguments, protocol, study)
    rows, resources = [], []
    wall_start = perf_counter()
    try:
        dataset = load_dataset(protocol,args.data_home)
        artifacts.dataset(dataset)
        write_json(artifacts.path / "control_audit.json",audit)
        fingerprint = data_fingerprint(dataset["X"],dataset["y"],dataset["row_ids"])
        for seed in args.seeds:
            seed_everything(seed)
            canonical = make_splits(dataset,protocol,seed)[0]
            split = smoke_split(dataset,canonical,seed,400,120,.2) if args.smoke else canonical
            partitions = prepare_partitions(dataset,split)
            root = artifacts.fold_path("shoppers",split["fold"]) / ("seed-"+str(seed))
            root.mkdir(parents=True,exist_ok=True)
            write_json(root / "split.json",{**split,"canonical_test_indices":canonical["test_indices"]})
            core = frozen_flow(dataset,split,partitions,root,seed,None if args.smoke else Path(args.reference_run))
            masks, mask_metadata = feature_masks(dataset["X"].iloc[split["fit_indices"]],seed)
            write_json(root / "masks.json",{**mask_metadata,"masks":masks,"fit_row_ids":dataset["row_ids"][split["fit_indices"]]})
            selections = {}
            for role in ("inner","outer"):
                role_indices = split["fit_indices" if role == "inner" else "train_indices"]
                real_frame = dataset["X"].iloc[role_indices].reset_index(drop=True)
                schema = ShoppersSchema(real_frame)
                role_root = root / role
                role_root.mkdir(parents=True,exist_ok=True)
                write_json(role_root / "generator_role.json",{"fit_indices":role_indices,
                    "fit_row_ids":dataset["row_ids"][role_indices],"synthetic_rows":0,
                    "validation_indices":split["validation_indices"] if role == "inner" else [],
                    "test_indices":split["test_indices"],"former_inner_validation_now_training":role=="outer",
                    "selection_before_final_outer_training":role=="outer", "inner_selection":selections if role=="outer" else None})
                start = perf_counter()
                generator = fit_tabdiff(real_frame,dataset["y"][role_indices],dataset["row_ids"][role_indices],
                    seed + 1889 + (role=="outer")*3001,role_root / "generator",generator_config,"auto")
                generator_total_seconds = perf_counter()-start
                resources.append({"seed":seed,"role":role,"generator_training_seconds":generator.training_audit["training_seconds"],
                    "generator_total_seconds":generator_total_seconds,"generator_checkpoint":str(generator.checkpoint_path.resolve()),
                    "generator_checkpoint_sha256":generator.training_audit["checkpoint_sha256"],"reused_native_classifiers":3,"reused_masks":2})
                for algorithm in ALGORITHMS:
                    control = controls[(seed,algorithm)]
                    if not args.smoke and sha256(root / "flow-core.pt") != control["f1_checkpoint_sha256"]:
                        raise ValueError("F1 checkpoint differs from locked actual retained configuration")
                    folder = root / algorithm / role
                    folder.parent.mkdir(parents=True, exist_ok=True)
                    metadata = {"dataset":"shoppers","dataset_fingerprint":fingerprint,
                        "split_fingerprint":split_fingerprint(dataset,split),"seed":seed,"algorithm":algorithm,
                        "budget_mode":"matched","tuning_budget":0,"candidate_configs":[control["parameters"]],
                        "experiment_type":experiment_type,"fold":split["fold"]}
                    write_json(root / algorithm / "control.json",{**control,"new_U5_means_U2_only":True,"OOF":False})
                    generated, resource = run_round(dataset,split,partitions,role,algorithm,seed,control,generator,masks,
                        schema,cap,nominal,protocol,folder,core,metadata)
                    rows.extend(generated)
                    resources.append({"seed":seed,"role":role,"algorithm":algorithm,**resource})
                    if role == "inner":
                        by_arm = {r["arm"]:r for r in generated}
                        choice = max(("B","C"),key=lambda arm:(by_arm[arm]["metrics"]["average_precision"],arm=="B"))
                        selections[algorithm] = {"selected_arm":choice,"inner_AP":{a:by_arm[a]["metrics"]["average_precision"] for a in ARMS},
                            "selected_mask_columns":masks[choice],"source":"inner_validation_only","outer_test_seen":False}
                        write_json(root / algorithm / "mask_selection.json",selections[algorithm])
                    else:
                        chosen = next(r for r in generated if r["arm"]==selections[algorithm]["selected_arm"])
                        rows.append({**chosen,"variant_id":"outer_selected_TabDiff","selected_arm":chosen["arm"],
                            "selected_record_derived":True,"additional_training":False,
                            "config":{**chosen["config"],"selection":selections[algorithm]},"native_fit_count":0})
                        if not args.smoke:
                            rows.extend(import_context(audit,metadata,seed,algorithm,protocol))
                    print(f"seed={seed} role={role} classifier={algorithm}: A/B/C common count={generated[0]['augmentation']['actual_count']}",flush=True)
            export_tables(artifacts.path,rows)
        write_json(artifacts.path / "resources.json",resources)
        write_json(artifacts.path / "invocation_costs.json",{"wall_seconds":perf_counter()-wall_start,
            "generator_fits":2*len(args.seeds),"shared_training_not_charged_per_arm":True,
            "F1_training_repeated":args.smoke,"historical_timing_not_compared":True})
        artifacts.finish(rows)
    except Exception as error:
        write_json(artifacts.path / "resources.json",resources)
        artifacts.finish(rows,error=str(error),status="failed")
        raise
    print(f"Artifacts: {artifacts.path}",flush=True)
    return artifacts.path


if __name__ == "__main__":
    main()
