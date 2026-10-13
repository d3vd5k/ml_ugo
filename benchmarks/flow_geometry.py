"""Shoppers U2 random pools: matched within-group input versus Flow geometry."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from benchmarks.pool_random_f1 import (frozen_flow, prepare_role, trajectory,
                                      main as reference_smoke, sha256)
from benchmarks.protocols import load_dataset, load_protocol, make_splits
from benchmarks.results import BenchmarkArtifacts, enforce_stage_gate, load_study
from benchmarks.run import candidate_configs, prepare_partitions, smoke_split, split_fingerprint
from benchmarks.validation_guided import ALGORITHMS, REFERENCES, numeric_metrics, prediction_metrics
from experiments import data_fingerprint, seed_everything, write_json
from samplers.flow_geometry import FlowParentCache, GroupGeometryProposer
from samplers.flow_gmm import _checksum

HISTORICAL = (*REFERENCES, "outer_iterative_pool_random", "outer_iterative_pool_random_f1")
PRIMARY = "outer_iterative_pool_random_group_flow"
CONTROL = "outer_iterative_pool_random_group_input"


def export_comparisons(path, rows):
    flat = []
    for row in rows:
        item = {key: row.get(key) for key in ("algorithm", "seed", "variant_id", "role", "phase",
            "runtime_seconds", "native_fit_count", "native_fit_seconds", "source_run_id")}
        item["accepted"] = row["augmentation"]["actual_count"]
        if row["evaluation_role"] == "outer_test":
            item.update({f"test_{key}": value for key, value in numeric_metrics(row["metrics"]).items()})
        item.update({f"validation_{key}": value for key, value in numeric_metrics(row.get("validation_metrics", {})).items()})
        flat.append(item)
    frame = pd.DataFrame(flat)
    frame.to_csv(path / "all_metrics.csv", index=False)
    summaries = []
    for (algorithm, variant), group in frame.groupby(["algorithm", "variant_id"]):
        for key in [column for column in group if column.startswith(("test_", "validation_"))]:
            values = group[key].dropna()
            if len(values):
                summaries.append({"algorithm": algorithm, "variant_id": variant, "metric": key,
                    "mean": values.mean(), "sample_sd": values.std(ddof=1), "seeds": len(values)})
    pd.DataFrame(summaries).to_csv(path / "metric_summary.csv", index=False)
    index = {(row["algorithm"], row["seed"], row["variant_id"]): row for row in rows}
    deltas = []
    for row in rows:
        if row["variant_id"] != PRIMARY:
            continue
        for variant in (*HISTORICAL, CONTROL):
            reference = index[(row["algorithm"], row["seed"], variant)]
            for metric, value in numeric_metrics(row["metrics"]).items():
                deltas.append({"algorithm": row["algorithm"], "seed": row["seed"], "variant_id": PRIMARY,
                    "reference_variant": variant, "metric": metric,
                    "selected_value": value, "reference_value": reference["metrics"][metric],
                    "delta": value - reference["metrics"][metric],
                    "selected_count": row["augmentation"]["actual_count"],
                    "reference_count": reference["augmentation"]["actual_count"]})
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
    result.add_argument("--study", default="configs/ablations/stage1_flow_geometry.yaml")
    result.add_argument("--reference-run", default="results/runs/20261009T170013-d5be50bc2107")
    result.add_argument("--data-home", default=".cache/benchmarks")
    result.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    result.add_argument("--output-dir")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    protocol, study = load_protocol(args.protocol), load_study(args.study)
    if protocol["dataset"] != "shoppers" or protocol["status"] != "ready":
        raise ValueError("Flow geometry requires the resolved Shoppers protocol")
    if protocol["split"] != {"type": "stratified_holdout", "test_fraction": .2, "outer_seed": 2026} or protocol["validation_fraction"] != .2:
        raise ValueError("Preserve original Shoppers data access")
    if not args.seeds or any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Require unique nonnegative matched seeds")
    experiment_type = "setup_smoke" if args.smoke else "development"
    enforce_stage_gate(study, experiment_type)
    invocation_start = perf_counter()
    if args.smoke:
        # Fresh prior-protocol controls are development-only; never import official
        # test predictions into a smoke comparison or fit the Flow on its rows.
        source = Path(reference_smoke(["--smoke", "--seeds", *map(str, args.seeds),
            "--protocol", args.protocol, "--data-home", args.data_home]))
    else:
        source = Path(args.reference_run)
        if not json.loads((source / "independent_verification.json").read_text())["passed"]:
            raise ValueError("Historical reference source must pass independent verification")
    source_rows = {(row["seed"], row["algorithm"], row["variant_id"]): row
                   for row in json.loads((source / "benchmark_records.json").read_text())}
    settings = {"device": "cpu", "mlp_epochs": 160,
        "augmentation": {"mode": "matched", "total": 40 if args.smoke else 1000},
        "ugo_max_iterations": 2 if args.smoke else 20,
        "method_parameters": {**protocol.get("method_parameters", {}),
                              **(protocol.get("smoke_method_parameters", {}) if args.smoke else {})}}
    arguments = {**vars(args), "experiment_type": experiment_type, "effective_settings": settings,
        "effective_classifiers": ALGORITHMS, "effective_seeds": args.seeds,
        "effective_tuning_budget": 1 if args.smoke else 10,
        "reference_source": str(source.resolve()), "native_anchor_and_final_classifier": True,
        "no_utility_or_surrogate": True, "no_F1_rejection": True,
        "categorical_projection": "original_U2_partner_categories", "primary_comparison": f"{PRIMARY} minus {CONTROL}"}
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
            core = frozen_flow(dataset, split, partitions, root, seed, source)
            frozen_checksum = _checksum(core.model)
            caches = {}
            for role in ("inner", "outer"):
                indices = split["fit_indices" if role == "inner" else "train_indices"]
                X = partitions["X_fit" if role == "inner" else "X_train"]
                caches[role] = FlowParentCache(core, partitions[role], dataset["X"].iloc[indices],
                                             X, dataset["row_ids"][indices])
                caches[role].save(root / "geometry-cache" / role)
            for algorithm in ALGORITHMS:
                folder = root / algorithm
                folder.mkdir(parents=True, exist_ok=True)
                candidates = candidate_configs(protocol, algorithm, seed, 1 if args.smoke else 10)
                metadata = {"dataset": "shoppers", "dataset_fingerprint": fingerprint,
                    "split_fingerprint": split_fingerprint(dataset, split), "seed": seed, "algorithm": algorithm,
                    "budget_mode": "matched", "tuning_budget": len(candidates), "candidate_configs": candidates,
                    "experiment_type": experiment_type, "fold": split["fold"]}
                for variant in HISTORICAL:
                    original = source_rows[(seed, algorithm, variant)]
                    if (original["dataset_fingerprint"] != fingerprint or original["split_fingerprint"] != metadata["split_fingerprint"]
                            or original["candidate_configs"] != candidates):
                        raise ValueError("Historical controls must match data access and canonical settings search")
                    origin = (source / "shoppers" / "fold-0" / f"seed-{seed}" / algorithm / variant
                              if variant in REFERENCES else Path(original["trajectory_path"]))
                    destination = (folder / variant if variant in REFERENCES else
                                   folder / "outer" / variant.removeprefix("outer_iterative_"))
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.symlink_to(origin.resolve(), target_is_directory=True)
                    reference = {**original, **metadata, "role": "outer", "phase": "reference",
                        "evaluation_role": "outer_test", "source_run_id": original["run_id"],
                        "source_record_id": original["record_id"], "source_artifact_path": str(origin.resolve()),
                        "earlier_source_run_id": original.get("source_run_id"),
                        "earlier_source_record_id": original.get("source_record_id"),
                        "imported_reference": True, "reference_runtime_reused_not_rerun": True}
                    reference["metrics"] = prediction_metrics(pd.read_csv(reference["predictions_path"], float_precision="round_trip"), protocol)
                    rows.append(reference)
                selected = json.loads((folder / "U2" / "selection.json").read_text())
                write_json(folder / "selection.json", {"canonical_U2_selection": {
                    key: selected["inner"][key] for key in ("parameters", "sampler_parameters")},
                    "canonical_U2_source": str((folder / "U2" / "selection.json").resolve()),
                    "source_selection_sha256": sha256(folder / "U2" / "selection.json"),
                    "native_parameters_locked": True, "group_alpha_locked": True,
                    "no_additional_tuning": True, "no_test_selection": True})
                for role in ("inner", "outer"):
                    prepared = prepare_role(dataset, split, partitions, role, seed, selected, settings, folder / role)
                    for mode in ("input", "latent"):
                        proposer = GroupGeometryProposer(caches[role], prepared[3]["row_group_ids"], mode=mode)
                        destination = folder / role / proposer.variant_name
                        row = trajectory(dataset, split, partitions, role, seed, algorithm, selected,
                            settings, protocol, destination, prepared, metadata, proposal_generator=proposer)
                        row["config"].update(generation_geometry=mode, parent_scope="same_U2_group_real_parents",
                                             categorical_projection="original_partner_categories", no_F1_rejection=True)
                        rows.append(row)
                if _checksum(core.model) != frozen_checksum:
                    raise RuntimeError("Flow generation changed the frozen checkpoint")
                export_comparisons(artifacts.path, rows)
                print(f"seed={seed} classifier={algorithm}: both geometry trajectories complete", flush=True)
        write_json(artifacts.path / "invocation_costs.json", {"observed_total_seconds": perf_counter() - invocation_start,
            "fresh_smoke_reference_subrun_included": args.smoke, "historical_reference_tuning_included": False,
            "frozen_flow_training_repeated": False,
            "parent_cache_build_once_per_seed_role": True, "cache_builds": 2 * len(args.seeds)})
        artifacts.finish(rows)
    except Exception as error:
        artifacts.finish(rows, error=str(error), status="failed")
        raise
    print(f"Artifacts: {artifacts.path}", flush=True)
    return artifacts.path


if __name__ == "__main__":
    main()
