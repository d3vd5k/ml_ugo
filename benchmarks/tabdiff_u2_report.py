"""Independently report verified no-F1 Shoppers U2/TabDiff saved-pool replays.

Example: python -m benchmarks.tabdiff_u2_report --root RESULTS \
    --verification RESULTS/independent_verification.json \
    --output RESULTS/independent_report.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef, roc_curve)

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261011
ALGORITHMS = {"logistic_regression": "LR", "random_forest": "RF", "lightgbm": "LightGBM"}
CURRENT = ("inner_A", "inner_B", "inner_C", "outer_A", "outer_B", "outer_C",
           "outer_selected_TabDiff")
PRIOR = ("prior_f1_A", "prior_f1_B", "prior_f1_C", "prior_f1_selected_TabDiff")
HISTORICAL = ("historical_U0", "historical_U2", "historical_F1", "historical_U5_F1")
LABELS = {"inner_A": "Inner A", "inner_B": "Inner B25", "inner_C": "Inner C50",
          "outer_A": "A no-F1", "outer_B": "B25 no-F1", "outer_C": "C50 no-F1",
          "outer_selected_TabDiff": "Inner-selected no-F1", "prior_f1_A": "Prior A+F1",
          "prior_f1_B": "Prior B25+F1", "prior_f1_C": "Prior C50+F1",
          "prior_f1_selected_TabDiff": "Prior inner-selected+F1",
          "historical_U0": "Original UGO", "historical_U2": "Original pure U2",
          "historical_F1": "Standalone U2+F1", "historical_U5_F1": "Historical OOF+U2+F1"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y, probabilities):
    """The durable independent 24-metric recipe; no producer imports."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if (p.shape != (len(y), 2) or not np.isfinite(p).all()
            or not np.all((p >= 0) & (p <= 1)) or not np.allclose(p.sum(axis=1), 1)
            or set(np.unique(y)) != {0, 1}):
        raise ValueError("Metrics require finite binary probabilities and both classes")
    pred = (p[:, 1] >= .5).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    f = f1_score(y, pred, labels=[0, 1], average=None, zero_division=0)
    prevalence = np.array([(y == k).mean() for k in [0, 1]])
    fpr, tpr, _ = roc_curve(y, p[:, 1], drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= .05)
    best = eligible[np.argmax(tpr[eligible])]
    weights = np.where(y == 0, 1 / (2 * prevalence[0]), 1 / (2 * prevalence[1]))
    return dict(
        average_precision=float(average_precision_score(y, p[:, 1])),
        macro_average_precision=float(np.mean([average_precision_score(y == k, p[:, k]) for k in [0, 1]])),
        minority_f1=float(f[1]), minority_recall=tp / (tp + fn),
        macro_f1=float(f.mean()), weighted_f1=float(prevalence @ f),
        sos_inverse_support_f1=float((1 - prevalence) @ f),
        balanced_accuracy=(tp / (tp + fn) + tn / (tn + fp)) / 2,
        mcc=float(matthews_corrcoef(y, pred)), threshold=.5,
        fp=fp, fn=fn, tp=tp, tn=tn, cost=10 * fp + 500 * fn,
        realized_recall_test=tp / (tp + fn), realized_fpr_test=fp / (fp + tn),
        recall_at_fpr_test=float(tpr[best]), roc_attained_fpr_test=float(fpr[best]),
        roc_fpr_limit=.05, accuracy=float((y == pred).mean()),
        precision=tp / (tp + fp) if tp + fp else 0.,
        ordinary_cross_entropy=float(log_loss(y, p, labels=[0, 1])),
        balanced_cross_entropy=float(log_loss(y, p, labels=[0, 1], sample_weight=weights)))


def paired_bootstrap(y, predictions, selected, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Share each stratified outer-row and matched-seed draw across every arm/model."""
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    algorithms = sorted(predictions)
    seeds = sorted(next(iter(predictions.values())))
    if any(sorted(predictions[a]) != seeds for a in algorithms):
        raise ValueError("Bootstrap requires matched seeds")
    strata = [np.flatnonzero(y == k) for k in [0, 1]]
    arms = ("B", "C", "inner_selected")
    samples = {(a, arm): {"row": np.empty(draws), "joint": np.empty(draws)}
               for a in algorithms for arm in arms}
    estimates = {(a, arm): np.array([
        average_precision_score(y, predictions[a][s][selected[a][s] if arm == "inner_selected" else arm])
        - average_precision_score(y, predictions[a][s]["A"]) for s in seeds])
        for a in algorithms for arm in arms}
    for draw in range(draws):
        row_sample = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in strata])
        weights = np.bincount(row_sample, minlength=len(y))
        seed_sample = rng.integers(len(seeds), size=len(seeds))
        for a in algorithms:
            ap = {arm: np.array([average_precision_score(y, predictions[a][s][arm], sample_weight=weights)
                                 for s in seeds]) for arm in ("A", "B", "C")}
            ap["inner_selected"] = np.array([ap[selected[a][s]][i] for i, s in enumerate(seeds)])
            for arm in arms:
                delta = ap[arm] - ap["A"]
                samples[a, arm]["row"][draw] = delta.mean()
                samples[a, arm]["joint"][draw] = delta[seed_sample].mean()
    return {"draws": draws, "precommitted_seed": seed,
            "interpretation": "Conditional on shared training and one realized outer holdout, frozen inner choices and exploratory seeds. Not independent-fold intervals, population superiority evidence or an automatic retention decision.",
            "comparisons": [{"algorithm": a, "comparison": arm + " vs A",
                "matched_seed_deltas": dict(zip(map(str, seeds), estimates[a, arm].tolist())),
                "mean_delta": float(estimates[a, arm].mean()),
                "paired_stratified_shared_row_percentile_CI95": np.quantile(values["row"], [.025, .975]).tolist(),
                "shared_row_and_matched_seed_joint_percentile_CI95": np.quantile(values["joint"], [.025, .975]).tolist()}
                for (a, arm), values in samples.items()]}


def read_bound_json(path, hashes):
    path = Path(path).resolve()
    hashes[str(path)] = sha(path)
    return json.loads(path.read_text())


def json_records(frame):
    # A one-seed smoke has undefined sample SD, represented by null, never zero.
    return json.loads(frame.to_json(orient="records", double_precision=15))


def summarize(frame, grouping, names):
    means = frame.groupby(grouping, sort=True)[names].mean()
    sd = frame.groupby(grouping, sort=True)[names].std(ddof=1).add_suffix("_seed_sd")
    counts = frame.groupby(grouping, sort=True).size().rename("seed_count")
    return means.join(sd).join(counts).reset_index()


def report(root, verification_path, output, prior_root=None, markdown=None, smoke_verification=None):
    root = Path(root).resolve()
    verification_path = Path(verification_path).resolve()
    output = Path(output).resolve()
    hashes = {}
    proof = read_bound_json(verification_path, hashes)
    manifest = read_bound_json(root / "manifest.json", hashes)
    if (proof["status"] != "verified" or Path(proof["root"]).resolve() != root
            or manifest["status"] != "completed"):
        raise ValueError("Only an independently verified completed actual root can be reported")
    for filename, digest in proof["actual_file_sha256"].items():
        if sha(filename) != digest:
            raise ValueError("Verification binding changed: " + filename)
    arguments = manifest["arguments"]
    smoke = arguments["smoke"]
    seeds = arguments["effective_seeds"]
    if arguments["F1_enabled"] is not False or (not smoke and seeds != list(range(5))):
        raise ValueError("Require F1-disabled replay and all five full-run seeds")
    if set(arguments["effective_classifiers"]) != set(ALGORITHMS):
        raise ValueError("All three native classifiers are required")
    if smoke and len(seeds) != 1:
        raise ValueError("Smoke contract requires one seed and 33 records")
    source = Path(arguments["source_root"]).resolve()
    if prior_root is not None and Path(prior_root).resolve() != source:
        raise ValueError("Prior root must be the actual replay source")
    parent_manifest = read_bound_json(source / "manifest.json", hashes)
    parent_proof = read_bound_json(source / "independent_verification.json", hashes)
    if (parent_manifest["status"] != "completed" or parent_proof["status"] != "verified"
            or Path(parent_proof["root"]).resolve() != source
            or sha(source / "independent_verification.json") != arguments["source_verification_sha256"]):
        raise ValueError("Replay source verification is not bound")
    recipe = source / "recipes" / "tabdiff_report.py"
    hashes[str(recipe)] = sha(recipe)
    records = read_bound_json(root / "benchmark_records.json", hashes)
    variants = CURRENT + PRIOR + (() if smoke else HISTORICAL)
    expected = {(a, s, v) for a in ALGORITHMS for s in seeds for v in variants}
    actual = [(r["algorithm"], r["seed"], r["variant_id"]) for r in records]
    if len(records) != (33 if smoke else 225) or set(actual) != expected or len(actual) != len(set(actual)):
        raise ValueError("Incomplete or duplicate 225/33-record evidence contract")
    flat, diagnostics, quotas, provenance = [], [], [], []
    predictions, selected, masks = {}, {}, {}
    outer_ids = outer_y = None
    audit_fields = ("proposal_count", "schema_valid_count", "eligible_count", "accepted_count",
                    "count_decode_changed_cells", "exact_real_duplicates", "exact_pool_duplicates",
                    "near_duplicates", "sampling_seconds", "filtering_seconds", "diagnostic_seconds",
                    "source_sampling_seconds", "native_fit_seconds", "mask_fraction")
    for record in records:
        prediction_path = Path(record["predictions_path"])
        hashes[str(prediction_path)] = sha(prediction_path)
        frame = pd.read_csv(prediction_path, float_precision="round_trip")
        y = frame["label" if "label" in frame else "y_true"].to_numpy()
        p = frame[["probability_0", "probability_1"]].to_numpy()
        recomputed = metrics(y, p)
        for name, value in recomputed.items():
            if not np.isclose(value, record["metrics"][name], atol=2e-12, rtol=2e-10):
                raise ValueError("Independent metric mismatch: " + record["variant_id"] + ":" + name)
        row = {k: record.get(k) for k in ("algorithm", "seed", "variant_id", "role", "phase")}
        augmentation = record["augmentation"]
        row.update(recomputed, nominal_requested=augmentation["requested_count"],
                   accepted=augmentation["actual_count"], quota_shortfall=augmentation["underspend"],
                   nominal_final_target=augmentation.get("nominal_final_target", augmentation["requested_count"]),
                   historical_label=record.get("historical_label"))
        row["source_run_id"] = record.get("source_run_id")
        row["native_parameters"] = json.dumps(record["config"].get("native_parameters", record["config"].get("source_native_parameters")), sort_keys=True)
        row["sampler_parameters"] = json.dumps(record["config"].get("source_sampler_parameters"), sort_keys=True)
        row["group_alpha"] = record["config"].get("group_alpha")
        row["nominal_target_shortfall"] = row["nominal_final_target"] - row["accepted"]
        row["not_proposed_in_fixed_round"] = row["nominal_final_target"] - row["nominal_requested"]
        if record["variant_id"] in CURRENT:
            if record["config"]["F1_enabled"] is not False:
                raise ValueError("Current F1 must be disabled")
            row["F1_enabled"] = False
            row["mask_columns"] = json.dumps(record["config"]["mask_columns"])
            row["mask_fraction"] = record["config"]["mask_fraction"]
        if record["variant_id"] in CURRENT[:-1]:
            folder = Path(record["trajectory_path"])
            audit = read_bound_json(folder / "audit.json", hashes)
            if audit["F1_enabled"] is not False or audit["sampling_seconds"] != 0:
                raise ValueError("Current replay cannot incur F1 or fresh sampling")
            row.update({name: audit[name] for name in audit_fields})
            row.update(schema_rejected=audit["proposal_count"] - audit["schema_valid_count"],
                       uniform_trimmed=audit["schema_valid_count"] - audit["accepted_count"],
                       schema_failure_counts=json.dumps(audit["schema_failure_counts"], sort_keys=True))
            pool_path = folder / "proposals.npz"
            hashes[str(pool_path)] = sha(pool_path)
            with np.load(pool_path, allow_pickle=False) as pool:
                chosen = pool["selected"].astype(bool)
                if (any("f1" in name.lower() for name in pool.files)
                        or not np.array_equal(pool["eligible"], pool["schema_valid"])
                        or np.any(chosen & ~pool["schema_valid"])):
                    raise ValueError("Only schema-valid eligibility/selection is allowed")
                diagnostic = {"algorithm": record["algorithm"], "seed": record["seed"],
                              "role": record["role"], "arm": record["arm"], "F1_enabled": False,
                              **{name: audit[name] for name in audit_fields},
                              "accepted_schema_invalid": int((chosen & ~pool["schema_valid"]).sum()),
                              "accepted_real_duplicates": int(pool["exact_real_duplicate"][chosen].sum()),
                              "accepted_pool_duplicates": int(pool["exact_pool_duplicate"][chosen].sum()),
                              "accepted_near_duplicates": int(pool["near_duplicate"][chosen].sum()),
                              "accepted_distinct_anchors": int(np.unique(pool["anchor_indices"][chosen]).size),
                              "accepted_distinct_groups": int(np.unique(pool["group_ids"][chosen]).size),
                              "proposed_distinct_anchors": int(np.unique(pool["anchor_indices"]).size),
                              "proposed_distinct_groups": int(np.unique(pool["group_ids"]).size),
                              "schema_failure_counts": row["schema_failure_counts"]}
            diagnostics.append(diagnostic)
        if record["variant_id"] in ("outer_A", "outer_B", "outer_C"):
            if outer_ids is None:
                outer_ids, outer_y = frame.row_id.to_numpy(), y
            if not np.array_equal(outer_ids, frame.row_id.to_numpy()) or not np.array_equal(outer_y, y):
                raise ValueError("Bootstrap requires shared outer rows and labels")
            predictions.setdefault(record["algorithm"], {}).setdefault(record["seed"], {})[record["arm"]] = p[:, 1]
        if record["variant_id"] == "outer_selected_TabDiff":
            selected.setdefault(record["algorithm"], {})[record["seed"]] = record["selected_arm"]
        row["selected_arm"] = record.get("selected_arm")
        row["native_fit_count"] = record.get("native_fit_count", 0)
        flat.append(row)
    names = list(recomputed)
    if len(names) != 24:
        raise ValueError("Durable metric contract must contain 24 metrics")
    index = {(r["algorithm"], r["seed"], r["variant_id"]): r for r in flat}
    paired, descriptive, decisions = [], [], []
    for seed in seeds:
        base = root / "shoppers" / "fold-0" / f"seed-{seed}"
        masks[str(seed)] = read_bound_json(base / "masks.json", hashes)
        for algorithm in ALGORITHMS:
            decision = read_bound_json(base / algorithm / "mask_selection.json", hashes)
            expected_choice = max(("B", "C"), key=lambda arm: (index[algorithm, seed, "inner_" + arm]["average_precision"], arm == "B"))
            if decision["selected_arm"] != expected_choice or selected[algorithm][seed] != expected_choice:
                raise ValueError("Frozen selection must use current no-F1 inner AP only")
            a = index[algorithm, seed, "outer_A"]
            chosen = index[algorithm, seed, "outer_selected_TabDiff"]
            previous = index[algorithm, seed, "prior_f1_selected_TabDiff"]
            for arm in ("B", "C", "selected_TabDiff"):
                target = index[algorithm, seed, "outer_" + arm]
                paired.append({"algorithm": algorithm, "seed": seed,
                               "comparison": ("inner_selected" if arm == "selected_TabDiff" else arm) + " vs A",
                               **{name: target[name] - a[name] for name in names}})
            descriptive.append({"algorithm": algorithm, "seed": seed,
                "comparison": "noF1 inner-selected minus priorF1 inner-selected (descriptive only)",
                "current_accepted": chosen["accepted"], "prior_F1_accepted": previous["accepted"],
                "accepted_delta": chosen["accepted"] - previous["accepted"],
                "current_selected_arm": expected_choice, "prior_F1_selected_arm": previous["selected_arm"],
                **{name: chosen[name] - previous[name] for name in names}})
            decisions.append({"algorithm": algorithm, "seed": seed, "selected_arm": expected_choice,
                              "prior_F1_selected_arm": previous["selected_arm"],
                              **{f"inner_{arm}_AP": index[algorithm, seed, "inner_" + arm]["average_precision"] for arm in ("A", "B", "C")},
                              **{f"outer_{arm}_AP": index[algorithm, seed, "outer_" + arm]["average_precision"] for arm in ("A", "B", "C")},
                              "selected_AP": chosen["average_precision"],
                              "B_minus_A_AP": index[algorithm, seed, "outer_B"]["average_precision"] - a["average_precision"],
                              "C_minus_A_AP": index[algorithm, seed, "outer_C"]["average_precision"] - a["average_precision"],
                              "selected_minus_A_AP": chosen["average_precision"] - a["average_precision"],
                              "accepted": a["accepted"], "quota_shortfall": a["quota_shortfall"]})
            for role in ("inner", "outer"):
                folder = base / algorithm / role
                source_info = read_bound_json(folder / "source.json", hashes)
                for filename, digest in source_info["paths_sha256"].items():
                    if sha(filename) != digest:
                        raise ValueError("Source pointer binding changed: " + filename)
                parent_cell = Path(source_info["parent_cell"])
                control = read_bound_json(parent_cell.parent / "control.json", hashes)
                checkpoint = read_bound_json(Path(source_info["parent_seed"]) / role / "generator" / "training_audit.json", hashes)
                provenance.append({"algorithm": algorithm, "seed": seed, "role": role,
                    "source_json": str(folder / "source.json"), "parent_cell": str(parent_cell),
                    "checkpoint_path": checkpoint["checkpoint_path"], "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "actual_epochs": checkpoint["actual_epochs"], "actual_updates": checkpoint["actual_updates"],
                    "already_paid_generator_training_seconds": checkpoint["training_seconds"],
                    "native_parameters": control["parameters"], "group_alpha": control["group_alpha"],
                    "F1_enabled": False, "generator_retrained": False})
                quota = read_bound_json(folder / "matched_quotas.json", hashes)
                quotas.extend({"algorithm": algorithm, "seed": seed, "role": role, "F1_enabled": False, **q} for q in quota["groups"])
    resources = read_bound_json(root / "resources.json", hashes)
    costs = read_bound_json(root / "invocation_costs.json", hashes)
    fresh_native = sum(row["native_fit_seconds"] for row in flat if row["variant_id"] in CURRENT[:-1])
    unique_checkpoints = {p["checkpoint_sha256"]: p for p in provenance}
    timings = {"fresh_generator_training_seconds": sum(r["generator_training_seconds"] for r in resources),
               "fresh_sampling_seconds": sum(d["sampling_seconds"] for d in diagnostics),
               "fresh_pool_load_seconds": sum(r["pool_load_seconds"] for r in resources),
               "fresh_matching_seconds": sum(r["matching_seconds"] for r in resources),
               "fresh_native_arm_fit_seconds": fresh_native, "fresh_wall_seconds": costs["wall_seconds"],
               "already_paid_source_generator_training_seconds": sum(p["already_paid_generator_training_seconds"] for p in unique_checkpoints.values()),
               "already_paid_source_sampling_seconds": sum(d["source_sampling_seconds"] for d in diagnostics),
               "accounting": "Replay pays only pool loading, schema-only group matching, new native fits and orchestration. Parent generation/sampling was already paid; checkpoint cost deduplicated across classifiers/arms. Selected alias and references add no fits."}
    if timings["fresh_generator_training_seconds"] != 0 or timings["fresh_sampling_seconds"] != 0:
        raise ValueError("Saved-pool replay must not regenerate or sample")
    table = pd.DataFrame(flat)
    paired_table, descriptive_table = pd.DataFrame(paired), pd.DataFrame(descriptive)
    summary = summarize(table, ["algorithm", "variant_id"], names)
    paired_summary = summarize(paired_table, ["algorithm", "comparison"], names)
    descriptive_summary = summarize(descriptive_table, ["algorithm", "comparison"], names)
    bootstrap = ({"status": "not_scientific_smoke_no_inference", "precommitted_draws": BOOTSTRAP_DRAWS,
                  "precommitted_seed": BOOTSTRAP_SEED} if smoke else paired_bootstrap(outer_y, predictions, selected))
    outputs = {"all_24_metrics_csv": output.with_suffix(".metrics.csv"),
               "all_metrics_summary_csv": root / "all_metrics_summary.csv",
               "all_24_paired_deltas_csv": root / "paired_metric_deltas.csv",
               "paired_metrics_summary_csv": root / "paired_metrics_summary.csv",
               "descriptive_prior_F1_deltas_csv": root / "descriptive_prior_f1_deltas.csv",
               "descriptive_prior_F1_summary_csv": root / "descriptive_prior_f1_summary.csv",
               "diagnostics_csv": root / "diagnostics.csv", "group_quotas_csv": root / "group_quotas.csv",
               "per_seed_decisions_csv": root / "per_seed_decisions.csv"}
    result = {"status": "independent_report_completed", "mode": "setup_smoke_only" if smoke else "five_exploratory_matched_seeds_single_fixed_outer_holdout",
              "root": str(root), "verification": str(verification_path), "source_root": str(source),
              "record_count": len(records), "metric_names": names, "F1_enabled": False,
              "per_seed_records": flat, "all_24_metric_means": json_records(summary[["algorithm", "variant_id", *names]]),
              "all_24_metric_means_and_seed_sd": json_records(summary),
              "all_24_paired_outer_deltas": paired, "all_24_paired_summary": json_records(paired_summary),
              "descriptive_noF1_vs_priorF1_selected": descriptive,
              "descriptive_noF1_vs_priorF1_summary": json_records(descriptive_summary),
              "paired_AP_uncertainty": bootstrap, "inner_selected_arms": selected,
              "per_seed_decisions": decisions, "masks_by_seed": masks,
              "diagnostics": diagnostics, "group_quotas": quotas, "source_provenance": provenance,
              "timings": timings, "generator_shared_resources": resources,
              "source_invocation_costs": costs["source_invocation"],
              "actual_input_sha256": hashes, "report_code_sha256": sha(__file__),
              **{k: str(v) for k, v in outputs.items()},
              "historical_context_labels": {k: LABELS[k] for k in HISTORICAL},
              "limitations": ["A is the matched schema-only no-F1 fixed-round interpolation control, NOT the original end-to-end pure-U2 pipeline.",
                  "Pure U2 anchors, quotas, original per-anchor caps and native controls are retained; historical OOF U5 is not relabeled as current U2.",
                  "No relocation, learned rejection, utility ranking, regeneration, refilling or iterative advancement; duplicate checks are diagnostic only.",
                  "No-F1 versus prior-F1 selected differences are descriptive: accepted counts and independently validation-chosen masks may differ.",
                  "One shared outer holdout; five exploratory seeds are not independent folds. Mean AP is not a superiority or retention decision.",
                  "Existing retained U2+F1 and phase-unfrozen state are unchanged."]}
    # Publication gating happens before any outputs: no fabricated or partial root reports.
    if markdown is not None:
        if smoke or smoke_verification is None:
            raise ValueError("Markdown requires full evidence plus verified actual no-F1 smoke")
        smoke_proof = read_bound_json(smoke_verification, hashes)
        smoke_root = Path(smoke_proof["root"])
        smoke_manifest = read_bound_json(smoke_root / "manifest.json", hashes)
        if (smoke_proof["status"] != "verified" or smoke_manifest["status"] != "completed"
                or not smoke_manifest["arguments"]["smoke"] or smoke_manifest["arguments"]["F1_enabled"] is not False):
            raise ValueError("Actual independently verified no-F1 smoke is required")
        for filename, digest in smoke_proof["actual_file_sha256"].items():
            if sha(filename) != digest:
                raise ValueError("Smoke verification binding changed: " + filename)
        result["smoke_verification"] = str(Path(smoke_verification).resolve())
        result["smoke_root"] = str(smoke_root.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    for key, data in (("all_24_metrics_csv", table), ("all_metrics_summary_csv", summary),
                      ("all_24_paired_deltas_csv", paired_table), ("paired_metrics_summary_csv", paired_summary),
                      ("descriptive_prior_F1_deltas_csv", descriptive_table), ("descriptive_prior_F1_summary_csv", descriptive_summary),
                      ("diagnostics_csv", pd.DataFrame(diagnostics)), ("per_seed_decisions_csv", pd.DataFrame(decisions))):
        data.to_csv(outputs[key], index=False)
    pd.json_normalize(quotas).to_csv(outputs["group_quotas_csv"], index=False)
    result["independent_report_json"] = str(output)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if markdown is not None:
        publish(result, markdown)
    print(json.dumps({"output": str(output), "records": len(records), "metrics": len(names),
                      "paired_AP_comparisons": len(bootstrap.get("comparisons", []))}))
    return result


def markdown_table(headers, rows):
    return ["| " + " | ".join(map(str, headers)) + " |", "|" + "---|" * len(headers),
            *["| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in rows], ""]


def publish(result, output):
    """Called only after report() has bound both scientific evidence and smoke."""
    root = Path(result["root"])
    output = Path(output).resolve()
    if result["mode"] == "setup_smoke_only" or "smoke_verification" not in result:
        raise ValueError("Scientific Markdown needs verified actual smoke and full run")
    names = result["metric_names"]
    summary = {(r["algorithm"], r["variant_id"]): r for r in result["all_24_metric_means_and_seed_sd"]}
    def link(path):
        import os
        return os.path.relpath(path, output.parent)
    lines = ["# Shoppers: pure U2 + TabDiff, F1 disabled", "", "## Protocol and evidence", "",
        f"Completed independently verified full replay: `{root}`; 225 records, five matched seeds, LR/RF/LightGBM. Actual no-F1 setup smoke: `{result['smoke_root']}` (33 records; no inference).", "",
        "Only F1 is removed. Current A uses original U2 interpolation; B25/C50 inpaint fixed original-column masks with the already-fitted frozen TabDiff generator. Preprocessing/generator fitting, real U2 anchor schedule, group quotas and original per-anchor caps are invariant to mask strength/selection. Saved complete pre-F1 raw pools and exact coordinates are reused, with schema-only eligibility, per-group minimum counts and uniform thinning. No regeneration, learned rejection, relocation, utility ranking, refill or iterative advancement. Evaluation labels and outer-test data never feed generation or matching; only current inner-validation AP selects B/C.", "",
        "A is the no-F1 schema-only matched fixed-round control, not the original end-to-end pure-U2 pipeline. Native parameters and group alpha lock canonical pure-U2 controls (without historical U1 OOF). Common `np.rint` ties-even count decoding and all existing schema/count-duration checks remain. B/C is newly selected on current no-F1 inner-validation AP (B wins ties), then the frozen chosen outer pool is evaluated; selected adds no fit.", "",
        "Original UGO/pure-U2/standalone-U2+F1 and historical OOF+U2+F1 are imported references with their original end-to-end counts, lineages and tuning, not refits or matched causal controls. Prior TabDiff+F1 A/B/C/selected retain the prior fixed-round run's F1 gates, smaller matched accepted counts and its independently frozen inner choice. Their differences from no-F1 are descriptive only: both counts and selected masks can differ.", "",
        "## Primary paired average precision", "",
        "Nine comparisons use 2,000 draws, seed 20261011. A stratified resample of the same outer row IDs is shared across every seed, classifier and arm; one matched-seed resample per draw is also shared for joint intervals. Inner choices remain frozen. These are conditional fixed-holdout intervals, not independent-fold population superiority evidence.", ""]
    rows = []
    for c in result["paired_AP_uncertainty"]["comparisons"]:
        delta = list(c["matched_seed_deltas"].values())
        rows.append([ALGORITHMS[c["algorithm"]], c["comparison"], f'{c["mean_delta"]:+.6f}',
                     f'{sum(v > 0 for v in delta)}/5',
                     *[f'[{c[k][0]:+.6f}, {c[k][1]:+.6f}]' for k in
                       ("paired_stratified_shared_row_percentile_CI95", "shared_row_and_matched_seed_joint_percentile_CI95")]])
    lines += markdown_table(["Classifier", "Comparison", "Mean ΔAP", "Positive seeds", "Shared-row 95% CI", "Shared-row + matched-seed 95% CI"], rows)
    lines += ["## All 24 metrics: mean ± sample seed SD", "",
              "SD uses five matched exploratory seeds (ddof=1), not independent holdout replications. Threshold is 0.5; class 1 is minority. ROC recall uses the best observed ROC point at FPR ≤0.05, not a tuned operating threshold.", ""]
    for a, label in ALGORITHMS.items():
        lines += [f"### {label}", ""]
        for heading, variants in (("Current outer arms", CURRENT[3:]), ("Inner validation", CURRENT[:3]),
                                  ("Prior F1 fixed-round context", PRIOR), ("Original end-to-end context", HISTORICAL)):
            lines += [f"#### {heading}", ""]
            lines += markdown_table(["Metric", *[LABELS[v] for v in variants]],
                [[name, *[f'{summary[a, v][name]:.6f} ± {summary[a, v][name + "_seed_sd"]:.6f}' for v in variants]] for name in names])
    lines += ["## Context counts and tuning (not matched causal comparisons)", ""]
    counts = pd.DataFrame(result["per_seed_records"])
    rows = []
    for (a, variant), part in counts[counts.variant_id.isin(CURRENT[3:] + PRIOR + HISTORICAL)].groupby(["algorithm", "variant_id"]):
        rows.append([ALGORITHMS[a], LABELS[variant], f'{part.accepted.mean():.1f} ± {part.accepted.std(ddof=1):.1f}',
                     ", ".join(str(int(v)) for v in part.sort_values("seed").accepted),
                     "; ".join(sorted(set(part.source_run_id.dropna().astype(str)))) or "prior fixed-round source / replay",
                     "; ".join(sorted(set(part.native_parameters.astype(str))))])
    lines += markdown_table(["Classifier", "Context", "Accepted mean ± seed SD", "Accepted seeds 0–4", "Source lineage", "Native configurations"], rows)
    lines += ["Source sampler/group/F1 tuning is retained in the independently recomputed metrics CSV and locked control audit; differing original tuning/counting is not normalized away.", ""]
    lines += ["## Per-seed AP, paired deltas and frozen masks", ""]
    lines += markdown_table(["Classifier", "Seed", "Current / prior F1 choice", "Inner A/B/C AP", "Outer A/B/C AP", "Selected AP", "B−A / C−A / selected−A", "Accepted / round shortfall"],
        [[ALGORITHMS[d["algorithm"]], d["seed"], d["selected_arm"] + " / " + d["prior_F1_selected_arm"],
          " / ".join(f'{d[f"inner_{arm}_AP"]:.6f}' for arm in ("A", "B", "C")),
          " / ".join(f'{d[f"outer_{arm}_AP"]:.6f}' for arm in ("A", "B", "C")), f'{d["selected_AP"]:.6f}',
          " / ".join(f'{d[k]:+.6f}' for k in ("B_minus_A_AP", "C_minus_A_AP", "selected_minus_A_AP")),
          f'{d["accepted"]} / {d["quota_shortfall"]}'] for d in result["per_seed_decisions"]])
    lines += markdown_table(["Seed", "B original columns", "C original columns"],
        [[seed, *[", ".join(mask["masks"][arm]) for arm in ("B", "C")]] for seed, mask in result["masks_by_seed"].items()])
    lines += ["## Descriptive no-F1 versus prior-F1 inner-selected", "",
              "No inferential intervals or matched-count F1-ablation claim: counts and masks differ. All 24 descriptive deltas and mean/SD are linked below.", ""]
    lines += markdown_table(["Classifier", "Seed", "Current / prior accepted", "Current / prior mask", "ΔAP", "Δminority F1", "ΔMCC", "Δordinary CE", "Δbalanced CE"],
        [[ALGORITHMS[d["algorithm"]], d["seed"], f'{d["current_accepted"]} / {d["prior_F1_accepted"]}',
          d["current_selected_arm"] + " / " + d["prior_F1_selected_arm"],
          *[f'{d[k]:+.6f}' for k in ("average_precision", "minority_f1", "mcc", "ordinary_cross_entropy", "balanced_cross_entropy")]]
         for d in result["descriptive_noF1_vs_priorF1_selected"]])
    lines += ["## Schema, accepted counts, duplicates and coverage", "",
              "F1 is disabled in every current cell. Finite/nonnegative values, decoded integer counts, bounded rates/SpecialDay, observed training categories and zero-count/zero-duration consistency remain enforced. No empirical maximum or BounceRates≤ExitRates rule is added. Exact/near duplicates remain diagnostic-only (near: classifier-standardized Euclidean distance ≤0.001); selected diagnostics are recomputed from saved selected masks, not inherited prior F1 selections. Constraint failure counts overlap.", ""]
    diagnostic = pd.DataFrame(result["diagnostics"])
    columns = ("proposal_count", "schema_valid_count", "accepted_count", "accepted_schema_invalid",
               "accepted_real_duplicates", "accepted_pool_duplicates", "accepted_near_duplicates")
    lines += markdown_table(["Classifier", "Outer arm", "Proposals", "Schema eligible", "Accepted", "Accepted invalid", "Exact real dup", "Exact pool dup", "Near dup"],
        [[ALGORITHMS[a], arm, *[int(part[k].sum()) for k in columns]]
         for (a, arm), part in diagnostic[diagnostic.role == "outer"].groupby(["algorithm", "arm"])])
    lines += ["Every seed/classifier/role/group's original quota, A/B/C availability, matched accepted count and shortfall is in group_quotas.csv; per-arm accepted/proposed anchor and group coverage and all invalid/duplicate/near diagnostics are in diagnostics.csv.", "",
              "## Frozen checkpoints and source provenance", "",
              f"Parent root: `{result['source_root']}`. Each current `source.json` SHA-binds original real/preprocessor/groups/schedule, split/masks, control, generator checkpoint/audit and every complete A/B/C proposals/raw/decoded/generation/audit input. Parent raw pools and checkpoints are never changed. Source pointers, not copied or refitted source parents, supply shared preprocessing and scheduling.", ""]
    unique = {(p["seed"], p["role"]): p for p in result["source_provenance"]}
    lines += markdown_table(["Seed", "Role", "Checkpoint SHA256", "Epochs", "Updates", "Already-paid fit seconds"],
        [[seed, role, p["checkpoint_sha256"], p["actual_epochs"], p["actual_updates"], f'{p["already_paid_generator_training_seconds"]:.3f}'] for (seed, role), p in sorted(unique.items())])
    controls = {(p["algorithm"], p["seed"]): p for p in result["source_provenance"]}
    lines += markdown_table(["Classifier", "Seed", "Canonical native parameters", "U2 group alpha", "F1 enabled"],
        [[ALGORITHMS[a], seed, json.dumps(p["native_parameters"], sort_keys=True), p["group_alpha"], "false"] for (a, seed), p in sorted(controls.items())])
    lines += ["## Incremental runtime versus already-paid source costs", "", result["timings"]["accounting"], ""]
    lines += markdown_table(["Component", "Observed seconds"],
        [[k, f'{v:.3f}'] for k, v in result["timings"].items() if isinstance(v, (float, int))])
    lines += ["## Decision boundary and durable evidence", "",
              "Mean AP does not decide superiority. Five seeds share training and one realized holdout. No automatic retention or phase transition is performed: current retained U2+F1 and phase-unfrozen state are preserved.", ""]
    for key in ("all_24_metrics_csv", "all_metrics_summary_csv", "all_24_paired_deltas_csv",
                "paired_metrics_summary_csv", "descriptive_prior_F1_deltas_csv", "descriptive_prior_F1_summary_csv",
                "per_seed_decisions_csv", "diagnostics_csv", "group_quotas_csv"):
        lines.append(f'- [{Path(result[key]).name}]({link(result[key])}): {key.replace("_", " ")}.')
    lines += [f'- [Independent verification]({link(result["verification"])}): actual completed-run evidence.',
              f'- [Actual no-F1 smoke verification]({link(result["smoke_verification"])}).',
              f'- [Independent report JSON]({link(result["independent_report_json"])}): all metrics, conditional uncertainty and SHA-bound provenance.', ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_positional", nargs="?")
    parser.add_argument("--root")
    parser.add_argument("--verification", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prior-root", help="Actual prior F1 source root (must match manifest)")
    parser.add_argument("--markdown", help="Optional scientific Markdown output, after actual verified smoke")
    parser.add_argument("--smoke-verification", help="Required with --markdown")
    args = parser.parse_args(argv)
    root = args.root or args.root_positional
    if not root or (args.root and args.root_positional and Path(args.root).resolve() != Path(args.root_positional).resolve()):
        parser.error("Supply one unambiguous --root or positional root")
    report(root, args.verification, args.output, args.prior_root, args.markdown, args.smoke_verification)


if __name__ == "__main__":
    main()
