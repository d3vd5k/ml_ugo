"""Report completed independently verified native synthetic-downweight trials."""
import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef, roc_curve)

from benchmarks.tabdiff_u2_report import json_records, markdown_table, sha, summarize

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261012
ALGORITHMS = {"logistic_regression": "LR", "random_forest": "RF",
              "lightgbm": "LightGBM", "mlp": "Native MLP"}
FAMILIES = ("U2_F1", "U2", "pool_random")
WEIGHTS = (.25, .5, 1.)


def native_metrics(y, probabilities):
    """The durable independent 24-metric recipe with native CE input precision."""
    y = np.asarray(y, dtype=int)
    raw = np.asarray(probabilities)
    p = raw.astype(np.float64, copy=False)
    if (p.shape != (len(y), 2) or not np.isfinite(p).all()
            or not np.all((p >= 0) & (p <= 1)) or not np.allclose(p.sum(axis=1), 1)
            or set(np.unique(y)) != {0, 1}):
        raise ValueError("Metrics require finite binary probabilities and both classes")
    pred = (p[:, 1] >= .5).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    f = f1_score(y, pred, labels=[0, 1], average=None, zero_division=0)
    prevalence = np.array([(y == k).mean() for k in (0, 1)])
    fpr, tpr, _ = roc_curve(y, p[:, 1], drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= .05)
    best = eligible[np.argmax(tpr[eligible])]
    weights = np.where(y == 0, 1 / (2 * prevalence[0]), 1 / (2 * prevalence[1]))
    # Native threshold/ROC formulas cast to double; CE uses the original dtype
    # so float32 MLP clipping is preserved rather than silently reinterpreted.
    return dict(
        average_precision=float(average_precision_score(y, p[:, 1])),
        macro_average_precision=float(np.mean([average_precision_score(y == k, p[:, k]) for k in (0, 1)])),
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
        ordinary_cross_entropy=float(log_loss(y, raw, labels=[0, 1])),
        balanced_cross_entropy=float(log_loss(y, raw, labels=[0, 1], sample_weight=weights)))


def paired_bootstrap(y, predictions, comparisons, draws=BOOTSTRAP_DRAWS,
                     seed=BOOTSTRAP_SEED):
    """Freeze choices and share stratified test-row and matched-seed draws.

    predictions[algorithm][seed][variant] holds positive-class probabilities.
    Each comparison names an algorithm, comparison label, left and right variant.
    Shared-row intervals condition on the observed seeds; joint intervals also
    resample those matched seeds. Neither treats seeds as independent holdouts.
    """
    y = np.asarray(y, dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Bootstrap requires both test classes")
    algorithms = sorted(predictions)
    seeds = sorted(predictions[algorithms[0]])
    if not seeds or any(sorted(predictions[a]) != seeds for a in algorithms):
        raise ValueError("Bootstrap requires matched seeds across classifiers")
    required = {a: set() for a in algorithms}
    samples, estimates = {}, {}
    for comparison in comparisons:
        a = comparison["algorithm"]
        key = (a, comparison["comparison"])
        if key in samples:
            raise ValueError("Duplicate bootstrap comparison")
        required[a].update((comparison["left"], comparison["right"]))
        samples[key] = {"row": np.empty(draws), "joint": np.empty(draws)}
    for a in algorithms:
        for s in seeds:
            for variant in required[a]:
                p = np.asarray(predictions[a][s][variant], dtype=float)
                if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                    raise ValueError("Bootstrap predictions must align with shared test rows")
    observed = {a: {v: np.array([average_precision_score(y, predictions[a][s][v])
                               for s in seeds]) for v in sorted(required[a])}
                for a in algorithms}
    for comparison in comparisons:
        a = comparison["algorithm"]
        estimates[a, comparison["comparison"]] = (
            observed[a][comparison["left"]] - observed[a][comparison["right"]])
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(y == label) for label in (0, 1)]
    for draw in range(draws):
        row_sample = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in strata])
        weights = np.bincount(row_sample, minlength=len(y))
        seed_sample = rng.integers(len(seeds), size=len(seeds))
        ap = {a: {v: np.array([average_precision_score(y, predictions[a][s][v],
                                                      sample_weight=weights)
                              for s in seeds]) for v in sorted(required[a])}
              for a in algorithms}
        for comparison in comparisons:
            a = comparison["algorithm"]
            delta = ap[a][comparison["left"]] - ap[a][comparison["right"]]
            values = samples[a, comparison["comparison"]]
            values["row"][draw] = delta.mean()
            values["joint"][draw] = delta[seed_sample].mean()
    return {"draws": draws, "precommitted_seed": seed,
            "interpretation": "Conditional on shared training and one realized outer holdout, frozen inner choices and exploratory matched seeds. Not independent-fold intervals, population superiority evidence or an automatic retention decision.",
            "comparisons": [{"algorithm": c["algorithm"], "comparison": c["comparison"],
                "left": c["left"], "right": c["right"],
                "matched_seed_deltas": dict(zip(map(str, seeds), estimates[c["algorithm"], c["comparison"]].tolist())),
                "mean_delta": float(estimates[c["algorithm"], c["comparison"]].mean()),
                "paired_stratified_shared_row_percentile_CI95": np.quantile(samples[c["algorithm"], c["comparison"]]["row"], [.025, .975]).tolist(),
                "shared_row_and_matched_seed_joint_percentile_CI95": np.quantile(samples[c["algorithm"], c["comparison"]]["joint"], [.025, .975]).tolist()}
                for c in comparisons]}


def read_bound_json(path, hashes):
    path = Path(path).resolve()
    hashes[str(path)] = sha(path)
    return json.loads(path.read_text())


def verified_root(root, verification, hashes):
    """Reject incomplete roots and any stale independently verified input."""
    root, verification = Path(root).resolve(), Path(verification).resolve()
    proof = read_bound_json(verification, hashes)
    manifest = read_bound_json(root / "manifest.json", hashes)
    if (proof["status"] != "verified" or Path(proof["root"]).resolve() != root
            or manifest["status"] != "completed"):
        raise ValueError("Only independently verified completed actual roots can be reported")
    bound = proof["actual_file_sha256"]
    if not bound:
        raise ValueError("Independent verification must bind actual inputs")
    for filename, digest in bound.items():
        path = Path(filename).resolve()
        if sha(path) != digest:
            raise ValueError("Verification binding changed: " + str(path))
        hashes[str(path)] = digest
    for path in (root / "manifest.json", root / "benchmark_records.json"):
        if str(path) not in hashes or str(path) not in bound:
            raise ValueError("Independent verification does not bind " + str(path))
    return manifest, proof


def evidence_json(path, hashes):
    path = Path(path).resolve()
    if str(path) not in hashes or sha(path) != hashes[str(path)]:
        raise ValueError("Artifact is not bound by independent verification: " + str(path))
    return json.loads(path.read_text())


def report(root, verification_path, output, markdown=None, smoke_verification=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    hashes = {}
    manifest, proof = verified_root(root, verification_path, hashes)
    for package in ("numpy", "scikit-learn"):
        if version(package) != manifest["dependencies"][package]:
            raise ValueError("Use the actual native experiment environment: " + package
                             + " version differs from the completed root")
    smoke = manifest["arguments"]["smoke"]
    seeds = [0] if smoke else list(range(5))
    arguments = manifest["arguments"]
    if (set(arguments["effective_classifiers"]) != set(ALGORITHMS)
            or arguments["effective_seeds"] != seeds
            or arguments["bootstrap_draws"] != BOOTSTRAP_DRAWS
            or arguments["bootstrap_seed"] != BOOTSTRAP_SEED
            or arguments["bagging_deferred"] is not True
            or arguments["no_automatic_retention"] is not True):
        raise ValueError("Root does not preserve the declared comparison/precommitment")
    smoke_evidence = {}
    if markdown is not None:
        if smoke or smoke_verification is None:
            raise ValueError("Scientific Markdown requires full run plus actual independently verified smoke")
        smoke_proof_path = Path(smoke_verification).resolve()
        smoke_proof = read_bound_json(smoke_proof_path, hashes)
        smoke_root = Path(smoke_proof["root"]).resolve()
        smoke_manifest, _ = verified_root(smoke_root, smoke_proof_path, hashes)
        smoke_records = evidence_json(smoke_root / "benchmark_records.json", hashes)
        if (not smoke_manifest["arguments"]["smoke"] or len(smoke_records) != 80
                or {r["algorithm"] for r in smoke_records} != set(ALGORITHMS)
                or {r["seed"] for r in smoke_records} != {0}):
            raise ValueError("Actual seed-0 smoke must cover all four native classifiers")
        smoke_evidence = {"smoke_root": str(smoke_root), "smoke_verification": str(smoke_proof_path)}
    records = evidence_json(root / "benchmark_records.json", hashes)
    expected = {"inner_real_only", "outer_real_only"}
    for family in FAMILIES:
        expected.update(f"inner_{family}_w{weight:g}" for weight in WEIGHTS)
        expected.update(f"outer_{family}_{arm}" for arm in ("unit", "selected", "parent"))
    index, rows, audits, native_fits = {}, [], [], []
    predictions, outer_ids, outer_y = {}, None, None
    inner_rows = {}
    for record in records:
        algorithm, seed, variant = record["algorithm"], record["seed"], record["variant_id"]
        key = (algorithm, seed, variant)
        if algorithm not in ALGORITHMS or seed not in seeds or key in index or variant not in expected:
            raise ValueError("Unexpected or duplicate cell: " + str(key))
        if record["status"] != "completed":
            raise ValueError("Incomplete record: " + str(key))
        path = Path(record["predictions_path"]).resolve()
        if str(path) not in hashes or sha(path) != hashes[str(path)]:
            raise ValueError("Predictions not independently bound: " + str(path))
        frame = pd.read_csv(path, float_precision="round_trip")
        ids, y = frame["row_id"].to_numpy(), frame["label"].to_numpy(dtype=int)
        # Pandas float32 CSV formatting round-trips only after dtype restoration.
        p = frame[["probability_0", "probability_1"]].to_numpy(
            dtype=np.float32 if algorithm == "mlp" else np.float64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Repeated evaluation row IDs")
        values = native_metrics(y, p)
        numeric_names = {name for name, value in record["metrics"].items()
                         if isinstance(value, (int, float)) and name != "primary_metric_value"}
        if len(values) != 24 or set(values) != numeric_names:
            raise ValueError("Expected exactly the canonical 24 metrics")
        if any(not np.isclose(values[name], record["metrics"][name], rtol=1e-10, atol=1e-12)
               for name in values):
            raise ValueError("Saved metrics disagree with saved predictions")
        if record["role"] == "outer":
            if outer_ids is None:
                outer_ids, outer_y = ids, y
            elif not np.array_equal(ids, outer_ids) or not np.array_equal(y, outer_y):
                raise ValueError("All outer arms must share ordered test row IDs and labels")
            predictions.setdefault(algorithm, {}).setdefault(seed, {})[variant] = p[:, 1]
        elif record["role"] == "inner":
            prior = inner_rows.setdefault(seed, (ids, y))
            if not np.array_equal(ids, prior[0]) or not np.array_equal(y, prior[1]):
                raise ValueError("Inner arms must share validation row IDs and labels per seed")
        else:
            raise ValueError("Unknown record role")
        folder = Path(record["trajectory_path"]).resolve()
        audit = evidence_json(folder / "audit.json", hashes)
        accepted = record["augmentation"]["actual_count"]
        weight = record["synthetic_weight"]
        if (audit["accepted_count"] != accepted
                or not np.isclose(audit["added_weight"], weight * accepted)
                or not np.isclose(record["augmentation"]["added_weight"], weight * accepted)):
            raise ValueError("Accepted counts/added weights disagree across record and actual audit")
        row = {"algorithm": algorithm, "seed": seed, "variant_id": variant,
               "role": record["role"], "phase": record["phase"], "family": record["family"],
               "synthetic_weight": weight, "accepted": accepted,
               "added_weight": weight * accepted,
               "rounds": 0 if record["phase"] == "parent_duplication" else len(audit["rounds"]),
               "stop_reason": "parent_duplication" if record["phase"] == "parent_duplication" else audit["stop_reason"],
               "native_fit_count": record["native_fit_count"],
               "native_fit_seconds": record["native_fit_seconds"],
               "runtime_seconds": record["runtime_seconds"],
               "trajectory_path": str(folder), **values}
        rows.append(row)
        index[key] = row
        audits.append({"algorithm": algorithm, "seed": seed, "variant_id": variant,
                       "audit_path": str(folder / "audit.json"),
                       "audit_sha256": hashes[str(folder / "audit.json")]})
        if algorithm == "mlp":
            if record["phase"] == "parent_duplication":
                native_fits.append({"seed": seed, "variant_id": variant,
                    "fit_path": str(folder / "audit.json"), "fit_kind": "parent_duplication",
                    "architecture": audit["architecture"], "actual_epochs": audit["actual_epochs"],
                    "device": audit["device"], "parameters": audit["native_parameters"]})
            else:
                fits = sorted((folder / "fits").glob("fit-*.json"))
                if len(fits) != record["native_fit_count"]:
                    raise ValueError("Native MLP fit evidence count mismatch")
                for fit_path in fits:
                    fit = evidence_json(fit_path, hashes)
                    native_fits.append({"seed": seed, "variant_id": variant,
                                        "fit_path": str(fit_path), **fit})
    if set(index) != {(a, seed, variant) for a in ALGORITHMS for seed in seeds for variant in expected}:
        raise ValueError("Root does not contain every required matched classifier/seed/arm")
    decisions, controls, divergence = [], [], []
    for a in ALGORITHMS:
        for seed in seeds:
            cell = Path(index[a, seed, "inner_real_only"]["trajectory_path"]).parents[1]
            selection = evidence_json(cell / "selection.json", hashes)
            control = evidence_json(cell / "control.json", hashes)
            comparison = evidence_json(cell / "comparison.json", hashes)
            controls.append({"algorithm": a, "seed": seed, "selection": selection, "control": control})
            divergence.append({"algorithm": a, "seed": seed, "comparison": comparison})
            for family in FAMILIES:
                inner = {w: index[a, seed, f"inner_{family}_w{w:g}"] for w in WEIGHTS}
                selected_weight = max((.25, .5), key=lambda w: (inner[w]["average_precision"], w))
                selected = index[a, seed, f"outer_{family}_selected"]
                unit = index[a, seed, f"outer_{family}_unit"]
                real = index[a, seed, "outer_real_only"]
                parent = index[a, seed, f"outer_{family}_parent"]
                frozen = selection["families"][family]
                if (selection["no_test_selection"] is not True
                        or selection["all_inner_selections_before_outer"] is not True
                        or frozen["no_test_selection"] is not True
                        or frozen["selected_weight"] != selected_weight
                        or selected["synthetic_weight"] != selected_weight
                        or unit["synthetic_weight"] != 1):
                    raise ValueError("Outer coefficients do not match inner-only weighted selection/unit control")
                trials = {trial["weight"]: trial for trial in frozen["trials"]}
                if set(trials) != set(WEIGHTS) or any(
                        not np.isclose(trials[w]["AP"], inner[w]["average_precision"], rtol=1e-10, atol=1e-12)
                        or Path(trials[w]["path"]).resolve() != Path(inner[w]["trajectory_path"])
                        for w in WEIGHTS):
                    raise ValueError("Saved selection trials disagree with inner real-validation evidence")
                decisions.append({"algorithm": a, "seed": seed, "family": family,
                    "selected_weight": selected_weight,
                    "inner_w0.25_AP": inner[.25]["average_precision"],
                    "inner_w0.5_AP": inner[.5]["average_precision"],
                    "inner_w1_AP": inner[1.]["average_precision"],
                    "outer_selected_AP": selected["average_precision"],
                    "outer_unit_AP": unit["average_precision"],
                    "outer_real_only_AP": real["average_precision"],
                    "outer_parent_AP": parent["average_precision"],
                    "selected_minus_unit_AP": selected["average_precision"] - unit["average_precision"],
                    "selected_minus_real_only_AP": selected["average_precision"] - real["average_precision"],
                    "unit_minus_real_only_AP": unit["average_precision"] - real["average_precision"],
                    "selected_accepted": selected["accepted"], "unit_accepted": unit["accepted"],
                    "selected_added_weight": selected["added_weight"], "unit_added_weight": unit["added_weight"],
                    "selected_rounds": selected["rounds"], "unit_rounds": unit["rounds"]})
    resources = {row["trajectory_path"]: {"algorithm": row["algorithm"], "seed": row["seed"],
                 "purpose": row["variant_id"], "trajectory_path": row["trajectory_path"],
                 **{key: row[key] for key in ("native_fit_count", "native_fit_seconds", "runtime_seconds")}}
                 for row in rows}
    flows = []
    for seed in seeds:
        cell = Path(index["mlp", seed, "inner_real_only"]["trajectory_path"]).parents[1]
        canonical = evidence_json(cell / "canonical" / "selection.json", hashes)
        tau_search = evidence_json(cell / "tau" / "selection.json", hashes)
        for purpose, paths in (("canonical_MLP_U2_search", [t["trajectory_path"] for t in canonical["trials"]]),
                               ("MLP_unit_F1_tau_search", [cell / "tau" / f"tau-{t['tau']:g}" for t in tau_search["trials"]])):
            for trajectory in paths:
                trajectory = str(Path(trajectory).resolve())
                if trajectory in resources:
                    resources[trajectory]["purpose"] += "; " + purpose + " (reused)"
                    continue
                audit = evidence_json(Path(trajectory) / "audit.json", hashes)
                resources[trajectory] = {"algorithm": "mlp", "seed": seed, "purpose": purpose,
                    "trajectory_path": trajectory,
                    **{key: audit[key] for key in ("native_fit_count", "native_fit_seconds", "runtime_seconds")}}
        flows.append({"seed": seed, **evidence_json(cell.parent / "flow.json", hashes)})
    resource_frame = pd.DataFrame(resources.values())
    resource_totals = resource_frame.groupby(["algorithm", "seed"], sort=True)[
        ["native_fit_count", "native_fit_seconds", "runtime_seconds"]].sum().reset_index()
    names = list(values)
    frame = pd.DataFrame(rows)
    grouping = ["algorithm", "role", "variant_id"]
    summary = summarize(frame, grouping, names)
    trajectory_names = ["accepted", "added_weight", "rounds", "native_fit_count", "native_fit_seconds", "runtime_seconds"]
    trajectory_summary = summarize(frame, grouping, trajectory_names)
    comparisons, deltas = [], []
    for a in ALGORITHMS:
        for family in FAMILIES:
            for left, right in ((f"outer_{family}_selected", f"outer_{family}_unit"),
                                (f"outer_{family}_selected", "outer_real_only"),
                                (f"outer_{family}_unit", "outer_real_only"),
                                (f"outer_{family}_selected", f"outer_{family}_parent")):
                label = left + " vs " + right
                comparisons.append({"algorithm": a, "comparison": label, "left": left, "right": right})
                for seed in seeds:
                    l, r = index[a, seed, left], index[a, seed, right]
                    deltas.append({"algorithm": a, "seed": seed, "comparison": label,
                                   "left": left, "right": right,
                                   **{name: l[name] - r[name] for name in names}})
    delta_frame = pd.DataFrame(deltas)
    delta_summary = summarize(delta_frame, ["algorithm", "comparison"], names)
    uncertainty = ({"status": "not_scientific_smoke_no_inference", "precommitted_draws": BOOTSTRAP_DRAWS,
                    "precommitted_seed": BOOTSTRAP_SEED} if smoke else
                   paired_bootstrap(outer_y, predictions, comparisons))
    result = {"root": str(root), "verification": str(Path(verification_path).resolve()),
              "mode": "setup_smoke_only" if smoke else "five_matched_seed_scientific_run",
              "metric_names": names, "seeds": seeds, "record_count": len(records),
              "per_seed_records": json_records(frame),
              "all_24_metric_means_and_seed_sd": json_records(summary),
              "all_24_paired_metric_deltas": json_records(delta_frame),
              "paired_metric_means_and_seed_sd": json_records(delta_summary),
              "paired_AP_uncertainty": uncertainty, "per_seed_decisions": decisions,
              "trajectory_means_and_seed_sd": json_records(trajectory_summary),
              "trajectory_audits": audits, "trajectory_divergence": divergence,
              "native_MLP_fits": native_fits, "canonical_controls_and_selection": controls,
              "physical_native_resources": json_records(resource_frame),
              "native_fit_runtime_totals_by_seed": json_records(resource_totals),
              "shared_frozen_flow": flows,
              "bagging": "not_run", "retention": "pending_no_automatic_retention_or_phasefreeze",
              "independent_report_json": str(output)}
    result.update(smoke_evidence)
    tables = {"all_24_metrics_csv": frame, "all_metrics_summary_csv": summary,
              "all_24_paired_deltas_csv": delta_frame, "paired_metrics_summary_csv": delta_summary,
              "per_seed_decisions_csv": pd.DataFrame(decisions),
              "trajectory_summary_csv": trajectory_summary,
              "trajectory_divergence_csv": pd.json_normalize(divergence),
              "native_MLP_fits_csv": pd.json_normalize(native_fits),
              "physical_native_resources_csv": resource_frame,
              "native_fit_runtime_totals_csv": resource_totals}
    output.parent.mkdir(parents=True, exist_ok=True)
    for key, table in tables.items():
        path = output.with_name(output.stem + "." + key.removesuffix("_csv") + ".csv")
        table.to_csv(path, index=False)
        result[key] = str(path)
    result["input_sha256"] = hashes
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if markdown is not None:
        publish(result, markdown)
    print(json.dumps({"output": str(output), "records": len(records), "metrics": len(names),
                      "paired_AP_comparisons": len(uncertainty.get("comparisons", []))}))
    return result



def publish(result, output):
    """Publish only full, independently bound run plus seed-0 smoke evidence."""
    if result["mode"] != "five_matched_seed_scientific_run" or "smoke_verification" not in result:
        raise ValueError("Scientific Markdown requires independently verified full and smoke roots")
    output = Path(output).resolve()
    def link(path):
        return os.path.relpath(path, output.parent)
    def cell(mean, sd):
        return f"{mean:.6f} ± {sd:.6f}" if sd is not None else f"{mean:.6f} ± undefined"
    def compact(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    names = result["metric_names"]
    summary = {(r["algorithm"], r["variant_id"]): r for r in result["all_24_metric_means_and_seed_sd"]}
    lines = ["# Shoppers: native synthetic-only downweighting", "", "## Scope and verified evidence", "",
        f"Completed independently verified root: \u0060{result['root']}\u0060 ({result['record_count']} records; seeds 0–4; LR, RF, LightGBM and native MLP). Independently verified seed-0 setup smoke: \u0060{result['smoke_root']}\u0060 (all four classifiers; no inference).", "",
        "Real rows retain weight 1; only accepted synthetic rows receive 0.25 or 0.5. Corresponding unit-weight 1 trajectories and independently fitted real-only controls remain distinct. Weighted selection uses only current real inner-validation AP (higher synthetic weight wins exact ties), never test AP. Every family/classifier/seed freezes that inner choice before outer testing.", "",
        "Families are current pure U2+F1 (U2_F1), U2 without F1 (U2), and U2 plus a K=5 random candidate pool without F1 (pool_random). Native model scoring is repeated inside the iterative trajectory, retaining quota caps, generation/relocation, no refill, Welch stopping and the 20-round limit. Reweighting changes fitted native scores and can change later proposals, accepted rows, counts and stopping; these are not fixed-pool or tuple-count-matched ablations. No historical OOF U5 semantics or historical mean substitutions are used.", "",
        "Each classifier preprocessor fits only its role's real training rows, never synthetic or test rows. The outer role includes former inner-validation rows in real training only after all choices are frozen; saved outer validation metrics overlap training and are diagnostic, not new selection or holdout evidence. The published outer metrics use the untouched shared test rows.", "",
        "LR/RF/LightGBM freeze canonical pure-U2 native parameters and group alpha from the SHA-bound tabdiff_controls.json audit. F1 freezes the original retained-fit 80% real inner-fit core and leaves its calibration 20% unused. The exact audited f1_tau and f1_tau_source are retained: the existing tau originated in the historical OOF+U2+F1 source search, not a new pure-F1 reselection. New generation and native scoring use pure U2 without OOF. Native MLP obtains its own canonical U2 parameters/group alpha from the same joint-search budget; its F1 tau is selected only on inner unit-weight AP and then locked. Full native MLP fits use fixed epochs rather than holdout training or early stopping. Exact locked controls and source hashes are retained in the report JSON.", "",
        "Parent duplication is an additional diagnostic control using each selected trajectory's final accepted batch; it is not bagging, a new generator, or a matched trajectory.", "",
        "## Paired outer-test average precision", "",
        "2,000 bootstrap draws, seed 20261012. Each stratified resample of the shared ordered test rows is reused across every classifier, matched seed and arm. One matched-seed resample per draw is also shared for joint intervals. Inner choices stay frozen. Shared-row intervals condition on the observed seeds; joint intervals additionally resample those matched seeds. These are conditional fixed-holdout intervals, not independent-fold population superiority evidence.", ""]
    ci_rows = []
    for comparison in result["paired_AP_uncertainty"]["comparisons"]:
        deltas = list(comparison["matched_seed_deltas"].values())
        ci_rows.append([ALGORITHMS[comparison["algorithm"]], comparison["comparison"],
                       f"{comparison['mean_delta']:+.6f}", f"{sum(d > 0 for d in deltas)}/{len(deltas)}",
                       *[f"[{comparison[key][0]:+.6f}, {comparison[key][1]:+.6f}]" for key in
                         ("paired_stratified_shared_row_percentile_CI95",
                          "shared_row_and_matched_seed_joint_percentile_CI95")]])
    lines += markdown_table(["Classifier", "Comparison", "Mean ΔAP", "Positive matched seeds",
                             "Shared-row 95% CI", "Shared-row + matched-seed 95% CI"], ci_rows)
    lines += ["## All 24 metrics: mean ± sample seed SD", "",
        "All metrics are independently recomputed from saved probabilities in the recorded NumPy/scikit-learn environment. MLP CSV values are restored to their native float32 precision, including the original float32 cross-entropy clipping; other native classifiers retain float64. SD uses ddof=1 across five matched exploratory seeds sharing the same holdout, not independent fold replications. The threshold is 0.5, class 1 is minority, and ROC recall uses the best observed point at FPR ≤0.05. The linked all-24-metrics CSV contains every per-seed inner/outer/parent record; the paired CSV contains every per-seed delta for all 24 metrics.", ""]
    for algorithm, label in ALGORITHMS.items():
        lines += [f"### {label}", ""]
        for family in FAMILIES:
            for role, variants in (("Outer test", [f"outer_{family}_{arm}" for arm in ("unit", "selected", "parent")] + ["outer_real_only"]),
                                   ("Inner real validation", [f"inner_{family}_w{w:g}" for w in WEIGHTS] + ["inner_real_only"])):
                lines += [f"#### {family}: {role}", ""]
                lines += markdown_table(["Metric", *variants],
                    [[name, *[cell(summary[algorithm, variant][name], summary[algorithm, variant][name + "_seed_sd"])
                              for variant in variants]] for name in names])
    lines += ["## Locked native controls and F1 provenance", ""]
    lines += markdown_table(["Classifier", "Seed", "Native parameters", "U2 group alpha", "F1 tau", "Exact tau source"],
        [[ALGORITHMS[d["algorithm"]], d["seed"], compact(d["control"]["parameters"]),
          d["control"]["group_alpha"], d["control"]["f1_tau"],
          d["control"]["source"] if d["algorithm"] == "mlp" else d["control"]["f1_tau_source"]]
         for d in result["canonical_controls_and_selection"]])
    lines += ["## Per-seed selection and AP comparisons", ""]
    lines += markdown_table(["Classifier", "Seed", "Family", "Frozen synthetic weight", "Inner AP 0.25 / 0.5 / 1",
                             "Outer AP selected / unit / real-only / parent", "ΔAP selected−unit / selected−real / unit−real"],
        [[ALGORITHMS[d["algorithm"]], d["seed"], d["family"], d["selected_weight"],
          " / ".join(f"{d[key]:.6f}" for key in ("inner_w0.25_AP", "inner_w0.5_AP", "inner_w1_AP")),
          " / ".join(f"{d[key]:.6f}" for key in ("outer_selected_AP", "outer_unit_AP", "outer_real_only_AP", "outer_parent_AP")),
          " / ".join(f"{d[key]:+.6f}" for key in ("selected_minus_unit_AP", "selected_minus_real_only_AP", "unit_minus_real_only_AP"))]
         for d in result["per_seed_decisions"]])
    lines += ["## Accepted counts, added weight, rounds and observed runtime", "",
        "Times are observed on a shared workstation; the first full seed may overlap regression checks, so no isolated-speed claim is made. Added weight equals accepted synthetic count × synthetic coefficient (parent controls instead duplicate real anchors at that coefficient), not extra tuple count. Native-fit timing excludes generation and artifact I/O. Iterative trajectory timing includes generation, native fits and in-loop artifacts but excludes later metric export; parent runtime is only its native-fit window. These windows are not silently treated as pipeline wall time. Canonical MLP and tau-search costs are included once by physical trajectory path below; reused unit/tau trajectories are not double-counted. Already-paid frozen-flow costs are separate. Per-seed values are retained in the metrics CSV; the report JSON indexes SHA-bound full round audits without duplicating their large candidate-coordinate logs.", ""]
    trajectory_fields = ("accepted", "added_weight", "rounds", "native_fit_count", "native_fit_seconds", "runtime_seconds")
    lines += markdown_table(["Classifier", "Arm", "Accepted", "Added weight", "Rounds", "Native fits", "Native fit seconds", "Trajectory seconds"],
        [[ALGORITHMS[r["algorithm"]], r["variant_id"], *[cell(r[key], r[key + "_seed_sd"]) for key in trajectory_fields]]
         for r in result["trajectory_means_and_seed_sd"]])
    lines += markdown_table(["Classifier", "Seed", "Unique native fits incl. tuning", "Native fit seconds", "Sum of recorded trajectory windows (not wall time)"],
        [[ALGORITHMS[r["algorithm"]], r["seed"], int(r["native_fit_count"]),
          f"{r['native_fit_seconds']:.3f}", f"{r['runtime_seconds']:.3f}"]
         for r in result["native_fit_runtime_totals_by_seed"]])
    lines += markdown_table(["Seed", "Frozen flow source checkpoint", "Imported", "New flow fits", "Already-paid training seconds", "Effective device"],
        [[r["seed"], r["source_checkpoint"], r["imported_frozen_checkpoint"], r["fit_count_this_run"],
          f"{r['training_seconds']:.3f}", r["effective_device"]] for r in result["shared_frozen_flow"]])
    lines += markdown_table(["Classifier", "Seed", "Family", "Accepted selected / unit", "Added weight selected / unit", "Rounds selected / unit"],
        [[ALGORITHMS[d["algorithm"]], d["seed"], d["family"],
          f"{d['selected_accepted']} / {d['unit_accepted']}",
          f"{d['selected_added_weight']:g} / {d['unit_added_weight']:g}",
          f"{d['selected_rounds']} / {d['unit_rounds']}"] for d in result["per_seed_decisions"]])
    lines += ["## Native trajectory divergence", "",
        "Initial pool equality is checked separately from later trajectory divergence. Identical first pools do not imply identical accepted batches, native scores or stopping after synthetic rows enter training. The complete SHA-bound per-cell comparison is shown without assuming count matching.", ""]
    lines += markdown_table(["Classifier", "Seed", "Observed comparison evidence"],
        [[ALGORITHMS[d["algorithm"]], d["seed"], compact(d["comparison"])]
         for d in result["trajectory_divergence"]])
    lines += ["## Native MLP architecture, epochs and device", "",
        "Every native MLP fit's saved architecture, actual epochs, device, parameters and model-state evidence are retained in the linked fit-evidence CSV and report JSON; iterative and final fits are not replaced by a differentiable surrogate.", ""]
    mlp_signatures = {}
    for fit in result["native_MLP_fits"]:
        signature = compact({key: fit[key] for key in ("architecture", "actual_epochs", "device")})
        mlp_signatures.setdefault((fit["seed"], fit["variant_id"], signature), 0)
        mlp_signatures[fit["seed"], fit["variant_id"], signature] += 1
    lines += markdown_table(["Seed", "Arm", "Native architecture / actual epochs / device", "Observed fit count"],
        [[seed, variant, signature, count] for (seed, variant, signature), count in sorted(mlp_signatures.items())])
    lines += ["## Decision boundary and durable artifacts", "",
        "Bagging was not run. Retention is pending: no automatic retention or phasefreeze is performed, and mean AP alone does not decide superiority. Five exploratory matched seeds share training and one realized outer holdout.", ""]
    for key in ("all_24_metrics_csv", "all_metrics_summary_csv", "all_24_paired_deltas_csv",
                "paired_metrics_summary_csv", "per_seed_decisions_csv", "trajectory_summary_csv",
                "trajectory_divergence_csv", "native_MLP_fits_csv", "physical_native_resources_csv",
                "native_fit_runtime_totals_csv", "independent_report_json",
                "verification", "smoke_verification"):
        lines.append(f"- [{Path(result[key]).name}]({link(result[key])}): {key.replace('_', ' ')}.")
    lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_positional", nargs="?")
    parser.add_argument("--root")
    parser.add_argument("--verification", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown")
    parser.add_argument("--smoke-verification")
    args = parser.parse_args(argv)
    root = args.root or args.root_positional
    if root is None:
        parser.error("Supply a completed independently verified root")
    return report(root, args.verification, args.output, args.markdown, args.smoke_verification)


if __name__ == "__main__":
    main()

