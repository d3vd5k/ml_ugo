"""Independently verify and summarize the native CLIMB UGO/Flow F1 run."""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score)


METHODS = ("original_ugo", "u2_flow_gate")
DATASETS = ("ozone-level-8hr", "wilt", "mammography", "dis", "Satellite")
ALGORITHMS = ("mlp", "logistic_regression", "xgboost", "random_forest")
METRICS = ("minority_f1", "minority_precision", "minority_recall", "average_precision",
           "balanced_accuracy", "mcc", "tp", "fp", "fn", "tn", "f1_at_0_5")
RESOURCES = ("runtime_seconds", "classifier_fit_count", "accepted_count", "requested_count",
             "flow_fit_seconds_shared")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _load(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _same_number(actual, expected, message):
    _require(isinstance(actual, (int, float, np.integer, np.floating))
             and np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-12, atol=1e-12),
             f"{message}: saved={actual!r}, recomputed={expected!r}")


def _count(value, name):
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
             f"{name} must be a nonnegative integer")
    return value


def _ids(values, name):
    _require(isinstance(values, list) and values, f"Missing or empty {name}")
    _require(all(isinstance(value, (str, int)) and not isinstance(value, bool) for value in values),
             f"Invalid row identity in {name}")
    result = [str(value) for value in values]
    _require(len(result) == len(set(result)), f"Duplicate row identities in {name}")
    return result


def _cell(row):
    return tuple(row[key] for key in ("dataset", "algorithm", "seed", "method"))


def _best_threshold(y, probability):
    """Whole tied-score groups only; exact rational F1 ties use distance then height."""
    order = np.argsort(-probability, kind="stable")
    scores, labels = probability[order], y[order]
    cumulative = np.cumsum(labels)
    endpoints = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])
    positives = int(labels.sum())
    best = None
    for end in endpoints:
        tp = int(cumulative[end])
        denominator = int(end) + 1 + positives
        threshold = float(scores[end])
        tie_key = (abs(threshold - .5), -threshold)
        if (best is None or tp * best[1] > best[0] * denominator
                or (tp * best[1] == best[0] * denominator and tie_key < best[3])):
            best = (tp, denominator, threshold, tie_key)
    return best[2], 2 * best[0] / best[1]


def _metrics(y, probability, threshold):
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "minority_f1": float(f1_score(y, predicted, zero_division=0)),
        "minority_precision": float(precision_score(y, predicted, zero_division=0)),
        "minority_recall": float(recall_score(y, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "mcc": float(matthews_corrcoef(y, predicted)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "f1_at_0_5": float(f1_score(y, probability >= .5, zero_division=0)),
    }


def _prediction(run_dir, relative, expected_ids, threshold):
    _require(isinstance(relative, str) and relative, "Missing prediction path")
    path = Path(relative)
    _require(not path.is_absolute() and path.suffix == ".csv", f"Invalid prediction path: {relative}")
    resolved = (run_dir / path).resolve()
    _require(resolved.is_relative_to(run_dir.resolve()), f"Prediction path escapes run: {relative}")
    frame = pd.read_csv(resolved, float_precision="round_trip", dtype={"row_id": str})
    required = {"row_id", "y_true", "probability", "y_pred", "threshold"}
    _require(required <= set(frame), f"Prediction columns missing: {relative}")
    _require(frame["row_id"].tolist() == expected_ids, f"Prediction role/order mismatch: {relative}")
    _require(not frame.isna().any().any(), f"Missing prediction values: {relative}")
    _require(set(frame["y_true"].unique()) == {0, 1}, f"Both binary positive-indicator labels required: {relative}")
    y = frame["y_true"].to_numpy(dtype=int)
    probability = frame["probability"].to_numpy(dtype=float)
    _require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(),
             f"Invalid native probabilities: {relative}")
    _require(np.all(frame["threshold"].to_numpy() == threshold), f"CSV threshold mismatch: {relative}")
    _require(np.array_equal(frame["y_pred"].to_numpy(), probability >= threshold),
             f"Saved native predictions disagree with threshold: {relative}")
    return y, probability


def _verify_splits(entries, datasets, seeds):
    _require(isinstance(entries, list), "splits.json must contain a list")
    splits = {}
    for entry in entries:
        key = entry["dataset"], entry["seed"]
        _require(key not in splits, f"Duplicate split: {key}")
        roles = {role: _ids(entry["row_ids"][role], f"{key}/{role}")
                 for role in ("train", "fit", "validation", "test")}
        for role, ids in roles.items():
            _require(_ids(entry[f"{role}_indices"], f"{key}/{role}_indices") == ids,
                     f"Indices are not original role row identities: {key}/{role}")
        train, fit, validation, test = (set(roles[role]) for role in ("train", "fit", "validation", "test"))
        _require(not train & test and not fit & validation and fit | validation == train,
                 f"Leaking or incomplete role partition: {key}")
        flow_fit = _ids(entry["flow_fit_row_ids"], f"{key}/flow_fit")
        flow_calibration = _ids(entry["flow_calibration_row_ids"], f"{key}/flow_calibration")
        _require(not set(flow_fit) & set(flow_calibration)
                 and set(flow_fit) | set(flow_calibration) == fit,
                 f"Flow fitting/calibration must partition inner real fit: {key}")
        _require(isinstance(entry["split_fingerprint"], str) and entry["split_fingerprint"],
                 f"Missing split fingerprint: {key}")
        digest = sha256()
        for role in ("train_indices", "fit_indices", "validation_indices", "test_indices"):
            digest.update(json.dumps({role: entry[role]}, sort_keys=True, default=str).encode())
        _require(entry["split_fingerprint"] == digest.hexdigest(),
                 f"Split fingerprint does not bind original role identities: {key}")
        splits[key] = {**entry, "roles": roles}
    _require(set(splits) == set(product(datasets, seeds)), "Declared split coverage is incomplete or has extra splits")
    return splits


def _verify_audit(audit, role, split, method, config, label):
    _require(audit["role"] == role, f"Wrong training role: {label}")
    ids = split["row_ids"]["fit" if role == "inner" else "train"]
    _require(audit["real_count"] == len(ids), f"Real training count mismatch: {label}")
    identity_hash = sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()
    _require(audit["real_row_ids_sha256"] == identity_hash,
             f"Real-training identity hash does not bind the declared role: {label}")
    accepted = _count(audit["accepted_count"], f"{label}/accepted_count")
    requested = _count(audit["requested_count"], f"{label}/requested_count")
    _require(requested <= config["augmentation_total"] and accepted <= requested,
             f"Accepted/requested dose exceeds declared cap: {label}")
    rounds = _count(audit["round_count"], f"{label}/round_count")
    _require(rounds <= config["max_iterations"], f"Iteration cap exceeded: {label}")
    _same_number(audit["synthetic_weight"], 1.0, f"Unit synthetic weight {label}")
    round_audits = audit["rounds"]
    _require(isinstance(round_audits, list) and len(round_audits) == rounds,
             f"Round audit coverage mismatch: {label}")
    proposed_sum = accepted_sum = rejected_sum = 0
    for index, round_audit in enumerate(round_audits):
        proposed_round = _count(round_audit["proposed_count"], f"{label}/round {index}/proposed")
        accepted_round = _count(round_audit["accepted_count"], f"{label}/round {index}/accepted")
        rejected_round = _count(round_audit["rejected_count"], f"{label}/round {index}/rejected")
        _require(accepted_round + rejected_round == proposed_round,
                 f"Round accepted/rejected accounting mismatch: {label}/{index}")
        proposed_sum += proposed_round
        accepted_sum += accepted_round
        rejected_sum += rejected_round
    _require(accepted_sum == accepted and proposed_sum <= requested,
             f"Round dose/proposal cap accounting mismatch: {label}")
    if "proposal_budget" in audit:
        budget = audit["proposal_budget"]
        _require(budget["cap"] == requested and budget["attempted"] == proposed_sum
                 and budget["accepted"] == accepted_sum and budget["rejected"] == rejected_sum,
                 f"Proposal-budget summary mismatch: {label}")
    fits = _count(audit["classifier_fit_count"], f"{label}/classifier_fit_count")
    _require(fits > 0, f"No native classifier fits: {label}")
    ancestry = audit["ancestry"]
    _require(ancestry["validated"] is True and ancestry["all_roots_in_real_role"] is True,
             f"Real-role-rooted ancestry validation failed: {label}")
    _require(_count(ancestry["entry_count"], f"{label}/ancestry entries") == accepted,
             f"Accepted ancestry coverage mismatch: {label}")
    _require(isinstance(ancestry["provenance_sha256"], str) and len(ancestry["provenance_sha256"]) == 64,
             f"Missing compact ancestry evidence: {label}")
    _count(ancestry["direct_real_parent_count"], f"{label}/real parents")
    synthetic_parents = _count(ancestry["synthetic_parent_count"], f"{label}/synthetic parents")
    roots = ancestry["root_row_ids"]
    _require(isinstance(roots, list), f"Missing compact actual root identities: {label}")
    if accepted:
        actual_roots = _ids(roots, f"{label}/ancestry roots")
        _require(set(actual_roots) <= set(str(value) for value in ids),
                 f"Ancestry contains roots outside its real training role: {label}")
    else:
        _require(not roots, f"Empty augmentation has nonempty ancestry roots: {label}")
    if method == "u2_flow_gate":
        _require(ancestry["real_only_direct_parents"] is True and synthetic_parents == 0,
                 f"Gated U2 requires exclusively real parents: {label}")
    return fits


def _verify_selection(selection, record, split, config):
    key, method = _cell(record), record["method"]
    _same_number(selection["threshold"], record["threshold"], f"Selection threshold {key}")
    proposals = selection["candidate_configs"]
    _require(isinstance(proposals, list) and len(proposals) == config["tuning_candidates"],
             f"Incorrect common learner proposal budget: {key}")
    canonical = selection["canonical_candidates"] if method == "u2_flow_gate" else selection["candidates"]
    _require(len(canonical) == len(proposals), f"Missing canonical candidates: {key}")
    fit_count = 0
    for index, candidate in enumerate(canonical):
        _require(candidate["candidate_id"] == index and candidate["parameters"] == proposals[index],
                 f"Canonical candidate parameters/identity mismatch: {key}/{index}")
        score = candidate["validation_score"]
        _require(np.isfinite(score) and 0 <= score <= 1, f"Invalid candidate F1: {key}/{index}")
        if method == "u2_flow_gate":
            _same_number(candidate["sampler_parameters"]["group_alpha"],
                         config["group_alphas"][index % len(config["group_alphas"])],
                         f"Canonical alpha cycle: {key}/{index}")
        fit_count += _verify_audit(candidate["audit"], "inner", split, method, config, f"{key}/canonical/{index}")
    canonical_best = max(range(len(canonical)), key=lambda index: canonical[index]["validation_score"])
    _require(selection["selected_candidate_id"] == canonical_best, f"Canonical F1 selection inconsistent: {key}")
    chosen = canonical[canonical_best]
    if method == "u2_flow_gate":
        candidates = selection["candidates"]
        _require(len(candidates) == len(config["taus"]), f"Incomplete tau candidates: {key}")
        for index, candidate in enumerate(candidates):
            _require(candidate["candidate_id"] == index and candidate["parameters"] == chosen["parameters"],
                     f"Learner changed after canonical lock: {key}/{index}")
            _same_number(candidate["sampler_parameters"]["group_alpha"],
                         chosen["sampler_parameters"]["group_alpha"], f"Alpha lock: {key}/{index}")
            _same_number(candidate["sampler_parameters"]["f1_tau"], config["taus"][index], f"Tau grid: {key}/{index}")
            _require(np.isfinite(candidate["validation_score"]) and 0 <= candidate["validation_score"] <= 1,
                     f"Invalid tau candidate F1: {key}/{index}")
            fit_count += _verify_audit(candidate["audit"], "inner", split, method, config, f"{key}/tau/{index}")
        best = max(range(len(candidates)), key=lambda index: candidates[index]["validation_score"])
        _require(selection["selected_tau_id"] == best, f"Tau F1 selection inconsistent: {key}")
        chosen = candidates[best]
        flow = selection["flow_evidence"]
        _require(_ids(flow["fit_row_ids"], f"{key}/flow fit") == [str(x) for x in split["flow_fit_row_ids"]]
                 and _ids(flow["calibration_row_ids"], f"{key}/flow calibration") == [str(x) for x in split["flow_calibration_row_ids"]]
                 and _ids(flow["validation_row_ids"], f"{key}/flow validation") == split["roles"]["validation"],
                 f"Flow role evidence mismatch: {key}")
        _require(flow["roles_disjoint"] is True and flow["fit_count_shared"] == 1
                 and flow["frozen_core_shared_across_roles"] is True,
                 f"Flow core was not frozen/shared with excluded calibration and validation: {key}")
        _require(isinstance(flow["core_sha256"], str) and len(flow["core_sha256"]) == 64,
                 f"Missing frozen Flow core fingerprint: {key}")
        _require(np.isfinite(flow["fit_seconds"]) and flow["fit_seconds"] >= 0,
                 f"Invalid shared Flow duration: {key}")
    else:
        _require(selection["selected_tau_id"] is None, f"Original UGO cannot have a Flow tau selection: {key}")
    _same_number(selection["selected_validation_score"], chosen["validation_score"], f"Selected validation F1 {key}")
    _same_number(selection["threshold"], chosen["threshold"], f"Selected validation threshold {key}")
    _require(selection["selected_inner_audit"] == chosen["audit"], f"Selected inner audit mismatch: {key}")
    outer = selection["outer_audit"]
    fit_count += _verify_audit(outer, "outer", split, method, config, f"{key}/outer")
    _require(record["classifier_fit_count"] == fit_count, f"Native fit accounting mismatch: {key}")
    for field in ("accepted_count", "requested_count"):
        _require(record[field] == outer[field], f"Outer native dose mismatch: {key}/{field}")


def _summaries(frame, datasets):
    rows = []
    for (dataset, algorithm, method), group in frame.groupby(["dataset", "algorithm", "method"], sort=False):
        row = {"dataset": dataset, "algorithm": algorithm, "method": method, "seeds": len(group), "datasets": 1}
        for metric in (*METRICS, *RESOURCES):
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sample_sd"] = group[metric].std(ddof=1)
        rows.append(row)
    # One value per seed: average all dataset values equally before describing split variation.
    for (algorithm, method), group in frame.groupby(["algorithm", "method"], sort=False):
        seed_macros = group.groupby("seed")[[*METRICS, *RESOURCES]].mean()
        row = {"dataset": "__macro__", "algorithm": algorithm, "method": method,
               "seeds": len(seed_macros), "datasets": len(datasets)}
        for metric in (*METRICS, *RESOURCES):
            row[f"{metric}_mean"] = seed_macros[metric].mean()
            row[f"{metric}_sample_sd"] = seed_macros[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def _paired(frame):
    rows = []
    for (dataset, algorithm, seed), group in frame.groupby(["dataset", "algorithm", "seed"], sort=False):
        methods = group.set_index("method")
        original, gated = methods.loc["original_ugo"], methods.loc["u2_flow_gate"]
        row = {"dataset": dataset, "algorithm": algorithm, "seed": seed,
               "split_fingerprint": original["split_fingerprint"], "data_fingerprint": original["data_fingerprint"]}
        for metric in (*METRICS, *RESOURCES):
            row[f"original_ugo_{metric}"] = original[metric]
            row[f"u2_flow_gate_{metric}"] = gated[metric]
            row[f"delta_{metric}"] = gated[metric] - original[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def _table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(value) for value in row) + " |" for row in rows)]


def _format(mean, sd):
    return f"{mean:.4f} ± {sd:.4f}" if pd.notna(sd) else f"{mean:.4f} (SD unavailable: one seed)"


def _markdown(manifest, summary, deltas, verification):
    config = manifest["config"]
    lines = ["# Native CLIMB UGO versus U2 + FlowGMM", "",
             f"Mode: **{manifest['mode']}**. Independent artifact verification: **passed** ({verification['verified_cells']} cells).",
             f"Datasets: {', '.join(config['datasets'])}. Classifiers: {', '.join(config['classifiers'])}. Seeds: {', '.join(map(str, config['seeds']))}.", "",
             "Minority F1 is the primary outcome. Thresholds maximize inner-validation F1 over whole tied-probability groups; exact ties choose closest to 0.5, then higher. Outer-test probabilities never select thresholds or settings. Precision, recall, AP and runtime are secondary descriptions, not substitutes for F1.", "",
             "## Equal-dataset macro averages", "",
             "Each dataset has equal weight; values first average datasets within each seed. ± is sample SD across repeated stratified splits, not an independent-replicate standard error. A subset macro covers only its declared datasets, not the full five-dataset suite.", ""]
    headers = ["Classifier", "Method", "Minority F1 ± SD", "Precision", "Recall", "AP", "Runtime s", "Accepted / requested", "Native fits"]
    def summary_row(row, include_dataset=False):
        values = [row["algorithm"], row["method"], _format(row["minority_f1_mean"], row["minority_f1_sample_sd"]),
                  f"{row['minority_precision_mean']:.4f}", f"{row['minority_recall_mean']:.4f}",
                  f"{row['average_precision_mean']:.4f}", f"{row['runtime_seconds_mean']:.2f}",
                  f"{row['accepted_count_mean']:.1f} / {row['requested_count_mean']:.1f}",
                  f"{row['classifier_fit_count_mean']:.1f}"]
        return [row["dataset"], *values] if include_dataset else values
    macro = summary[summary["dataset"] == "__macro__"]
    lines.extend(_table(headers, [summary_row(row) for _, row in macro.iterrows()]))
    lines.extend(["", "## Per-dataset native results", ""])
    lines.extend(_table(["Dataset", *headers], [summary_row(row, True) for _, row in summary[summary["dataset"] != "__macro__"].iterrows()]))
    lines.extend(["", "## Paired descriptive deltas", "", "Deltas are U2 + FlowGMM minus original UGO, paired on dataset, classifier, seed and the exact split. Positive F1 is favorable; runtime and dose differences are descriptive.", ""])
    delta_rows = []
    for algorithm, group in deltas.groupby("algorithm", sort=False):
        per_seed = group.groupby("seed")[[f"delta_{metric}" for metric in ("minority_f1", "minority_precision", "minority_recall", "average_precision", "runtime_seconds", "accepted_count")]].mean()
        means = per_seed.mean()
        delta_rows.append([algorithm, _format(means["delta_minority_f1"], per_seed["delta_minority_f1"].std(ddof=1)),
                           *(f"{means[f'delta_{metric}']:.4f}" for metric in ("minority_precision", "minority_recall", "average_precision")),
                           f"{means['delta_runtime_seconds']:.2f}", f"{means['delta_accepted_count']:.1f}"])
    lines.extend(_table(["Classifier", "Δ minority F1 ± SD", "Δ precision", "Δ recall", "Δ AP", "Δ runtime s", "Δ accepted"], delta_rows))
    lines.extend(["", "## Scope and interpretation", "",
                  f"Both methods receive {config['tuning_candidates']} common learner proposals and an augmentation cap of {config['augmentation_total']}, with at most {config['max_iterations']} rounds. U2 cycles canonical alpha, locks the winning learner/alpha, then selects tau on validation F1. A common cap is **not** a matched accepted dose: no count-matched quality claim is supported.",
                  "The original UGO is the uncorrected baseline; its synthetic parents are permitted only with validated ancestry rooted in its real training role. Gated U2 uses real parents only. Unit synthetic weights and the fixed native learner protocol are recorded by the run; no historical U1, corrected UGO, downweighting or ensemble is compared.",
                  "Flow is fitted only to its declared inner real-fit core. Calibration, validation and test rows are excluded; the frozen core is shared for outer replay. Shared Flow fit seconds are exposed separately and are not summed repeatedly as independent fits.",
                  "Native test and selected pre-refit validation CSVs independently reproduce the metrics and selected threshold. Compact hashed fit/provenance audits are checked for role identity, accounting and recorded validation assertions; they are **not** a replay of generation or an independent reconstruction of omitted raw ancestry. Candidate ranking is checked against recorded candidate scores; only the selected candidate's validation probabilities are retained and independently rescored.",
                  "Seeds are descriptive repeated-split evidence, not independent significance tests. No superiority or compatibility verdict is inferred solely from mean AP. Smoke results are training-pipeline checks, not scientific evidence.", "",
                  "Artifacts: `verification.json`, `summary.csv`, `paired_deltas.csv`; source predictions, selections, splits and the effective configuration remain in this run folder.", ""])
    return "\n".join(lines)


def _build(run_dir):
    manifest = _load(run_dir / "manifest.json")
    _require(manifest["status"] == "completed", "Run is not complete")
    config = manifest["config"]
    datasets, algorithms, seeds = config["datasets"], config["classifiers"], config["seeds"]
    for values, allowed, label in ((datasets, DATASETS, "datasets"), (algorithms, ALGORITHMS, "classifiers")):
        _require(isinstance(values, list) and values and len(values) == len(set(values))
                 and set(values) <= set(allowed), f"Invalid effective {label}")
    _require(isinstance(seeds, list) and seeds and len(seeds) == len(set(seeds)), "Invalid effective seeds")
    for field in ("tuning_candidates", "augmentation_total", "max_iterations"):
        _require(_count(config[field], field) > 0, f"Empty declared budget: {field}")
    _require(manifest["mode"] in ("smoke", "run"), "Missing explicit smoke/scientific mode")
    _require(isinstance(manifest["source_hashes"], dict) and manifest["source_hashes"], "Missing source fingerprints")
    metadata = {}
    for item in manifest["datasets"]:
        _require(item["name"] not in metadata, f"Duplicate dataset metadata: {item['name']}")
        metadata[item["name"]] = item
    _require(set(metadata) == set(datasets), "Dataset metadata coverage does not match effective configuration")
    splits = _verify_splits(_load(run_dir / "splits.json"), datasets, seeds)
    records, selections = _load(run_dir / "records.json"), _load(run_dir / "selections.json")
    _require(isinstance(records, list) and isinstance(selections, list), "Records and selections must be lists")
    expected = set(product(datasets, algorithms, seeds, METHODS))
    index, selected = {}, {}
    for row in records:
        key = _cell(row)
        _require(key not in index, f"Duplicate result cell: {key}")
        index[key] = row
    for row in selections:
        key = _cell(row)
        _require(key not in selected, f"Duplicate selection cell: {key}")
        selected[key] = row
    _require(set(index) == expected, "Expected complete declared dataset/classifier/seed/method cells; missing or unexpected methods/cells")
    _require(set(selected) == expected, "Selection coverage does not match result cells")
    csv_rows = pd.read_csv(run_dir / "metrics.csv", float_precision="round_trip").to_dict("records")
    csv_index = {}
    for row in csv_rows:
        key = _cell(row)
        _require(key not in csv_index, f"Duplicate metrics.csv cell: {key}")
        csv_index[key] = row
    _require(set(csv_index) == expected, "metrics.csv coverage does not match native records")
    for key, record in index.items():
        csv_row = csv_index[key]
        for field in ("threshold", *METRICS, *RESOURCES):
            _same_number(csv_row[field], record[field], f"metrics.csv parity {key}/{field}")
        for field in ("status", "positive_class", "split_fingerprint", "data_fingerprint",
                      "predictions_path", "validation_predictions_path"):
            _require(csv_row[field] == record[field], f"metrics.csv parity {key}/{field}")
    verified, prediction_paths, dataset_fingerprints = [], set(), {}
    for key, record in index.items():
        _require(record["status"] == "completed", f"Incomplete cell: {key}")
        split = splits[(record["dataset"], record["seed"])]
        _require(record["split_fingerprint"] == split["split_fingerprint"], f"Split fingerprint mismatch: {key}")
        fingerprint = record["data_fingerprint"]
        _require(isinstance(fingerprint, str) and fingerprint, f"Missing data fingerprint: {key}")
        prior = dataset_fingerprints.setdefault(record["dataset"], fingerprint)
        _require(prior == fingerprint, f"Native dataset fingerprint differs across cells: {key}")
        dataset_metadata = metadata[record["dataset"]]
        _require(dataset_metadata["raw_fingerprint"] == fingerprint,
                 f"Native data fingerprint is not bound to loader metadata: {key}")
        _require(record["positive_class"] in [item["encoded"] for item in dataset_metadata["label_mapping"]],
                 f"Positive class is not an encoded native label: {key}")
        threshold = record["threshold"]
        _require(np.isfinite(threshold) and 0 <= threshold <= 1, f"Invalid threshold: {key}")
        for field in RESOURCES:
            _require(np.isfinite(record[field]) and record[field] >= 0, f"Invalid native resource: {key}/{field}")
        selection = selected[key]
        _verify_selection(selection, record, split, config)
        for field in ("predictions_path", "validation_predictions_path"):
            _require(record[field] not in prediction_paths, f"Reused native prediction file: {record[field]}")
            prediction_paths.add(record[field])
        y, probability = _prediction(run_dir, record["predictions_path"], split["roles"]["test"], threshold)
        metrics = _metrics(y, probability, threshold)
        for field, value in metrics.items():
            _same_number(record[field], value, f"Native test metric {key}/{field}")
        validation_y, validation_p = _prediction(run_dir, record["validation_predictions_path"], split["roles"]["validation"], threshold)
        optimum, score = _best_threshold(validation_y, validation_p)
        _require(threshold == optimum,
                 f"Independent validation threshold optimum mismatch: {key}: {threshold!r} != {optimum!r}")
        _same_number(selection["selected_validation_score"], score, f"Independent validation F1 {key}")
        verified.append({**record, **metrics})
    # Verify the two methods and all native learners see identical held-out labels and identities.
    for dataset, seed in product(datasets, seeds):
        cells = [index[(dataset, algorithm, seed, method)] for algorithm, method in product(algorithms, METHODS)]
        reference = cells[0]
        for record in cells[1:]:
            _require(record["positive_class"] == reference["positive_class"], f"Positive class mismatch: {dataset}/{seed}")
            for field in ("predictions_path", "validation_predictions_path"):
                a = pd.read_csv(run_dir / reference[field], dtype={"row_id": str}, float_precision="round_trip")
                b = pd.read_csv(run_dir / record[field], dtype={"row_id": str}, float_precision="round_trip")
                _require(a[["row_id", "y_true"]].equals(b[["row_id", "y_true"]]),
                         f"Native held-out role/label pairing mismatch: {dataset}/{seed}/{field}")
        gated = [selected[(dataset, algorithm, seed, "u2_flow_gate")]["flow_evidence"] for algorithm in algorithms]
        _require(len({item["core_sha256"] for item in gated}) == 1, f"Flow core not shared across native learners: {dataset}/{seed}")
        fit_seconds = gated[0]["fit_seconds"]
        for algorithm, evidence in zip(algorithms, gated):
            _same_number(evidence["fit_seconds"], fit_seconds, f"Shared Flow duration {dataset}/{seed}/{algorithm}")
            original = index[(dataset, algorithm, seed, "original_ugo")]
            flow_record = index[(dataset, algorithm, seed, "u2_flow_gate")]
            _same_number(original["flow_fit_seconds_shared"], 0, f"Original has no Flow fit {dataset}/{seed}/{algorithm}")
            _same_number(flow_record["flow_fit_seconds_shared"], fit_seconds if algorithm == algorithms[0] else 0,
                         f"Flow shared-fit duration charged exactly once {dataset}/{seed}/{algorithm}")
        for algorithm in algorithms:
            a, b = (selected[(dataset, algorithm, seed, method)] for method in METHODS)
            _require(a["candidate_configs"] == b["candidate_configs"], f"Unequal learner proposals: {dataset}/{algorithm}/{seed}")
    frame = pd.DataFrame(verified)
    summary, deltas = _summaries(frame, datasets), _paired(frame)
    verification = {"status": "passed", "success": True, "verified_cells": len(frame),
                    "expected_cells": len(expected), "datasets": datasets, "classifiers": algorithms,
                    "seeds": seeds, "mode": manifest["mode"], "full_five_dataset_suite": set(datasets) == set(DATASETS),
                    "independent_native_test_metrics": True, "independent_selected_validation_thresholds": True,
                    "role_identity_disjointness": True, "paired_native_labels": True,
                    "compact_provenance_audits_checked": True, "raw_generation_replayed": False,
                    "candidate_scores_independently_recomputed": "selected candidate only",
                    "source_hashes": manifest["source_hashes"],
                    "artifacts": ["summary.csv", "paired_deltas.csv", "RESULTS.md"]}
    summary.to_csv(run_dir / "summary.csv", index=False)
    deltas.to_csv(run_dir / "paired_deltas.csv", index=False)
    (run_dir / "RESULTS.md").write_text(_markdown(manifest, summary, deltas, verification), encoding="utf-8")
    _json(run_dir / "verification.json", verification)
    return verification


def build_report(run_dir: Path) -> dict:
    """Verify saved native evidence, write reports, and raise on any missing/invalid evidence."""
    run_dir = Path(run_dir)
    _require(run_dir.is_dir(), f"Run directory does not exist: {run_dir}")
    try:
        return _build(run_dir)
    except Exception as exc:
        _json(run_dir / "verification.json", {"status": "failed", "success": False,
                                             "error_type": type(exc).__name__, "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = build_report(args.run_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
