"""Saved-artifact reporting for the fixed-weight-one Shoppers ablation.

Usage: python -m benchmarks.shoppers_ablation_report RUN_DIR [--verify]
Only report files are written; saved experiment inputs are never modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score,
)


CLASSIFIERS = ("mlp", "random_forest", "xgboost")
VARIANTS = tuple(f"A{i}" for i in range(8))
STRATEGIES = ("flow", "random_round", "random_round_group")
PRIMARY = ("minority_f1", "precision", "minority_recall", "average_precision",
           "balanced_accuracy", "mcc")
COUNTERS = ("accepted_count", "requested_count", "proposal_count", "rejected_count",
            "round_count", "classifier_fit_count", "native_fit_seconds", "runtime_seconds")
CONFUSION = ("tp", "fp", "tn", "fn")
HALF = {"minority_f1": "f1_at_0_5", "precision": "precision_at_0_5",
        "minority_recall": "minority_recall_at_0_5",
        "average_precision": "average_precision_at_0_5",
        "balanced_accuracy": "balanced_accuracy_at_0_5", "mcc": "mcc_at_0_5"}
PREDICTIVE = (*PRIMARY, *CONFUSION, *HALF.values(),
              *(f"{key}_at_0_5" for key in CONFUSION))
VALUES = (*PREDICTIVE, *COUNTERS)
KEY = ["classifier", "fold", "seed", "variant", "result_type", "replay_strategy", "mask_seed"]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _path(root, relative):
    _require(isinstance(relative, str) and bool(relative), "Missing artifact path")
    relative = Path(relative)
    resolved = (root / relative).resolve()
    _require(not relative.is_absolute() and resolved.is_relative_to(root.resolve()),
             f"Artifact path escapes run directory: {relative}")
    return resolved


def _npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _close(actual, expected, message):
    _require(np.isfinite(float(actual)) and np.isclose(float(actual), float(expected),
                                                     rtol=1e-10, atol=1e-12),
             f"{message}: saved={actual!r}, recomputed={expected!r}")


def _best_threshold(frame):
    """Exact rational F1; tied-score groups plus .5, nearest .5 then higher."""
    y = frame.y_true.to_numpy(dtype=np.int64)
    probability = frame.probability.to_numpy(dtype=np.float64)
    positives = int(y.sum())
    _require(positives > 0, "Validation must contain the minority class")
    order = np.argsort(-probability, kind="stable")
    scores = probability[order]
    cumulative = np.cumsum(y[order])
    endpoints = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(y) - 1]
    at_half = probability >= .5
    best = (int(y[at_half].sum()), int(at_half.sum()) + positives, .5)
    for end in endpoints:
        item = (int(cumulative[end]), int(end) + 1 + positives, float(scores[end]))
        comparison = item[0] * best[1] - best[0] * item[1]
        if comparison > 0 or (comparison == 0 and
                (-abs(item[2] - .5), item[2]) > (-abs(best[2] - .5), best[2])):
            best = item
    return best[2]


def _classification(y, predicted):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"minority_f1": float(f1_score(y, predicted, zero_division=0)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "minority_recall": float(recall_score(y, predicted, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "mcc": float(matthews_corrcoef(y, predicted)),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def _metrics(frame, classifier):
    y = frame.y_true.to_numpy(dtype=np.int64)
    probability = frame.probability.to_numpy(dtype=np.float32 if classifier == "mlp" else np.float64)
    values = _classification(y, frame.y_pred.to_numpy(dtype=np.int64))
    values["average_precision"] = float(average_precision_score(y, probability))
    half = _classification(y, frame.y_pred_05.to_numpy(dtype=np.int64))
    half["average_precision"] = values["average_precision"]
    for key, value in half.items():
        values[HALF.get(key, f"{key}_at_0_5")] = value
    return values


def _prediction(root, relative, classifier, threshold):
    path = _path(root, relative)
    frame = pd.read_csv(path, dtype={"row_id": str}, float_precision="round_trip")
    required = {"row_id", "y_true", "probability", "threshold", "y_pred", "y_pred_05"}
    _require(required <= set(frame), f"Missing prediction columns: {relative}")
    _require(len(frame) > 0 and not frame[list(required)].isna().any().any(),
             f"Empty/missing predictions: {relative}")
    _require(frame.row_id.is_unique, f"Repeated row identities: {relative}")
    for key in ("y_true", "y_pred", "y_pred_05"):
        _require(set(frame[key].unique()) <= {0, 1}, f"Nonbinary {key}: {relative}")
    probability = frame.probability.to_numpy(dtype=np.float32 if classifier == "mlp" else np.float64)
    _require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(),
             f"Invalid probabilities: {relative}")
    _require(np.isfinite(threshold) and np.all(frame.threshold.to_numpy() == threshold),
             f"Inconsistent threshold: {relative}")
    # Preserve native float32 comparison semantics as well as AP tie grouping.
    _require(np.array_equal(frame.y_pred.to_numpy(), probability >= threshold),
             f"Threshold predictions disagree: {relative}")
    _require(np.array_equal(frame.y_pred_05.to_numpy(), probability >= .5),
             f"0.5 predictions disagree: {relative}")
    return frame


def _read_run(run_dir):
    root = Path(run_dir)
    _require(root.is_dir(), f"Missing run directory: {root}")
    manifest = _load(root / "manifest.json")
    mode = manifest.get("mode")
    _require(mode in ("scientific", "smoke"), "Manifest must explicitly distinguish scientific/smoke mode")
    _require(manifest.get("status") in ("complete", "completed"), "Run is not complete")
    frame = pd.read_csv(root / "metrics.csv", keep_default_na=False, float_precision="round_trip")
    required = {*KEY, "threshold", *PRIMARY, "f1_at_0_5", *CONFUSION, *COUNTERS,
                "backend", "predictions_path", "validation_predictions_path", "artifact_dir", "status"}
    _require(required <= set(frame), f"Missing metric columns: {sorted(required - set(frame))}")
    for name in ("fold", "seed", "mask_seed"):
        values = pd.to_numeric(frame[name], errors="raise")
        _require(np.equal(values, values.astype(int)).all(), f"Noninteger {name}")
        frame[name] = values.astype(int)
    _require(not frame.duplicated(KEY).any(), "Duplicate metric record identity")
    _require((frame.status == "complete").all(), "Incomplete metric records")
    _require(set(frame.classifier) <= set(CLASSIFIERS), "Historical/unsupported classifier records")
    _require(set(frame.variant) <= set(VARIANTS), "Historical/unsupported variants")
    _require(set(frame.result_type) <= {"main", "replay"}, "Unknown result type")
    expected = {"main": 360, "replay": 945}
    if mode == "smoke":
        _require(isinstance(manifest.get("expected_counts"), dict), "Smoke manifest needs expected_counts")
        expected = manifest["expected_counts"]
    elif "expected_counts" in manifest:
        _require(manifest["expected_counts"] == expected, "Scientific expected counts must be 360/945")
    for kind in ("main", "replay"):
        _require(int((frame.result_type == kind).sum()) == int(expected[kind]),
                 f"Expected {expected[kind]} complete {kind} records")
    config = manifest.get("config", {})
    classifiers = config.get("classifiers", list(CLASSIFIERS))
    seeds = config.get("seeds", [0, 1, 2])
    folds_value = config.get("outer_folds", 5)
    folds = list(range(folds_value)) if isinstance(folds_value, int) else list(folds_value)
    masks = config.get("replay_mask_seeds", list(range(10)))
    if mode == "scientific":
        _require(set(classifiers) == set(CLASSIFIERS) and list(seeds) == [0, 1, 2]
                 and folds == list(range(5)) and list(masks) == list(range(10)),
                 "Scientific grid differs from the registered 3 classifiers/5 folds/3 seeds/10 masks")
    _require(len(set(classifiers)) == len(classifiers) and len(set(seeds)) == len(seeds)
             and len(set(folds)) == len(folds) and len(set(masks)) == len(masks), "Duplicate configured grid values")
    if mode == "scientific":
        fixed = {"dataset": "shoppers", "n_candidates": 10, "mlp_epochs": 160, "flow_epochs": 60,
                 "flow_fit_fraction": .8, "flow_reserve_calibration_used": False, "group_alpha": .5,
                 "gate_tau": .6, "proposal_cap": 1000, "max_rounds": 20,
                 "real_weight": 1., "synthetic_weight": 1., "refill": False}
        _require(all(config.get(key) == value for key, value in fixed.items()),
                 "Scientific configuration differs from the fixed-weight-one registered protocol")
    expected_keys = set()
    for classifier in classifiers:
        for fold in folds:
            for seed in seeds:
                for variant in VARIANTS:
                    # Main replay_strategy is intentionally empty; mask_seed is -1.
                    expected_keys.add((classifier, fold, seed, variant, "main", "", -1))
                for strategy in STRATEGIES:
                    for mask in ([-1] if strategy == "flow" else masks):
                        expected_keys.add((classifier, fold, seed, "A4", "replay", strategy, mask))
    actual_keys = set(frame[KEY].itertuples(index=False, name=None))
    _require(actual_keys == expected_keys,
             f"Incomplete/unexpected experiment grid: missing={len(expected_keys - actual_keys)}, extra={len(actual_keys - expected_keys)}")
    return root, manifest, frame


def _oof(root, records):
    rows = []
    group_keys = ["classifier", "seed", "variant", "result_type", "replay_strategy", "mask_seed"]
    for key, group in records.groupby(group_keys, sort=True, dropna=False):
        classifier = key[0]
        predictions = [_prediction(root, row.predictions_path, classifier, float(row.threshold))
                       for row in group.sort_values("fold").itertuples()]
        pooled = pd.concat(predictions, ignore_index=True)
        _require(pooled.row_id.is_unique, f"A row appears in multiple outer folds: {key}")
        row = dict(zip(group_keys, key))
        row.update(_metrics(pooled, classifier))
        row.update({name: float(group[name].sum()) for name in COUNTERS})
        row.update(oof_rows=len(pooled), outer_folds=len(group),
                   threshold_min=float(group.threshold.min()), threshold_max=float(group.threshold.max()),
                   threshold_fold_mean=float(group.threshold.mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate(frame, keys, values=VALUES):
    rows = []
    for identity, group in frame.groupby(keys, sort=True, dropna=False):
        identity = identity if isinstance(identity, tuple) else (identity,)
        row = dict(zip(keys, identity))
        row["training_seeds"] = len(group)
        _require(group.seed.is_unique, f"Non-independent duplicate seed values: {identity}")
        for name in values:
            row[f"{name}_mean"] = float(group[name].mean())
            row[f"{name}_sample_sd"] = float(group[name].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _replay_seed_means(oof):
    replay = oof[oof.result_type == "replay"]
    keys = ["classifier", "seed", "replay_strategy"]
    averaged = replay.groupby(keys, as_index=False)[list(VALUES)].mean()
    counts = replay.groupby(keys, as_index=False).agg(masks=("mask_seed", "size"))
    return averaged.merge(counts, on=keys, validate="one_to_one")


def _paired(oof, replay):
    rows = []
    contrasts = (("A4-A3", "A4", "A3"), ("A2-A1", "A2", "A1"),
                 ("A4-A2", "A4", "A2"), ("A4-A5", "A4", "A5"),
                 ("A4-A6", "A4", "A6"), ("A4-A7", "A4", "A7"))
    for (classifier, seed), group in oof[oof.result_type == "main"].groupby(["classifier", "seed"]):
        indexed = group.set_index("variant")
        for label, left, right in contrasts:
            row = {"classifier": classifier, "seed": seed, "contrast": label}
            row.update({name: float(indexed.loc[left, name] - indexed.loc[right, name]) for name in VALUES})
            rows.append(row)
        row = {"classifier": classifier, "seed": seed, "contrast": "interaction_(A4-A3)-(A2-A1)"}
        row.update({name: float(indexed.loc["A4", name] - indexed.loc["A3", name]
                               - indexed.loc["A2", name] + indexed.loc["A1", name]) for name in VALUES})
        rows.append(row)
    for (classifier, seed), group in replay.groupby(["classifier", "seed"]):
        indexed = group.set_index("replay_strategy")
        for control in STRATEGIES[1:]:
            row = {"classifier": classifier, "seed": seed, "contrast": f"replay_flow-{control}_mask_mean"}
            row.update({name: float(indexed.loc["flow", name] - indexed.loc[control, name]) for name in VALUES})
            rows.append(row)
    return pd.DataFrame(rows)


def _format(mean, sd):
    return f"{mean:.4f} ± {sd:.4f}" if pd.notna(sd) else f"{mean:.4f} (one seed; SD unavailable)"


def _table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows)]


def _markdown(manifest, records, summary, paired_summary, replay_summary, oof, verification):
    mode = manifest["mode"]
    lines = ["# Shoppers: fixed synthetic weight 1 ablation", "",
             f"Mode: **{mode}**. Saved main records: {int((records.result_type == 'main').sum())}; "
             f"saved replay records: {int((records.result_type == 'replay').sum())}.", ""]
    if mode == "smoke":
        lines += ["**Smoke run: execution/artifact evidence only, not scientific performance evidence.**", ""]
    lines += ["Minority F1 is the primary outcome. Precision, minority recall, average precision (AP), "
              "balanced accuracy (BA), and MCC are secondary. Thresholds are selected on matching real inner "
              "validation predictions; outer labels never select settings or thresholds. Classifier candidates "
              "are selected once on real-only validation F1 and locked across arms. All real and synthetic weights are 1.", "",
              "Every metric is recomputed after pooling five held-out folds into one OOF prediction vector per "
              "classifier/training seed/arm, preserving saved fold-specific decisions. AP uses native float32 "
              "probabilities for MLP. Random replay metrics are computed separately for each mask, then averaged "
              "within the training seed. Tables show mean ± sample SD across training seeds, never independent "
              "folds or independent masks. No probability ensembling, confidence intervals, or significance claims.", "",
              "## Main OOF results (validation-selected thresholds)", ""]
    headers = ["Classifier", "Arm", "F1", "Precision", "Recall", "AP", "BA", "MCC"]
    lines += _table(headers, [[row.classifier, row.variant,
                              *(_format(getattr(row, f"{metric}_mean"), getattr(row, f"{metric}_sample_sd"))
                                for metric in PRIMARY)] for row in summary.itertuples()])
    lines += ["", "## Main OOF results (threshold 0.5)", ""]
    lines += _table(headers, [[row.classifier, row.variant,
                              *(_format(getattr(row, f"{HALF[metric]}_mean"), getattr(row, f"{HALF[metric]}_sample_sd"))
                                for metric in PRIMARY)] for row in summary.itertuples()])
    lines += ["", "## Paired seed-level contrasts", "", "Differences are left minus right; the interaction is "
              "(A4−A3)−(A2−A1). It is descriptive and conditional on the chosen metric/score scale, not a "
              "scale-invariant causal interaction.", ""]
    lines += _table(["Classifier", "Contrast", *headers[2:]],
                    [[row.classifier, row.contrast,
                      *(_format(getattr(row, f"{metric}_mean"), getattr(row, f"{metric}_sample_sd"))
                        for metric in PRIMARY)] for row in paired_summary.itertuples()])
    lines += ["", "### Paired contrasts at threshold 0.5", ""]
    lines += _table(["Classifier", "Contrast", *headers[2:]],
                    [[row.classifier, row.contrast,
                      *(_format(getattr(row, f"{HALF[metric]}_mean"), getattr(row, f"{HALF[metric]}_sample_sd"))
                        for metric in PRIMARY)] for row in paired_summary.itertuples()])
    lines += ["", "## Fixed-pool replay", "",
              "Flow and matched random masks operate on the **saved A4 proposal stream**, conditional on "
              "that stream and its valid proposals. These controls do not represent independently generated "
              "random trajectories. Round controls match accepted counts within round; round+group controls "
              "match within both round and original group. Each replay selects its own threshold on its "
              "matching inner-role replay. Flow replay must reproduce A4 predictions.", ""]
    lines += _table(["Classifier", "Replay", *headers[2:]],
                    [[row.classifier, row.replay_strategy,
                      *(_format(getattr(row, f"{metric}_mean"), getattr(row, f"{metric}_sample_sd"))
                        for metric in PRIMARY)] for row in replay_summary.itertuples()])
    lines += ["", "### Replay at threshold 0.5", ""]
    lines += _table(["Classifier", "Replay", *headers[2:]],
                    [[row.classifier, row.replay_strategy,
                      *(_format(getattr(row, f"{HALF[metric]}_mean"), getattr(row, f"{HALF[metric]}_sample_sd"))
                        for metric in PRIMARY)] for row in replay_summary.itertuples()])
    lines += ["", "## Counters and coverage", "",
              "Counters below sum outer fitting/evaluation records across folds before seed-level aggregation; "
              "shared detector fitting is not multiplied or imputed. Exact counts, all confusion counts, both "
              "threshold policies, paired 0.5 contrasts, and all replay-mask OOF metrics are in the CSV outputs.", ""]
    lines += _table(["Classifier", "Arm", "Accepted", "Requested", "Proposed", "Rejected", "Rounds", "Classifier fits", "Native fit s", "Runtime s"],
                    [[row.classifier, row.variant,
                      *(_format(getattr(row, f"{metric}_mean"), getattr(row, f"{metric}_sample_sd"))
                        for metric in COUNTERS)] for row in summary.itertuples()])
    coverage = oof.groupby(["classifier", "seed"]).agg(min_rows=("oof_rows", "min"), max_rows=("oof_rows", "max"),
                                                     min_folds=("outer_folds", "min"), max_folds=("outer_folds", "max"))
    lines += ["", *_table(["Classifier", "Training seed", "OOF rows min/max", "Folds min/max"],
                          [[classifier, seed, f"{row.min_rows}/{row.max_rows}", f"{row.min_folds}/{row.max_folds}"]
                           for (classifier, seed), row in coverage.iterrows()]), "", "## Protocol and audit scope", "",
              "A0: real only; A1: base class-wide interpolation/pass-through with native MMA; A2: base+Flow; "
              "A3: canonical U2/pass-through; A4: canonical U2+Flow; A5: A4 with positive-stability allocation "
              "multiplier set to 1, original eligibility/caps retained; A6: A4 with nonbinding local caps inside "
              "the global cap; A7: native U2 post-detection+relocation.", "",
              "Registered scientific protocol: five duplicate-group-disjoint outer folds; matched seeds 0/1/2; "
              "real nested validation; KMeans8/5 repeats/80% canonical grouping, group alpha 0.5; Flow tau 0.6, "
              "60 epochs, frozen after fitting real-only 80% of inner-fit with an unused 20% reserve; "
              "proposal cap 1000, at most 20 rounds, no refill; MLP 160 epochs; MLP/XGBoost/Flow and neighbor "
              "math on CUDA, sklearn RF on CPU. Smoke settings are not scientific substitutions. "
              "No historical or different-weight results are mixed into this report.", ""]
    if verification is None:
        lines += ["Independent artifact verification: **not requested**. Run with `--verify`; descriptive report "
                  "generation alone is not a provenance or leakage verification."]
    else:
        lines += [f"Independent saved-artifact verification: **{verification['status']}**. "
                  "See `verification.json` for exactly checked evidence and limitations. "
                  "Saved audits are checked, not a replay of GPU fitting or a proof of unsaved execution history."]
    return "\n".join(lines) + "\n"


def summarize(run_dir):
    """Recompute OOF outcomes, average masks within seed, and write descriptive reports."""
    root, manifest, records = _read_run(run_dir)
    oof = _oof(root, records)
    main = oof[oof.result_type == "main"]
    summary = _aggregate(main, ["classifier", "variant"])
    replay = _replay_seed_means(oof)
    replay_summary = _aggregate(replay, ["classifier", "replay_strategy"])
    paired = _paired(oof, replay)
    paired_summary = _aggregate(paired, ["classifier", "contrast"])
    outputs = {"oof_metrics.csv": oof, "summary.csv": summary, "paired_deltas.csv": paired,
               "paired_summary.csv": paired_summary, "replay_summary.csv": replay_summary}
    for name, frame in outputs.items():
        frame.to_csv(root / name, index=False)
    # Never present a stale verification from a different invocation as current evidence.
    (root / "RESULTS.md").write_text(
        _markdown(manifest, records, summary, paired_summary, replay_summary, oof, None), encoding="utf-8")
    return {"paths": [*outputs, "RESULTS.md"], "main_records": len(records[records.result_type == "main"]),
            "replay_records": len(records[records.result_type == "replay"]), "oof_records": len(oof),
            "mode": manifest["mode"]}


def _indices(value, size, label):
    values = np.asarray(value)
    _require(values.ndim == 1 and np.issubdtype(values.dtype, np.integer), f"Noninteger indices: {label}")
    _require(len(values) > 0 and len(np.unique(values)) == len(values), f"Empty/duplicate indices: {label}")
    _require(((values >= 0) & (values < size)).all(), f"Out-of-range indices: {label}")
    return values.astype(np.int64, copy=False)


def _check_roles(shared, dataset, fold, seed):
    row_ids = np.asarray([str(value) for value in dataset["row_ids"]])
    y = np.asarray(dataset["y"])
    groups = np.asarray(dataset["duplicate_groups"])
    _require(len(row_ids) == len(y) == len(groups) and len(np.unique(row_ids)) == len(y),
             "Dataset identity/label/group dimensions differ or IDs repeat")
    _require(set(np.unique(y)) == {0, 1}, "Dataset labels must be binary positive indicators")
    split = _npz(shared / "split.npz")
    roles = {key: _indices(split[key], len(y), f"fold={fold},seed={seed},{key}")
             for key in ("fit", "validation", "train", "test", "flow_fit", "flow_calibration")}
    for left, right in (("fit", "validation"), ("fit", "test"), ("validation", "test"),
                        ("train", "test"), ("flow_fit", "flow_calibration")):
        _require(not np.intersect1d(roles[left], roles[right]).size, f"Row leakage: {shared}/{left}/{right}")
        _require(not np.intersect1d(groups[roles[left]], groups[roles[right]]).size,
                 f"Duplicate predictor-group leakage: {shared}/{left}/{right}")
    _require(set(roles["fit"]) | set(roles["validation"]) == set(roles["train"]),
             f"Inner roles do not partition train: {shared}")
    _require(set(roles["train"]) | set(roles["test"]) == set(range(len(y))),
             f"Outer roles do not cover dataset: {shared}")
    _require(set(roles["flow_fit"]) | set(roles["flow_calibration"]) == set(roles["fit"]),
             f"Flow roles do not partition inner-fit: {shared}")
    features = _npz(shared / "role_X.npz") if (shared / "role_X.npz").is_file() else None
    if features is not None:
        for role in ("fit", "validation", "train", "test"):
            _require(role in features and len(features[role]) == len(roles[role])
                     and np.isfinite(features[role]).all(), f"Invalid shared role features: {shared}/{role}")
    return roles, features


def _flow_audit(shared, roles):
    metadata = _load(shared / "metadata.json")
    _require(isinstance(metadata.get("source_hashes"), dict) and metadata["source_hashes"],
             f"Missing source hashes: {shared}")
    flow = metadata.get("flow")
    _require(isinstance(flow, dict), f"Missing frozen detector audit: {shared}")
    _require(flow.get("frozen") is True, f"Flow is not audited frozen: {shared}")
    _require("cuda" in str(flow.get("backend", "")).lower(), f"Flow is not audited CUDA: {shared}")
    _require(flow.get("fit_call_count") == 1, f"Flow fit count is not one per shared split: {shared}")
    _require(flow.get("unused_calibration") is True, f"Flow reserve not explicitly unused: {shared}")
    _require(flow.get("fit_count") == len(roles["flow_fit"]), f"Flow fit size mismatch: {shared}")
    _require(np.array_equal(np.asarray(flow.get("fit_indices", [])), roles["flow_fit"]),
             f"Flow audit training indices differ from frozen role: {shared}")
    _require(flow.get("synthetic_training_count") == 0 and flow.get("calibration_used") is False
             and metadata.get("calibration_used") is False,
             f"Flow training/reserve audit violates real-only unused-reserve protocol: {shared}")
    _require((shared / "flow.pt").is_file(), f"Missing frozen Flow artifact: {shared}")
    return metadata["source_hashes"]


def _training(root, artifact, role, split, features, y, variant, proposal_cap):
    audit = _load(artifact / "audit.json")
    train = _npz(artifact / "final_training.npz")
    _require({"X", "y"} <= set(train), f"Missing final training arrays: {artifact}")
    X, labels = train["X"], train["y"]
    indices = split["fit" if role == "inner" else "train"]
    real_count = len(indices)
    accepted = len(labels) - real_count
    _require(X.ndim == 2 and len(X) == len(labels) and accepted >= 0 and np.isfinite(X).all(),
             f"Malformed final training arrays: {artifact}")
    _require(np.array_equal(labels[:real_count], y[indices]), f"Real label prefix changed: {artifact}")
    _require(np.all(labels[real_count:] == 1), f"Accepted labels are not minority-positive: {artifact}")
    if features is not None:
        _require(np.array_equal(X[:real_count], features["fit" if role == "inner" else "train"]),
                 f"Real feature prefix changed: {artifact}")
    _require(audit.get("real_count") == real_count and audit.get("accepted_count") == accepted,
             f"Training counts disagree with audit: {artifact}")
    _require(audit.get("role") == role and audit.get("real_prefix_count") == real_count,
             f"Wrong training role/prefix audit: {artifact}")
    _require(audit.get("requested_count") == (0 if variant == "A0" else proposal_cap),
             f"Wrong requested proposal budget: {artifact}")
    _require(audit.get("real_weight") == 1 and audit.get("synthetic_weight") == 1,
             f"Non-unit or missing weights: {artifact}")
    for key in ("sample_weight", "weights"):
        if key in train:
            _require(len(train[key]) == len(labels) and np.all(train[key] == 1), f"Non-unit saved weights: {artifact}")
    for key in ("requested_count", "proposal_count", "rejected_count"):
        _require(isinstance(audit.get(key), int) and audit[key] >= 0, f"Invalid audit {key}: {artifact}")
    _require(accepted <= audit["proposal_count"] <= proposal_cap,
             f"Proposal cap/acceptance violation: {artifact}")
    _require(audit["proposal_count"] - accepted == audit["rejected_count"],
             f"Proposal accounting mismatch: {artifact}")
    if variant == "A0":
        _require(accepted == audit["proposal_count"] == 0, f"Real-only arm contains synthesis: {artifact}")
    if variant in ("A2", "A4", "A5", "A6"):
        _require(audit.get("gate_no_relocation") is True, f"Flow gate relocation not excluded: {artifact}")
        _require(audit.get("original_parents") is True, f"Original-parent provenance missing: {artifact}")
    if "source_trace" not in audit:
        _require(audit.get("flow_state_before") == audit.get("flow_state_after")
                 and isinstance(audit.get("flow_state_before"), str),
                 f"Frozen Flow state changed or missing: {artifact}")
    if variant != "A0" and "source_trace" not in audit:
        summary = audit.get("support_summary", {})
        flags = summary.get("variant_flags", {})
        _require(flags.get("refill") is False and flags.get("all_sample_weights_one") is True,
                 f"Refill/unit-weight protocol flags missing: {artifact}")
        _require(flags.get("flow_gate") == (variant in ("A2", "A4", "A5", "A6"))
                 and flags.get("native_relocation") == (variant == "A7"),
                 f"Wrong gate/relocation arm: {artifact}")
        budget = summary.get("proposal_budget", {})
        _require(budget.get("cap") == proposal_cap and budget.get("attempted") == audit["proposal_count"]
                 and budget.get("accepted") == accepted,
                 f"Compact proposal budget differs from training evidence: {artifact}")
        ancestry = audit.get("provenance")
        _require(isinstance(ancestry, list) and len(ancestry) == accepted,
                 f"Missing or misaligned synthetic ancestry: {artifact}")
        for offset, item in enumerate(ancestry):
            _require(item.get("row_index") == real_count + offset and item.get("class") == 1,
                     f"Synthetic ancestry order/class mismatch: {artifact}")
            for name in ("seed_index", "partner_index"):
                parent = item.get(name, -1)
                _require(0 <= parent < real_count and labels[parent] == 1,
                         f"Non-real or cross-class interpolation parent: {artifact}")
            relocation = item.get("relocation_index", -1)
            _require(relocation == -1 or (variant == "A7" and 0 <= relocation < real_count
                                         and labels[relocation] == 1),
                     f"Relocation outside native A7 real support: {artifact}")
            if relocation >= 0:
                _require(np.array_equal(X[real_count + offset], X[relocation]),
                         f"Native relocation does not copy its real donor: {artifact}")
        use_u2 = variant in ("A3", "A4", "A5", "A6", "A7")
        groups = summary.get("group_support", {})
        _require(flags.get("u2_grouping") == use_u2 and groups.get("enabled") == use_u2,
                 f"Wrong subgroup component: {artifact}")
        _require(flags.get("stability_weights_removed") == (variant == "A5")
                 and flags.get("nonbinding_seed_group_caps") == (variant == "A6"),
                 f"Wrong component knockout flags: {artifact}")
        if use_u2:
            grouping = groups.get("configuration", {})
            _require(all(grouping.get(key) == value for key, value in
                         {"grouping": "kmeans", "n_clusters": 8, "repeats": 5, "sample_fraction": .8}.items()),
                     f"Noncanonical grouping: {artifact}")
            allocation = groups.get("allocation", {})
            _require(all(allocation.get(key) == value for key, value in
                         {"allocation": "stability", "group_alpha": .5, "epsilon": .001}.items()),
                     f"Allocation weights changed beyond specified intervention: {artifact}")
            seed_cap, group_cap = allocation.get("seed_cap", 0), allocation.get("group_cap_multiplier", 0)
            _require((seed_cap >= proposal_cap and group_cap >= proposal_cap) if variant == "A6"
                     else (seed_cap == 4 and group_cap == 2.),
                     f"Wrong subgroup/anchor caps: {artifact}")
            multiplicity = groups.get("attempted_seed_multiplicity", {})
            _require(multiplicity.get("maximum", proposal_cap + 1) <= seed_cap,
                     f"Attempted anchor cap exceeded: {artifact}")
            for group in groups.get("groups", []):
                original = group.get("original_stability")
                _require(original == group.get("stability"),
                         f"Knockout changed fitted subgroup stability: {artifact}")
                effective = 1. if variant == "A5" and original > 0 else original
                _require(group.get("effective_stability") == effective,
                         f"Wrong stability allocation intervention: {artifact}")
                supported = group["real_support"] >= 2 and group["distinct_coordinates"] >= 2 and original > 0
                _require(group.get("supported") == supported,
                         f"Knockout changed original support eligibility: {artifact}")
                _require(group.get("attempted_generated", -1) <= int(group_cap * group["real_support"]),
                         f"Attempted subgroup cap exceeded: {artifact}")
    return audit, train


def _trace(artifact, audit, training, real_count, max_rounds):
    trace = _npz(artifact / "proposal_trace.npz")
    required = {"X", "y", "round", "group_id", "seed_index", "partner_index", "accepted", "valid", "log_ratio"}
    _require(required <= set(trace), f"Incomplete proposal trace: {artifact}")
    count = len(trace["y"])
    _require(all(len(trace[key]) == count for key in required), f"Trace dimension mismatch: {artifact}")
    _require(count == audit["proposal_count"], f"Trace/audit proposal count mismatch: {artifact}")
    for key in ("accepted", "valid"):
        _require(trace[key].dtype == np.bool_ and trace[key].ndim == 1, f"Trace {key} must be boolean: {artifact}")
    for key in ("round", "group_id", "seed_index", "partner_index", "y"):
        _require(np.issubdtype(trace[key].dtype, np.integer) and trace[key].ndim == 1,
                 f"Trace {key} must be integer vector: {artifact}")
    _require(not np.any(trace["accepted"] & ~trace["valid"]), f"Accepted invalid proposals: {artifact}")
    _require(int(trace["accepted"].sum()) == audit["accepted_count"], f"Trace accepted count mismatch: {artifact}")
    _require(np.all(trace["y"] == 1), f"Nonminority proposals: {artifact}")
    for key in ("seed_index", "partner_index"):
        values = trace[key]
        _require(((values >= 0) & (values < real_count)).all(), f"Nonoriginal trace parents: {artifact}")
        _require(np.all(training["y"][values] == trace["y"]), f"Cross-class trace parents: {artifact}")
    _require(len(np.unique(trace["round"])) <= max_rounds and (trace["round"] >= 0).all(),
             f"Too many/invalid generation rounds: {artifact}")
    _require(np.isfinite(trace["X"][trace["valid"]]).all(), f"Valid proposals contain nonfinite coordinates: {artifact}")
    _require(np.isfinite(trace["log_ratio"][trace["accepted"]]).all(), f"Accepted proposals lack Flow scores: {artifact}")
    _require(np.all(trace["log_ratio"][trace["accepted"]] >= np.log(.6) - np.log1p(-.6)),
             f"Accepted proposal fails frozen tau=0.6 gate score: {artifact}")
    _require(np.array_equal(training["X"][real_count:], trace["X"][trace["accepted"]])
             and np.array_equal(training["y"][real_count:], trace["y"][trace["accepted"]]),
             f"Gate moved/reordered accepted proposals: {artifact}")
    # The actual parent indices and unchanged accepted coordinates are verified;
    # unsaved interpolation RNG and rejected postprocessing are not reconstructed.
    return trace


def _mask_check(root, artifact, trace, strategy, training, real_count, source_trace):
    mask = np.load(artifact / "replay_mask.npy", allow_pickle=False)
    _require(mask.dtype == np.bool_ and mask.shape == trace["accepted"].shape,
             f"Malformed replay mask: {artifact}")
    _require(not np.any(mask & ~trace["valid"]), f"Replay selects invalid original proposals: {artifact}")
    if strategy == "flow":
        _require(np.array_equal(mask, trace["accepted"]), f"Flow replay mask differs from original gate: {artifact}")
    else:
        keys = ["round"] if strategy == "random_round" else ["round", "group_id"]
        strata = np.column_stack([trace[key] for key in keys])
        for stratum in np.unique(strata, axis=0):
            selected = np.all(strata == stratum, axis=1)
            _require(int(mask[selected].sum()) == int(trace["accepted"][selected].sum()),
                     f"Replay accepted-stratum count mismatch: {artifact}/{tuple(stratum)}")
    _require(np.array_equal(training["X"][real_count:], trace["X"][mask])
             and np.array_equal(training["y"][real_count:], trace["y"][mask]),
             f"Replay final training set differs from mask: {artifact}")
    audit = _load(artifact / "audit.json")
    _require(_path(root, audit.get("source_trace")) == source_trace.resolve(),
             f"Replay does not identify matching original trace: {artifact}")


def _selection(root, cell, baseline_validation):
    selection = _load(cell / "selection.json")
    candidates = selection.get("candidates", selection.get("candidate_configs"))
    _require(isinstance(candidates, list) and bool(candidates), f"Missing candidate selection evidence: {cell}")
    selected = selection.get("selected_parameters")
    _require(isinstance(selected, dict), f"Missing locked selected parameters: {cell}")
    identities = [candidate.get("candidate_id") for candidate in candidates]
    _require(None not in identities and len(set(identities)) == len(identities),
             f"Missing/duplicate candidate identities: {cell}")
    selected_id = selection.get("selected_candidate_id")
    _require(selected_id in identities, f"Selected candidate identity is absent: {cell}")
    best = None
    for candidate in candidates:
        _require(isinstance(candidate.get("parameters"), dict), f"Missing candidate parameters: {cell}")
        for metric in ("validation_f1", "validation_ap", "threshold"):
            _require(np.isfinite(candidate.get(metric, np.nan)), f"Missing candidate {metric}: {cell}")
        _require(0 <= candidate["validation_f1"] <= 1 and 0 <= candidate["validation_ap"] <= 1,
                 f"Invalid candidate validation metrics: {cell}")
        numerator, denominator = candidate.get("f1_numerator"), candidate.get("f1_denominator")
        _require(isinstance(numerator, int) and isinstance(denominator, int) and denominator > 0,
                 f"Missing exact candidate F1 counts: {cell}")
        _close(candidate["validation_f1"], numerator / denominator, f"Candidate F1 fraction: {cell}")
        if best is None:
            best = candidate
        else:
            comparison = numerator * best["f1_denominator"] - best["f1_numerator"] * denominator
            if comparison > 0 or (comparison == 0 and candidate["validation_ap"] > best["validation_ap"]):
                best = candidate
    _require(best["candidate_id"] == selected_id and best["parameters"] == selected,
             f"Selected classifier does not follow exact real-only F1/AP selection: {cell}")
    matched = best
    metrics = _metrics(baseline_validation, cell.parent.name)
    _close(matched["validation_f1"], metrics["minority_f1"], f"Selected candidate baseline F1: {cell}")
    _close(matched["validation_ap"], metrics["average_precision"], f"Selected candidate baseline AP: {cell}")
    _close(matched["threshold"], baseline_validation.threshold.iloc[0], f"Selected candidate threshold: {cell}")
    return selected, len(candidates)


def _verify_impl(run_dir):
    root, manifest, records = _read_run(run_dir)
    dataset = _load(root / "dataset.json")
    y = np.asarray(dataset["y"])
    if manifest["mode"] == "scientific":
        _require(len(y) == 12330, "Scientific Shoppers run does not cover all 12,330 source rows")
    row_ids = np.asarray([str(value) for value in dataset["row_ids"]])
    config = manifest.get("config", {})
    proposal_cap = int(config.get("proposal_cap", 1000))
    max_rounds = int(config.get("max_rounds", 20))
    _require(0 < proposal_cap <= 1000 and 0 < max_rounds <= 20, "Proposal budget exceeds registered cap")
    context = {}
    source_hashes = {}
    missing_role_features = []
    for fold, seed in records[["fold", "seed"]].drop_duplicates().itertuples(index=False, name=None):
        shared = root / "shared" / f"fold-{fold}_seed-{seed}"
        split, features = _check_roles(shared, dataset, fold, seed)
        source_hashes[f"fold-{fold}_seed-{seed}"] = _flow_audit(shared, split)
        _require(source_hashes[f"fold-{fold}_seed-{seed}"] == manifest.get("source_hashes"),
                 f"Shared source hashes differ from run manifest: {shared}")
        if features is None:
            missing_role_features.append(str(shared.relative_to(root)))
        context[(fold, seed)] = (split, features)
    for seed in records.seed.unique():
        tests = np.concatenate([split["test"] for (fold_key, seed_key), (split, _) in context.items() if seed_key == seed])
        _require(len(tests) == len(y) and len(np.unique(tests)) == len(y),
                 f"Outer test roles fail unique full dataset coverage: seed={seed}")
    test_reference = {}
    baseline_validation = {}
    trace_cache = {}
    selected_by_cell = {}
    support_profiles = {}
    candidate_counts = {}
    backend_expected = {"mlp": "pytorch_cuda", "xgboost": "xgboost_cuda", "random_forest": "sklearn_cpu"}
    # Main records first so every replay can refer to an already checked A4 stream.
    ordered = pd.concat([records[records.result_type == "main"], records[records.result_type == "replay"]])
    for record in ordered.itertuples(index=False):
        identity = (record.classifier, record.fold, record.seed, record.variant, record.result_type,
                    record.replay_strategy, record.mask_seed)
        _require(record.backend == backend_expected[record.classifier], f"Wrong native backend: {identity}")
        for counter in COUNTERS:
            value = float(getattr(record, counter))
            _require(np.isfinite(value) and value >= 0, f"Invalid counter {counter}: {identity}")
            if counter not in ("native_fit_seconds", "runtime_seconds"):
                _require(value == int(value), f"Noninteger count {counter}: {identity}")
        _require(record.proposal_count <= proposal_cap and record.round_count <= max_rounds,
                 f"Record proposal/round cap exceeded: {identity}")
        _require(record.accepted_count + record.rejected_count == record.proposal_count,
                 f"Record proposal accounting mismatch: {identity}")
        split, features = context[(record.fold, record.seed)]
        threshold = float(record.threshold)
        outer = _prediction(root, record.predictions_path, record.classifier, threshold)
        inner = _prediction(root, record.validation_predictions_path, record.classifier, threshold)
        _require(threshold == _best_threshold(inner), f"Threshold is not selected on matching real validation role: {identity}")
        for role, predictions, indices in (("outer", outer, split["test"]), ("inner", inner, split["validation"])):
            _require(predictions.row_id.tolist() == row_ids[indices].tolist(), f"Prediction identity/order mismatch: {identity}/{role}")
            _require(np.array_equal(predictions.y_true.to_numpy(), y[indices]), f"Prediction labels differ: {identity}/{role}")
        reference_key = (record.fold, record.seed)
        identities = (outer.row_id.tolist(), outer.y_true.tolist())
        if reference_key in test_reference:
            _require(identities == test_reference[reference_key], f"Unmatched test identities across arms/classifiers: {identity}")
        else:
            test_reference[reference_key] = identities
        measured = _metrics(outer, record.classifier)
        for name in (*PRIMARY, *CONFUSION, "f1_at_0_5"):
            _close(getattr(record, name), measured[name], f"Native test metric {identity}/{name}")
        outer_artifact = _path(root, record.artifact_dir)
        # Derive inner role solely from the declared outer path's known unit hierarchy.
        cell = root / "cells" / record.classifier / f"fold-{record.fold}_seed-{record.seed}"
        expected_artifact = cell / "outer" / record.variant if record.result_type == "main" else (
            cell / "outer" / "replay" / record.replay_strategy / f"mask-{record.mask_seed}")
        _require(outer_artifact == expected_artifact.resolve(), f"Artifact directory differs from record identity: {identity}")
        inner_artifact = cell / "inner" / record.variant if record.result_type == "main" else (
            cell / "inner" / "replay" / record.replay_strategy / f"mask-{record.mask_seed}")
        _require(_path(root, record.predictions_path).parent == outer_artifact
                 and _path(root, record.validation_predictions_path).parent == inner_artifact.resolve(),
                 f"Prediction files are not in matching training artifact roles: {identity}")
        frozen_threshold = _load(inner_artifact / "threshold.json")
        _require(frozen_threshold.get("threshold") == threshold
                 and frozen_threshold.get("selection") == "real_validation_only"
                 and frozen_threshold.get("selected_before_outer_fit") is True,
                 f"Missing pre-outer real-validation threshold audit: {identity}")
        _close(frozen_threshold.get("validation_f1", np.nan), _metrics(inner, record.classifier)["minority_f1"],
               f"Frozen threshold validation F1: {identity}")
        for role, artifact in (("inner", inner_artifact), ("outer", outer_artifact)):
            audit, training = _training(root, artifact, role, split, features, y, record.variant, proposal_cap)
            if record.result_type == "main" and record.variant in ("A3", "A4", "A5", "A6", "A7"):
                group_support = audit["support_summary"]["group_support"]
                profile = (group_support["row_group_ids"], [
                    (group["id"], group["indices"], group["real_support"], group["distinct_coordinates"],
                     group["original_stability"], group["supported"])
                    for group in group_support["groups"]])
                profile_key = (record.fold, record.seed, role)
                previous_profile = support_profiles.setdefault(profile_key, profile)
                _require(profile == previous_profile,
                         f"Subgroup memberships/support changed across knockouts/classifiers: {identity}/{role}")
            real_count = len(split["fit" if role == "inner" else "train"])
            if role == "outer":
                for name in ("accepted_count", "requested_count", "proposal_count", "rejected_count"):
                    _require(getattr(record, name) == audit[name], f"Record/outer audit {name} mismatch: {identity}")
                for name in ("round_count", "classifier_fit_count", "native_fit_seconds", "runtime_seconds"):
                    _close(getattr(record, name), audit.get(name, np.nan), f"Record/outer native counter {name}: {identity}")
            cell_key = (record.classifier, record.fold, record.seed)
            _require(isinstance(audit.get("native_parameters"), dict),
                     f"Missing locked classifier parameters: {identity}/{role}")
            previous = selected_by_cell.setdefault(cell_key, audit["native_parameters"])
            _require(previous == audit["native_parameters"], f"Classifier setting changed across roles/arms: {identity}")
            if record.result_type == "main" and record.variant == "A4":
                trace_cache[(*cell_key, role)] = _trace(artifact, audit, training, real_count, max_rounds)
            elif record.result_type == "replay":
                trace = trace_cache[(*cell_key, role)]
                source_trace = cell / role / "A4" / "proposal_trace.npz"
                _mask_check(root, artifact, trace, record.replay_strategy, training, real_count, source_trace)
        cell_key = (record.classifier, record.fold, record.seed)
        if record.result_type == "main" and record.variant == "A0":
            baseline_validation[cell_key] = inner
        if record.result_type == "replay" and record.replay_strategy == "flow":
            main_record = records[(records.classifier == record.classifier) & (records.fold == record.fold)
                                  & (records.seed == record.seed) & (records.variant == "A4")
                                  & (records.result_type == "main")].iloc[0]
            _require(threshold == float(main_record.threshold), f"Flow replay threshold differs from A4: {identity}")
            for predictions, relative in ((outer, main_record.predictions_path), (inner, main_record.validation_predictions_path)):
                original = _prediction(root, relative, record.classifier, float(main_record.threshold))
                _require(np.array_equal(predictions[["probability", "y_pred", "y_pred_05"]].to_numpy(),
                                        original[["probability", "y_pred", "y_pred_05"]].to_numpy()),
                         f"Flow replay prediction parity failure: {identity}")
    for cell_key, validation in baseline_validation.items():
        classifier, fold, seed = cell_key
        cell = root / "cells" / classifier / f"fold-{fold}_seed-{seed}"
        selected, count = _selection(root, cell, validation)
        _require(count == int(config.get("n_candidates", 10)), f"Wrong baseline candidate count: {cell}")
        candidate_counts[f"{classifier}/fold-{fold}_seed-{seed}"] = count
        _require(selected_by_cell[cell_key] == selected, f"Training settings differ from real-only selection: {cell}")
    # Reusing the summarizer's pooling path also independently checks each arm's OOF uniqueness.
    oof = _oof(root, records)
    _require((oof.oof_rows == len(y)).all(), "Some arms lack full dataset OOF coverage")
    return {"status": "passed", "mode": manifest["mode"], "main_records": int((records.result_type == "main").sum()),
            "replay_records": int((records.result_type == "replay").sum()), "oof_records": len(oof),
            "dataset_rows": len(y), "shared_split_contexts": len(context), "frozen_flow_audits": len(context),
            "native_predictive_metrics_recomputed": True, "mlp_ap_restores_float32": True,
            "threshold_predictions_exact": True, "ordered_identities_and_labels_matched": True,
            "duplicate_groups_and_roles_disjoint": True, "full_dataset_oof_coverage": True,
            "unit_weights_and_minority_accepted_labels_checked": True,
            "proposal_caps_original_parent_indices_and_unchanged_accepted_gate_coordinates_checked": True,
            "replay_valid_subsets_and_exact_stratum_counts_checked": True,
            "flow_replay_prediction_parity_checked": True, "backend_records_checked": True,
            "real_feature_prefix_unavailable": missing_role_features,
            "candidate_counts": candidate_counts, "source_hashes": source_hashes,
            "limitations": ["No refitting or regeneration; execution history is represented by saved audits.",
                            "Candidate scores independently recomputed only for the selected real-only candidate.",
                            "Interpolation RNG, unsaved rejected postprocessing, and GPU kernel execution are not reconstructed.",
                            "Fixed-pool replay is conditional on the A4 proposal stream, not independent random trajectories."]}


def verify(run_dir):
    """Verify saved identities, metrics, splits, training prefixes, proposals and replays.

    A failed verification is saved before the original exception is raised.
    """
    root = Path(run_dir)
    try:
        result = _verify_impl(root)
    except Exception as error:
        if root.is_dir():
            _save_json(root / "verification.json", {"status": "failed", "error_type": type(error).__name__,
                                                    "error": str(error)})
        raise
    _save_json(root / "verification.json", result)
    # Refresh only the report (not any experiment input) with this invocation's result.
    root, manifest, records = _read_run(root)
    oof = _oof(root, records)
    summary = _aggregate(oof[oof.result_type == "main"], ["classifier", "variant"])
    replay_summary = _aggregate(_replay_seed_means(oof), ["classifier", "replay_strategy"])
    paired_summary = _aggregate(_paired(oof, _replay_seed_means(oof)), ["classifier", "contrast"])
    (root / "RESULTS.md").write_text(
        _markdown(manifest, records, summary, paired_summary, replay_summary, oof, result), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--verify", action="store_true", help="Also verify saved experiment evidence")
    args = parser.parse_args()
    summary = summarize(args.run_dir)
    if args.verify:
        summary["verification"] = verify(args.run_dir)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
