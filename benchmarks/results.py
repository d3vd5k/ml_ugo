"""Protocol-bound metrics and indexes linked to the existing experiment registry.

Threshold selection consumes inner validation only. Test ROC operating points are
performance descriptions, never deployable threshold selection. Summaries are
strictly descriptive; overlapping folds are not independent observations.
"""
import csv
import fcntl
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef)

from experiments import RunArtifacts, write_json


EXPERIMENT_TYPES = {"setup_smoke", "setup_prepare", "development", "confirmatory", "reproduction"}
METRIC_KEYS = {"average_precision", "macro_average_precision", "minority_f1", "minority_recall", "macro_f1",
               "weighted_f1", "sos_inverse_support_f1", "balanced_accuracy", "mcc", "fp", "fn", "tp", "tn",
               "cost", "recall_at_fpr_test", "roc_attained_fpr_test",
               "realized_recall_test", "realized_fpr_test", "primary_metric_value"}
IDENTITY_FIELDS = ("protocol_id", "dataset_fingerprint", "split_fingerprint", "seed",
                   "algorithm", "method", "variant_id", "budget_mode", "config_sha256")
INDEX_FIELDS = ("run_id", "record_id", "configuration_id", "artifact_path", "status", "error",
                "experiment_type", "study_id", "study_type", "protocol_sha256", "study_sha256",
                "record_scope",
                "dataset", *IDENTITY_FIELDS, "tuning_budget", "candidate_configs",
                "evaluation_policy_sha256", "augmentation", "metrics", "runtime_seconds",
                "peak_rss_bytes", "cuda_peak_torch_allocated_bytes", "cuda_peak_torch_reserved_bytes",
                "memory_scope", "predictions_path")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON metadata: {type(value).__name__}")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=_json_default, allow_nan=False)


def snapshot_sha256(value):
    """Hash the canonical JSON snapshot, not a mutable file path."""
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def load_study(path):
    with Path(path).open(encoding="utf-8") as handle:
        study = yaml.safe_load(handle)
    if not isinstance(study, dict) or not study.get("study_id"):
        raise ValueError("Study requires study_id")
    if study.get("study_type") not in {"baseline_control", "component_ablation"}:
        raise ValueError("Study must separate baseline controls from component ablations")
    return study


def require_variant(study, variant_id):
    variants = study.get("variants", {})
    if variant_id not in variants:
        raise ValueError(f"Unknown variant {variant_id!r} in {study.get('study_id')}")
    variant = variants[variant_id]
    if not variant.get("available", False) or variant.get("missing_components"):
        missing = ", ".join(variant.get("missing_components", [])) or "unavailable implementation"
        raise ValueError(f"Variant {variant_id} unavailable: {missing}")
    decisions = study.get("manual_review", {}).get("decisions", {})
    pending = [name for name in variant.get("requires_user_decisions", [])
               if decisions.get(name) not in {"keep", "discard"}]
    if pending:
        raise ValueError(f"Variant {variant_id} awaits user decisions: {", ".join(pending)}")
    return variant


def enforce_stage_gate(study, experiment_type):
    if experiment_type not in EXPERIMENT_TYPES:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")
    freeze = study.get("freeze", {})
    if study.get("stage_gate", {}).get("stage2_allowed") and freeze.get("status") != "frozen":
        raise ValueError("Stage 2 cannot be enabled before the evidence-backed Stage 1 freeze")
    if freeze.get("status") == "frozen":
        review = study.get("manual_review")
        if review is not None and any(review.get("decisions", {}).get(name) not in {"keep", "discard"}
                                      for name in ("U1", "U2", "U3", "U4", "U5")):
            raise ValueError("Phase 1 cannot freeze before explicit user keep/discard decisions")
        evidence = freeze.get("evidence", {})
        required = ("component_justifications", "completed_ablations", "generation_vs_W5_R5",
                    "simulation_diagnostics", "configuration_sha256")
        if any(not evidence.get(key) for key in required):
            raise ValueError("Frozen status requires Stage 1 gate evidence, not a declaration")
        if study.get("study_type") != "component_ablation":
            raise ValueError("Baseline control studies cannot certify the component-ablation freeze gate")
        for variant_id in ("U0", "U1", "U2", "U3", "U4", "U5", "W5", "R5"):
            require_variant(study, variant_id)
        if not {"U0", "U1", "U2", "U3", "U4", "U5", "W5", "R5"}.issubset(set(evidence["completed_ablations"])):
            raise ValueError("Freeze gate requires the complete declared ablation set")
    if experiment_type == "confirmatory" and freeze.get("status") != "frozen":
        raise ValueError("Confirmatory evaluation requires an evidence-backed frozen configuration")


def _binary_inputs(y, probabilities):
    y = np.asarray(y)
    probabilities = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or probabilities.shape != y.shape or not len(y):
        raise ValueError("Binary labels and probabilities must be nonempty aligned vectors")
    if not np.isin(y, [0, 1, False, True]).all() or len(np.unique(y)) != 2:
        raise ValueError("Operating-point evaluation requires both binary classes")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    return y.astype(bool), probabilities


def _attainable_points(y, probabilities):
    """All >= thresholds, with tied scores entering as a single block."""
    y, probabilities = _binary_inputs(y, probabilities)
    order = np.argsort(-probabilities, kind="stable")
    scores, labels = probabilities[order], y[order]
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    tp = np.r_[0, np.cumsum(labels)[ends]]
    fp = np.r_[0, np.cumsum(~labels)[ends]]
    thresholds = np.r_[np.nextafter(scores[0], np.inf), scores[ends]]
    return thresholds, tp, fp, int(y.sum()), int((~y).sum())


def choose_threshold(yval, prob_positive, rule):
    """Select on validation only; ties favor the highest (most conservative) threshold."""
    options = rule if isinstance(rule, dict) else {"name": rule}
    name = options.get("name", options.get("rule", "default"))
    if name in {"default", "fixed", "fixed_0.5"}:
        return float(options.get("threshold", .5))
    thresholds, tp, fp, positives, negatives = _attainable_points(yval, prob_positive)
    if name in {"cost", "aps_cost"}:
        costs = options.get("fp_cost", 10) * fp + options.get("fn_cost", 500) * (positives - tp)
        index = int(np.argmin(costs))
    elif name in {"fpr", "recall_at_fpr", "recall_at_fpr_0.05"}:
        limit = float(options.get("fpr", .05))
        if not 0 <= limit <= 1:
            raise ValueError("FPR must be in [0, 1]")
        eligible = np.flatnonzero(fp / negatives <= limit)
        index = int(eligible[np.argmax(tp[eligible])])
    else:
        raise ValueError(f"Unknown threshold rule: {name}")
    return float(thresholds[index])


def metric_results(y_true, probabilities, classes, positive_class, threshold,
                   primary_metric, fpr=.05):
    """Evaluate a fixed threshold and attainable test ROC point separately.

    recall_at_fpr_test is max TPR over observed >= thresholds with FPR <= fpr,
    without interpolation or randomized tie breaking. It is not the recall of
    the validation-selected threshold, reported as realized_recall_test.
    """
    y_true, classes = np.asarray(y_true), np.asarray(classes)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 1 or probabilities.shape != (len(y_true), len(classes)) or not len(y_true):
        raise ValueError("Expected aligned labels and an n_rows by n_classes probability matrix")
    if classes.ndim != 1 or len(classes) < 2 or len(np.unique(classes)) != len(classes) or not np.isin(y_true, classes).all():
        raise ValueError("At least two unique classes must contain all evaluation labels")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Probability rows must sum to one")
    if positive_class not in classes:
        raise ValueError("positive_class must be present in classes")
    binary = len(classes) == 2
    positive_index = int(np.flatnonzero(classes == positive_class)[0])
    scores = probabilities[:, positive_index]
    if binary:
        if threshold is None or not np.isfinite(threshold):
            raise ValueError("Binary evaluation needs a fixed finite validation-selected threshold")
        negative = classes[1 - positive_index]
        prediction = np.where(scores >= threshold, positive_class, negative)
    else:
        prediction = classes[np.argmax(probabilities, axis=1)]
    class_f1 = f1_score(y_true, prediction, labels=classes, average=None, zero_division=0)
    prevalence = np.asarray([np.mean(y_true == label) for label in classes])
    inverse_support_weights = (1 - prevalence) / (len(classes) - 1)
    class_ap = [float(average_precision_score(y_true == label, probabilities[:, index]))
                if np.any(y_true == label) else None for index, label in enumerate(classes)]
    positive = y_true == positive_class
    minority_predicted = prediction == positive_class
    tp = int(np.sum(positive & minority_predicted))
    fn = int(np.sum(positive & ~minority_predicted))
    fp = int(np.sum(~positive & minority_predicted))
    tn = int(np.sum(~positive & ~minority_predicted))
    result = {
        "average_precision": float(average_precision_score(positive, scores)) if positive.any() else None,
        "macro_average_precision": float(np.mean(class_ap)) if all(value is not None for value in class_ap) else None,
        "minority_f1": float(f1_score(positive, minority_predicted, zero_division=0)),
        "minority_recall": tp / (tp + fn) if tp + fn else None,
        "macro_f1": float(f1_score(y_true, prediction, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, prediction, labels=classes, average="weighted", zero_division=0)),
        "sos_inverse_support_f1": float(np.dot(inverse_support_weights, class_f1)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "threshold": float(threshold) if binary else None,
        "fp": fp if binary else None, "fn": fn if binary else None,
        "tp": tp if binary else None, "tn": tn if binary else None,
        "cost": 10 * fp + 500 * fn if binary else None,
        "realized_recall_test": tp / (tp + fn) if binary and tp + fn else None,
        "realized_fpr_test": fp / (fp + tn) if binary and fp + tn else None,
        "recall_at_fpr_test": None, "roc_attained_fpr_test": None,
        "roc_operating_point_definition": "maximum attainable TPR with FPR <= limit; >= threshold; ties intact; no interpolation",
        "roc_fpr_limit": float(fpr),
    }
    if not 0 <= fpr <= 1:
        raise ValueError("FPR must be in [0, 1]")
    if binary and positive.any() and (~positive).any():
        _, roc_tp, roc_fp, positives, negatives = _attainable_points(positive, scores)
        eligible = np.flatnonzero(roc_fp / negatives <= fpr)
        index = int(eligible[np.argmax(roc_tp[eligible])])
        result.update(recall_at_fpr_test=float(roc_tp[index] / positives),
                      roc_attained_fpr_test=float(roc_fp[index] / negatives))
    aliases = {"ap": "average_precision", "pr_auc": "average_precision",
               "recall_at_fpr": "recall_at_fpr_test", "aps_cost": "cost"}
    primary = aliases.get(primary_metric, primary_metric)
    if primary not in METRIC_KEYS or primary == "primary_metric_value":
        raise ValueError(f"Unknown primary metric: {primary_metric}")
    result.update(primary_metric=primary, primary_metric_value=result[primary])
    return result


class BenchmarkArtifacts(RunArtifacts):
    """RunArtifacts plus immutable metadata and a linked, locked append-only index.

    results/runs uses results/benchmark_runs.csv; arbitrary output roots keep the
    index local. The index points to the SAME run_id and artifact directory as
    experiment_registry.csv and is not an independent run registry.
    """
    def __init__(self, output_root, arguments, protocol, study, benchmark_index=None):
        experiment_type = arguments.get("experiment_type", "development")
        enforce_stage_gate(study, experiment_type)
        if not protocol.get("protocol_id"):
            raise ValueError("Protocol requires protocol_id")
        self._protocol_json = _canonical(protocol)
        self._study_json = _canonical(study)
        self._arguments_json = _canonical(arguments)
        super().__init__(output_root, json.loads(self._arguments_json))
        repo = Path(__file__).resolve().parents[1]
        canonical_root = repo / "results" / "runs"
        self.benchmark_index = (Path(benchmark_index) if benchmark_index else
                                (repo / "results" / "benchmark_runs.csv"
                                 if self.root.resolve() == canonical_root else self.root / "benchmark_runs.csv"))
        self.protocol_sha256 = snapshot_sha256(protocol)
        self.study_sha256 = snapshot_sha256(study)
        self._finished = False
        self.manifest.update(protocol_snapshot=json.loads(self._protocol_json),
                             study_snapshot=json.loads(self._study_json),
                             protocol_sha256=self.protocol_sha256, study_sha256=self.study_sha256,
                             experiment_type=experiment_type, benchmark_index=str(self.benchmark_index))
        write_json(self.path / "manifest.json", self.manifest)

    @property
    def protocol(self):
        return json.loads(self._protocol_json)

    @property
    def study(self):
        return json.loads(self._study_json)

    def finish(self, rows, error=None, status=None):
        if self._finished:
            raise RuntimeError("An append-only benchmark run can be finished only once")
        run_status = status or ("failed" if error is not None else "completed")
        if run_status not in {"completed", "failed", "prepared", "unsupported"}:
            raise ValueError(f"Unsupported run status: {run_status}")
        records = []
        seen = set()
        arguments = json.loads(self._arguments_json)
        if not rows:
            if run_status == "completed":
                raise ValueError("A completed benchmark requires evaluated records")
            # A failed acquisition has no dataset/split fingerprint. These are
            # explicit unavailable markers, never invented scientific identities.
            rows = [{"record_scope": "run_setup", "status": run_status, "error": error,
                     "dataset": self.protocol.get("dataset", "unavailable"),
                     "dataset_fingerprint": "unavailable:not_loaded",
                     "split_fingerprint": "unavailable:not_created",
                     "seed": arguments.get("random_state", "unavailable:not_selected"),
                     "algorithm": "unavailable:not_selected", "method": "unavailable:not_selected",
                     "variant_id": "unavailable:not_selected",
                     "budget_mode": arguments.get("effective_settings", {}).get("augmentation", {}).get("mode", "matched")} ]
        for raw in rows:
            row = {**arguments, **raw}
            row.setdefault("record_scope", "benchmark_cell")
            row_status = raw.get("status", run_status)
            if row_status not in {"completed", "failed", "prepared", "unsupported"}:
                raise ValueError(f"Unsupported record status: {row_status}")
            row_error = raw.get("error", error if row_status in {"failed", "unsupported"} else None)
            if row_status in {"failed", "unsupported"} and not row_error:
                raise ValueError("Failed/unsupported records require an explicit error")
            augmentation = row.get("augmentation", {})
            budget_mode = row.get("budget_mode", augmentation.get("mode", augmentation.get("budget_mode")))
            if budget_mode not in {"matched", "native"}:
                raise ValueError("Record budget_mode must be matched or native")
            # Seeds/splits/runtime paths are separate identity/provenance fields,
            # not method configuration: otherwise seed summaries never aggregate.
            config = {"protocol": self.protocol, "study": self.study,
                      "effective_settings": arguments.get("effective_settings", {}),
                      "selected_config": row.get("config", {}),
                      "augmentation_budget": {key: augmentation[key] for key in ("mode", "total", "requested_count") if key in augmentation},
                      "tuning_budget": row.get("tuning_budget", self.protocol.get("tuning_budget")),
                      "candidate_configs": row.get("candidate_configs", [])}
            policy = {"protocol": self.protocol, "study": self.study,
                      "effective_settings": arguments.get("effective_settings", {}),
                      "tuning_budget": config["tuning_budget"]}
            policy_sha256 = snapshot_sha256(policy)
            identity = {key: row.get(key) for key in IDENTITY_FIELDS}
            identity.update(protocol_id=self.protocol["protocol_id"], budget_mode=budget_mode,
                            config_sha256=snapshot_sha256(config))
            if any(identity[key] is None or identity[key] == "" for key in IDENTITY_FIELDS):
                raise ValueError(f"Missing record identity fields: {[key for key in IDENTITY_FIELDS if identity[key] is None or identity[key] == '']}")
            configuration_id = snapshot_sha256(identity)
            if configuration_id in seen:
                raise ValueError("Duplicate benchmark cell identity in a run")
            seen.add(configuration_id)
            metrics = {key: row[key] for key in METRIC_KEYS | {"threshold", "primary_metric", "roc_operating_point_definition", "roc_fpr_limit"} if key in row}
            if isinstance(row.get("metrics"), dict):
                metrics.update(row["metrics"])
            if row_status != "completed":
                metrics = {}
                for key in METRIC_KEYS:
                    row.pop(key, None)
            record = {**row, **identity, "run_id": self.run_id,
                      "record_id": snapshot_sha256({"run_id": self.run_id, **identity}),
                      "configuration_id": configuration_id, "artifact_path": str(self.path.resolve()),
                      "status": row_status, "error": row_error,
                      "experiment_type": arguments.get("experiment_type", "development"),
                      "study_id": self.study["study_id"], "study_type": self.study["study_type"],
                      "protocol_sha256": self.protocol_sha256, "study_sha256": self.study_sha256,
                      "evaluation_policy_sha256": policy_sha256,
                      "tuning_budget": config["tuning_budget"], "candidate_configs": config["candidate_configs"],
                      "augmentation": augmentation, "metrics": metrics,
                      "strategy": identity["variant_id"], "fold": identity["split_fingerprint"]}
            records.append(record)
        # Existing registry owns runs and archival provenance. Its row JSON retains
        # per-cell status even when the invocation fails part way through.
        super().finish(records, error=error)
        self.manifest.update(status=run_status)
        write_json(self.path / "manifest.json", self.manifest)
        write_json(self.path / "benchmark_records.json", records)
        self.benchmark_index.parent.mkdir(parents=True, exist_ok=True)
        with self.benchmark_index.open("a+", newline="", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0, os.SEEK_END)
            writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
            if handle.tell() == 0:
                writer.writeheader()
            for record in records:
                writer.writerow({key: _canonical(record[key]) if isinstance(record.get(key), (dict, list))
                                 else record.get(key, "") for key in INDEX_FIELDS})
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle, fcntl.LOCK_UN)
        self._finished = True
        return records


def read_records(path):
    """Read either a linked CSV index or a run's benchmark_records.json."""
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("metrics", "augmentation", "candidate_configs", "tuning_budget"):
            if row.get(key):
                try:
                    row[key] = json.loads(row[key])
                except json.JSONDecodeError:
                    pass
    return rows


def _metric(row, metric):
    value = row.get("metrics", {}).get(metric, row.get(metric))
    return float(value) if value is not None else None


def summarize_records(records, metric="average_precision", include_smoke=False):
    """Group compatible cells; average folds within seeds, then describe seeds.

    No standard error, hypothesis test, or independent-fold inference is made.
    Protocol/study/evaluation-policy hashes, dataset identity and budget tracks never mix.
    Failed/unsupported/prepared counts are retained without success metrics.
    """
    fields = ("protocol_id", "protocol_sha256", "dataset", "dataset_fingerprint", "study_id",
              "study_sha256", "experiment_type", "algorithm", "method", "variant_id",
              "budget_mode", "evaluation_policy_sha256")
    groups = defaultdict(list)
    for row in records:
        if not include_smoke and row.get("experiment_type") in {"setup_smoke", "setup_prepare"}:
            continue
        compatible = {**row, "evaluation_policy_sha256": row.get("evaluation_policy_sha256") or row.get("config_sha256")}
        if not compatible["evaluation_policy_sha256"]:
            raise ValueError("Summary requires a policy hash or conservative legacy configuration hash")
        groups[tuple(compatible.get(key) for key in fields)].append(row)
    summaries = []
    for key, rows in groups.items():
        seeds = defaultdict(list)
        seen = set()
        for row in rows:
            if row.get("status") != "completed":
                continue
            cell = (str(row["seed"]), row["split_fingerprint"])
            if cell in seen:
                raise ValueError("Duplicate split/seed cells; choose one run explicitly before summarizing")
            seen.add(cell)
            value = _metric(row, metric)
            if value is not None:
                seeds[cell[0]].append(value)
        values = [float(np.mean(folds)) for folds in seeds.values()]
        summaries.append({**dict(zip(fields, key)), "metric": metric,
                          "selected_config_sha256s": sorted({row.get("config_sha256", "") for row in rows}),
                          "completed_cells": sum(row.get("status") == "completed" for row in rows),
                          "failed_cells": sum(row.get("status") in {"failed", "unsupported"} for row in rows),
                          "prepared_cells": sum(row.get("status") == "prepared" for row in rows),
                          "seed_count": len(values), "mean": float(np.mean(values)) if values else None,
                          "std_across_seed_means": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                          "aggregation": "fold means within seed, descriptive mean/sample SD across seeds; no independent-fold inference"})
    return summaries


def paired_summary(records, reference_variant, metric="average_precision", include_smoke=False):
    """Return per-split/seed paired deltas, requiring shared tuning and count budget.

    No-resampling/weighting are non-generation controls, not matched mechanism
    claims. Native budgets never assert count matching. Actual underspending is
    visible and disqualifies equal-count mechanism claims when counts differ.
    """
    fields = ("protocol_id", "protocol_sha256", "dataset_fingerprint", "split_fingerprint",
              "seed", "algorithm", "study_id", "study_sha256", "experiment_type", "budget_mode", "evaluation_policy_sha256")
    groups = defaultdict(list)
    for row in records:
        if row.get("status") == "completed" and (include_smoke or row.get("experiment_type") not in {"setup_smoke", "setup_prepare"}):
            compatible = {**row, "evaluation_policy_sha256": row.get("evaluation_policy_sha256") or row.get("config_sha256")}
            if not compatible["evaluation_policy_sha256"]:
                raise ValueError("Pairing requires a policy hash or conservative legacy configuration hash")
            groups[tuple(str(compatible.get(key)) for key in fields)].append(row)
    result = []
    for key, rows in groups.items():
        references = [row for row in rows if row["variant_id"] == reference_variant]
        if len(references) > 1:
            raise ValueError("Duplicate reference cells; select one run before pairing")
        if not references:
            continue
        reference = references[0]
        for row in rows:
            if row is reference:
                continue
            same_tuning = row.get("tuning_budget") == reference.get("tuning_budget")
            same_candidates = row.get("candidate_configs") == reference.get("candidate_configs")
            actual = row.get("augmentation", {}).get("actual_counts")
            reference_actual = reference.get("augmentation", {}).get("actual_counts")
            counts_equal = actual is not None and reference_actual is not None and actual == reference_actual
            mechanism = row.get("method") not in {"baseline", "class_weight"} and reference.get("method") not in {"baseline", "class_weight"}
            matched = row.get("budget_mode") == "matched" and counts_equal and same_tuning and same_candidates and mechanism
            value, baseline = _metric(row, metric), _metric(reference, metric)
            result.append({**dict(zip(fields, key)), "variant_id": row["variant_id"],
                           "reference_variant": reference_variant, "metric": metric,
                           "delta": value - baseline if value is not None and baseline is not None else None,
                           "same_tuning_budget": same_tuning, "actual_counts_equal": counts_equal,
                           "same_candidate_configs": same_candidates,
                           "matched_count_comparison": matched,
                           "actual_counts": actual, "reference_actual_counts": reference_actual})
    return result


def sos_weighted_f1_summary(records, protocol):
    """Aggregate ONLY a resolved exact author layout declared in the protocol.

    Four classifiers/five seeds are supported, not assumed. Repeated-seed
    author trials require explicit repetition IDs; two-classifier layouts stay
    distinct from four-classifier targets. No unresolved comparison is claimed.
    """
    if protocol.get("status", protocol.get("reproduction_status")) != "resolved":
        raise ValueError("SOS published aggregation requires a resolved author protocol")
    aggregation = protocol.get("aggregation", {})
    classifiers = aggregation.get("classifiers", [])
    if not classifiers or len(set(classifiers)) != len(classifiers):
        raise ValueError("SOS requires explicit unique author classifiers")
    repetitions = aggregation.get("repetitions")
    seeds = aggregation.get("seeds")
    if repetitions:
        unit_field = "repetition_id"
        units_expected = {str(item["id"]): str(item["seed"]) for item in repetitions}
        if len(units_expected) != len(repetitions):
            raise ValueError("SOS repetition IDs must be unique")
    elif seeds and len(set(seeds)) == len(seeds):
        unit_field = "seed"
        units_expected = {str(seed): str(seed) for seed in seeds}
    else:
        raise ValueError("SOS requires explicit unique seeds or identified repetitions")
    rows = [row for row in records if row.get("status") == "completed" and row.get("experiment_type") == "reproduction"]
    layout = {(classifier, unit) for classifier in classifiers for unit in units_expected}
    observed = {(row["algorithm"], str(row.get(unit_field))) for row in rows}
    if len(rows) != len(layout) or observed != layout:
        raise ValueError("SOS requires exactly one cell per declared classifier/trial")
    if any(str(row["seed"]) != units_expected[str(row[unit_field])] for row in rows):
        raise ValueError("SOS trial seed does not match author declaration")
    for field in ("protocol_id", "protocol_sha256", "dataset_fingerprint", "study_id", "study_sha256", "method", "variant_id", "budget_mode", "evaluation_policy_sha256"):
        if len({row.get(field) for row in rows}) != 1:
            raise ValueError(f"Incompatible SOS {field}")
    if rows[0]["protocol_id"] != protocol["protocol_id"] or rows[0].get("protocol_sha256") != snapshot_sha256(protocol):
        raise ValueError("SOS records do not match the exact supplied protocol snapshot")
    if any(len({row["split_fingerprint"] for row in rows if str(row[unit_field]) == unit}) != 1 for unit in units_expected):
        raise ValueError("SOS classifiers must share each trial's split")
    metric = aggregation.get("metric")
    if metric not in {"weighted_f1", "sos_inverse_support_f1"}:
        raise ValueError("SOS must distinguish support-weighted from inverse-support F1")
    values = [_metric(row, metric) for row in rows]
    if any(value is None for value in values):
        raise ValueError(f"SOS requires {metric} for every cell")
    order = aggregation.get("order")
    if order in {"classifier_mean_per_seed_then_seed_mean", "classifier_mean_per_trial_then_trial_mean"}:
        spread_units = [np.mean([_metric(row, metric) for row in rows if str(row[unit_field]) == unit]) for unit in units_expected]
    elif order in {"seed_mean_per_classifier_then_classifier_mean", "trial_mean_per_classifier_then_classifier_mean"}:
        spread_units = [np.mean([_metric(row, metric) for row in rows if row["algorithm"] == classifier]) for classifier in classifiers]
    elif order == "all_cells":
        spread_units = values
    else:
        raise ValueError("SOS author aggregation order remains unresolved")
    spread = aggregation.get("spread")
    if spread not in {"sample_std", "population_std"}:
        raise ValueError("SOS author spread definition remains unresolved")
    return {"metric": metric, "mean": float(np.mean(values)),
            "spread": float(np.std(spread_units, ddof=1 if spread == "sample_std" else 0)) if len(spread_units) > 1 else None,
            "classifier_count": len(classifiers), "seed_count": len({str(row["seed"]) for row in rows}),
            "trial_count": len(units_expected), "cell_count": len(rows),
            "aggregation_order": order, "spread_definition": spread,
            "published_comparability": "verified"}
