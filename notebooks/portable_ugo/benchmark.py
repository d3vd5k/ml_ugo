"""Raw-OpenML, F1-first six-method benchmark embedded in the portable notebook.

Scientific and training-only smoke runs are distinctly marked. No test observation
participates in preprocessing, proposal scoring, learner selection or thresholding.
Sklearn LR/RF/RBF-SVM are explicit CPU controls; Torch/Flow/XGBoost obey DEVICE.
"""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from numbers import Integral
from pathlib import Path
import platform
import re
import sys
from time import perf_counter
import traceback
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import torch

from .models import (CLASSIFIERS, as_array, candidate_parameters, fit_estimator,
                     make_estimator, resolve_device)
from .preprocessing import categorical_columns, make_preprocessor
from .ugo_original import OriginalUGOConfig, original_ugo
from .u2 import prepare_u2, run_u2
from .flow_gmm import FlowGMMConfig, FlowGate, fit_flow_gmm
from .sampling import SAMPLER_DEFAULTS, resample_control, validate_sampler_parameters

METHODS = ("baseline", "original_ugo", "u2_flow_gate", "smote", "kmeans_smote", "rwo")
SAMPLER_PARAMETERS = SAMPLER_DEFAULTS
DEFAULTS = dict(DATASET_RANGE="1-32", EXCLUDED_DATASETS=(10, 26),
    CLASSIFIERS=list(CLASSIFIERS), SEEDS=[42, 43, 44],
    N_CANDIDATES=10, TEST_FRACTION=.2, VALIDATION_FRACTION=.2,
    AUGMENTATION_CAP=1000, MAX_ROUNDS=20, MLP_EPOCHS=160, FLOW_EPOCHS=60,
    GROUP_ALPHAS=[0., .5, 1.], GATE_TAUS=[.5, .6, .7, .8], DEVICE="auto",
    OUTPUT_DIR="ugo_results", DATA_HOME="ugo_data", N_JOBS=4, SMOKE=False,
    MAX_ROWS=None, SAMPLER_PARAMETERS=SAMPLER_PARAMETERS)
METRICS = ("minority_precision", "minority_recall", "minority_f1", "average_precision",
           "balanced_accuracy", "mcc", "f1_at_0_5", "accepted_count", "requested_count",
           "runtime_seconds", "classifier_fit_count")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_json_default,
                                    allow_nan=False).encode()).hexdigest()


def _dump(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
        default=_json_default, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _catalogue(catalogue):
    rows = [dict(row) for row in catalogue]
    if not rows:
        raise ValueError("The dataset catalogue is empty")
    for index, row in enumerate(rows, 1):
        if row.get("number") != index or isinstance(row.get("number"), bool):
            raise ValueError("Catalogue numbers must be consecutive and 1-based")
        if not isinstance(row.get("name"), str) or not row["name"]:
            raise ValueError("Each catalogue entry requires a dataset name")
        if isinstance(row.get("openml_id"), bool) or not isinstance(row.get("openml_id"), Integral):
            raise ValueError("Each catalogue entry requires an integer OpenML source ID")
        if "imbalance_type" not in row:
            raise ValueError("Each catalogue entry requires its imbalance type")
    return rows


def show_catalogue(catalogue):
    """Display the numbered raw OpenML sources, without downloading data."""
    return pd.DataFrame(_catalogue(catalogue))[["number", "openml_id", "name", "imbalance_type"]]


def _excluded_dataset_numbers(numbers):
    if not isinstance(numbers, (list, tuple)) or any(
            isinstance(number, (bool, np.bool_)) or not isinstance(number, Integral) or number < 1
            for number in numbers):
        raise ValueError("EXCLUDED_DATASETS must be a unique list/tuple of positive dataset numbers")
    if len(set(numbers)) != len(numbers):
        raise ValueError("EXCLUDED_DATASETS must not repeat dataset numbers")
    return [int(number) for number in numbers]


def select_catalogue(catalogue, dataset_range, *, excluded_numbers=()):
    """Inclusive stable-number x-y selection, omitting explicitly excluded IDs."""
    rows = _catalogue(catalogue)
    if not isinstance(dataset_range, str) or not re.fullmatch(r"\s*[1-9][0-9]*\s*-\s*[1-9][0-9]*\s*", dataset_range):
        raise ValueError(f"DATASET_RANGE must be an inclusive 'x-y' string between 1 and {len(rows)}")
    start, end = map(int, dataset_range.split("-"))
    if not 1 <= start <= end <= len(rows):
        raise ValueError(f"DATASET_RANGE must satisfy 1 <= x <= y <= {len(rows)}")
    excluded = _excluded_dataset_numbers(excluded_numbers)
    if any(number > len(rows) for number in excluded):
        raise ValueError(f"EXCLUDED_DATASETS contains numbers outside 1-{len(rows)}")
    selected = [row for row in rows[start - 1:end] if row["number"] not in excluded]
    if not selected:
        raise ValueError("No datasets remain in DATASET_RANGE after EXCLUDED_DATASETS; change the range or exclusions")
    return selected


def _configuration(config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    result = {**DEFAULTS, **config}
    result["EXCLUDED_DATASETS"] = _excluded_dataset_numbers(result["EXCLUDED_DATASETS"])
    if result["SAMPLER_PARAMETERS"] is None:
        raise ValueError("SAMPLER_PARAMETERS must provide exactly smote, kmeans_smote and rwo")
    result["SAMPLER_PARAMETERS"] = validate_sampler_parameters(result["SAMPLER_PARAMETERS"])
    for key in ("N_CANDIDATES", "MAX_ROUNDS", "MLP_EPOCHS", "FLOW_EPOCHS", "N_JOBS"):
        if isinstance(result[key], (bool, np.bool_)) or not isinstance(result[key], Integral) or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    cap = result["AUGMENTATION_CAP"]
    if isinstance(cap, (bool, np.bool_)) or not isinstance(cap, Integral) or cap < 0:
        raise ValueError("AUGMENTATION_CAP must be a nonnegative integer")
    limit = result["MAX_ROWS"]
    if limit is not None and (isinstance(limit, (bool, np.bool_)) or not isinstance(limit, Integral) or limit < 2):
        raise ValueError("MAX_ROWS must be None or an explicit integer of at least two")
    for key in ("TEST_FRACTION", "VALIDATION_FRACTION"):
        if isinstance(result[key], (bool, np.bool_)) or not np.isfinite(result[key]) or not 0 < result[key] < 1:
            raise ValueError(f"{key} must be strictly between zero and one")
    learners = result["CLASSIFIERS"]
    if not isinstance(learners, (list, tuple)) or not learners or len(set(learners)) != len(learners) or set(learners) - set(CLASSIFIERS):
        raise ValueError(f"CLASSIFIERS must be a nonempty unique subset of {CLASSIFIERS}")
    seeds = result["SEEDS"]
    if not isinstance(seeds, (list, tuple)) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("SEEDS must be a nonempty unique sequence")
    if any(isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or not 0 <= seed < 2**32 - 7919 for seed in seeds):
        raise ValueError("SEEDS must be nonnegative integers below 2**32 - 7919")
    for key, lower, upper in (("GROUP_ALPHAS", 0., 1.), ("GATE_TAUS", 0., 1.)):
        values = result[key]
        if not isinstance(values, (list, tuple)) or not values or len(set(values)) != len(values):
            raise ValueError(f"{key} must be a nonempty unique sequence")
        if any(isinstance(v, (bool, np.bool_)) or not np.isfinite(v) or v < lower or v > upper for v in values):
            raise ValueError(f"{key} values must be finite in [{lower}, {upper}]")
        if key == "GATE_TAUS" and any(v in (0, 1) for v in values):
            raise ValueError("GATE_TAUS must be strictly between zero and one")
    if not isinstance(result["SMOKE"], (bool, np.bool_)):
        raise ValueError("SMOKE must be boolean")
    resolve_device(result["DEVICE"])
    for key in ("OUTPUT_DIR", "DATA_HOME"):
        if not isinstance(result[key], (str, Path)) or not str(result[key]):
            raise ValueError(f"{key} must be a nonempty filesystem path")
        result[key] = str(Path(result[key]).expanduser().resolve())
    if result["SMOKE"]:
        result.update(N_CANDIDATES=2, MAX_ROUNDS=2, AUGMENTATION_CAP=40,
                      MLP_EPOCHS=2, FLOW_EPOCHS=2, SEEDS=[int(seeds[0])])
    return result


def _fingerprint(features, target, row_ids):
    digest = hashlib.sha256()
    digest.update(json.dumps([(str(c), str(features[c].dtype)) for c in features]).encode())
    for values in (features, pd.Series(target), pd.Series(row_ids)):
        digest.update(pd.util.hash_pandas_object(values, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _load_dataset(entry, config):
    """Cache OpenML raw artifacts only; never use CLIMB globally-scaled arrays."""
    import openml
    cache = Path(config["DATA_HOME"]) / "openml"
    cache.mkdir(parents=True, exist_ok=True)
    previous = openml.config.get_config_as_dict()["cachedir"]
    openml.config.set_root_cache_directory(str(cache))
    try:
        raw = openml.datasets.get_dataset(int(entry["openml_id"]), download_data=True,
                                         download_qualities=False)
        features, target, _, _ = raw.get_data(target=raw.default_target_attribute,
            include_row_id=True, dataset_format="dataframe")
        if target is None or np.asarray(target).ndim != 1:
            raise ValueError("Dataset requires exactly one target column")
        target = np.asarray(target)
        if pd.isna(target).any():
            raise ValueError("Raw target contains missing labels")
        y, labels = pd.factorize(target, sort=True)
        y = np.asarray(y, dtype=np.int64)
        if len(labels) != 2:
            raise ValueError(f"Expected a binary CLIMB source, got {len(labels)} classes")
        row_attribute = raw.row_id_attribute
        row_columns = [row_attribute] if isinstance(row_attribute, str) else list(row_attribute or [])
        if row_columns:
            identity_frame, _, _, _ = raw.get_data(include_row_id=True,
                include_ignore_attribute=True, dataset_format="dataframe")
            if len(row_columns) == 1:
                row_ids = identity_frame[row_columns[0]].to_numpy(copy=True)
            else:
                row_ids = np.asarray([json.dumps(list(values), default=_json_default, ensure_ascii=False)
                    for values in identity_frame[row_columns].itertuples(index=False, name=None)])
            features = features.drop(columns=row_columns, errors="ignore")
        else:
            row_ids = np.arange(len(features), dtype=np.int64)
        if pd.isna(row_ids).any() or len(set(row_ids.tolist())) != len(row_ids):
            raise ValueError("Original raw row identities must be nonmissing and unique")
        if features.empty or not len(features.columns):
            raise ValueError("Raw source has no usable features")
        full_fingerprint = _fingerprint(features, target, row_ids)
        metadata = {"source": "openml_raw", "openml_id": int(entry["openml_id"]),
            "catalogue_number": int(entry["number"]), "catalogue_name": entry["name"],
            "imbalance_type": entry["imbalance_type"], "openml_name": raw.name,
            "version": int(raw.version), "openml_md5_checksum": raw.md5_checksum,
            "target_attribute": raw.default_target_attribute, "row_id_attributes": row_columns,
            "row_id_source": "attribute" if row_columns else "original_row_position",
            "raw_fingerprint": full_fingerprint,
            "fingerprint_algorithm": "sha256_pandas_raw_values_v1",
            "label_mapping": [{"label": label, "encoded": index} for index, label in enumerate(labels)],
            "feature_dtypes": {str(column): str(dtype) for column, dtype in features.dtypes.items()},
            "source_n_rows": len(features), "n_features": features.shape[1],
            "explicit_development_row_cap": config["MAX_ROWS"], "development_subset": False}
        limit = config["MAX_ROWS"]
        if limit is not None and limit < len(y):
            chosen, _ = train_test_split(np.arange(len(y)), train_size=int(limit), stratify=y, random_state=42)
            chosen = np.sort(chosen)
            features, target, y, row_ids = features.iloc[chosen].copy(), target[chosen], y[chosen], row_ids[chosen]
            metadata["development_subset"] = True
        metadata.update(n_rows=len(y), data_fingerprint=_fingerprint(features, target, row_ids),
                        class_counts=_counts(y))
        return {**entry, "X": features, "y": y, "row_ids": row_ids, "metadata": metadata}
    finally:
        openml.config.set_root_cache_directory(previous)


def _counts(y):
    classes, counts = np.unique(as_array(y), return_counts=True)
    return {int(label): int(count) for label, count in zip(classes, counts)}


def _split(dataset, seed, config):
    rows = np.arange(len(dataset["y"]))
    train, untouched_test = train_test_split(rows, test_size=config["TEST_FRACTION"],
        stratify=dataset["y"], random_state=seed)
    if config["SMOKE"]:
        # The study's outer test remains entirely unconsumed in smoke.
        if len(train) > 1200:
            subset, _ = train_test_split(train, train_size=1200,
                stratify=dataset["y"][train], random_state=seed)
        else:
            subset = train
        train, test = train_test_split(subset, test_size=config["TEST_FRACTION"],
            stratify=dataset["y"][subset], random_state=seed)
    else:
        test = untouched_test
    fit, validation = train_test_split(train, test_size=config["VALIDATION_FRACTION"],
        stratify=dataset["y"][train], random_state=seed)
    result = {"train": train, "fit": fit, "validation": validation, "test": test}
    if any(len(np.unique(dataset["y"][indices])) != 2 for indices in result.values()):
        raise ValueError("Every real split must contain both classes")
    if config["SMOKE"] and set(test.tolist()) & set(untouched_test.tolist()):
        raise ValueError("Smoke must not consume any scientific outer-test row")
    return result


def _partitions(dataset, split):
    inner = make_preprocessor(dataset["X"].iloc[split["fit"]])
    outer = make_preprocessor(dataset["X"].iloc[split["train"]])
    return {"inner": inner, "outer": outer,
        "X_fit": inner.fit_transform(dataset["X"].iloc[split["fit"]]).astype(np.float32),
        "X_validation": inner.transform(dataset["X"].iloc[split["validation"]]).astype(np.float32),
        "X_train": outer.fit_transform(dataset["X"].iloc[split["train"]]).astype(np.float32),
        "X_test": outer.transform(dataset["X"].iloc[split["test"]]).astype(np.float32)}


def _categorical_groups(preprocessor):
    """Recover each complete fitted one-hot variable, not one merged block."""
    columns = preprocessor.named_steps["columns"]
    block = columns.output_indices_["categorical"]
    if block.start == block.stop:
        return ()
    encoder = columns.named_transformers_["categorical"].named_steps["encode"]
    groups, start = [], block.start
    for categories in encoder.categories_:
        stop = start + len(categories)
        groups.append(tuple(range(start, stop)))
        start = stop
    if start != block.stop:
        raise ValueError("One-hot category widths do not match the fitted output block")
    return tuple(groups)


def _targets(y, cap):
    """Matched total deficit budget, clipped at real majority balance."""
    counts = _counts(y)
    maximum = max(counts.values())
    deficits = {label: maximum - count for label, count in counts.items()}
    total = min(int(cap), sum(deficits.values()))
    if not total:
        return counts
    desired = {label: total * value / sum(deficits.values()) for label, value in deficits.items()}
    allocated = {label: int(np.floor(value)) for label, value in desired.items()}
    for label in sorted(counts, key=lambda c: (-(desired[c] - allocated[c]), c))[:total - sum(allocated.values())]:
        allocated[label] += 1
    return {label: counts[label] + allocated[label] for label in counts}


def choose_f1_threshold(y_true, probability):
    """O(n log n), exact rational F1 comparisons over whole tied-score groups.

    Candidates are observed score cut points and .5. Exact F1 ties prefer the
    threshold closest to .5, then the higher threshold (never split a score tie).
    """
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Threshold selection requires aligned nonempty vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Threshold selection requires binary labels and finite probabilities")
    positives = int(y.sum())
    if not positives:
        raise ValueError("Validation must contain the minority class")
    order = np.argsort(-p, kind="stable")
    ordered = p[order]
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(y) - 1]
    cumulative = np.cumsum(y[order])
    at_half = p >= .5
    best = (int(y[at_half].sum()), int(at_half.sum()) + positives, .5)
    for end in ends:
        item = (int(cumulative[end]), int(end) + 1 + positives, float(ordered[end]))
        comparison = item[0] * best[1] - best[0] * item[1]
        if comparison > 0 or (comparison == 0 and
                (-abs(item[2] - .5), item[2]) > (-abs(best[2] - .5), best[2])):
            best = item
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


def _positive_probability(model, X, positive):
    columns = np.flatnonzero(np.asarray(model.classes_) == positive)
    if len(columns) != 1:
        raise ValueError("Learner has no unique minority-class probability column")
    probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
    if probabilities.ndim != 2 or len(probabilities) != len(X):
        raise ValueError("Learner produced misaligned class probabilities")
    probability = probabilities[:, columns[0]]
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Learner produced invalid minority probabilities")
    return probability


def _compact(value):
    """No row arrays, proposal logs, checkpoints or large group vectors on disk."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in ("candidate_log", "provenance", "training_losses", "fit_row_ids",
                       "calibration_row_ids", "validation_row_ids", "real_row_ids"):
                continue
            if key in ("seed_multiplicity", "attempted_seed_multiplicity"):
                multiplicities = [int(count) for count in item.values()]
                result[str(key)] = {"unique_seed_count": len(multiplicities),
                    "total": sum(multiplicities), "maximum": max(multiplicities, default=0),
                    "sha256": _hash(item)}
            else:
                result[str(key)] = _compact(item)
        return result
    if isinstance(value, np.ndarray):
        return {"count": len(value), "sha256": _hash(value)}
    if isinstance(value, (list, tuple)):
        if len(value) > 32:
            return {"count": len(value), "sha256": _hash(value)}
        return [_compact(item) for item in value]
    return value


def _audit(sampled, y, row_ids, targets, real_only, gate=None, tau=None):
    labels = as_array(sampled.y)
    n = len(y)
    provenance = sampled.provenance
    accepted = len(labels) - n
    requested = sum(targets.values()) - n
    if len(provenance) != accepted or not 0 <= accepted <= requested:
        raise ValueError("Augmentation counts do not match ancestry/cap")
    if not np.array_equal(labels[:n], y):
        raise ValueError("Augmentation changed real labels/order")
    ancestry, used_roots = [], set()
    direct, synthetic = 0, 0
    for offset, entry in enumerate(provenance):
        row = n + offset
        if int(entry["row_index"]) != row or int(entry["class"]) != int(labels[row]):
            raise ValueError("Synthetic ancestry is not contiguous and class-aligned")
        parents = [int(entry["seed_index"]), int(entry["partner_index"])]
        relocation = int(entry.get("relocation_index", -1))
        if relocation >= 0:
            parents.append(relocation)
        roots = set()
        for parent in parents:
            if not 0 <= parent < row or (real_only and parent >= n):
                raise ValueError("Synthetic parent outside allowed training ancestry")
            if labels[parent] != labels[row]:
                raise ValueError("Synthetic parent has the wrong class")
            direct += parent < n
            synthetic += parent >= n
            roots.update({parent} if parent < n else ancestry[parent - n])
        ancestry.append(roots)
        used_roots.update(roots)
    if any(_counts(labels).get(label, 0) > maximum for label, maximum in targets.items()):
        raise ValueError("Sampler exceeded a per-class target")
    log = sampled.support_summary.get("candidate_log")
    gate_evidence = None
    if log is not None:
        if len(log) > requested:
            raise ValueError("Proposals exceeded the matched augmentation budget")
        for proposal in log:
            for key in ("seed_index", "partner_index"):
                parent = int(proposal[key])
                if not 0 <= parent < n or int(labels[parent]) != int(proposal["class"]):
                    raise ValueError("Proposal parent is not same-class real training")
            if gate is not None and proposal["accepted"]:
                ratio = proposal.get("log_ratio")
                if ratio is None or not np.isfinite(ratio) or ratio < np.log(tau) - np.log1p(-tau):
                    raise ValueError("Accepted proposal log violates the fixed log-ratio threshold")
        accepted_entries = [item for item in log if item["accepted"]]
        if len(accepted_entries) != accepted:
            raise ValueError("Candidate acceptance log does not reconcile final ancestry")
        for proposal, final in zip(accepted_entries, provenance):
            for key in ("seed_index", "partner_index", "class", "proposal_id"):
                if proposal[key] != final[key]:
                    raise ValueError("Accepted proposal identity differs from final ancestry")
        budget = sampled.support_summary["proposal_budget"]
        if (budget["attempted"] != len(log) or budget["accepted"] != accepted or
                budget["rejected"] != len(log) - accepted):
            raise ValueError("Attempt/accept/reject quota accounting does not reconcile")
        groups = sampled.support_summary["group_support"]
        if any(int(count) > 4 for count in groups["attempted_seed_multiplicity"].values()):
            raise ValueError("U2 attempted seed multiplicity exceeds canonical cap 4")
        for group in groups["groups"]:
            if int(group["attempted_generated"]) > 2 * len(group["indices"]):
                raise ValueError("U2 attempted group quota exceeds canonical cap 2n")
    if gate is not None:
        for entry in provenance:
            if int(entry.get("relocation_index", -1)) != -1:
                raise ValueError("FlowGate must replace native relocation")
        if accepted:
            checked = gate.predict(as_array(sampled.X[n:]), labels[n:], tau)
            if not bool(np.asarray(checked["accepted"]).all()):
                raise ValueError("Final synthetic coordinates violate the frozen FlowGate")
        gate_evidence = {"accepted_coordinates_checked": accepted, "all_pass": True,
                         "tau": tau, "no_relocation": True}
    root_ids = np.asarray(row_ids)[sorted(used_roots)]
    return {"accepted_count": accepted, "requested_count": requested,
        "underspend": requested - accepted, "real_count": n, "real_counts": _counts(y),
        "actual_counts": _counts(labels), "stop_reason": sampled.stop_reason,
        "round_count": len(sampled.rounds), "synthetic_weights": 1,
        "support_summary": _compact(sampled.support_summary),
        "ancestry": {"validated": True, "real_only_parents": real_only,
            "direct_real_parent_references": int(direct), "synthetic_parent_references": int(synthetic),
            "unique_real_root_count": len(used_roots), "real_root_row_ids_sha256": _hash(root_ids),
            "training_row_ids_sha256": _hash(np.asarray(row_ids)),
            "provenance_sha256": _hash(provenance)}, "gate_acceptance": gate_evidence}


def _fit(dataset, split, partitions, classifier, method, parameters, sampler, config,
         seed, role, prepared, gate=None):
    indices = split["fit" if role == "inner" else "train"]
    features = partitions["X_fit" if role == "inner" else "X_train"]
    y = dataset["y"][indices]
    targets = _targets(y, config["AUGMENTATION_CAP"])
    fits = 0
    latest_model, latest_count = None, None
    def train_round(X, labels):
        nonlocal fits, latest_model, latest_count
        model = make_estimator(classifier, seed, parameters, device=config["DEVICE"],
            epochs=config["MLP_EPOCHS"], n_jobs=config["N_JOBS"])
        model = fit_estimator(model, X, labels)
        fits += 1
        latest_model, latest_count = model, len(labels)
        return model
    if method == "baseline":
        model = train_round(features, y)
        return model, {"accepted_count": 0, "requested_count": 0, "real_count": len(y),
            "real_counts": _counts(y), "actual_counts": _counts(y), "classifier_fit_count": fits,
            "stop_reason": "no_oversampling", "synthetic_weights": 1}
    if method in SAMPLER_PARAMETERS:
        # Fixed training-only pools are identical across learner candidates/families.
        pools = prepared.setdefault("control_pools", {})
        key = (role, method)
        if key not in pools:
            started = perf_counter()
            X_res, y_res, audit = resample_control(method, features, y, targets, seed,
                categorical_groups=_categorical_groups(partitions[role]),
                n_jobs=config["N_JOBS"], parameters=sampler)
            audit = {**audit, "pool_scope": "dataset_seed_role_method",
                "pool_role": role, "pool_seed": seed,
                "shared_generation_seconds": perf_counter() - started}
            pools[key] = X_res, y_res, audit
        X_res, y_res, audit = pools[key]
        model = train_round(X_res, y_res)
        return model, {**audit, "classifier_fit_count": fits}
    categorical = categorical_columns(partitions[role])
    if method == "original_ugo":
        options = OriginalUGOConfig(max_iterations=config["MAX_ROUNDS"], random_state=seed,
            device=config["DEVICE"], categorical_columns=categorical,
            categorical_projection=bool(categorical))
        sampled = original_ugo(features, y, train_round, options, target_counts=targets)
    elif method in ("u2_native", "u2_flow_gate"):
        if role not in prepared:
            prepared[role] = prepare_u2(features, y, seed, targets, sampler)
        sampled = run_u2(features, y, train_round, seed=seed, targets=targets,
            parameters=sampler, max_iterations=config["MAX_ROUNDS"], device=config["DEVICE"],
            categorical_columns=categorical, prepared=prepared[role], flow_support=gate)
    else:
        raise ValueError(f"Unknown benchmark method: {method}")
    unchanged = method == "u2_flow_gate" and latest_count == len(sampled.y)
    model = latest_model if unchanged else train_round(sampled.X, sampled.y)
    audit = _audit(sampled, y, dataset["row_ids"][indices], targets,
                   method != "original_ugo", gate, sampler.get("f1_tau"))
    audit["classifier_fit_count"] = fits
    return model, audit


def _candidate(dataset, split, partitions, classifier, method, parameters, sampler,
               config, seed, positive, candidate_id, prepared, gate=None):
    try:
        model, audit = _fit(dataset, split, partitions, classifier, method, parameters, sampler,
                           config, seed, "inner", prepared, gate)
        probability = _positive_probability(model, partitions["X_validation"], positive)
        truth = dataset["y"][split["validation"]] == positive
        threshold, score = choose_f1_threshold(truth, probability)
        metrics = _metrics(truth, probability, threshold)
        return {"candidate_id": candidate_id, "parameters": parameters,
            "sampler_parameters": sampler, "threshold": threshold, "validation_f1": score,
            "validation_ap": metrics["average_precision"],
            "f1_numerator": 2 * metrics["tp"],
            "f1_denominator": 2 * metrics["tp"] + metrics["fp"] + metrics["fn"],
            "audit": audit}, probability
    except Exception as error:
        context = {"role": "inner", "method": method, "candidate_id": candidate_id,
                   "parameters": parameters, "sampler_parameters": sampler}
        raise RuntimeError(f"Native candidate failed: {json.dumps(context, default=_json_default)}") from error


def _better(item, best):
    if best is None:
        return True
    comparison = item["f1_numerator"] * best["f1_denominator"] - best["f1_numerator"] * item["f1_denominator"]
    return comparison > 0 or (comparison == 0 and item["validation_ap"] > best["validation_ap"])


def _core_hash(core):
    digest = hashlib.sha256()
    for name, value in core.model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _flow_roles(dataset, split, partitions, seed, config):
    fit, calibration = train_test_split(split["fit"], test_size=.2,
        stratify=dataset["y"][split["fit"]], random_state=seed + 7919)
    core = fit_flow_gmm(dataset["X"].iloc[fit], dataset["y"][fit],
        dataset["row_ids"][fit], seed, FlowGMMConfig(epochs=config["FLOW_EPOCHS"], device=config["DEVICE"]))
    gates = []
    for role in ("inner", "outer"):
        gate = FlowGate(core, partitions[role], calibration_row_ids=dataset["row_ids"][calibration],
                        validation_row_ids=dataset["row_ids"][split["validation"]])
        gate.role = role
        gate.calibration_class_counts = _counts(dataset["y"][calibration])
        gates.append(gate)
    evidence = _flow_evidence(gates, dataset, split)
    return gates, evidence


def _flow_evidence(gates, dataset, split):
    inner, outer = gates
    core = inner.core
    if outer.core is not core or core.model.training or any(p.requires_grad for p in core.model.parameters()):
        raise ValueError("Flow core must remain frozen and shared across outer refit")
    roles = [set(np.asarray(ids).tolist()) for ids in (core.fit_row_ids,
        inner.calibration_row_ids, inner.validation_row_ids, dataset["row_ids"][split["test"]])]
    if any(roles[a] & roles[b] for a in range(4) for b in range(a + 1, 4)):
        raise ValueError("Flow fit/calibration/validation/test roles overlap")
    if roles[0] | roles[1] != set(dataset["row_ids"][split["fit"]].tolist()):
        raise ValueError("Flow fit and reserved calibration do not partition inner real fit")
    if roles[2] != set(dataset["row_ids"][split["validation"]].tolist()):
        raise ValueError("Flow validation identities do not match real validation")
    if not np.array_equal(outer.calibration_row_ids, inner.calibration_row_ids):
        raise ValueError("Outer replay changed reserved flow calibration identities")
    audit = core.audit
    if audit["synthetic_training_count"] != 0 or audit["calibration_used"] or not audit["frozen"]:
        raise ValueError("Flow must be fitted only on real fit rows and frozen")
    current_hash = _core_hash(core)
    if current_hash != audit["model_checksum"]:
        raise ValueError("Flow state changed since the real-only fit")
    return {"core_sha256": current_hash, "fit_row_ids_sha256": _hash(core.fit_row_ids),
        "calibration_row_ids_sha256": _hash(inner.calibration_row_ids),
        "validation_row_ids_sha256": _hash(inner.validation_row_ids),
        "fit_count": len(core.fit_row_ids), "calibration_count": len(inner.calibration_row_ids),
        "validation_count": len(inner.validation_row_ids), "roles_disjoint": True,
        "fit_count_shared": 1, "frozen_core_shared_across_roles": True,
        "fit_seconds": float(audit["training_seconds"]), "calibration_used": False,
        "synthetic_training_count": 0, "resolved_device": audit["resolved_device"],
        "model_config": audit["model_config"], "source_sha256": audit["source_sha256"]}


def _safe_name(entry, classifier, seed, method):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", entry["name"]).strip("._-")[:70] or "dataset"
    return f"{int(entry['number']):02d}_{name}__{classifier}__s{seed}__{method}"


def _predictions(path, row_ids, labels, probability, threshold, positive):
    pd.DataFrame({"row_id": row_ids, "y_true": (labels == positive).astype(np.int64),
        "y_true_encoded": labels, "minority_class": positive, "probability": probability,
        "y_pred": (probability >= threshold).astype(np.int64),
        "y_pred_encoded": np.where(probability >= threshold, positive, 1 - positive),
        "threshold": threshold}).to_csv(path, index=False, float_format="%.17g")


def _cell(dataset, split, partitions, classifier, seed, method, config, candidates,
          run_dir, prepared, gates=None, flow_evidence=None):
    started = perf_counter()
    counts = _counts(dataset["y"][split["train"]])
    positive = min(counts, key=lambda label: (counts[label], label))
    selection = {"dataset": dataset["name"], "dataset_number": dataset["number"],
        "classifier": classifier, "seed": seed, "method": method,
        "positive_class": positive, "canonical_candidates": [], "candidates": [],
        "selection_objective": "validation_minority_f1_then_average_precision",
        "threshold_policy": "whole_tied_score_groups_plus_0.5_exact_rational_f1_ties_nearest_0.5_then_higher"}
    best, best_probability = None, None
    if method not in METHODS:
        raise ValueError(f"Unknown benchmark method: {method}")
    if method != "u2_flow_gate":
        sampler = config["SAMPLER_PARAMETERS"][method] if method in SAMPLER_PARAMETERS else {}
        for index, parameters in enumerate(candidates):
            item, probability = _candidate(dataset, split, partitions, classifier, method,
                parameters, sampler, config, seed, positive, index, prepared)
            selection["candidates"].append(item)
            if _better(item, best):
                best, best_probability = item, probability
        selection["selected_candidate_id"] = best["candidate_id"]
    else:
        locked = None
        for index, parameters in enumerate(candidates):
            sampler = {"group_alpha": config["GROUP_ALPHAS"][index % len(config["GROUP_ALPHAS"]) ]}
            item, _ = _candidate(dataset, split, partitions, classifier, "u2_native", parameters,
                sampler, config, seed, positive, index, prepared)
            selection["canonical_candidates"].append(item)
            if _better(item, locked):
                locked = item
        selection["selected_candidate_id"] = locked["candidate_id"]
        selection["locked_learner_parameters"] = locked["parameters"]
        selection["locked_group_alpha"] = locked["sampler_parameters"]["group_alpha"]
        for index, tau in enumerate(config["GATE_TAUS"]):
            sampler = {**locked["sampler_parameters"], "f1_tau": tau}
            item, probability = _candidate(dataset, split, partitions, classifier, method,
                locked["parameters"], sampler, config, seed, positive, index, prepared, gates[0])
            selection["candidates"].append(item)
            if _better(item, best):
                best, best_probability = item, probability
        selection["selected_tau_id"] = best["candidate_id"]
        selection["flow_evidence"] = flow_evidence
    threshold = best["threshold"]
    basename = _safe_name(dataset, classifier, seed, method)
    validation_path = Path("predictions") / f"{basename}__validation.csv"
    test_path = Path("predictions") / f"{basename}__test.csv"
    # Preserve actual selected inner predictions before any outer model exists.
    _predictions(run_dir / validation_path, dataset["row_ids"][split["validation"]],
        dataset["y"][split["validation"]], best_probability, threshold, positive)
    try:
        model, outer_audit = _fit(dataset, split, partitions, classifier, method, best["parameters"],
            best["sampler_parameters"], config, seed, "outer", prepared,
            gates[1] if method == "u2_flow_gate" else None)
        probability = _positive_probability(model, partitions["X_test"], positive)
    except Exception as error:
        context = {"role": "outer", "parameters": best["parameters"],
                   "sampler_parameters": best["sampler_parameters"], "frozen_threshold": threshold}
        raise RuntimeError(f"Selected outer refit failed: {json.dumps(context, default=_json_default)}") from error
    if method == "u2_flow_gate" and _flow_evidence(gates, dataset, split) != flow_evidence:
        raise ValueError("Frozen detector evidence changed during inner selection/outer refit")
    _predictions(run_dir / test_path, dataset["row_ids"][split["test"]],
        dataset["y"][split["test"]], probability, threshold, positive)
    fits = sum(item["audit"]["classifier_fit_count"] for item in
        selection["canonical_candidates"] + selection["candidates"]) + outer_audit["classifier_fit_count"]
    selection.update(selected_parameters=best["parameters"], selected_sampler_parameters=best["sampler_parameters"],
        threshold=threshold, selected_validation_f1=best["validation_f1"],
        selected_validation_ap=best["validation_ap"], selected_inner_audit=best["audit"],
        outer_audit=outer_audit, classifier_fit_count=fits, backend=model.backend_,
        prediction_paths={"validation": str(validation_path), "test": str(test_path)})
    split_hash = _hash({role: dataset["row_ids"][indices] for role, indices in split.items()})
    record = {"dataset_number": dataset["number"], "dataset": dataset["name"],
        "openml_id": dataset["openml_id"], "classifier": classifier, "seed": seed,
        "method": method, "status": "completed", "run_kind": "training_only_smoke" if config["SMOKE"] else "study",
        "development_subset": dataset["metadata"]["development_subset"], "positive_class": positive,
        "threshold": threshold, **_metrics(dataset["y"][split["test"]] == positive, probability, threshold),
        "validation_f1": best["validation_f1"], "validation_ap": best["validation_ap"],
        "accepted_count": outer_audit["accepted_count"], "requested_count": outer_audit["requested_count"],
        "n_inner_fit": len(split["fit"]), "n_validation": len(split["validation"]),
        "n_outer_train": len(split["train"]), "n_test": len(split["test"]),
        "runtime_seconds": perf_counter() - started, "classifier_fit_count": fits,
        "backend": model.backend_,
        "sampler_device": "cpu" if method in SAMPLER_PARAMETERS else resolve_device(config["DEVICE"]),
        "flow_fit_seconds_shared": flow_evidence["fit_seconds"] if method == "u2_flow_gate" else 0.,
        "split_fingerprint": split_hash, "data_fingerprint": dataset["metadata"]["data_fingerprint"],
        "predictions_path": str(test_path), "validation_predictions_path": str(validation_path)}
    return record, selection


def _tables(records, selected, config):
    """Keep missing coverage explicit; deltas always mean method minus reference."""
    results = pd.DataFrame(records)
    rows, comparison, paired = [], [], []
    lookup = {(row["dataset_number"], row["classifier"], row["seed"], row["method"]): row for row in records}
    run_kind = "training_only_smoke" if config["SMOKE"] else "study"
    expected = len(config["SEEDS"])
    for entry in selected:
        for classifier in config["CLASSIFIERS"]:
            identity = {"dataset_number": entry["number"], "dataset": entry["name"],
                        "classifier": classifier, "run_kind": run_kind}
            wide = dict(identity)
            complete = True
            for method in METHODS:
                subset = [lookup[(entry["number"], classifier, seed, method)]
                    for seed in config["SEEDS"]
                    if (entry["number"], classifier, seed, method) in lookup]
                row = {**identity, "method": method, "expected_seeds": expected,
                    "completed_seeds": len(subset), "failed_or_missing_seeds": expected - len(subset),
                    "coverage_complete": len(subset) == expected,
                    "status": "complete" if len(subset) == expected else "incomplete"}
                for metric in METRICS:
                    values = np.asarray([item[metric] for item in subset], dtype=float)
                    row[metric + "_mean"] = float(values.mean()) if len(values) else np.nan
                    row[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
                rows.append(row)
                for field in ("minority_f1_mean", "minority_f1_std", "completed_seeds",
                              "expected_seeds", "failed_or_missing_seeds", "coverage_complete"):
                    wide[f"{method}_{field}"] = row[field]
                complete = complete and row["coverage_complete"]
            # No winner/rank is inferred from incomplete or differently covered means.
            wide.update(coverage_complete=complete, status="complete" if complete else "incomplete")
            comparison.append(wide)
            for seed in config["SEEDS"]:
                for index, reference in enumerate(METHODS):
                    for method in METHODS[index + 1:]:
                        row = lookup.get((entry["number"], classifier, seed, method))
                        other = lookup.get((entry["number"], classifier, seed, reference))
                        matched = row is not None and other is not None
                        if matched and (row["split_fingerprint"] != other["split_fingerprint"] or
                                        row["data_fingerprint"] != other["data_fingerprint"]):
                            raise ValueError("Paired comparison attempted across different real data or splits")
                        paired.append({**identity, "seed": seed, "method": method,
                            "reference": reference, "status": "complete" if matched else "incomplete",
                            "method_completed": row is not None, "reference_completed": other is not None,
                            **{metric + "_delta": row[metric] - other[metric] if matched else np.nan
                               for metric in METRICS},
                            "split_fingerprint": row["split_fingerprint"] if matched else None,
                            "data_fingerprint": row["data_fingerprint"] if matched else None})
    paired_columns = ["dataset_number", "dataset", "classifier", "seed", "method", "reference",
        "run_kind", "status", "method_completed", "reference_completed"] + [
        metric + "_delta" for metric in METRICS] + ["split_fingerprint", "data_fingerprint"]
    if results.empty:
        results = pd.DataFrame(columns=["dataset_number", "dataset", "classifier", "seed", "method", "status"] + list(METRICS))
    return results, pd.DataFrame(rows), pd.DataFrame(paired, columns=paired_columns), pd.DataFrame(comparison)


def _dependencies():
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "torch", "xgboost", "openml", "threadpoolctl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def _source_hashes():
    directory = Path(__file__).resolve().parent
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.glob("*.py"))}


def run_benchmark(config, catalogue):
    """Run every requested real-data cell, flushing success/failure evidence per cell.

    Returns run_dir and results/summary/paired_deltas/comparison DataFrames.
    Dataset or training failures are explicit failures.json entries; they never
    change learners or sampler settings, omit gating, fabricate metrics or stop other cells.
    """
    config = _configuration(dict(config))
    selected = select_catalogue(catalogue, config["DATASET_RANGE"],
                                excluded_numbers=config["EXCLUDED_DATASETS"])
    root = Path(config["OUTPUT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{timestamp}_{'smoke' if config['SMOKE'] else 'study'}_{uuid.uuid4().hex[:10]}"
    run_dir.mkdir()
    (run_dir / "predictions").mkdir()
    started = perf_counter()
    expected = len(selected) * len(config["CLASSIFIERS"]) * len(config["SEEDS"]) * len(METHODS)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "selected_catalogue": selected, "methods": list(METHODS), "dependencies": _dependencies(),
        "source_sha256": _source_hashes(), "python": sys.version, "platform": platform.platform(),
        "cuda": {"available": torch.cuda.is_available(), "torch_cuda_version": torch.version.cuda,
                 "resolved_device": resolve_device(config["DEVICE"]),
                 "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "backend_policy": {"mlp": "pytorch_DEVICE", "xgboost": "xgboost_DEVICE_hist",
            "logistic_regression": "sklearn_cpu", "random_forest": "sklearn_cpu",
            "svm": "sklearn_cpu_probability_training_calibration",
            "control_samplers": "smote_kmeans_smote_rwo_cpu; downstream_native_learner_DEVICE_unchanged"},
        "scientific_result": not config["SMOKE"] and config["MAX_ROWS"] is None,
        "run_kind": "training_only_smoke" if config["SMOKE"] else "study",
        "datasets": [], "coverage": {"expected_cells": expected, "completed_cells": 0, "complete": False},
        "status": "running", "runtime_seconds": 0., "shared_flow_fits": [],
        "protocol": {"validation": "real_inner_holdout_only", "threshold": "frozen_before_outer_refit",
            "minority_class": "least_frequent_encoded_class_in_real_outer_train",
            "selection": "minority_F1_primary_AP_secondary", "mlp": "fixed_epochs_no_validation_stopping",
            "synthetic_weights": 1, "raw_cache_only": True,
            "original_ugo": "paper_derived_reference_not_author_reproduction; mixed_features_use_explicit_partner_category_projection",
            "flow": "single_real_inner_fit_core_per_dataset_seed_reserved_calibration_unused_frozen_outer_bridge",
            "control_samplers": {
                "parameters": config["SAMPLER_PARAMETERS"],
                "tuning": "same_native_learner_candidates_and_F1_threshold_opportunities_as_baseline",
                "generation": "real_role_training_only_once_per_dataset_seed_role_method; shared_in_memory_across_candidates_and_classifiers",
                "runtime": "cell_wall_time_includes_pool_generation_only_for_first_consumer; audit_shared_generation_seconds_is_not_a_per_cell_fit",
                "weights": 1,
                "targets": "same_real_class_deficit_targets_and_augmentation_cap_as_UGO",
                "mixed_data_adapter": "SMOTE_and_KMeansSMOTE_on_numeric_plus_full_onehot_coordinates_then_argmax_each_whole_nominal_block_for_synthetics_only; not_SMOTENC",
                "kmeans_smote_budget": "deterministically_trim_ceiling_overproduction_only; never_refill",
                "default_cluster_balance_threshold": "prespecified_inclusive_0.0_allows_rare_positive_clusters; not_a_fallback_or_dataset_search",
                "rwo": "independent_real_seed_plus_Normal_times_real_class_population_std_over_sqrt_n; empirical_real_class_nominal_roulette; no_clipping_rounding_or_chaining",
                "rwo_std_reference": "https://raw.githubusercontent.com/analyticalmindsltd/smote_variants/master/smote_variants/oversampling/_rwo_sampling.py",
                "rwo_nominal_reference": "https://raw.githubusercontent.com/cran/imbalance/master/R/rwo.R",
                "rwo_reference_difference": "independently_written; population_std_convention_follows_Python_reference_not_bitwise_R_variance_variant",
                "failure_semantics": "infeasible_neighbors_clusters_or_other_native_errors_are_logged_failed_cells; no_ROS_fallback_parameter_relaxation_or_synthetic_metrics",
                "ancestry": "imblearn_parent_indices_not_exposed_and_not_fabricated"},
            "paired_deltas": "all_unordered_METHODS_pairs_in_method_order; later_method_minus_earlier_reference; identical_data_and_split_required; missing_pairs_have_NaN_deltas",
            "comparison": "per_dataset_classifier_F1_mean_sample_std_and_completed_expected_seed_coverage_for_every_method; no_partial_coverage_winner",
            "smoke": "all_smoke_observations_drawn_only_from_scientific_outer_train"}}
    records, selections, failures = [], {}, []
    def flush():
        results, summary, paired, comparison = _tables(records, selected, config)
        results.to_csv(run_dir / "per_cell_metrics.csv", index=False, float_format="%.17g")
        summary.to_csv(run_dir / "summary.csv", index=False, float_format="%.17g")
        paired.to_csv(run_dir / "paired_deltas.csv", index=False, float_format="%.17g")
        comparison.to_csv(run_dir / "comparison.csv", index=False, float_format="%.17g")
        manifest["coverage"] = {"expected_cells": expected, "completed_cells": len(records),
            "failed_or_missing_cells": expected - len(records), "complete": len(records) == expected,
            "failure_events": len(failures)}
        manifest["runtime_seconds"] = perf_counter() - started
        _dump(run_dir / "manifest.json", manifest)
        _dump(run_dir / "selections.json", selections)
        _dump(run_dir / "failures.json", failures)
        return results, summary, paired, comparison
    def failed(kind, context, error):
        failures.append({"kind": kind, **context, "exception_type": type(error).__name__,
                         "message": str(error), "traceback": traceback.format_exc()})
        print(f"FAILED {kind}: {context}: {type(error).__name__}: {error}", flush=True)
        flush()
    flush()
    original_threads = torch.get_num_threads()
    torch.set_num_threads(config["N_JOBS"])
    try:
        with threadpool_limits(limits=config["N_JOBS"]):
            for entry in selected:
                context = {"dataset_number": entry["number"], "dataset": entry["name"], "openml_id": entry["openml_id"]}
                try:
                    dataset = _load_dataset(entry, config)
                    manifest["datasets"].append(dataset["metadata"])
                    flush()
                except Exception as error:
                    failed("dataset_error", {**context, "affected_cells": len(config["CLASSIFIERS"]) * len(config["SEEDS"]) * len(METHODS)}, error)
                    continue
                for seed in config["SEEDS"]:
                    try:
                        split = _split(dataset, seed, config)
                        partitions = _partitions(dataset, split)
                    except Exception as error:
                        failed("dataset_seed_error", {**context, "seed": seed,
                            "affected_cells": len(config["CLASSIFIERS"]) * len(METHODS)}, error)
                        continue
                    prepared, gates, flow_evidence, flow_error = {}, None, None, None
                    for classifier in config["CLASSIFIERS"]:
                        candidates = candidate_parameters(classifier, seed, config["N_CANDIDATES"], config["SMOKE"])
                        for method in METHODS:
                            cell_context = {**context, "classifier": classifier, "seed": seed, "method": method,
                                "candidate_configs": candidates, "config": config}
                            try:
                                if method == "u2_flow_gate":
                                    if flow_error is not None:
                                        raise RuntimeError(f"Shared real-only FlowGMM fit failed: {flow_error}")
                                    if gates is None:
                                        try:
                                            gates, flow_evidence = _flow_roles(dataset, split, partitions, seed, config)
                                            manifest["shared_flow_fits"].append({
                                                "dataset_number": entry["number"], "dataset": entry["name"],
                                                "seed": seed, "fit_seconds": flow_evidence["fit_seconds"],
                                                "core_sha256": flow_evidence["core_sha256"], "physical_fit_count": 1})
                                        except Exception as error:
                                            flow_error = f"{type(error).__name__}: {error}"
                                            raise
                                record, selection = _cell(dataset, split, partitions, classifier, seed,
                                    method, config, candidates, run_dir, prepared, gates, flow_evidence)
                                records.append(record)
                                selections[_safe_name(dataset, classifier, seed, method)] = selection
                                flush()
                                print(f"{entry['number']:02d} {entry['name']} | {classifier} | seed {seed} | {method}: "
                                    f"F1={record['minority_f1']:.4f} AP={record['average_precision']:.4f} "
                                    f"accepted={record['accepted_count']}/{record['requested_count']} backend={record['backend']}", flush=True)
                            except Exception as error:
                                failed("cell_error", cell_context, error)
                    del partitions, prepared, gates
    finally:
        torch.set_num_threads(original_threads)
    manifest["status"] = "completed" if len(records) == expected else "incomplete"
    results, summary, paired, comparison = flush()
    coverage = manifest["coverage"]
    print(f"Saved {run_dir}\nCoverage: {coverage['completed_cells']}/{expected} cells; "
          f"{'COMPLETE' if coverage['complete'] else 'INCOMPLETE — inspect failures.json'}. "
          f"{'Training-only smoke, not scientific evidence.' if config['SMOKE'] else ''}", flush=True)
    return {"run_dir": str(run_dir), "results": results, "summary": summary,
            "paired_deltas": paired, "comparison": comparison}
