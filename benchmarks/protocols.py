"""Raw Stage 1 datasets and leakage-isolated, shared split contracts.

No learned preprocessing or resampling is performed here. Archive/member
checksums are pinned in manifest.yaml; Kaggle credentials are never accessed.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from datasets.scripts.climb import _raw_fingerprint

MANIFEST_PATH = Path(__file__).with_name("manifest.yaml")
METHODS = {"baseline", "class_weight", "ros", "smote", "borderline_smote", "kmeans_smote",
           "ugo_original", "ugo_corrected", "ugo_oof_mma", "ugo_groups_only", "ugo_constraints_only",
           "ugo_stopping_only", "ugo_combined"}
CLASSIFIERS = {"logistic_regression", "random_forest", "lightgbm", "balanced_random_forest",
               "decision_tree", "adaboost", "mlp", "xgboost", "knn", "svm_rbf"}


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_protocol(protocol):
    """Validate executable configuration structure, without resolving availability."""
    if not isinstance(protocol, dict):
        raise ValueError("Protocol must be a mapping")
    required = {"protocol_id", "dataset", "track", "status", "primary_metric", "split",
                "validation_fraction", "classifiers", "seeds", "tuning_budget", "threshold", "methods"}
    missing = required - protocol.keys()
    if missing:
        raise ValueError(f"Missing protocol fields: {sorted(missing)}")
    if not isinstance(protocol["protocol_id"], str) or not protocol["protocol_id"].strip():
        raise ValueError("protocol_id must be nonempty")
    if protocol["dataset"] not in {"shoppers", "aps", "baf_base", "satimage"}:
        raise ValueError("Unsupported Stage 1 dataset")
    if protocol["track"] not in {"controlled", "published"} or protocol["status"] not in {"ready", "unresolved"}:
        raise ValueError("Invalid track/status")
    if protocol["status"] == "unresolved" and not protocol.get("missing_prerequisites"):
        raise ValueError("Unresolved protocols must list exact missing_prerequisites")
    if protocol["primary_metric"] not in {"average_precision", "aps_cost", "recall_at_fpr", "macro_f1",
                                          "sos_inverse_support_f1", "accuracy"}:
        raise ValueError("Unsupported primary metric")
    fraction = protocol["validation_fraction"]
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    for key, choices in (("classifiers", CLASSIFIERS), ("methods", METHODS)):
        values = protocol[key]
        if not isinstance(values, list) or not values or any(not isinstance(x, str) for x in values):
            raise ValueError(f"{key} must be a nonempty list of names")
        if len(set(values)) != len(values) or set(values) - choices:
            raise ValueError(f"Unknown or duplicate {key}: {values}")
    seeds = protocol["seeds"]
    if not isinstance(seeds, list) or not seeds or any(not _integer(s) or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    budget = protocol["tuning_budget"]
    if not isinstance(budget, dict) or not _integer(budget.get("n_candidates")) or budget["n_candidates"] < 1:
        raise ValueError("tuning_budget.n_candidates must be a positive integer")
    if not _integer(budget.get("sampler_seed", 0)) or budget.get("sampler_seed", 0) < 0:
        raise ValueError("sampler_seed must be a nonnegative integer")
    threshold = protocol["threshold"]
    if not isinstance(threshold, dict) or threshold.get("name") not in {"fixed_0.5", "aps_cost", "recall_at_fpr"}:
        raise ValueError("threshold must declare a supported name")
    if threshold["name"] == "recall_at_fpr" and threshold.get("fpr") != 0.05:
        raise ValueError("BAF operating point is FPR=0.05")
    split = protocol["split"]
    if not isinstance(split, dict) or split.get("type") not in {"stratified_holdout", "official", "temporal", "author_fixed"}:
        raise ValueError("Unsupported split type")
    if split["type"] == "stratified_holdout":
        if not 0 < split.get("test_fraction", 0) < 1 or not _integer(split.get("outer_seed")):
            raise ValueError("Holdout requires test_fraction and fixed outer_seed")
    expected_split = {"shoppers": "stratified_holdout", "aps": "official", "baf_base": "temporal", "satimage": "official"}
    if protocol["track"] == "controlled" and split["type"] != expected_split[protocol["dataset"]]:
        raise ValueError("Controlled dataset split violates its official/temporal contract")
    if protocol["track"] == "controlled":
        objective = {"shoppers": "average_precision", "aps": "aps_cost",
                     "baf_base": "recall_at_fpr", "satimage": "macro_f1"}
        rule = {"shoppers": "fixed_0.5", "aps": "aps_cost",
                "baf_base": "recall_at_fpr", "satimage": "fixed_0.5"}
        if protocol["primary_metric"] != objective[protocol["dataset"]] or threshold["name"] != rule[protocol["dataset"]]:
            raise ValueError("Controlled dataset objective/threshold must remain prespecified")
    if threshold["name"] == "aps_cost" and (threshold.get("fp_cost", 10) != 10 or threshold.get("fn_cost", 500) != 500):
        raise ValueError("APS cost is exactly 10FP+500FN")
    if protocol["dataset"] == "baf_base" and split["type"] == "temporal":
        if split.get("train_months") != list(range(6)) or split.get("test_months") != [6, 7] or split.get("validation_months") != [5]:
            raise ValueError("BAF requires train months0..5, test6..7, validation5")
    for key in ("classifier_parameters", "search_space", "candidate_parameters"):
        if key in protocol and not isinstance(protocol[key], dict):
            raise ValueError(f"{key} must be a classifier-keyed mapping")
        if key in protocol and set(protocol[key]) - set(protocol["classifiers"]):
            raise ValueError(f"{key} contains an undeclared classifier")
    for classifier, candidates in protocol.get("candidate_parameters", {}).items():
        if not isinstance(candidates, list) or len(candidates) != budget["n_candidates"] or any(not isinstance(c, dict) for c in candidates):
            raise ValueError(f"{classifier} must have exactly n_candidates parameter dictionaries")
    augmentation = protocol.get("augmentation", {"mode": "matched", "total": 1000})
    if not isinstance(augmentation, dict) or augmentation.get("mode") not in {"matched", "native"}:
        raise ValueError("augmentation.mode must be matched or native")
    if not _integer(augmentation.get("total")) or augmentation["total"] < 0:
        raise ValueError("augmentation.total must be a nonnegative integer")
    return protocol


def load_protocol(path):
    with Path(path).open(encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    return validate_protocol(protocol)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive(spec, data_home, download):
    path = Path(data_home) / spec["archive"]
    # Audit acquisition and normal runner caches share the same immutable bytes.
    if not path.exists() and (path.parent / "audit" / path.name).exists():
        path = path.parent / "audit" / path.name
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Missing {path}; pass download=True to acquire public UCI data")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            with urllib.request.urlopen(spec["download_url"], timeout=180) as response, temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            if _sha(temporary) != spec["archive_sha256"]:
                raise ValueError(f"Source archive checksum mismatch: {path.name}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if _sha(path) != spec["archive_sha256"]:
        raise ValueError(f"Source archive checksum mismatch: {path.name}")
    return path


def _member(archive, name, expected):
    with zipfile.ZipFile(archive) as handle:
        matches = [member for member in handle.namelist() if Path(member).name == name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one archive member {name}")
        data = handle.read(matches[0])
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch for member {name}")
    return io.BytesIO(data)


def _csv(path, **kwargs):
    # Reject duplicate header labels before pandas can silently suffix them.
    import csv
    if isinstance(path, Path):
        with path.open(encoding="utf-8") as handle:
            header = next(csv.reader(handle))
    else:
        position = path.tell()
        header = next(csv.reader(io.StringIO(path.read().decode("utf-8"))), None)
        path.seek(position)
    if header is None or len(set(header)) != len(header):
        raise ValueError("Missing or duplicate CSV column names")
    return pd.read_csv(path, **kwargs)


def _dataset(name, frames, spec, source_checksums, target, labelmap, categorical):
    frame = pd.concat(frames, ignore_index=True)
    if frame.columns.has_duplicates or set(frame) != set(spec["columns"]):
        raise ValueError(f"{name}: schema mismatch; expected {spec['columns']}, got {list(frame)}")
    if len(frame) != spec["rows"]:
        raise ValueError(f"{name}: expected {spec['rows']} rows, got {len(frame)}")
    y = frame.pop(target).map(labelmap)
    if y.isna().any():
        raise ValueError(f"{name}: missing or unmapped target values")
    y = y.to_numpy(dtype=np.int64)
    counts = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    if spec.get("class_counts") and counts != {str(k): v for k, v in spec["class_counts"].items()}:
        raise ValueError(f"{name}: unexpected class counts {counts}")
    for column in spec["numeric_features"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if np.isinf(frame[column].to_numpy(dtype=np.float64)).any():
            raise ValueError(f"{name}: nonfinite numeric predictor {column}")
    for column, token in spec.get("missing_values", {}).items():
        frame[column] = frame[column].replace(token, np.nan)
    for column in categorical:
        frame[column] = frame[column].astype("category")
    metadata = {"source": spec["source"], "version": spec["version"], "license": spec["license"],
                "source_checksums": source_checksums, "label_mapping": {str(k): v for k, v in labelmap.items()},
                "class_counts": counts, "categorical_features": categorical,
                "missingness": {column: int(value) for column, value in frame.isna().sum().items()},
                "entity_fields": spec["entity_fields"], "time_fields": spec["time_fields"],
                "climb_overlap": spec["climb_overlap"], "preprocessing": "raw; fit transforms on fit_indices only"}
    ids = np.concatenate([np.array([f"{name}:{part}:{i}" for i in range(len(rows))], dtype=object)
                          for part, rows in enumerate(frames)])
    result = {"name": name, "X": frame, "y": y, "row_ids": ids, "metadata": metadata}
    if len(frames) == 2:
        result["official_train_indices"] = np.arange(len(frames[0]), dtype=np.int64)
        result["official_test_indices"] = np.arange(len(frames[0]), len(frame), dtype=np.int64)
    return result


def validate_dataset(dataset):
    X, y, ids = dataset["X"], np.asarray(dataset["y"]), np.asarray(dataset["row_ids"])
    if not isinstance(X, pd.DataFrame) or X.columns.has_duplicates or X.columns.isna().any():
        raise ValueError("X must have unique nonmissing DataFrame column names")
    if y.dtype != np.dtype("int64") or y.ndim != 1 or len(y) != len(X):
        raise ValueError("y must be an aligned int64 vector")
    if not np.array_equal(np.unique(y), np.arange(len(np.unique(y)))) or len(np.unique(y)) < 2:
        raise ValueError("Labels must be contiguous and contain at least two classes")
    if ids.ndim != 1 or len(ids) != len(X) or pd.isna(ids).any() or pd.Index(ids).has_duplicates:
        raise ValueError("row_ids must be unique, nonmissing and aligned")
    if any(isinstance(value, str) and not value.strip() for value in ids):
        raise ValueError("row_ids cannot be blank")
    if "time_values" in dataset and (len(dataset["time_values"]) != len(X) or pd.isna(dataset["time_values"]).any()):
        raise ValueError("time_values must be nonmissing and aligned")
    return dataset


def load_dataset(protocol, data_home, local_data=None, download=False):
    validate_protocol(protocol)
    if protocol["status"] != "ready":
        raise ValueError("Unresolved protocol: " + "; ".join(protocol["missing_prerequisites"]))
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)["datasets"][protocol["dataset"]]
    name = protocol["dataset"]
    if name == "baf_base":
        if local_data is None:
            raise FileNotFoundError("BAF requires local official Kaggle Base.csv, Base.parquet or release ZIP; no credentials are read")
        path = Path(local_data)
        if path.is_dir():
            files = [p for p in path.iterdir() if p.name in {"Base.csv", "Base.parquet"}]
            if len(files) != 1:
                raise ValueError("Supply exactly one Base.csv or Base.parquet")
            path = files[0]
        checksums = {path.name: _sha(path)}
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                members = [n for n in archive.namelist() if Path(n).name == "Base.csv"]
                if len(members) != 1:
                    raise ValueError("Official release ZIP must contain exactly one Base.csv")
                raw = archive.read(members[0])
            checksums["Base.csv"] = hashlib.sha256(raw).hexdigest()
            frame = _csv(io.BytesIO(raw))
        elif path.name == "Base.csv":
            frame = _csv(path)
        elif path.name == "Base.parquet":
            frame = pd.read_parquet(path)
        else:
            raise ValueError("Only official Base.csv, Base.parquet or release ZIP is accepted")
        source_ids = None
        if "Unnamed: 0" in frame:
            source_ids = frame.pop("Unnamed: 0").to_numpy()
        elif path.name == "Base.parquet" and not isinstance(frame.index, pd.RangeIndex):
            source_ids = frame.index.to_numpy()
        result = _dataset(name, [frame], spec, checksums, "fraud_bool", {0: 0, 1: 1}, spec["categorical_features"])
        if source_ids is not None:
            result["row_ids"] = source_ids
        months = result["X"]["month"].to_numpy()
        if not np.array_equal(np.unique(months), np.arange(8)):
            raise ValueError("BAF month encoding must contain exactly zero-based 0..7")
        result["time_values"] = months
        result["official_train_indices"] = np.flatnonzero(months < 6)
        result["official_test_indices"] = np.flatnonzero(months >= 6)
        result["metadata"]["release_identity"] = "Local user-supplied official Kaggle Base; upstream checksum unavailable"
    else:
        if local_data is not None:
            raise ValueError("UCI loaders use pinned cached sources; local_data is reserved for Kaggle Base")
        archive = None if name == "aps" else _archive(spec, data_home, download)
        checksums = dict(spec["data_members"])
        if archive is not None:
            checksums[archive.name] = spec["archive_sha256"]
        frames = []
        for member, expected in spec["data_members"].items():
            if name == "aps":
                raw = _archive({"archive": member, "archive_sha256": expected,
                                "download_url": spec["download_urls"][member]}, data_home, download)
                frames.append(pd.read_csv(raw, skiprows=20, na_values=["na"]))
            else:
                raw = _member(archive, member, expected)
                if name == "satimage":
                    frames.append(pd.read_csv(raw, sep=r"\s+", header=None, names=spec["columns"]))
                else:
                    frames.append(_csv(raw))
        maps = {"shoppers": {False: 0, True: 1}, "aps": {"neg": 0, "pos": 1},
                "satimage": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 7: 5}}
        result = _dataset(name, frames, spec, checksums,
                          spec["target"], maps[name], spec["categorical_features"])
        for rows, expected_count in zip(frames, spec["partition_rows"]):
            if len(rows) != expected_count:
                raise ValueError(f"{name}: official partition row count mismatch")
        for indices, expected_counts in zip([result.get("official_train_indices", np.arange(len(result["y"]))),
                                            result.get("official_test_indices", [])], spec.get("partition_class_counts", [])):
            counts = dict(zip(*np.unique(result["y"][indices], return_counts=True)))
            if counts != {int(key): value for key, value in expected_counts.items()}:
                raise ValueError(f"{name}: official partition class-count mismatch")
    validate_dataset(result)
    result["metadata"]["raw_fingerprint"] = _raw_fingerprint(result["X"], result["y"], result["row_ids"])
    result["metadata"]["rows"] = len(result["y"])
    result["metadata"]["predictors"] = len(result["X"].columns)
    return result


def _indices(values, n, label):
    result = np.asarray(values)
    if result.ndim != 1 or not np.issubdtype(result.dtype, np.integer) or not len(result):
        raise ValueError(f"{label} must be a nonempty integer index vector")
    if np.any(result < 0) or np.any(result >= n) or len(np.unique(result)) != len(result):
        raise ValueError(f"{label} contains out-of-bounds or duplicate indices")
    return result.astype(np.int64, copy=False)


def make_splits(dataset, protocol, seed):
    validate_protocol(protocol)
    validate_dataset(dataset)
    if protocol["status"] != "ready":
        raise ValueError("Cannot split an unresolved published protocol")
    if dataset["name"] != protocol["dataset"]:
        raise ValueError("Dataset name does not match protocol dataset")
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    y, n = dataset["y"], len(dataset["y"])
    split = protocol["split"]
    if split["type"] == "stratified_holdout":
        train, test = train_test_split(np.arange(n), test_size=split["test_fraction"],
                                      random_state=split["outer_seed"], stratify=y)
    elif split["type"] in {"official", "temporal"}:
        train = _indices(dataset["official_train_indices"], n, "official training")
        test = _indices(dataset["official_test_indices"], n, "official test")
    else:
        raise ValueError("Author fixed protocol is unresolved; no invented split")
    train, test = _indices(train, n, "outer training"), _indices(test, n, "outer test")
    if np.intersect1d(train, test).size or len(train) + len(test) != n:
        raise ValueError("Outer partitions must cover all rows exactly once")
    if split["type"] == "temporal":
        time = np.asarray(dataset["time_values"])
        if not np.array_equal(np.unique(time), np.arange(8)) or not np.all(time[train] < 6) or not np.all(time[test] >= 6):
            raise ValueError("BAF official partitions violate zero-based temporal boundary")
        validation = train[time[train] == 5]
        fit = train[time[train] < 5]
    else:
        fit, validation = train_test_split(train, test_size=protocol["validation_fraction"],
                                           random_state=seed, stratify=y[train])
    fit, validation = _indices(fit, n, "fit"), _indices(validation, n, "validation")
    if np.intersect1d(fit, validation).size or not np.array_equal(np.sort(np.r_[fit, validation]), np.sort(train)):
        raise ValueError("Fit and validation must partition outer training exactly")
    for label, indices in (("fit", fit), ("validation", validation), ("test", test)):
        if not np.array_equal(np.unique(y[indices]), np.unique(y)):
            raise ValueError(f"Every class must be present in {label}")
    return [{"fold": 0, "train_indices": train, "test_indices": test,
             "fit_indices": fit, "validation_indices": validation}]


def main():
    parser = argparse.ArgumentParser(description="Acquire and audit real Shoppers/APS splits without fitting any model")
    parser.add_argument("--data-home", default=".cache/benchmarks")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", default="results/smoke/protocol_acquisition.json")
    args = parser.parse_args()
    report = {"experiment_type": "setup_smoke", "model_fits": 0, "datasets": {}}
    root = Path(__file__).resolve().parents[1]
    for name in ("shoppers", "aps"):
        protocol = load_protocol(root / "configs" / "benchmarks" / f"{name}_controlled.yaml")
        data = load_dataset(protocol, args.data_home, download=args.download)
        splits = make_splits(data, protocol, protocol["seeds"][0])
        partition = splits[0]
        report["datasets"][name] = {"metadata": data["metadata"],
                                     "split_counts": {key: len(values) for key, values in partition.items() if key.endswith("indices")},
                                     "split_class_counts": {key: dict(zip(map(str, np.unique(data["y"][values])),
                                                                         map(int, np.unique(data["y"][values], return_counts=True)[1])))
                                                             for key, values in partition.items() if key.endswith("indices")},
                                     "split_sha256": {key: hashlib.sha256(values.tobytes()).hexdigest()
                                                      for key, values in partition.items() if key.endswith("indices")}}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "model_fits": 0,
                      "datasets": {name: value["split_counts"] for name, value in report["datasets"].items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
