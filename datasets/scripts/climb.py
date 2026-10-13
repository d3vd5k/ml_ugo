import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


CATALOG_PATH = Path(__file__).resolve().parents[1] / "climb_catalog.csv"


def _normalize_missing(X):
    """Normalize nullable pandas values without learning from any partition."""
    result = X.copy()
    for column in result:
        values = result[column]
        if pd.api.types.is_numeric_dtype(values.dtype) and not pd.api.types.is_bool_dtype(values.dtype):
            result[column] = values.to_numpy(dtype=np.float64, na_value=np.nan)
            result[column] = result[column].replace([np.inf, -np.inf], np.nan)
        else:
            result[column] = values.map(lambda value: str(value) if pd.notna(value) else np.nan).astype(object)
    return result


def _dense_float32(X):
    return np.asarray(X, dtype=np.float32)


def make_preprocessor(X_train):
    """Build an unfitted transformer; fit it only on the actual training fold.

    Numeric features use median imputation and standardization. Categorical
    features use mode imputation and dense one-hot encoding; unseen categories
    become all-zero blocks. Empty training columns remain represented.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("make_preprocessor requires a raw pandas DataFrame.")
    numeric = [column for column in X_train if pd.api.types.is_numeric_dtype(X_train[column].dtype)
               and not pd.api.types.is_bool_dtype(X_train[column].dtype)]
    categorical = [column for column in X_train if column not in numeric]
    columns = ColumnTransformer(
        [
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]), numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
            ]), categorical),
        ],
        sparse_threshold=0,
    )
    return Pipeline([
        ("normalize", FunctionTransformer(_normalize_missing, feature_names_out="one-to-one")),
        ("columns", columns),
        ("float32", FunctionTransformer(_dense_float32, feature_names_out="one-to-one")),
    ])


def categorical_columns(preprocessor):
    """Return one-hot output indices from an already fitted preprocessor."""
    block = preprocessor.named_steps["columns"].output_indices_["categorical"]
    return tuple(range(block.start, block.stop))


def _raw_fingerprint(features, target, row_ids):
    digest = hashlib.sha256()
    schema = [(str(column), str(features[column].dtype)) for column in features]
    digest.update(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
    for values in (features, pd.Series(target), pd.Series(row_ids)):
        digest.update(pd.util.hash_pandas_object(values, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def load_climb_datasets(
    dataset_names=None,
    target_type="binary",
    max_datasets=None,
    data_home=".cache/climb",
):
    """Load raw cached OpenML tables, never IMBENS' preprocessed NPZ files.

    No feature imputation, scaling, encoding or row filtering happens here.
    Targets are mapped deterministically to contiguous integers. Original
    OpenML row IDs (or original row positions) survive every later split.
    """
    import openml

    cache_path = Path(data_home).resolve() / "openml"
    cache_path.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(CATALOG_PATH)
    if target_type != "all":
        catalog = catalog[catalog["target_type"] == target_type]
    if dataset_names:
        requested = set(dataset_names)
        catalog = catalog[catalog["dataset_name"].isin(requested)]
        missing = requested - set(catalog["dataset_name"])
        if missing:
            raise ValueError(f"Unknown CLIMB dataset names: {sorted(missing)}")
    if max_datasets is not None:
        catalog = catalog.head(max_datasets)
    if catalog.empty:
        raise ValueError("No CLIMB datasets match the requested filters.")

    datasets = []
    previous_cache = openml.config.get_config_as_dict()["cachedir"]
    openml.config.set_root_cache_directory(cache_path)
    try:
        for row in catalog.itertuples(index=False):
            raw = openml.datasets.get_dataset(int(row.openml_id), download_data=True)
            features, target, _, _ = raw.get_data(
                target=raw.default_target_attribute, include_row_id=True, dataset_format="dataframe"
            )
            if target is None or np.asarray(target).ndim != 1:
                raise ValueError(f"Dataset {row.dataset_name} requires a single target column.")
            target = np.asarray(target)
            if pd.isna(target).any():
                raise ValueError(f"Dataset {row.dataset_name} has missing target labels.")
            y, labels = pd.factorize(target, sort=True)
            y = np.asarray(y, dtype=np.int64)
            row_attribute = raw.row_id_attribute
            row_columns = ([row_attribute] if isinstance(row_attribute, str) else list(row_attribute or []))
            # Some metadata marks a row ID as ignored too; read it explicitly.
            if row_columns:
                identity_frame, _, _, _ = raw.get_data(
                    include_row_id=True, include_ignore_attribute=True, dataset_format="dataframe"
                )
                if len(row_columns) == 1:
                    row_ids = identity_frame[row_columns[0]].to_numpy(copy=True)
                else:
                    row_ids = np.empty(len(identity_frame), dtype=object)
                    row_ids[:] = list(identity_frame[row_columns].itertuples(index=False, name=None))
                features = features.drop(columns=row_columns, errors="ignore")
            else:
                row_ids = features.index.to_numpy(copy=True)
            metadata = {
                "source": "openml_raw",
                "openml_id": int(row.openml_id),
                "openml_name": raw.name,
                "version": int(raw.version),
                "target_attribute": raw.default_target_attribute,
                "row_id_attributes": row_columns,
                "row_id_source": "attribute" if row_columns else "original_row_position",
                "raw_fingerprint": _raw_fingerprint(features, target, row_ids),
                "fingerprint_algorithm": "sha256_pandas_raw_values_v1",
                "openml_md5_checksum": raw.md5_checksum,
                "label_mapping": [{"label": label.item() if isinstance(label, np.generic) else label,
                                   "encoded": index} for index, label in enumerate(labels)],
                "feature_dtypes": {str(column): str(dtype) for column, dtype in features.dtypes.items()},
                "n_rows": len(features),
                "n_features": features.shape[1],
            }
            datasets.append({
                "name": row.dataset_name,
                "openml_id": int(row.openml_id),
                "imbalance_type": row.imbalance_type,
                "X": features,
                "y": y,
                "row_ids": row_ids,
                "metadata": metadata,
            })
    finally:
        openml.config.set_root_cache_directory(previous_cache)
    return datasets
