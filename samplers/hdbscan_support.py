"""H1 model-based support gate fitted once to real minority training rows.

The caller supplies training-standardized continuous coordinates and complete
one-hot categorical coordinates. Continuous coordinates are left unchanged;
every categorical coordinate is scaled by 1/sqrt(2). Thus changing category in
one complete one-hot feature block contributes exactly 1 to squared Euclidean
distance (two coordinates change). This is not a density or correctness guarantee.
Only flat column indices are available here, so binary categorical values are
checked, but one-hot block sums cannot be checked: the caller's domain validator
must enforce them. Synthetic queries never fit or restandardize any geometry.
"""
from importlib.metadata import version
from numbers import Integral
from time import perf_counter

import numpy as np


class H1Support:
    """Frozen HDBSCAN geometry; only inference counters change on prediction.

    Real fitting labels and approximate real assignments are distinct diagnostics.
    Membership strength is not label-correctness probability or a GLOSH score.
    With tied real class counts, the first class in numpy.unique order is used.
    """

    def __init__(self, X, y, categorical_columns=(), parameters=None, real_row_ids=None):
        parameters = dict(parameters or {})
        config = {}
        for name, default, minimum in (("h1_min_cluster_size", 10, 2),
                                       ("h1_min_samples", 5, 1)):
            value = parameters.get(name, default)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                    or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}")
            config[name] = int(value)
        source, labels = np.asarray(X), np.asarray(y)
        if (source.ndim != 2 or source.shape[1] == 0 or labels.ndim != 1
                or len(source) != len(labels) or not np.issubdtype(source.dtype, np.number)
                or np.iscomplexobj(source) or not np.isfinite(source).all()):
            raise ValueError("H1 requires finite real numeric features and matching one-dimensional labels")
        if (np.issubdtype(labels.dtype, np.number)
                and (np.iscomplexobj(labels) or not np.isfinite(labels).all())):
            raise ValueError("H1 requires finite real binary class labels")
        classes, counts = np.unique(labels, return_counts=True)
        if len(classes) != 2:
            raise ValueError("H1 requires exactly two real training classes")
        columns = tuple(categorical_columns)
        if (any(isinstance(c, (bool, np.bool_)) or not isinstance(c, Integral)
                or c < 0 or c >= source.shape[1] for c in columns)
                or len(set(columns)) != len(columns)):
            raise ValueError("categorical_columns must contain distinct valid column indices")
        self.categorical_columns = tuple(int(c) for c in columns)
        if columns and not np.isin(source[:, columns], (0, 1)).all():
            raise ValueError("Real categorical coordinates must be 0 or 1 in complete one-hot blocks")
        row_ids = None if real_row_ids is None else np.asarray(real_row_ids)
        if row_ids is not None and (row_ids.ndim != 1 or len(row_ids) != len(labels)):
            raise ValueError("real_row_ids must match the real training rows")
        self._width = source.shape[1]
        self.minority_class = classes[np.argmin(counts)].item()
        indices = np.flatnonzero(labels == self.minority_class)
        self._fit_X = self._represent(source[indices])
        if not np.isfinite(self._fit_X).all():
            raise ValueError("Real minority features must remain finite in Float64")
        self._clusterer = None
        self.audit = {
            "configuration": {**config, "metric": "euclidean", "prediction_data": True,
                              "cluster_selection_method": "eom", "allow_single_cluster": False},
            "minority_class": self.minority_class,
            "minority_tie_policy": "first_class_in_numpy_unique_order",
            "fit_real_indices": indices.copy(),
            "fit_real_row_ids": None if row_ids is None else row_ids[indices].copy(),
            "real_fit_cluster_labels": np.full(len(indices), -1, dtype=np.int64),
            "real_prediction_labels": np.full(len(indices), -1, dtype=np.int64),
            "real_prediction_strengths": np.zeros(len(indices), dtype=np.float64),
            "usable_clusters": 0, "status": "no_usable_clusters",
            "training_seconds": 0., "inference_seconds": 0., "inference_count": 0,
            "synthetic_training_count": 0,
            "distance_representation": {
                "metric": "euclidean", "continuous": "caller_training_standardized_unchanged",
                "categorical_columns": list(self.categorical_columns),
                "categorical_coordinate_scale": float(1 / np.sqrt(2)),
                "squared_distance_per_valid_onehot_block_mismatch": 1.,
                "categorical_validation": "binary_coordinates_only_block_boundaries_not_supplied",
                "synthetic_restandardization": False,
            },
            "prediction_convention": "hdbscan.approximate_predict",
            "membership_strength_interpretation": "model_membership_not_label_probability_or_GLOSH",
        }
        import hdbscan

        self._hdbscan = hdbscan
        self.audit["package_version"] = version("hdbscan")
        start = perf_counter()
        try:
            # Prediction-data construction requires at least min_samples rows.
            # Such tiny partitions cannot supply two usable EOM clusters; report
            # this structural degeneracy explicitly, never relax the parameters.
            if len(indices) < max(2, config["h1_min_samples"]):
                self.audit["degenerate_reason"] = "insufficient_real_minority_rows"
                return
            self._clusterer = hdbscan.HDBSCAN(
                min_cluster_size=config["h1_min_cluster_size"],
                min_samples=config["h1_min_samples"], metric="euclidean",
                prediction_data=True, core_dist_n_jobs=1).fit(self._fit_X)
            self.audit["real_fit_cluster_labels"] = self._clusterer.labels_.copy()
            clusters = np.unique(self._clusterer.labels_[self._clusterer.labels_ >= 0])
            self.audit["usable_clusters"] = int(len(clusters))
            if not len(clusters) or not len(self._clusterer.prediction_data_.cluster_tree):
                self.audit["degenerate_reason"] = "real_minority_fit_has_no_predictable_clusters"
                self.audit["usable_clusters"] = 0
                return
            predicted, strengths = hdbscan.approximate_predict(self._clusterer, self._fit_X)
            self.audit["real_prediction_labels"] = predicted.copy()
            self.audit["real_prediction_strengths"] = strengths.copy()
            self.audit["status"] = "ready"
        except Exception as error:
            raise RuntimeError(f"H1 HDBSCAN real-minority fit/prediction failed: {error}") from error
        finally:
            # Cython fitting needs writable memoryviews; freeze after fitting.
            self._fit_X.setflags(write=False)
            self.audit["training_seconds"] = perf_counter() - start

    def _represent(self, X):
        points = np.array(X, dtype=np.float64, copy=True)
        if self.categorical_columns:
            points[:, self.categorical_columns] *= 1 / np.sqrt(2)
        return points

    def predict(self, X):
        """Return row-aligned diagnostics; accept iff valid & (labels != -1).

        Nonfinite rows and nonbinary categorical coordinates never enter package
        prediction. No usable clusters reject all otherwise-valid rows without
        repeatedly calling the package's warning-producing degenerate predictor.
        Empty batches are supported; inference_count counts candidate rows only.
        """
        points = np.asarray(X)
        if (points.ndim != 2 or points.shape[1] != self._width
                or not np.issubdtype(points.dtype, np.number) or np.iscomplexobj(points)):
            raise ValueError("H1 candidates must be a real numeric matrix with the fitted feature width")
        start = perf_counter()
        labels = np.full(len(points), -1, dtype=np.int64)
        strengths = np.zeros(len(points), dtype=np.float64)
        valid = np.isfinite(points).all(axis=1)
        reasons = np.full(len(points), "no_usable_clusters", dtype="<U24")
        reasons[~valid] = "nonfinite"
        if self.categorical_columns:
            categorical_valid = np.isin(points[:, self.categorical_columns], (0, 1)).all(axis=1)
            reasons[valid & ~categorical_valid] = "invalid_categorical"
            valid &= categorical_valid
        try:
            if self.audit["status"] == "ready" and valid.any():
                predicted, membership = self._hdbscan.approximate_predict(
                    self._clusterer, self._represent(points[valid]))
                labels[valid], strengths[valid] = predicted, membership
                reasons[valid] = np.where(predicted == -1, "hdbscan_noise", "accepted")
        except Exception as error:
            raise RuntimeError(f"H1 frozen candidate prediction failed: {error}") from error
        finally:
            self.audit["inference_seconds"] += perf_counter() - start
            self.audit["inference_count"] += len(points)
        return {"labels": labels, "strengths": strengths, "valid": valid, "reasons": reasons}


def prepare_h1_support(X, y, categorical_columns=(), parameters=None, real_row_ids=None):
    """Prepare one frozen real-only H1 detector for a single training partition."""
    return H1Support(X, y, categorical_columns, parameters, real_row_ids)
