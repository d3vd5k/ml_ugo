"""F2 class-conditional calibration around the unchanged, pinned F1 detector.

Likelihoods are continuous-proxy densities in the core's detector coordinates,
including its Jacobian; neither likelihood nor compatibility guarantees support.
"""
from dataclasses import asdict
from time import perf_counter

from joblib import hash as state_hash
import numpy as np
import pandas as pd

from samplers.flow_gmm import FlowGate, _checksum, _layout, _source_sha


QUANTILE_CONVENTION = (
    "numpy.quantile(method='linear') on all class rows, including closed-vocabulary zero-density -inf; "
    "exact ranks use the endpoint; interpolation from -inf to finite or -inf is -inf; "
    "finite endpoints use numpy linear; inclusive likelihood >= threshold")


def _audit_score(value):
    value = float(value)
    return "-inf" if np.isneginf(value) else value if np.isfinite(value) else None


def _support_quantile(values, q):
    ordered = np.sort(values)
    rank = (len(ordered) - 1) * q
    lower, upper = int(np.floor(rank)), int(np.ceil(rank))
    if lower == upper or np.isneginf(ordered[lower]):
        return float(ordered[lower])
    return float(np.quantile(ordered, q, method="linear"))


def _quantile(q):
    if q is None:
        return None
    if (isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, float, np.number))
            or not np.isfinite(q) or not 0 <= q <= 1):
        raise ValueError("support quantile must be None (off) or a finite number in [0, 1]")
    return float(q)


def _binding(core):
    return {"model_checksum": _checksum(core.model),
            "preprocessing_checksum": state_hash(core.preprocessor, hash_name="sha1"),
            "source_sha256": _source_sha(), "model_config": asdict(core.config),
            "classes": core.classes.tolist(), "fit_row_ids": core.fit_row_ids.tolist(),
            "seed": core.seed}


class FlowCalibration:
    """Immutable real-row scores, bound to an exact frozen model/preprocessor."""

    def __init__(self, core, row_ids, y, scores, known, binding, seconds):
        self.core = core
        self._model, self._preprocessor = core.model, core.preprocessor
        self.row_ids, self.y, self.known = row_ids.copy(), y.copy(), known.copy()
        self.scores = {key: np.asarray(value, dtype=np.float64).copy() for key, value in scores.items()}
        for value in (self.row_ids, self.y, self.known, *self.scores.values()):
            value.setflags(write=False)
        self.class_counts = {str(c): int((y == c).sum()) for c in core.classes}
        self.finite_class_counts = {str(c): int(((y == c) & known).sum()) for c in core.classes}
        self.unknown_class_counts = {str(c): int(((y == c) & ~known).sum()) for c in core.classes}
        self.binding = binding
        self.calibration_seconds = seconds
        self._thresholds = {}

    def assert_current(self, core):
        if (core is not self.core or core.model is not self._model
                or core.preprocessor is not self._preprocessor or _binding(core) != self.binding):
            raise ValueError("Stale FlowGMM calibration: model or preprocessing changed; recalibrate")

    def thresholds(self, q):
        q = _quantile(q)
        self.assert_current(self.core)
        if q is None:
            return {}
        key = str(q)
        if key not in self._thresholds:
            self._thresholds[key] = {str(c): _support_quantile(
                self.scores["log_likelihood_intended"][self.y == c], q)
                for c in self.core.classes}
        return dict(self._thresholds[key])

    @property
    def audit(self):
        self.assert_current(self.core)
        return {"calibration_row_ids": self.row_ids.tolist(),
                "calibration_classes": self.y.tolist(), "calibration_rows": len(self.row_ids),
                "calibration_class_counts": dict(self.class_counts),
                "original_calibration_class_counts": dict(self.class_counts),
                "calibration_total_class_counts": dict(self.class_counts),
                "calibration_finite_class_counts": dict(self.finite_class_counts),
                "calibration_unknown_class_counts": dict(self.unknown_class_counts),
                "calibration_score_kinds": np.where(self.known, "finite", "negative_infinity").tolist(),
                "calibration_reasons": np.where(self.known, "scored", "unknown_flow_category").tolist(),
                "unknown_category_support_convention": "closed_fit_vocabulary_zero_density; all unknown rows retained as -inf",
                "calibration_scores": {key: [_audit_score(value) for value in values]
                                       for key, values in self.scores.items()},
                "calibration_source": "exact_real_calibration_rows_only",
                "calibration_seconds": self.calibration_seconds,
                "fit_calibration_disjoint": True,
                "threshold_convention": QUANTILE_CONVENTION,
                "thresholds_per_q": {q: {c: _audit_score(value) for c, value in values.items()}
                                     for q, values in self._thresholds.items()},
                "source_binding": dict(self.binding)}


def calibrate_flow(core, raw_frame, y, row_ids):
    """Score precisely the supplied real calibration rows; never drop bad rows.

    The gate constructor checks these identities against its reserved calibration
    partition. Callers must subset raw data to those identities before calibration.
    """
    start = perf_counter()
    if not isinstance(raw_frame, pd.DataFrame) or raw_frame.empty:
        raise ValueError("Calibration requires a nonempty raw feature DataFrame")
    labels, ids = np.asarray(y), np.asarray(row_ids)
    if (ids.ndim != 1 or labels.ndim != 1 or len(ids) != len(raw_frame)
            or len(labels) != len(ids) or not len(ids) or pd.isna(ids).any()
            or len(set(ids.tolist())) != len(ids)):
        raise ValueError("Calibration requires nonempty unique aligned row IDs and labels")
    if set(ids.tolist()) & set(core.fit_row_ids.tolist()):
        raise ValueError("Flow fit and calibration rows must be disjoint")
    if not np.isin(labels, core.classes).all() or any(not np.any(labels == c) for c in core.classes):
        raise ValueError("Calibration requires both fitted real classes with no unknown labels")
    numeric, _, _, groups = _layout(core.preprocessor)
    if raw_frame.isna().to_numpy().any() or (numeric and not np.isfinite(
            raw_frame[numeric].to_numpy(dtype=np.float64)).all()):
        raise ValueError("Nonfinite calibration features cannot be imputed or dropped")
    if any(not value.isfinite().all().item() for value in core.model.state_dict().values()):
        raise ValueError("Nonfinite calibration model parameters")
    binding = _binding(core)
    transformed = np.asarray(core.preprocessor.transform(raw_frame), dtype=np.float64)
    if not np.isfinite(transformed).all():
        raise ValueError("Nonfinite calibration representation")
    known = np.ones(len(ids), dtype=bool)
    for group in groups:
        known &= transformed[:, group].sum(axis=1) == 1
    # Unknown fitted-vocabulary categories have zero closed-support density.
    # Keep them in their real class's calibration denominator, never refit or
    # invent finite scores. Ratio and Jacobian are undefined for those rows.
    required = {"log_likelihood_intended", "log_likelihood_opposing", "log_ratio", "log_determinant"}
    scores = {key: np.full(len(ids), -np.inf if key.startswith("log_likelihood") else np.nan)
              for key in required}
    if known.any():
        finite_scores = core.density(core.represent(raw_frame.iloc[np.flatnonzero(known)]), labels[known])
        if set(finite_scores) != required or any(np.asarray(values).shape != (int(known.sum()),)
                or not np.isfinite(values).all() for values in finite_scores.values()):
            raise ValueError("Nonfinite or missing calibration scores; calibration rows cannot be dropped")
        for key, values in finite_scores.items():
            scores[key][known] = values
    if _binding(core) != binding:
        raise ValueError("Flow model or preprocessing changed during calibration")
    return FlowCalibration(core, ids, labels, scores, known, binding, perf_counter() - start)


class CalibratedFlowGate:
    """F1 compatibility AND real-calibrated full-density support; None is off."""

    def __init__(self, base_gate, calibration, q=None):
        if not isinstance(base_gate, FlowGate) or not isinstance(calibration, FlowCalibration):
            raise ValueError("F2 requires an explicit FlowGate and FlowCalibration")
        calibration.assert_current(base_gate.core)
        if set(calibration.row_ids.tolist()) != set(base_gate.calibration_row_ids.tolist()):
            raise ValueError("Calibration must contain exactly the gate's reserved calibration rows")
        roles = [set(ids.tolist()) for ids in (base_gate.core.fit_row_ids,
                 calibration.row_ids, base_gate.validation_row_ids)]
        if any(roles[a] & roles[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Flow fit/calibration/validation row IDs must be disjoint")
        self.base_gate, self.calibration = base_gate, calibration
        self.core = base_gate.core
        self.q = _quantile(q)
        self.thresholds = calibration.thresholds(self.q)
        self.calibration_row_ids = base_gate.calibration_row_ids
        self.validation_row_ids = base_gate.validation_row_ids
        self.classifier_preprocessor = base_gate.classifier_preprocessor
        self._classifier_checksum = state_hash(self.classifier_preprocessor, hash_name="sha1")
        self.inference_seconds, self.inference_rows = 0., 0

    def _assert_current(self):
        self.calibration.assert_current(self.core)
        if (self.base_gate.core is not self.core
                or self.base_gate.classifier_preprocessor is not self.classifier_preprocessor
                or state_hash(self.base_gate.classifier_preprocessor, hash_name="sha1") != self._classifier_checksum):
            raise ValueError("Stale calibrated gate: classifier preprocessing or core changed")

    @property
    def audit(self):
        self._assert_current()
        return {**self.base_gate.audit, **self.calibration.audit,
                "role": self.base_gate.role, "roles_disjoint": True,
                "validation_row_ids": self.validation_row_ids.tolist(),
                "support_filter": self.q is not None, "support_quantile": self.q,
                "calibration_used": self.q is not None,
                "likelihood_thresholds": {c: _audit_score(value) for c, value in self.thresholds.items()},
                "model_checksum": self.calibration.binding["model_checksum"],
                "classifier_preprocessing_checksum": self._classifier_checksum,
                "inference_seconds": self.inference_seconds, "inference_rows": self.inference_rows}

    def predict(self, X, y, tau):
        start = perf_counter()
        self._assert_current()
        result = self.base_gate.predict(X, y, tau)
        compatible = result["accepted"].copy()
        threshold = np.full(len(compatible), np.nan, dtype=np.float64)
        likelihood = result["valid"].copy()
        if self.q is not None:
            labels = np.asarray(y)
            for c in self.core.classes:
                threshold[labels == c] = self.thresholds[str(c)]
            likelihood &= result["log_likelihood_intended"] >= threshold
        accepted = compatible & likelihood
        reasons = result["reasons"].copy()
        reasons[compatible & ~likelihood] = "flow_low_likelihood"
        self.inference_seconds += perf_counter() - start
        self.inference_rows += len(compatible)
        return {**result, "accepted": accepted, "reasons": reasons,
                "compatibility_accepted": compatible, "likelihood_accepted": likelihood,
                "likelihood_threshold": threshold}
