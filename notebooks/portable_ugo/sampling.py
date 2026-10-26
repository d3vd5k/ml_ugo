"""Training-role-only SMOTE, KMeansSMOTE and random walk oversampling.

SMOTE and KMeansSMOTE are the actual imbalanced-learn CPU algorithms on
numeric plus full one-hot coordinates, NOT SMOTENC. Only synthetic rows have
each complete categorical block projected to its argmax (first category breaks
ties). No validation/test observations or synthetic feedback fit any sampler.
KMeansSMOTE's ceil-rounded excess is deterministically trimmed, never refilled.

RWO is independently implemented from the numerical convention described at
https://raw.githubusercontent.com/analyticalmindsltd/smote_variants/master/smote_variants/oversampling/_rwo_sampling.py
and the nominal roulette convention described at
https://raw.githubusercontent.com/cran/imbalance/master/R/rwo.R .
For each class, a uniform real seed receives independent numeric noise with
scale std(real class, ddof=0) / sqrt(real class size). Each nominal variable is
sampled independently from real-class category frequencies. The R reference
uses a variance variant: this implementation follows the Python reference's
standard-deviation convention, not a bitwise reproduction of the R code.
There is no walk chain, synthetic-statistics update, numeric clipping, integer
rounding, ROS substitution, or copied third-party implementation code.
"""
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np


SAMPLER_DEFAULTS = {
    "smote": {"k_neighbors": 5},
    "kmeans_smote": {"k_neighbors": 2, "kmeans_estimator": 8,
                     "cluster_balance_threshold": 0.0, "density_exponent": "auto"},
    "rwo": {},
}


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _method_parameters(method, parameters):
    if method not in SAMPLER_DEFAULTS:
        raise ValueError(f"Unknown sampling control: {method!r}")
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, Mapping):
        raise ValueError(f"{method} parameters must be a mapping")
    unknown = set(parameters) - set(SAMPLER_DEFAULTS[method])
    if unknown:
        raise ValueError(f"Unsupported {method} parameters: {sorted(unknown, key=str)}; "
                         "use target_counts and seed for the sampling budget/random state")
    values = {**SAMPLER_DEFAULTS[method], **parameters}
    for name in ("k_neighbors", "kmeans_estimator"):
        if name in values:
            values[name] = _integer(values[name], f"{method}.{name}")
    for name in ("cluster_balance_threshold", "density_exponent"):
        if name not in values:
            continue
        value = values[name]
        if isinstance(value, str) and value == "auto":
            continue
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0
                or (name == "cluster_balance_threshold" and value > 1)):
            interval = "[0, 1]" if name == "cluster_balance_threshold" else "[0, infinity)"
            raise ValueError(f"{method}.{name} must be 'auto' or a finite number in {interval}")
        values[name] = float(value)
    return values


def validate_sampler_parameters(parameters=None):
    """Return independent JSON-safe defaults/options for exactly the three controls.

    Supplied top-level mappings must contain all three method keys. Within each
    method, omitted known options take their prespecified defaults. Estimator
    objects and competing random_state/sampling_strategy options are unsupported.
    The fixed inclusive KMeansSMOTE threshold 0.0 permits rare-positive clusters;
    it is not a per-dataset fallback or an extra searched configuration.
    """
    if parameters is None:
        parameters = SAMPLER_DEFAULTS
    if not isinstance(parameters, Mapping) or set(parameters) != set(SAMPLER_DEFAULTS):
        raise ValueError("SAMPLER_PARAMETERS must have exactly smote, kmeans_smote and rwo keys")
    return {method: _method_parameters(method, parameters[method]) for method in SAMPLER_DEFAULTS}


def _counts(y):
    labels, counts = np.unique(y, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def _targets(y, target_counts):
    original = _counts(y)
    if not isinstance(target_counts, Mapping):
        raise ValueError("target_counts must map class labels to final integer counts")
    supplied = {}
    for label, count in target_counts.items():
        label = _integer(label, "target_counts class label", minimum=np.iinfo(np.int64).min)
        if label not in original:
            raise ValueError("target_counts cannot introduce an absent real class")
        supplied[label] = _integer(count, "target_counts final count", minimum=0)
    targets = {label: supplied.get(label, count) for label, count in original.items()}
    if any(targets[label] < count for label, count in original.items()):
        raise ValueError("target_counts cannot remove real observations")
    return original, targets


def _categorical_groups(X, categorical_groups):
    groups, occupied = [], set()
    for group in categorical_groups:
        if isinstance(group, Integral):
            raise ValueError("categorical_groups requires complete one-hot blocks, not flat indices")
        columns = tuple(_integer(column, "categorical group index", minimum=0) for column in group)
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("Categorical blocks must be nonempty and contain distinct indices")
        if any(column >= X.shape[1] or column in occupied for column in columns):
            raise ValueError("Categorical blocks must be disjoint and within the feature matrix")
        block = X[:, columns]
        if not np.isin(block, (0, 1)).all() or not np.all(block.sum(axis=1) == 1):
            raise ValueError("Real training categorical blocks must be valid full one-hot encodings")
        occupied.update(columns)
        groups.append(columns)
    numeric = tuple(column for column in range(X.shape[1]) if column not in occupied)
    return tuple(groups), numeric


def _project_categories(synthetic, groups):
    for group in groups:
        categories = np.argmax(synthetic[:, group], axis=1)
        synthetic[:, group] = 0
        synthetic[np.arange(len(synthetic)), np.asarray(group)[categories]] = 1


def _rwo(X, y, original, targets, groups, numeric, seed):
    rng = np.random.default_rng(seed)
    count = sum(targets[label] - original[label] for label in original)
    synthetic = np.empty((count, X.shape[1]), dtype=X.dtype)
    labels = np.empty(count, dtype=y.dtype)
    offset = 0
    for label, real_count in original.items():
        needed = targets[label] - real_count
        if not needed:
            continue
        real = X[y == label]
        block = synthetic[offset:offset + needed]
        block[:] = real[rng.integers(real_count, size=needed)]
        if numeric:
            coordinates = real[:, numeric].astype(np.float64, copy=False)
            scale = np.std(coordinates, axis=0, ddof=0) / np.sqrt(real_count)
            # Constant columns stay constant even if floating-point reduction
            # would otherwise introduce a tiny nonzero standard deviation.
            scale[np.all(coordinates == coordinates[0], axis=0)] = 0
            block[:, numeric] += rng.normal(size=(needed, len(numeric))) * scale
        for group in groups:
            frequencies = real[:, group].sum(axis=0, dtype=np.float64) / real_count
            categories = rng.choice(len(group), size=needed, p=frequencies)
            block[:, group] = 0
            block[np.arange(needed), np.asarray(group)[categories]] = 1
        labels[offset:offset + needed] = label
        offset += needed
    return synthetic, labels


def resample_control(method, X, y, target_counts, seed, *, categorical_groups=(),
                     n_jobs=4, parameters=None):
    """Return (X_resampled, y_resampled, compact_audit), with unit row weights.

    Inputs are floating-point classifier-preprocessed REAL training observations
    only; target_counts contains final class counts, not added counts. Original
    feature/label bytes, dtype and order are retained, and inputs are not mutated.
    Zero additional demand returns the original arrays without importing/fitting
    imbalanced-learn. Infeasible neighbors/clusters raise without parameter
    weakening, fallback, hidden duplication or synthetic padding. The caller's
    existing failure logger should capture such failures.
    """
    values = _method_parameters(method, parameters)
    seed = _integer(seed, "seed", minimum=0)
    if seed >= 2 ** 32:
        raise ValueError("seed must be less than 2**32 for imbalanced-learn")
    if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(n_jobs, Integral) or n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    X, y = np.asarray(X), np.asarray(y)
    if (X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not len(y)
            or not X.shape[1] or not np.issubdtype(X.dtype, np.floating)
            or not np.isfinite(X).all() or not np.issubdtype(y.dtype, np.integer)):
        raise ValueError("Controls require aligned nonempty finite floating-point features and integer class labels")
    original, targets = _targets(y, target_counts)
    groups, numeric = _categorical_groups(X, categorical_groups)
    requested = sum(targets[label] - original[label] for label in original)
    implementations = {"smote": "imblearn.over_sampling.SMOTE",
                       "kmeans_smote": "imblearn.over_sampling.KMeansSMOTE",
                       "rwo": "independent_real_class_RWO_std_ddof0"}
    adapter = ("real_class_nominal_frequency_roulette" if method == "rwo" else
               "numeric_plus_full_onehot_then_categorical_argmax_projection") if groups else "numeric_native"
    audit = {"method": method, "implementation": implementations[method],
             "accepted_count": 0, "requested_count": requested, "underspend": requested,
             "real_count": len(y), "real_counts": original, "actual_counts": dict(original),
             "stop_reason": "target_counts_reached", "synthetic_weights": 1,
             "parameters": values, "categorical_adapter": adapter,
             "raw_generated_count": 0, "trimmed_count": 0}
    if not requested:
        return X, y, audit
    if method == "rwo":
        synthetic, labels = _rwo(X, y, original, targets, groups, numeric, seed)
    else:
        from imblearn.over_sampling import KMeansSMOTE, SMOTE

        strategy = {label: count for label, count in targets.items() if count > original[label]}
        options = {**values, "sampling_strategy": strategy, "random_state": seed}
        if method == "kmeans_smote":
            sampler = KMeansSMOTE(n_jobs=int(n_jobs), **options)
        else:
            sampler = SMOTE(**options)
        # Keep the caller's arrays untouched even across dependency versions.
        sampled_X, sampled_y = sampler.fit_resample(X.copy(), y.copy())
        if (sampled_X.ndim != 2 or sampled_X.shape[1] != X.shape[1]
                or sampled_y.ndim != 1 or len(sampled_X) != len(sampled_y)
                or len(sampled_y) < len(y)
                or not np.array_equal(sampled_X[:len(X)], X)
                or not np.array_equal(sampled_y[:len(y)], y)):
            raise ValueError("imbalanced-learn changed real rows/labels or returned misaligned output")
        synthetic, labels = sampled_X[len(X):], sampled_y[len(y):]
    raw_count = len(labels)
    if not np.isfinite(synthetic).all():
        raise ValueError(f"{method} generated nonfinite features")
    if set(_counts(labels)) - set(original):
        raise ValueError(f"{method} generated labels absent from the real training role")
    keep = np.ones(raw_count, dtype=bool)
    for label, real_count in original.items():
        indices = np.flatnonzero(labels == label)
        allowance = targets[label] - real_count
        if len(indices) > allowance:
            if method != "kmeans_smote":
                raise ValueError(f"{method} exceeded the requested class target")
            # Stable prefix selection trims only generated ceil excess; original
            # rows never enter this selection. No attempt fills any shortfall.
            keep[indices[allowance:]] = False
    if not keep.all():
        synthetic, labels = synthetic[keep], labels[keep]
    # Own the synthetic buffer before projection; original rows are never decoded.
    synthetic = np.array(synthetic, dtype=X.dtype, copy=True)
    if method != "rwo" and groups:
        _project_categories(synthetic, groups)
    if not np.isfinite(synthetic).all():
        raise ValueError(f"{method} generated features outside the finite input dtype domain")
    X_res = np.concatenate((X, synthetic), axis=0)
    y_res = np.concatenate((y, labels.astype(y.dtype, copy=False)))
    actual = _counts(y_res)
    accepted = len(labels)
    audit.update(accepted_count=accepted, underspend=requested - accepted,
                 actual_counts=actual, raw_generated_count=raw_count,
                 trimmed_count=raw_count - accepted,
                 stop_reason="target_counts_reached" if accepted == requested else "sampler_underspend")
    return X_res, y_res, audit
