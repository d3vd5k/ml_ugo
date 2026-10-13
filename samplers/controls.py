"""Training-only controls with explicit budgets and categorical adaptations.

SMOTE on mixed features is true SMOTENC (nominal categories, numeric distance
plus its variance-scaled one-hot encoding). BorderlineSMOTE, ADASYN and
KMeansSMOTE run their actual imbalanced-learn algorithms on numeric + one-hot
coordinates, then decode each categorical block by argmax. This is a declared
mixed-type adapter, NOT native SMOTENC variants or numeric category-code
interpolation. ROS copies whole observations. Raw categorical_columns refers to
one column per categorical variable. For pre-expanded dummy blocks, supply
categorical_groups (or parameters["categorical_groups"]); blocks are collapsed
to nominal codes for sampling and re-expanded without fractional labels.
"""
from copy import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder

from samplers.ugo import UGOConfig, UGOPipeline, allocate_group_budget, accept_relocation
from samplers.ugo_original import OriginalUGOConfig, original_ugo
from samplers.protected import protected_ugo
from samplers.constraints import constrained_ugo
from samplers.stopping import stopping_ugo
from samplers.hdbscan_support import prepare_h1_support
from samplers.flow_gmm import FlowGate
from samplers.flow_support import CalibratedFlowGate


@dataclass
class ControlResult:
    X: object
    y: object
    rounds: list
    provenance: list
    stop_reason: str
    support_summary: dict = field(default_factory=dict)


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _targets(y, target_counts):
    classes, counts = np.unique(y, return_counts=True)
    original = {int(c): int(n) for c, n in zip(classes, counts)}
    if target_counts is None:
        return original, {c: int(counts.max()) for c in original}
    supplied = dict(target_counts)
    if any(not isinstance(v, (int, np.integer)) for v in supplied.values()):
        raise ValueError("target_counts must contain integer final counts")
    targets = {int(c): int(supplied.get(c, n)) for c, n in original.items()}
    if set(supplied) - set(original) or any(targets[c] < original[c] for c in original):
        raise ValueError("target_counts cannot introduce classes or remove real observations")
    return original, targets


def _corrected(X, y, seed, categorical_columns, max_iterations, device, model_fn, targets, parameters, seed_score=None,
               *, generation_cap=None, stop=None, batch_limit=None, target_limit=None):
    values = dict(parameters)
    values.update(random_state=seed, categorical_columns=tuple(categorical_columns), max_iterations=max_iterations, device=device)
    config = UGOConfig(**values)
    scoring = {} if seed_score is None else {"seed_score": seed_score}
    options = {} if stop is None else {"stop": stop}
    run_options = {} if generation_cap is None else {"generation_cap": generation_cap}
    def limited(allocate):
        if batch_limit is None:
            return allocate
        def bounded_budget(support, scores, remaining, cfg):
            result = allocate(support, scores, remaining, cfg)
            return batch_limit(result, remaining) if result is not None else None
        return bounded_budget
    if targets is None:
        if batch_limit is not None:
            options["group_budget"] = limited(allocate_group_budget)
        return UGOPipeline(config, **scoring, **options).run(torch.as_tensor(X), torch.as_tensor(y), model_fn, **run_options)
    original, final = _targets(y, targets)
    used = {c: 0 for c in original}
    def budget(support, scores, remaining, cfg):
        active_final = final if target_limit is None else target_limit(sum(used.values()))
        bounded = copy(support)
        bounded.group_indices = {c: indices for c, indices in support.group_indices.items() if used[c] < active_final[c] - original[c]}
        result = allocate_group_budget(bounded, scores, remaining, cfg)
        if result is not None:
            result.count = min(result.count, active_final[result.target_class] - original[result.target_class] - used[result.target_class])
        return result
    def accept(support, candidates, cfg):
        result = accept_relocation(support, candidates, cfg)
        for c in used:
            used[c] += int((result.y == c).sum())
        return result
    result = UGOPipeline(config, group_budget=limited(budget), accept=accept, **scoring, **options).run(torch.as_tensor(X), torch.as_tensor(y), model_fn, **run_options)
    result.support_summary.update(target_counts=final, actual_counts={int(c): int((result.y == c).sum()) for c in original}, protocol_deviations=["matched_per_class_target_caps"])
    if all(used[c] == final[c] - original[c] for c in used):
        result.stop_reason = "target_counts_reached"
    return result


def resample(method, X, y, seed, categorical_columns=(), max_iterations=20,
             device="auto", model_fn=None, target_counts=None, parameters=None,
             categorical_groups=(), seed_score=None, group_partition=None, constraint_geometry=None,
             h1_support=None, flow_support=None):
    """No silently skipped dependencies, neighbor weakening, or quota padding.

    target_counts specifies final class counts, not generated counts. Native
    imbalanced-learn algorithms can underspend (Borderline has no danger seeds)
    or round allocations (ADASYN/KMeans). Matched mode trims only excess rows
    deterministically; it never invents rows to conceal an underspent budget.
    """
    X, y = _array(X), _array(y)
    parameters = dict(parameters or {})
    parameter_groups = parameters.pop("categorical_groups", ())
    if categorical_groups and parameter_groups:
        raise ValueError("Specify categorical_groups once")
    categorical_groups = categorical_groups or parameter_groups
    if seed_score is not None and method not in ("ugo_oof_mma", "ugo_combined", "ugo_combined_f1"):
        raise ValueError("External seed scores require ugo_oof_mma or ugo_combined")
    if group_partition is not None and method not in ("ugo_groups_only", "ugo_combined", "ugo_combined_f1", "ugo_h1", "ugo_f1", "ugo_f2", "ugo_n_reject"):
        raise ValueError("Real group partitions require a protected-group method")
    if h1_support is not None and method != "ugo_h1":
        raise ValueError("H1 support requires the explicit ugo_h1 method")
    if flow_support is not None and method not in ("ugo_f1", "ugo_f2", "ugo_combined_f1"):
        raise ValueError("Flow support requires the explicit ugo_f1 or ugo_f2 method")
    if method in ("ugo_f1", "ugo_combined_f1") and flow_support is None:
        raise ValueError("F1 requires a frozen real-fit-only FlowGMM and raw-coordinate bridge")
    if method == "ugo_f2" and not isinstance(flow_support, CalibratedFlowGate):
        raise ValueError("F2 requires an explicitly calibrated frozen FlowGMM gate")
    if method in ("ugo_f1", "ugo_combined_f1") and isinstance(flow_support, CalibratedFlowGate):
        raise ValueError("Calibrated F2 gates require ugo_f2, including support-filter-off")
    if method == "ugo_combined_f1" and (not isinstance(flow_support, FlowGate)
            or flow_support.core.model.training
            or any(parameter.requires_grad for parameter in flow_support.core.model.parameters())):
        raise ValueError("U5+F1 requires a plain frozen real-fit-only F1 FlowGMM gate")
    if method != "ugo_f2" and any(key in parameters for key in ("f2_q", "support_filter")):
        raise ValueError("F2 support parameters require the explicit ugo_f2 method")
    if constraint_geometry is not None and method != "ugo_constraints_only":
        raise ValueError("Constraint geometry requires the explicit ugo_constraints_only method")
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(np.unique(y)) < 2 or not np.isfinite(X).all():
        raise ValueError("Resampling requires finite features and at least two classes")
    categorical_columns = tuple(int(c) for c in categorical_columns)
    if len(set(categorical_columns)) != len(categorical_columns) or any(c < 0 or c >= X.shape[1] for c in categorical_columns):
        raise ValueError("Invalid categorical column indices")
    if method in ("baseline", "class_weight"):
        if parameters:
            raise ValueError("No-resampling controls do not accept sampler parameters")
        return ControlResult(X, y, [], [], "no_resampling", {"method": method})
    if method in ("ugo_original", "ugo_corrected", "ugo_oof_mma", "ugo_groups_only",
                  "ugo_constraints_only", "ugo_stopping_only", "ugo_combined", "ugo_combined_f1", "ugo_h1", "ugo_f1", "ugo_f2", "ugo_n_reject"):
        if model_fn is None:
            raise ValueError("UGO requires a model_fn training callback")
        if method in ("ugo_oof_mma", "ugo_combined", "ugo_combined_f1") and seed_score is None:
            raise ValueError("OOF MMA requires fold-isolated real-only seed scores")
        if method == "ugo_original":
            values = dict(parameters)
            values.update(random_state=seed, categorical_columns=categorical_columns, max_iterations=max_iterations, device=device)
            # Projection remains opt-in: runner records controlled-track deviation.
            return original_ugo(X, y, model_fn, OriginalUGOConfig(**values), target_counts)
        if method in ("ugo_groups_only", "ugo_combined", "ugo_combined_f1", "ugo_h1", "ugo_f1", "ugo_f2", "ugo_n_reject"):
            _, final = _targets(y, target_counts)
            if method == "ugo_h1" and h1_support is None:
                h1_support = prepare_h1_support(X, y, categorical_columns, parameters,
                    None if group_partition is None else group_partition.get("real_row_ids"))
            return protected_ugo(X, y, seed, categorical_columns, max_iterations, device, model_fn,
                                 final, parameters, group_partition, seed_score=seed_score, h1_support=h1_support,
                                 flow_support=flow_support, reject_neighbors=method == "ugo_n_reject")
        if method == "ugo_constraints_only":
            final = None if target_counts is None else _targets(y, target_counts)[1]
            return constrained_ugo(X, y, seed, categorical_columns, max_iterations, device, model_fn,
                                   final, parameters, categorical_groups, constraint_geometry)
        if method == "ugo_stopping_only":
            return stopping_ugo(X, y, seed, categorical_columns, max_iterations, device, model_fn,
                                target_counts, parameters)
        return _corrected(X, y, seed, categorical_columns, max_iterations, device, model_fn, target_counts, parameters, seed_score)
    if method not in ("ros", "smote", "borderline_smote", "adasyn", "kmeans_smote"):
        raise ValueError(f"Unknown resampling method: {method}")
    if categorical_groups:
        groups = [tuple(int(c) for c in group) for group in categorical_groups]
        flattened = [c for group in groups for c in group]
        if any(not group for group in groups) or len(set(flattened)) != len(flattened) or any(c < 0 or c >= X.shape[1] for c in flattened):
            raise ValueError("categorical_groups must be disjoint nonempty valid one-hot blocks")
        if categorical_columns and set(categorical_columns) != set(flattened):
            raise ValueError("categorical_columns must match the supplied one-hot blocks")
        for group in groups:
            block = X[:, group]
            if not np.isin(block, [0, 1]).all() or not np.all(block.sum(axis=1) == 1):
                raise ValueError("Training categorical_groups must be valid full one-hot blocks")
        numeric = [c for c in range(X.shape[1]) if c not in set(flattened)]
        collapsed = np.column_stack([X[:, numeric]] + [X[:, group].argmax(axis=1) for group in groups])
        sampled = resample(method, collapsed, y, seed,
                           categorical_columns=tuple(range(len(numeric), len(numeric) + len(groups))),
                           max_iterations=max_iterations, device=device, model_fn=model_fn,
                           target_counts=target_counts, parameters=parameters)
        expanded = np.zeros((len(sampled.X), X.shape[1]), dtype=X.dtype)
        expanded[:, numeric] = sampled.X[:, :len(numeric)]
        for offset, group in enumerate(groups, start=len(numeric)):
            codes = sampled.X[:, offset].astype(int)
            expanded[np.arange(len(expanded)), np.asarray(group)[codes]] = 1
        expanded[:len(X)] = X
        sampled.X = expanded
        sampled.support_summary["input_categorical_representation"] = "onehot_blocks_collapsed_to_nominal_then_reexpanded"
        sampled.support_summary["categorical_groups"] = groups
        return sampled
    try:
        from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, BorderlineSMOTE, ADASYN, KMeansSMOTE
    except ImportError as exc:
        raise ImportError(f"Requested {method} requires imbalanced-learn") from exc
    original, targets = _targets(y, target_counts)
    strategy = {c: n for c, n in targets.items() if n > original[c]}
    if not strategy:
        return ControlResult(X, y, [], [], "target_counts_reached", {"target_counts": targets, "actual_counts": original})
    if "sampling_strategy" in parameters or "random_state" in parameters:
        raise ValueError("Use target_counts and seed, not competing sampler budget/seed parameters")
    kwargs = dict(parameters, sampling_strategy=strategy, random_state=seed)
    encoder = None
    numeric = [c for c in range(X.shape[1]) if c not in categorical_columns]
    distance_X = X
    adapter = "numeric_native"
    if method == "ros":
        sampler = RandomOverSampler(**kwargs)
        adapter = "whole_observation_duplication"
    elif method == "smote" and categorical_columns:
        if not numeric:
            raise ValueError("SMOTENC requires numeric and categorical features; all-categorical SMOTEN is a different control")
        sampler = SMOTENC(categorical_features=list(categorical_columns), **kwargs)
        adapter = "native_SMOTENC"
    else:
        if categorical_columns:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="error")
            distance_X = np.column_stack((X[:, numeric], encoder.fit_transform(X[:, categorical_columns])))
            adapter = "numeric_plus_onehot_then_categorical_argmax_projection"
        constructor = {"smote": SMOTE, "borderline_smote": BorderlineSMOTE, "adasyn": ADASYN, "kmeans_smote": KMeansSMOTE}[method]
        sampler = constructor(**kwargs)
    # imbalanced-learn raises impossible class / cluster / neighbor errors; these
    # are intentionally not caught and not weakened to a different algorithm.
    generated_X, generated_y = sampler.fit_resample(distance_X, y)
    raw_counts = {int(c): int((generated_y == c).sum()) for c in original}
    if encoder is not None:
        decoded = np.empty((len(generated_X), X.shape[1]), dtype=np.result_type(X.dtype, float))
        decoded[:, numeric] = generated_X[:, :len(numeric)]
        position = len(numeric)
        for column, categories in zip(categorical_columns, encoder.categories_):
            width = len(categories)
            decoded[:, column] = categories[generated_X[:, position:position + width].argmax(axis=1)]
            position += width
        generated_X = decoded
    # Preserve real rows bit-for-bit, even if a library encoding was applied.
    generated_X[:len(X)] = X
    keep = list(range(len(X)))
    for c in original:
        indices = np.where(generated_y[len(y):] == c)[0] + len(y)
        keep.extend(indices[:targets[c] - original[c]].tolist() if target_counts is not None else indices.tolist())
    keep = np.asarray(keep, dtype=int)
    generated_X, generated_y = generated_X[keep], generated_y[keep]
    actual = {int(c): int((generated_y == c).sum()) for c in original}
    metadata = {"method": method, "implementation": type(sampler).__name__, "categorical_adapter": adapter, "target_counts": targets, "actual_counts": actual, "raw_algorithm_counts": raw_counts, "underspend": {c: max(0, targets[c] - actual[c]) for c in original}, "parameters": sampler.get_params(deep=False), "matched_count_trimming": target_counts is not None and raw_counts != actual}
    provenance = []
    for row, source in enumerate(keep[len(y):], start=len(y)):
        entry = {"row_index": row, "class": int(generated_y[row]), "method": method, "categorical_adapter": adapter}
        if method == "ros":
            entry["seed_index"] = int(sampler.sample_indices_[source])
        else:
            entry["parent_indices_status"] = "not_exposed_by_imbalanced_learn"
        provenance.append(entry)
    reason = "target_counts_reached" if all(actual[c] >= targets[c] for c in actual) else "algorithm_underspend"
    return ControlResult(generated_X, generated_y, [{"round": 1, "generated": len(generated_y) - len(y), "actual_counts": actual}], provenance, reason, metadata)
