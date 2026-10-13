"""U4 stopping-only budgets over the unchanged corrected UGO stages."""

from math import floor
from numbers import Real

import numpy as np

from samplers.allocation import integer_quotas
from samplers.ugo import UGOConfig


def stopping_configuration(parameters):
    """Return policy, checkpoint fractions, selected fraction and UGO parameters."""
    values = dict(parameters or {})
    policy = values.pop("policy", "checkpoint")
    if policy not in ("checkpoint", "fixed_grid"):
        raise ValueError("policy must be checkpoint or fixed_grid")
    supplied = values.pop("budget_fractions", (0., .25, .5, 1.))
    try:
        fractions = tuple(supplied)
    except TypeError as error:
        raise ValueError("budget_fractions must be an ascending sequence including 0 and 1") from error
    if (len(fractions) < 2 or any(isinstance(f, (bool, np.bool_)) or not isinstance(f, Real)
                                 or not np.isfinite(f) for f in fractions)
            or fractions[0] != 0 or fractions[-1] != 1
            or any(a >= b for a, b in zip(fractions, fractions[1:]))):
        raise ValueError("budget_fractions must be finite, strictly ascending and include 0 and 1")
    fractions = tuple(float(f) for f in fractions)
    fraction = values.pop("budget_fraction", 1.)
    if (isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, Real)
            or not np.isfinite(fraction) or fraction not in fractions):
        raise ValueError("budget_fraction must be a finite member of budget_fractions, not bool")
    return policy, fractions, float(fraction), values


def stopping_budget(y, targets, parameters):
    """Plan a real-label budget, retaining native phi semantics when targets is None."""
    from samplers.controls import _array, _targets

    policy, fractions, fraction, values = stopping_configuration(parameters)
    config = UGOConfig(**values)
    labels = _array(y)
    original, final = _targets(labels, targets)
    if targets is None:
        maximum = max(0, int(config.phi * max(original.values()) * len(original)) - len(labels))
    else:
        maximum = sum(final[c] - original[c] for c in original)
    selected = floor(maximum * fraction)
    checkpoints = tuple(sorted({0, selected, *(floor(maximum * f) for f in fractions if f <= fraction)}))
    selected_targets = None
    if targets is not None:
        classes = sorted(original)
        deficits = np.array([final[c] - original[c] for c in classes], dtype=np.int64)
        if policy == "fixed_grid":
            quotas = integer_quotas(deficits, deficits, selected)
        else:
            # Incremental remaining-deficit apportionment is monotone even on
            # grids where independent Hamilton budgets lose a class seat.
            quotas = np.zeros(len(classes), dtype=np.int64)
            previous = 0
            for count in checkpoints:
                remaining = deficits - quotas
                quotas += integer_quotas(remaining, remaining, count - previous)
                previous = count
        selected_targets = {c: original[c] + int(quota) for c, quota in zip(classes, quotas)}
    return maximum, selected, checkpoints, selected_targets


def stopping_ugo(X, y, seed, categorical_columns, max_iterations, device, model_fn, targets, parameters):
    """Replace Welch stopping only; checkpoints clip proposed batches, not output."""
    from samplers.controls import _corrected

    policy, fractions, fraction, values = stopping_configuration(parameters)
    maximum, selected, checkpoints, selected_targets = stopping_budget(y, targets, parameters)
    native_maximum = stopping_budget(y, None, parameters)[0]
    effective_cap = min(selected, native_maximum)
    checkpoint_targets = {}
    if policy == "checkpoint" and targets is not None:
        for checkpoint_fraction in fractions:
            if checkpoint_fraction <= fraction:
                checkpoint_parameters = dict(parameters, budget_fraction=checkpoint_fraction)
                _, count, _, caps = stopping_budget(y, targets, checkpoint_parameters)
                checkpoint_targets[count] = caps

    def stop(p_value, generated, cap, config):
        return "generation_cap" if generated >= cap else None

    def next_checkpoint(generated):
        return next(count for count in checkpoints if count > generated)

    def batch_limit(budget, remaining):
        # remaining is the accepted-row balance under both selected and native caps.
        generated = effective_cap - remaining
        budget.count = min(budget.count, next_checkpoint(generated) - generated)
        return budget

    def target_limit(generated):
        return checkpoint_targets[next_checkpoint(generated)]

    result = _corrected(X, y, seed, categorical_columns, max_iterations, device, model_fn,
                        selected_targets, values, generation_cap=selected, stop=stop,
                        batch_limit=batch_limit if policy == "checkpoint" else None,
                        target_limit=target_limit if checkpoint_targets else None)
    result.support_summary["stopping"] = {
        "policy": policy, "budget_fraction": fraction, "budget_fractions": fractions,
        "maximum_count": maximum, "requested_count": selected,
        "checkpoint_counts": checkpoints, "welch_bypassed": True,
        "per_class_caps": selected_targets, "checkpoint_class_caps": checkpoint_targets,
        "class_cap_allocation": "cumulative_remaining_deficit" if policy == "checkpoint"
                                else "independent_original_deficit",
    }
    return result
