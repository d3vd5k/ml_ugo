"""Real-only U2 quota planning; allocation does not change UGO geometry."""

from dataclasses import dataclass
from math import ceil, floor
from numbers import Integral

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True)
class AllocationConfig:
    allocation: str = "stability"
    group_alpha: float = .5
    epsilon: float = .001
    seed_cap: int = 4
    group_cap_multiplier: float = 2.

    def __post_init__(self):
        if self.allocation not in ("stability", "size", "uniform", "difficulty"):
            raise ValueError("allocation must be stability, size, uniform or difficulty")
        if not np.isfinite(self.group_alpha) or not 0 <= self.group_alpha <= 1:
            raise ValueError("group_alpha must be finite and in [0, 1]")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if isinstance(self.seed_cap, bool) or not isinstance(self.seed_cap, Integral) or self.seed_cap < 1:
            raise ValueError("seed_cap must be a positive integer")
        if not np.isfinite(self.group_cap_multiplier) or self.group_cap_multiplier <= 0:
            raise ValueError("group_cap_multiplier must be finite and positive")


def integer_quotas(weights, capacities, budget):
    """Capped proportional largest remainders, redistributing saturated shares.

    Zero weights never receive quota, including when positive-weight capacities
    are exhausted. Equal remainders are resolved in original input order.
    """
    weights = np.asarray(weights, dtype=float)
    capacities = np.asarray(capacities)
    if weights.ndim != 1 or capacities.shape != weights.shape:
        raise ValueError("weights and capacities must be matching vectors")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and nonnegative")
    if capacities.size and (not np.issubdtype(capacities.dtype, np.integer) or np.any(capacities < 0)):
        raise ValueError("capacities must be nonnegative integers")
    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    capacities = capacities.astype(np.int64, copy=False)
    quotas = np.zeros(len(weights), dtype=np.int64)
    eligible = np.flatnonzero((weights > 0) & (capacities > 0))
    remaining = min(int(budget), sum(int(capacities[i]) for i in eligible))
    while remaining and len(eligible):
        normalized = weights[eligible] / weights[eligible].max()
        shares = remaining * (normalized / normalized.sum())
        room = capacities[eligible] - quotas[eligible]
        saturated = shares >= room
        if np.any(saturated):
            filled = eligible[saturated]
            quotas[filled] = capacities[filled]
            remaining -= sum(int(n) for n in room[saturated])
            eligible = eligible[~saturated]
            continue
        whole = np.floor(shares).astype(np.int64)
        quotas[eligible] += whole
        remaining -= sum(int(n) for n in whole)
        order = np.argsort(-(shares - whole), kind="stable")
        quotas[eligible[order[:remaining]]] += 1
        remaining = 0
    return quotas


class GroupAllocator:
    """Plan repeated source seeds and account only accepted descendants.

    ``prepared_groups`` is the real-only output of ``prepare_groups``. Final
    class targets, seed caps and group caps remain fixed across every commit.
    Planning itself never reserves or consumes any capacity.
    """

    def __init__(self, prepared_groups, y, target_counts=None, config=None):
        self.config = config or AllocationConfig()
        self.y = np.asarray(y)
        if self.y.ndim != 1 or not len(self.y):
            raise ValueError("y must be a nonempty vector of real source labels")
        classes, counts = np.unique(self.y, return_counts=True)
        self.real_counts = {c.item(): int(n) for c, n in zip(classes, counts)}
        supplied = {} if target_counts is None else dict(target_counts)
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in supplied.values()):
            raise ValueError("target_counts must contain integer final counts")
        if set(supplied) - set(self.real_counts):
            raise ValueError("target_counts cannot introduce classes")
        self.target_counts = {c: int(counts.max()) if target_counts is None else int(supplied.get(c, n))
                              for c, n in self.real_counts.items()}
        if any(self.target_counts[c] < n for c, n in self.real_counts.items()):
            raise ValueError("target_counts cannot remove real observations")
        self.groups = prepared_groups["groups"]
        self.used_seed = np.zeros(len(self.y), dtype=np.int64)
        self.used_group = np.zeros(len(self.groups), dtype=np.int64)
        self.used_class = {c: 0 for c in self.real_counts}
        self.records = []
        self._seed_group = np.full(len(self.y), -1, dtype=np.int64)
        self._group_caps = np.zeros(len(self.groups), dtype=np.int64)
        self._eligible = np.zeros(len(self.groups), dtype=bool)
        seen_ids = set()
        for pos, group in enumerate(self.groups):
            indices = np.asarray(group["indices"])
            if indices.ndim != 1 or (indices.size and not np.issubdtype(indices.dtype, np.integer)):
                raise ValueError("group indices must be an integer vector")
            if np.any(indices < 0) or np.any(indices >= len(self.y)) or len(np.unique(indices)) != len(indices):
                raise ValueError("groups must contain distinct real source indices")
            if group["id"] in seen_ids or np.any(self._seed_group[indices] != -1):
                raise ValueError("group IDs and source memberships must be unique")
            seen_ids.add(group["id"])
            if group["class"] not in self.real_counts or np.any(self.y[indices] != group["class"]):
                raise ValueError("group class must match its real source labels")
            n = group["real_support"]
            if not isinstance(n, Integral) or n != len(indices):
                raise ValueError("real_support must equal the real source membership count")
            distinct = group["distinct_coordinates"]
            if not isinstance(distinct, Integral) or not 0 <= distinct <= n:
                raise ValueError("distinct_coordinates must be an integer bounded by real_support")
            stability = group["stability"]
            if not np.isfinite(stability) or not 0 <= stability <= 1:
                raise ValueError("stability must be finite and in [0, 1]")
            self._seed_group[indices] = pos
            self._group_caps[pos] = floor(self.config.group_cap_multiplier * n)
            self._eligible[pos] = n >= 2 and distinct >= 2 and stability > 0

    def allocate(self, scores, target_class, budget, selection_fraction):
        scores = np.asarray(scores, dtype=float)
        if scores.shape != self.y.shape or not np.all(np.isfinite(scores)):
            raise ValueError("scores must be one finite score per real source row")
        if target_class not in self.real_counts:
            raise ValueError("target_class must be an observed real class")
        if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 0:
            raise ValueError("budget must be a nonnegative integer")
        if not np.isfinite(selection_fraction) or not 0 < selection_fraction <= 1:
            raise ValueError("selection_fraction must be finite and in (0, 1]")
        class_indices = np.flatnonzero(self.y == target_class)
        percentiles = np.zeros(len(self.y), dtype=float)
        percentiles[class_indices] = (rankdata(scores[class_indices], method="average") - .5) / len(class_indices)
        remaining = self.target_counts[target_class] - self.real_counts[target_class] - self.used_class[target_class]
        weights = np.zeros(len(self.groups), dtype=float)
        capacities = np.zeros(len(self.groups), dtype=np.int64)
        selected = {}
        trace = []
        for pos, group in enumerate(self.groups):
            if group["class"] != target_class:
                continue
            indices = np.asarray(group["indices"], dtype=np.int64)
            n = group["real_support"]
            h = float(percentiles[indices].mean()) if n else 0.
            reasons = []
            if n < 2:
                reasons.append("insufficient_real_support")
            if group["distinct_coordinates"] < 2:
                reasons.append("insufficient_distinct_coordinates")
            if group["stability"] <= 0:
                reasons.append("zero_stability")
            if not reasons:
                if self.config.allocation == "stability":
                    weights[pos] = group["stability"] * (h + self.config.epsilon) / (n + self.config.epsilon) ** self.config.group_alpha
                elif self.config.allocation == "size":
                    weights[pos] = n
                elif self.config.allocation == "uniform":
                    weights[pos] = 1.
                else:
                    weights[pos] = h + self.config.epsilon
                available = indices[self.used_seed[indices] < self.config.seed_cap]
                order = np.argsort(-scores[available], kind="stable")
                chosen = available[order[:ceil(selection_fraction * n)]]
                selected[pos] = chosen
                seed_room = int((self.config.seed_cap - self.used_seed[chosen]).sum())
                group_room = int(self._group_caps[pos] - self.used_group[pos])
                capacities[pos] = min(seed_room, group_room, remaining)
                if seed_room == 0:
                    reasons.append("seed_cap_exhausted")
                if group_room == 0:
                    reasons.append("group_cap_exhausted")
                if remaining == 0:
                    reasons.append("class_target_reached")
            trace.append({"group_id": group["id"], "class": group["class"], "n_real": n,
                          "stability": float(group["stability"]), "h": h, "weight": float(weights[pos]),
                          "capacity": int(capacities[pos]), "quota": 0,
                          "used_group": int(self.used_group[pos]),
                          "used_seed": {int(i): int(self.used_seed[i]) for i in indices},
                          "exclusions": reasons})
        quotas = integer_quotas(weights, capacities, min(int(budget), remaining))
        plans = []
        for pos, group in enumerate(self.groups):
            quota = int(quotas[pos])
            if quota:
                seeds = selected[pos]
                seed_counts = integer_quotas(np.ones(len(seeds)), self.config.seed_cap - self.used_seed[seeds], quota)
                plans.append(np.repeat(seeds, seed_counts))
        for entry, pos in zip(trace, (i for i, g in enumerate(self.groups) if g["class"] == target_class)):
            entry["quota"] = int(quotas[pos])
        seeds = np.concatenate(plans) if plans else np.empty(0, dtype=np.int64)
        record = {"target_class": target_class, "allocation": self.config.allocation,
                  "requested": int(budget), "allocated": len(seeds), "unspent": int(budget) - len(seeds),
                  "class_remaining": remaining, "used_class": self.used_class[target_class],
                  "accepted": 0, "group_trace": trace}
        self.records.append(record)
        if not np.any((capacities > 0) & (weights > 0)):
            return None
        return {"target_class": target_class, "seed_indices": seeds, "count": len(seeds), "group_trace": trace}

    def commit(self, seed_indices):
        """Atomically consume caps for accepted real-source seeds only."""
        seeds = np.asarray(seed_indices)
        if seeds.ndim != 1 or (seeds.size and not np.issubdtype(seeds.dtype, np.integer)):
            raise ValueError("accepted seeds must be an integer vector")
        if not len(seeds):
            return
        if np.any(seeds < 0) or np.any(seeds >= len(self.y)):
            raise ValueError("accepted seeds must refer to real source rows")
        groups = self._seed_group[seeds]
        if np.any(groups < 0) or not np.all(self._eligible[groups]):
            raise ValueError("accepted seeds must belong to supported stable groups")
        classes = np.unique(self.y[seeds])
        if len(classes) != 1:
            raise ValueError("one commit must contain a single target class")
        target_class = classes[0].item()
        seed_delta = np.bincount(seeds, minlength=len(self.y))
        group_delta = np.bincount(groups, minlength=len(self.groups))
        if np.any(self.used_seed + seed_delta > self.config.seed_cap):
            raise ValueError("accepted seeds exceed the per-seed cap")
        if np.any(self.used_group + group_delta > self._group_caps):
            raise ValueError("accepted seeds exceed the group cap")
        if self.real_counts[target_class] + self.used_class[target_class] + len(seeds) > self.target_counts[target_class]:
            raise ValueError("accepted seeds exceed the final class target")
        self.used_seed += seed_delta
        self.used_group += group_delta
        self.used_class[target_class] += len(seeds)
        if self.records and self.records[-1]["target_class"] == target_class:
            self.records[-1]["accepted"] += len(seeds)
