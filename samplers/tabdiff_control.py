"""Fixed-round U2 schedules, original-column masks and common Shoppers decoding."""
from copy import deepcopy
from dataclasses import asdict, fields
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from samplers.allocation import AllocationConfig, GroupAllocator
from samplers.groups import GroupConfig
from samplers.ugo import (GroupBudget, RealSupport, UGOConfig, nearest_indices,
                          propose_interpolation, score_seeds)

COUNTS = ("Administrative", "Informational", "ProductRelated")
NUMERIC = tuple(name for count in COUNTS for name in (count, count + "_Duration")) + (
    "BounceRates", "ExitRates", "PageValues", "SpecialDay")
BOUNDED = ("BounceRates", "ExitRates", "SpecialDay")


class ShoppersSchema:
    def __init__(self, real_frame):
        self.columns = tuple(real_frame.columns)
        if not set(NUMERIC) <= set(self.columns):
            raise ValueError("Expected original Shoppers feature columns")
        self.categorical = tuple(c for c in self.columns if c not in NUMERIC)
        self.categories = {c: list(pd.unique(real_frame[c].dropna())) for c in self.categorical}
        self.real_keys = set(self.keys(real_frame))

    def keys(self, frame):
        canonical = pd.DataFrame(index=pd.RangeIndex(len(frame)))
        for column in self.columns:
            values = frame[column].reset_index(drop=True)
            canonical[column] = (pd.to_numeric(values, errors="coerce").astype(float)
                                 if column in NUMERIC else values.map(str))
        return list(canonical.itertuples(index=False, name=None))

    def audit(self):
        return {"columns": self.columns, "numeric": NUMERIC, "categorical": self.categorical,
                "observed_training_categories": self.categories,
                "count_decoder": "user_approved_np_rint_ties_even_all_arms",
                "bounds": "all_numeric>=0; BounceRates/ExitRates/SpecialDay<=1",
                "cross_fields": "zero_rounded_count_implies_exact_zero_duration",
                "not_enforced": ["observed_minmax", "BounceRates<=ExitRates"],
                "duplicates": "diagnostics_only_not_a_new_rejection_gate"}

    def decode(self, frame):
        decoded = frame.reset_index(drop=True).astype(object).copy()
        failures = {}
        numeric = frame.loc[:, NUMERIC].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        failures["nonfinite_numeric"] = ~np.isfinite(numeric).all(axis=1)
        failures["negative_raw_numeric"] = (numeric < 0).any(axis=1)
        failures["raw_rate_above_one"] = (frame.loc[:, BOUNDED].apply(
            pd.to_numeric, errors="coerce").to_numpy(dtype=float) > 1).any(axis=1)
        for column in self.categorical:
            failures["unknown_category:" + column] = ~frame[column].isin(self.categories[column]).to_numpy()
            known = decoded[column].where(~failures["unknown_category:" + column])
            decoded[column] = pd.Categorical(known, categories=self.categories[column])
        changes = np.zeros((len(frame), len(COUNTS)), dtype=bool)
        for i, column in enumerate(NUMERIC):
            values = numeric[:, i]
            if column in COUNTS:
                rounded = np.rint(values)
                changes[:, COUNTS.index(column)] = np.isfinite(values) & (rounded != values)
                values = rounded
            decoded[column] = values
        for column in COUNTS:
            failures["zero_count_nonzero_duration:" + column] = (
                (decoded[column].to_numpy(dtype=float) == 0) &
                (decoded[column + "_Duration"].to_numpy(dtype=float) != 0))
        valid = ~np.logical_or.reduce(list(failures.values()))
        reasons = [";".join(name for name, mask in failures.items() if mask[i]) or "valid"
                   for i in range(len(frame))]
        return decoded.loc[:, self.columns], valid, reasons, changes, failures

    def duplicates(self, frame, valid):
        real_match = np.zeros(len(frame), dtype=bool)
        pool_match = np.zeros(len(frame), dtype=bool)
        seen = set()
        for i, key in zip(np.flatnonzero(valid), self.keys(frame.iloc[np.flatnonzero(valid)])):
            real_match[i] = key in self.real_keys
            pool_match[i] = key in seen
            seen.add(key)
        return real_match, pool_match


def feature_masks(frame, seed):
    columns = tuple(frame.columns)
    blocks = [tuple((c, c + "_Duration")) for c in COUNTS]
    blocks += [("BounceRates", "ExitRates"), ("PageValues",), ("SpecialDay",)]
    grouped = set(c for block in blocks for c in block)
    blocks += [(c,) for c in columns if c not in grouped]
    if set(c for block in blocks for c in block) != set(columns):
        raise ValueError("Feature blocks must cover original features and exclude the label")
    subsets = [tuple(i for i in range(len(blocks)) if bits & (1 << i))
               for bits in range(1, 1 << len(blocks))]
    sizes = np.array([sum(len(blocks[i]) for i in subset) for subset in subsets])
    rng = np.random.default_rng(seed + 16201)
    distance = np.abs(sizes - .25 * len(columns))
    selected = subsets[int(rng.choice(np.flatnonzero(distance == distance.min())))]
    eligible = np.array([set(selected) <= set(subset) for subset in subsets])
    distance = np.where(eligible, np.abs(sizes - .50 * len(columns)), np.inf)
    nearest = np.flatnonzero(distance == distance.min())
    target_size = int(rng.choice(np.unique(sizes[nearest])))
    larger = subsets[int(rng.choice(nearest[sizes[nearest] == target_size]))]
    result = {arm: [c for c in columns if any(c in blocks[i] for i in subset)]
              for arm, subset in (("B", selected), ("C", larger))}
    return result, {"training_columns": columns, "blocks": blocks, "seed": seed + 16201,
                    "C_size_choice": "uniform_nearest_feasible_integer_size_then_uniform_nested_block_subset",
                    "actual_fractions": {arm: len(mask) / len(columns) for arm, mask in result.items()}}


def fixed_u2_schedule(X, y, model, prepared, parameters, seed, categorical_columns,
                      target_counts, proposal_cap):
    specialized = {f.name for cls in (GroupConfig, AllocationConfig) for f in fields(cls)}
    base = {key: value for key, value in parameters.items()
            if key not in specialized and key not in ("capture_proposals", "f1_tau")}
    cfg = UGOConfig(**{**base, "random_state": seed, "max_iterations": 1,
                       "device": "cpu", "categorical_columns": tuple(categorical_columns)})
    allocation = AllocationConfig(**{f.name: parameters[f.name] for f in fields(AllocationConfig)
                                     if f.name in parameters})
    features, labels = torch.as_tensor(X), torch.as_tensor(y, dtype=torch.long)
    scores = score_seeds(model, features, labels)
    classes, counts = np.unique(y, return_counts=True)
    minority = int(classes[counts.argmin()])
    supported = [g for g in prepared["groups"] if g["class"] == minority and
                 g["real_support"] >= 2 and g["distinct_coordinates"] >= 2 and g["stability"] > 0]
    indices = np.sort(np.concatenate([g["indices"] for g in supported])) if supported else np.empty(0, dtype=int)
    eligible = torch.zeros(len(y), dtype=torch.bool)
    eligible[torch.as_tensor(indices)] = True
    native_cap = max(0, int(cfg.phi * int(counts.max()) * len(classes)) - len(y))
    budget = min(int(proposal_cap), native_cap, max(1, round(len(y) * cfg.lambda_step)),
                 target_counts[minority] - int(counts.min()))
    allocator = GroupAllocator(prepared, y, target_counts, allocation)
    planned = allocator.allocate(scores.numpy(), minority, budget, cfg.lambda_select)
    if planned is None:
        raise ValueError("No supported canonical U2 minority group for fixed-round proposals")
    neighbors = nearest_indices(features[indices], features[indices], min(cfg.k_pre, len(indices)-1),
                                cfg.distance_batch_size, torch.arange(len(indices)), exclude_zero=True)
    support = RealSupport(features, labels, torch.unique(labels), eligible,
                          {minority: torch.as_tensor(indices)[neighbors]},
                          {minority: torch.as_tensor(indices)}, [])
    seeds = torch.as_tensor(planned["seed_indices"], dtype=torch.long)
    rng = np.random.default_rng(seed)
    replay = np.random.default_rng()
    replay.bit_generator.state = deepcopy(rng.bit_generator.state)
    proposal_start = perf_counter()
    candidates = propose_interpolation(support, GroupBudget(minority, seeds, len(seeds)), rng, cfg,
                                       seed_indices=seeds)
    original_proposal_seconds = perf_counter() - proposal_start
    if not torch.equal(candidates.seed_indices, seeds):
        raise RuntimeError("Supported U2 anchors unexpectedly lost a valid partner")
    replay.random(len(seeds))
    coefficients = replay.random((len(seeds), 1)).astype(np.float32)
    if replay.bit_generator.state != rng.bit_generator.state:
        raise RuntimeError("Original U2 partner/interpolation stream could not be replayed")
    return candidates, {"configuration": asdict(cfg), "allocation": asdict(allocation),
                        "proposal_cap": int(proposal_cap), "actual_round_budget": budget,
                        "minority_label": minority, "scores": scores.numpy(),
                        "allocation_record": allocator.records[-1],
                        "interpolation_coefficients": coefficients,
                        "rng_stream": "original_seed_PCG64_partner_draws_then_float32_lambda",
                        "rng_final_state": rng.bit_generator.state,
                        "original_proposal_seconds": original_proposal_seconds}


def original_raw_proposals(real_frame, candidates, coefficients):
    anchors, partners = candidates.seed_indices.numpy(), candidates.partner_indices.numpy()
    raw = real_frame.iloc[partners].reset_index(drop=True).copy()
    # Affine input geometry in physical units, avoiding false violations from
    # reversing already-rounded float32 classifier scalings of real zeros.
    a = real_frame.iloc[anchors].loc[:, NUMERIC].to_numpy(dtype=float)
    b = real_frame.iloc[partners].loc[:, NUMERIC].to_numpy(dtype=float)
    numeric = a + coefficients * (b - a)
    for i, name in enumerate(NUMERIC):
        raw[name] = numeric[:, i]
    return raw


def match_group_counts(eligible, group_ids, quota_trace, seed):
    selected = {arm: np.zeros(len(group_ids), dtype=bool) for arm in eligible}
    records = []
    generators = {arm: np.random.default_rng(seed + 23021 + i * 1009)
                  for i, arm in enumerate(eligible)}
    for entry in quota_trace:
        group, requested = entry["group_id"], int(entry["quota"])
        positions = {arm: np.flatnonzero(mask & (group_ids == group)) for arm, mask in eligible.items()}
        common = min(requested, *(len(p) for p in positions.values()))
        for arm, possible in positions.items():
            chosen = generators[arm].permutation(possible)[:common]
            selected[arm][chosen] = True
        records.append({"group_id": int(group), "nominal_quota": requested,
                        "available": {arm: len(p) for arm, p in positions.items()},
                        "matched_accepted": common, "quota_shortfall": requested - common})
    return selected, records
