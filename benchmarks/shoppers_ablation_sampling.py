"""Shoppers-only sampling interventions; production sampler defaults stay intact.

A4 and A7 retain canonical real-only U2 geometry and allocation. Flow gates
score original proposals, rejected attempts consume quotas, and no variant
refills. ``capture`` retains the original (pre-relocation) proposal stream for
conditional fixed-pool replay, not an independently generated trajectory.
"""
from dataclasses import asdict, replace

import numpy as np
import torch

from samplers.allocation import AllocationConfig, GroupAllocator
from samplers.geometry import _columns, _onehot_valid
from samplers.groups import GroupConfig, prepare_groups
from samplers.ugo import (
    Candidates, GroupBudget, RealSupport, UGOConfig, UGOPipeline,
    accept_relocation, allocate_group_budget, build_support, nearest_indices,
    noise_pre_detect, propose_interpolation,
)


VARIANTS = tuple(f"A{index}" for index in range(8))
_PROPOSAL_CAP = 1000
_FLOW_TAU = .6
_CANDIDATE_FIELDS = ("X", "y", "seed_indices", "partner_indices", "relocation_indices")


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _select(candidates, mask):
    return Candidates(*(getattr(candidates, key)[mask] for key in _CANDIDATE_FIELDS))


def prepare_ablation_groups(X, y, seed, targets):
    """Prepare canonical KMeans-8, five-repeat, 80% real-only support groups.

    Targets are final class counts. Unspecified classes in an explicit mapping
    retain their real counts, following the native allocator's target contract.
    """
    real_y = _numpy(y)
    counts = GroupAllocator({"groups": []}, real_y, targets).target_counts
    labels = [label for label, count in counts.items()
              if count > int(np.count_nonzero(real_y == label))]
    return prepare_groups(_numpy(X), real_y, labels, seed, GroupConfig())


def run_ablation(X, y, model_fn, *, variant, seed, targets, device,
                 prepared=None, flow_support=None, categorical_columns=(),
                 categorical_groups=(), max_iterations=20, capture=False):
    """Run an actual native learner trajectory and return ``(UGOResult, trace)``.

    A1/A2 use native class-wide pre-noise support and MMA top-half selection.
    A3 is canonical U2 with validity-only pass-through; A4 adds frozen Flow.
    A5 changes only positive stability multipliers to one; A6 changes only
    seed/group caps to finite nonbinding values. A7 uses native post-neighbor
    detection and relocation. A0 belongs to the real-only caller.

    ``model_fn`` fits the downstream learner on every native retraining call.
    The caller owns the final fit (and can reuse its last identical fit).
    ``trace['valid']`` means finite coordinates and valid complete one-hot
    blocks, independently of density-score validity or gate acceptance.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown Shoppers ablation variant: {variant!r}")
    if variant == "A0":
        raise ValueError("A0 is real-only and must be fitted by the experiment runner")
    use_u2 = variant not in ("A1", "A2")
    use_flow = variant in ("A2", "A4", "A5", "A6")
    relocate = variant == "A7"
    if use_flow != (flow_support is not None):
        raise ValueError("Exactly A2/A4/A5/A6 require a frozen flow_support gate")
    if max_iterations > 20:
        raise ValueError("The Shoppers ablation permits at most 20 native rounds")

    real_X, real_y = _numpy(X), _numpy(y)
    if real_X.ndim != 2:
        raise ValueError("Ablation features must be a real training matrix")
    columns = _columns(categorical_columns, real_X.shape[1], "categorical_columns")
    blocks = tuple(_columns(block, real_X.shape[1], "categorical_groups")
                   for block in categorical_groups)
    flattened = tuple(column for block in blocks for column in block)
    if blocks:
        if any(not block for block in blocks) or len(set(flattened)) != len(flattened):
            raise ValueError("categorical_groups must be disjoint nonempty full one-hot blocks")
        if columns and set(columns) != set(flattened):
            raise ValueError("categorical_columns must match the supplied full one-hot blocks")
        columns = tuple(sorted(flattened))
        if not _onehot_valid(real_X, blocks).all():
            raise ValueError("Real categorical_groups must be valid complete one-hot blocks")

    grouping = GroupConfig()
    canonical_allocation = AllocationConfig(group_alpha=.5)
    target_normalizer = GroupAllocator({"groups": []}, real_y, targets, canonical_allocation)
    target_counts = target_normalizer.target_counts
    real_counts = target_normalizer.real_counts
    target_labels = {label for label in target_counts
                     if target_counts[label] > real_counts[label]}
    proposal_cap = min(_PROPOSAL_CAP, sum(target_counts[c] - real_counts[c] for c in target_counts))
    allocation = canonical_allocation
    if variant == "A6":
        allocation = replace(allocation, seed_cap=max(allocation.seed_cap, proposal_cap),
                             group_cap_multiplier=max(allocation.group_cap_multiplier, float(proposal_cap)))
    config = UGOConfig(random_state=seed, device=device, max_iterations=max_iterations,
                       categorical_columns=columns)
    allocator = None
    supported_ids = set()
    protected_indices = np.empty(0, dtype=np.int64)
    if use_u2:
        prepared = (prepare_ablation_groups(real_X, real_y, seed, targets)
                    if prepared is None else prepared)
        if prepared["configuration"] != asdict(grouping):
            raise ValueError("Prepared groups must use canonical KMeans-8/five-repeat/80% grouping")
        supported = [group for group in prepared["groups"]
                     if group["real_support"] >= 2 and group["distinct_coordinates"] >= 2
                     and group["stability"] > 0]
        supported_ids = {group["id"] for group in supported}
        if supported:
            protected_indices = np.concatenate([group["indices"] for group in supported])
        allocation_prepared = prepared
        if variant == "A5":
            # The original eligibility set and n-based caps are unchanged.
            # Zero stability remains zero; only positive weight factors become 1.
            allocation_prepared = {**prepared, "groups": [
                {**group, "stability": 1. if group["stability"] > 0 else 0.}
                for group in prepared["groups"]]}
        allocator = GroupAllocator(allocation_prepared, real_y, targets, allocation)
        row_group_ids = np.asarray(prepared["row_group_ids"])
    else:
        # Class IDs describe class-wide support in the audit, not U2 clusters.
        row_group_ids = np.where(np.isin(real_y, list(target_labels)), real_y, -1).astype(np.int64)

    attempted_seed = (allocator.used_seed if allocator is not None
                      else np.zeros(len(real_y), dtype=np.int64))
    accepted_seed = np.zeros(len(real_y), dtype=np.int64)
    attempted_class = {label: 0 for label in real_counts}
    candidate_log, trace_parts = [], []
    base_records = []
    conflict = None
    support_eligible = None
    proposal_round = 0

    def support(features, labels, cfg):
        nonlocal conflict, support_eligible
        if not use_u2:
            real = build_support(features, labels, cfg)
            support_eligible = real.eligible.detach().cpu().numpy()
            return real
        classes = torch.unique(labels, sorted=True)
        conflict = noise_pre_detect(features, labels, cfg.k_pre, cfg.distance_batch_size)
        eligible = ~conflict
        for label in target_labels:
            eligible[labels == label] = False
        eligible[torch.as_tensor(protected_indices, device=features.device)] = True
        partners, indices_by_class, excluded = {}, {}, []
        for label in classes.tolist():
            indices = torch.nonzero((labels == label) & eligible, as_tuple=True)[0]
            if len(indices) < 2 or len(torch.unique(features[indices], dim=0)) < 2:
                eligible[indices] = False
                excluded.append(label)
                continue
            indices_by_class[label] = indices
            neighbors = nearest_indices(
                features[indices], features[indices], min(cfg.k_pre, len(indices) - 1),
                cfg.distance_batch_size, torch.arange(len(indices), device=features.device),
                exclude_zero=True,
            )
            partners[label] = indices[neighbors]
        support_eligible = eligible.detach().cpu().numpy()
        return RealSupport(features, labels, classes, eligible, partners, indices_by_class, excluded)

    def budget(real, scores, remaining, cfg):
        available = [label for label in real.group_indices if label in target_labels
                     and attempted_class[label] < target_counts[label] - real_counts[label]]
        remaining = min(remaining, proposal_cap - int(attempted_seed.sum()))
        if remaining <= 0 or not available:
            return None
        if not use_u2:
            restricted = replace(real, group_indices={label: real.group_indices[label] for label in available})
            planned = allocate_group_budget(restricted, scores, remaining, cfg)
            planned.count = min(planned.count, target_counts[planned.target_class]
                                - real_counts[planned.target_class] - attempted_class[planned.target_class])
            base_records.append({"target_class": planned.target_class, "requested": remaining,
                                 "allocated": planned.count, "attempted": 0, "accepted": 0,
                                 "ranked_seed_indices": planned.seed_indices.detach().cpu().numpy(),
                                 "selection_fraction": cfg.lambda_select})
            return planned
        count = min(remaining, max(1, round(len(real.y) * cfg.lambda_step)))
        score_values = scores.detach().cpu().numpy()
        for target in sorted(available,
                             key=lambda label: float(scores[real.group_indices[label]].mean()),
                             reverse=True):
            plan = allocator.allocate(score_values, target, count, cfg.lambda_select)
            if plan is not None:
                seeds = torch.as_tensor(plan["seed_indices"], device=real.X.device, dtype=torch.long)
                return GroupBudget(target, seeds, plan["count"])
        return None

    def propose(real, planned, rng, cfg):
        return propose_interpolation(real, planned, rng, cfg,
                                     seed_indices=planned.seed_indices if use_u2 else None)

    def accept(real, candidates, cfg):
        nonlocal proposal_round
        proposal_round += 1
        original = candidates.X.detach().cpu().numpy().copy()
        labels = candidates.y.detach().cpu().numpy()
        seeds = candidates.seed_indices.detach().cpu().numpy()
        partners = candidates.partner_indices.detach().cpu().numpy()
        # Commit attempts BEFORE any validity/gate/post-neighbor handling.
        if allocator is not None:
            allocator.commit(seeds)
        else:
            np.add.at(attempted_seed, seeds, 1)
        if len(labels):
            attempted_class[int(labels[0])] += len(labels)
        valid = np.isfinite(original).all(axis=1) & _onehot_valid(original, blocks)
        keep = valid.copy()
        scores = None
        log_ratio = np.full(len(labels), np.nan, dtype=np.float64)
        if use_flow:
            scores = flow_support.predict(original, labels, _FLOW_TAU)
            decisions = np.asarray(scores["accepted"], dtype=bool)
            if decisions.shape != seeds.shape:
                raise ValueError("Flow gate must return one acceptance decision per original proposal")
            log_ratio = np.asarray(scores["log_ratio"], dtype=np.float64)
            if log_ratio.shape != seeds.shape:
                raise ValueError("Flow gate must return one log ratio per original proposal")
            keep &= decisions
        elif relocate:
            # Invalid coordinates never enter native nearest-neighbor math.
            mask = torch.as_tensor(valid, device=candidates.X.device)
            relocated = accept_relocation(real, _select(candidates, mask), cfg)
            candidates.X[mask] = relocated.X
            candidates.relocation_indices[mask] = relocated.relocation_indices
        np.add.at(accepted_seed, seeds[keep], 1)
        records = allocator.records if allocator is not None else base_records
        if records:
            records[-1].update(attempted=len(seeds), accepted=int(keep.sum()))
        relocations = candidates.relocation_indices.detach().cpu().numpy()
        for index, source in enumerate(seeds):
            reason = ("invalid_features" if not valid[index] else
                      str(scores["reasons"][index]) if scores is not None else "accepted")
            entry = {"proposal_id": len(candidate_log), "round": proposal_round,
                     "seed_index": int(source), "partner_index": int(partners[index]),
                     "relocation_index": int(relocations[index]),
                     "category_donor_index": int(relocations[index] if relocations[index] >= 0
                                                 else partners[index]),
                     "group_id": int(row_group_ids[source]), "class": int(labels[index]),
                     "accepted": bool(keep[index]), "valid": bool(valid[index]), "reason": reason}
            if scores is not None:
                entry.update(threshold=_FLOW_TAU,
                             log_ratio_threshold=float(np.log(_FLOW_TAU) - np.log1p(-_FLOW_TAU)))
                for key in ("log_likelihood_intended", "log_likelihood_opposing", "log_ratio",
                            "compatibility_score", "log_determinant"):
                    value = float(scores[key][index])
                    entry[key] = value if np.isfinite(value) else None
            candidate_log.append(entry)
        if capture:
            trace_parts.append({"X": original, "y": labels.copy(),
                                "round": np.full(len(labels), proposal_round, dtype=np.int64),
                                "group_id": row_group_ids[seeds].astype(np.int64, copy=True),
                                "seed_index": seeds.copy(), "partner_index": partners.copy(),
                                "accepted": keep.copy(), "valid": valid.copy(),
                                "log_ratio": log_ratio.copy()})
        return _select(candidates, torch.as_tensor(keep, device=candidates.X.device))

    # Cap original attempts in budget(), not native accepted-row generation.
    # Keeping the native cap preserves its terminal quota-exhaustion retrain.
    result = UGOPipeline(config, support=support, group_budget=budget,
                         propose=propose, accept=accept).run(
                             torch.as_tensor(real_X), torch.as_tensor(real_y), model_fn)
    actual_counts = {label: int((result.y == label).sum()) for label in real_counts}
    groups = []
    if use_u2:
        conflict_numpy = conflict.detach().cpu().numpy()
        originals = {group["id"]: group for group in prepared["groups"]}
        for group in prepared["groups"]:
            indices = group["indices"]
            groups.append({**group, "supported": group["id"] in supported_ids,
                           "original_stability": group["stability"],
                           "effective_stability": (1. if variant == "A5" and group["stability"] > 0
                                                   else group["stability"]),
                           "cross_class_conflict_count": int(conflict_numpy[indices].sum()),
                           "attempted_generated": int(attempted_seed[indices].sum()),
                           "actual_generated": int(accepted_seed[indices].sum())})
        records = [{**record, "group_trace": [
            {**{key: value for key, value in trace.items() if key != "used_seed"},
             "original_stability": originals[trace["group_id"]]["stability"],
             "effective_stability": trace["stability"]}
            for trace in record["group_trace"]]} for record in allocator.records]
    else:
        conflict_numpy = np.empty(0, dtype=np.int64)
        records = base_records
        for label in sorted(target_labels):
            indices = np.flatnonzero(real_y == label)
            groups.append({"id": int(label), "class": int(label), "indices": indices,
                           "real_support": len(indices),
                           "eligible_real_support": int(support_eligible[indices].sum()),
                           "supported": bool(support_eligible[indices].any()),
                           "attempted_generated": int(attempted_seed[indices].sum()),
                           "actual_generated": int(accepted_seed[indices].sum())})
    attempted, accepted_count = int(attempted_seed.sum()), int(accepted_seed.sum())
    result.support_summary.update(
        variant=variant,
        variant_flags={"u2_grouping": use_u2, "native_mma_ranking": True,
                       "flow_gate": use_flow, "flow_tau": _FLOW_TAU if use_flow else None,
                       "native_neighbor_detection": relocate, "native_relocation": relocate,
                       "validity_checks": True, "stability_weights_removed": variant == "A5",
                       "original_support_eligibility_preserved": use_u2,
                       "nonbinding_seed_group_caps": variant == "A6", "refill": False,
                       "all_sample_weights_one": True},
        target_counts=target_counts, actual_counts=actual_counts,
        protocol_deviations=["matched_per_class_target_caps"] if targets is not None else [],
        group_support={
            "enabled": use_u2, "configuration": asdict(grouping) if use_u2 else None,
            "allocation": asdict(allocation) if use_u2 else None,
            "canonical_allocation": asdict(canonical_allocation) if use_u2 else None,
            "row_group_ids": row_group_ids, "groups": groups,
            "real_row_ids": prepared.get("real_row_ids") if use_u2 else None,
            "synthetic_training_count": 0,
            "cross_class_conflict_indices": np.flatnonzero(conflict_numpy) if use_u2 else None,
            "protected_conflict_indices": protected_indices[conflict_numpy[protected_indices]] if use_u2 else [],
            "seed_multiplicity": {str(i): int(accepted_seed[i]) for i in np.flatnonzero(accepted_seed)},
            "attempted_seed_multiplicity": {str(i): int(attempted_seed[i]) for i in np.flatnonzero(attempted_seed)},
            "allocation_records": records,
            "partner_scope": ("eligible_same_class_real_neighbors_not_group_restricted" if use_u2
                              else "native_pre_noise_eligible_class_wide_real_neighbors"),
        },
        candidate_log=candidate_log,
        proposal_budget={"cap": proposal_cap, "original_cap": _PROPOSAL_CAP,
                         "target_deficit": sum(target_counts[c] - real_counts[c] for c in target_counts),
                         "attempted": attempted, "accepted": accepted_count,
                         "rejected": attempted - accepted_count, "unspent_proposals": proposal_cap - attempted,
                         "allocation_accounting": "original proposals; rejected quotas never refilled",
                         "attempted_seed_multiplicity": {
                             str(i): int(attempted_seed[i]) for i in np.flatnonzero(attempted_seed)}},
        post_handler=("frozen_flow_gate" if use_flow else
                      "native_neighbor_relocation" if relocate else "validity_only_pass_through"),
        categorical_policy="copy_real_partner_before_post_handler; native_relocation_copies_real_donor_row",
        synthetic_sample_weight=1.,
    )
    for ancestry, proposal in zip(result.provenance,
                                  (entry for entry in candidate_log if entry["accepted"])):
        ancestry.update(group_id=proposal["group_id"], proposal_id=proposal["proposal_id"],
                        category_donor_index=proposal["category_donor_index"])
    if use_flow:
        result.support_summary["flow_support"] = flow_support.audit
    if result.stop_reason == "no_valid_partner" and candidate_log:
        result.stop_reason = "f1_no_candidates_accepted" if use_flow else "no_valid_proposals"
    if result.stop_reason == "no_eligible_group" and attempted >= proposal_cap:
        result.stop_reason = "proposal_cap_reached"
    if all(actual_counts[label] == target_counts[label] for label in actual_counts):
        result.stop_reason = "target_counts_reached"
    if result.rounds and "stop_reason" in result.rounds[-1]:
        result.rounds[-1]["stop_reason"] = result.stop_reason
    trace = None
    if capture:
        empty = {"X": np.empty((0, real_X.shape[1]), dtype=np.float32),
                 "y": np.empty(0, dtype=np.int64), "round": np.empty(0, dtype=np.int64),
                 "group_id": np.empty(0, dtype=np.int64), "seed_index": np.empty(0, dtype=np.int64),
                 "partner_index": np.empty(0, dtype=np.int64), "accepted": np.empty(0, dtype=bool),
                 "valid": np.empty(0, dtype=bool), "log_ratio": np.empty(0, dtype=np.float64)}
        trace = {key: np.concatenate([part[key] for part in trace_parts]) if trace_parts else value
                 for key, value in empty.items()}
    return result, trace
