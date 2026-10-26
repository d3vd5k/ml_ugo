"""Real-only canonical U2, with native relocation or a frozen FlowGMM gate.

Grouping, quotas, MMA scoring, interpolation, relocation and stopping are reused
from the bundled core. The gate replaces both post-neighbor detection and
relocation; rejected attempts consume quotas and are never refilled. Candidate
records contain ancestry and gate diagnostics, not feature-coordinate copies.
"""
from dataclasses import asdict, fields

import numpy as np
import torch

from portable_ugo.allocation import AllocationConfig, GroupAllocator
from portable_ugo.groups import GroupConfig, prepare_groups
from portable_ugo.ugo import (
    Candidates, GroupBudget, RealSupport, UGOConfig, UGOPipeline,
    accept_relocation, nearest_indices, noise_pre_detect, propose_interpolation,
)


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _configurations(parameters):
    values = dict(parameters)
    tau = values.pop("f1_tau", .5)
    group_names = {item.name for item in fields(GroupConfig)}
    allocation_names = {item.name for item in fields(AllocationConfig)}
    grouping = GroupConfig(**{name: values.pop(name) for name in group_names if name in values})
    allocation = AllocationConfig(**{name: values.pop(name) for name in allocation_names if name in values})
    if grouping != GroupConfig():
        raise ValueError("Canonical U2 requires KMeans-8 and five 80% real-row subsamples")
    if allocation != AllocationConfig(group_alpha=allocation.group_alpha):
        raise ValueError("Canonical U2 requires stability allocation, group cap 2n and seed cap 4")
    if values.get("lambda_select", .5) != .5:
        raise ValueError("Canonical U2 uses the top half of native MMA-ranked real seeds")
    return grouping, allocation, tau, values


def prepare_u2(X, y, seed, targets, parameters):
    """Prepare reusable real-only groups; alpha and gate tau do not affect them.

    ``targets`` contains final class counts, not synthetic counts. With an
    explicit partial mapping, unspecified classes retain their real counts.
    Without targets, every observed class targets the largest real class count.
    """
    grouping, _, _, _ = _configurations(parameters)
    real_X, real_y = _numpy(X), _numpy(y)
    # Reuse core validation and normalization even before group memberships exist.
    target_counts = GroupAllocator({"groups": []}, real_y, targets).target_counts
    labels = [label for label in target_counts
              if target_counts[label] > int(np.count_nonzero(real_y == label))]
    return prepare_groups(real_X, real_y, labels, seed, grouping)


def run_u2(X, y, model_fn, *, seed, targets, parameters, max_iterations, device,
           categorical_columns=(), prepared=None, flow_support=None):
    """Return the core UGOResult for native U2 or U2 plus a frozen FlowGate.

    ``model_fn(X_tensor, y_tensor)`` fits the actual downstream learner and must
    expose probabilities in consecutive class-label order. Native U2 is used
    for learner/alpha selection; supplying an already-fitted ``flow_support``
    invokes only its ``predict(X, y, tau)`` and never fits or calibrates it.
    Parent indices always reference the original real training matrix. All
    categorical output columns are copied together from the real partner before
    either post-handler. Accepted coordinates remain available in result.X;
    candidate_log deliberately does not duplicate proposed or final coordinates.
    """
    grouping, allocation, tau, base = _configurations(parameters)
    if flow_support is not None and (isinstance(tau, (bool, np.bool_))
            or not np.isfinite(tau) or not 0 < tau < 1):
        raise ValueError("f1_tau must be strictly between zero and one")
    base.update(random_state=seed, categorical_columns=tuple(categorical_columns),
                max_iterations=max_iterations, device=device)
    config = UGOConfig(**base)
    real_y = _numpy(y)
    prepared = (prepare_u2(X, real_y, seed, targets, parameters)
                if prepared is None else prepared)
    if prepared["configuration"] != asdict(grouping):
        raise ValueError("Prepared groups do not match canonical U2 grouping")
    allocator = GroupAllocator(prepared, real_y, targets, allocation)
    target_labels = {label for label in allocator.target_counts
                     if allocator.target_counts[label] > allocator.real_counts[label]}
    proposal_cap = sum(allocator.target_counts[label] - allocator.real_counts[label]
                       for label in allocator.target_counts)
    supported = [group for group in prepared["groups"]
                 if group["real_support"] >= 2 and group["distinct_coordinates"] >= 2
                 and group["stability"] > 0]
    supported_ids = {group["id"] for group in supported}
    protected_indices = (np.concatenate([group["indices"] for group in supported])
                         if supported else np.empty(0, dtype=np.int64))
    accepted_seed = np.zeros(len(real_y), dtype=np.int64)
    candidate_log = []
    conflict = None
    proposal_round = 0

    def support(features, labels, cfg):
        nonlocal conflict
        classes = torch.unique(labels, sorted=True)
        conflict = noise_pre_detect(features, labels, cfg.k_pre, cfg.distance_batch_size)
        eligible = ~conflict
        # Cross-class conflict is diagnostic, not a minority corruption label:
        # supported minority rows override pre-noise exclusion, unsupported ones
        # cannot become parents even when their local neighbors agree.
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
        return RealSupport(features, labels, classes, eligible, partners, indices_by_class, excluded)

    def budget(real, scores, remaining, cfg):
        available = [label for label in real.group_indices if label in target_labels
                     and allocator.used_class[label]
                     < allocator.target_counts[label] - allocator.real_counts[label]]
        # In gated runs generated counts only accepted rows. Bound by attempts
        # separately so rejection cannot turn fixed quotas into refill budgets.
        remaining = min(remaining, proposal_cap - int(allocator.used_seed.sum()))
        if remaining <= 0:
            return None
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
        return propose_interpolation(real, planned, rng, cfg, seed_indices=planned.seed_indices)

    def accept(real, candidates, cfg):
        nonlocal proposal_round
        proposal_round += 1
        # Commit actual proposals BEFORE the gate, never just accepted rows.
        # Distinct real-coordinate support ensures interpolation has a partner.
        seeds = candidates.seed_indices.detach().cpu().numpy()
        allocator.commit(seeds)
        scores = None
        if flow_support is None:
            accepted = accept_relocation(real, candidates, cfg)
            keep = np.ones(len(seeds), dtype=bool)
        else:
            accepted = candidates
            scores = flow_support.predict(candidates.X.detach().cpu().numpy(),
                                          candidates.y.detach().cpu().numpy(), tau)
            keep = np.asarray(scores["accepted"], dtype=bool)
            if keep.shape != seeds.shape:
                raise ValueError("FlowGate must return one acceptance decision per proposal")
        np.add.at(accepted_seed, seeds[keep], 1)
        if allocator.records:
            allocator.records[-1].update(attempted=len(seeds), accepted=int(keep.sum()))
        partners = accepted.partner_indices.detach().cpu().numpy()
        relocations = accepted.relocation_indices.detach().cpu().numpy()
        labels = accepted.y.detach().cpu().numpy()
        for index, source in enumerate(seeds):
            entry = {"proposal_id": len(candidate_log), "round": proposal_round,
                     "seed_index": int(source), "partner_index": int(partners[index]),
                     "relocation_index": int(relocations[index]),
                     "category_donor_index": int(relocations[index] if relocations[index] >= 0
                                                 else partners[index]),
                     "group_id": int(prepared["row_group_ids"][source]),
                     "class": int(labels[index]), "accepted": bool(keep[index]),
                     "reason": "accepted" if scores is None else str(scores["reasons"][index])}
            if scores is not None:
                entry.update(threshold=float(tau),
                             log_ratio_threshold=float(np.log(tau) - np.log1p(-tau)))
                for key in ("log_likelihood_intended", "log_likelihood_opposing", "log_ratio",
                            "compatibility_score", "log_determinant"):
                    value = float(scores[key][index])
                    entry[key] = value if np.isfinite(value) else None
            candidate_log.append(entry)
        if flow_support is not None:
            mask = torch.as_tensor(keep, device=accepted.X.device)
            accepted = Candidates(*(getattr(accepted, key)[mask] for key in
                                    ("X", "y", "seed_indices", "partner_indices", "relocation_indices")))
        return accepted

    result = UGOPipeline(config, support=support, group_budget=budget,
                         propose=propose, accept=accept).run(
                             torch.as_tensor(X), torch.as_tensor(y), model_fn)
    conflict_numpy = conflict.detach().cpu().numpy()
    actual_counts = {label: int((result.y == label).sum()) for label in allocator.real_counts}
    groups = []
    for group in prepared["groups"]:
        indices = group["indices"]
        groups.append({**group, "supported": group["id"] in supported_ids,
                       "cross_class_conflict_count": int(conflict_numpy[indices].sum()),
                       "attempted_generated": int(allocator.used_seed[indices].sum()),
                       "actual_generated": int(accepted_seed[indices].sum())})
    # Core traces include a per-group per-round seed dictionary. Final attempted
    # and accepted multiplicities already retain that information compactly.
    records = [{**record, "group_trace": [
        {key: value for key, value in trace.items() if key != "used_seed"}
        for trace in record["group_trace"]]} for record in allocator.records]
    attempted = int(allocator.used_seed.sum())
    accepted_count = int(accepted_seed.sum())
    result.support_summary.update(
        target_counts=allocator.target_counts, actual_counts=actual_counts,
        protocol_deviations=["matched_per_class_target_caps"] if targets is not None else [],
        group_support={
            "configuration": prepared["configuration"], "allocation": asdict(allocation),
            "row_group_ids": prepared["row_group_ids"], "groups": groups,
            "real_row_ids": prepared.get("real_row_ids"), "synthetic_training_count": 0,
            "cross_class_conflict_indices": np.flatnonzero(conflict_numpy),
            "protected_conflict_indices": protected_indices[conflict_numpy[protected_indices]],
            "seed_multiplicity": {str(i): int(accepted_seed[i]) for i in np.flatnonzero(accepted_seed)},
            "attempted_seed_multiplicity": {
                str(i): int(allocator.used_seed[i]) for i in np.flatnonzero(allocator.used_seed)},
            "allocation_records": records,
            "partner_scope": "eligible_same_class_real_neighbors_not_group_restricted",
        },
        candidate_log=candidate_log,
        proposal_budget={
            "cap": proposal_cap, "attempted": attempted, "accepted": accepted_count,
            "rejected": attempted - accepted_count, "unspent_proposals": proposal_cap - attempted,
            "allocation_accounting": "planned proposals; rejected quotas never refilled",
        },
        post_handler="native_neighbor_relocation" if flow_support is None else "frozen_flow_gate",
        categorical_policy="copy_real_partner_before_post_handler; native_relocation_copies_real_donor_row",
        synthetic_sample_weight=1.,
    )
    accepted_proposals = (entry for entry in candidate_log if entry["accepted"])
    for ancestry, proposal in zip(result.provenance, accepted_proposals):
        ancestry.update(group_id=proposal["group_id"], proposal_id=proposal["proposal_id"],
                        category_donor_index=proposal["category_donor_index"])
    if flow_support is not None:
        result.support_summary["flow_support"] = flow_support.audit
        if result.stop_reason == "no_valid_partner" and candidate_log:
            result.stop_reason = "f1_no_candidates_accepted"
    if result.stop_reason == "no_eligible_group" and attempted >= proposal_cap:
        result.stop_reason = "proposal_cap_reached"
    if all(actual_counts[label] == allocator.target_counts[label] for label in actual_counts):
        result.stop_reason = "target_counts_reached"
    # Keep the terminal round consistent when the wrapper gives a more precise
    # quota/gate explanation than the generic core pipeline.
    if result.rounds and "stop_reason" in result.rounds[-1]:
        result.rounds[-1]["stop_reason"] = result.stop_reason
    return result
