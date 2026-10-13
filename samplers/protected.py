"""Canonical U2 allocation with explicit, independently selected post handlers."""
from dataclasses import asdict, fields
from time import perf_counter

import numpy as np
import torch

from samplers.allocation import AllocationConfig, GroupAllocator
from samplers.groups import GroupConfig, prepare_groups
from samplers.flow_support import CalibratedFlowGate, _audit_score, _quantile
from samplers.flow_gmm import FlowGate
from samplers.pool_random import RandomPoolSelector
from samplers.ugo import (GroupBudget, RealSupport, UGOConfig, UGOPipeline,
                          _majority, accept_relocation, nearest_indices, noise_pre_detect,
                          propose_interpolation)


def group_configuration(parameters):
    return GroupConfig(**{field.name: parameters[field.name] for field in fields(GroupConfig)
                          if field.name in parameters})


def prepare_protected_groups(X, y, seed, targets, parameters):
    labels, counts = np.unique(y, return_counts=True)
    final = targets if targets is not None else {int(label): int(counts.max()) for label in labels}
    target_labels = [int(label) for label, count in zip(labels, counts) if final[int(label)] > count]
    return prepare_groups(X, y, target_labels, seed, group_configuration(parameters))


def neighbor_rejection_scores(real_X, real_y, candidate_X, candidate_y, k_after=1, batch_size=512):
    """The original neighbor detector on original coordinates, without relocation."""
    reference = torch.as_tensor(real_X, dtype=torch.float32)
    labels = torch.as_tensor(real_y, device=reference.device, dtype=torch.long)
    query = torch.as_tensor(candidate_X, device=reference.device, dtype=torch.float32)
    intended = torch.as_tensor(candidate_y, device=reference.device, dtype=torch.long)
    valid = torch.isfinite(query).all(dim=1)
    positions = torch.nonzero(valid, as_tuple=True)[0]
    detected = torch.full_like(intended, -1)
    if len(positions):
        neighbors = nearest_indices(query[positions], reference, k_after, batch_size)
        detected[positions] = _majority(labels[neighbors], len(torch.unique(labels)))
    keep = valid & (detected == intended)
    valid_np, keep_np = valid.cpu().numpy(), keep.cpu().numpy()
    reasons = np.where(keep_np, "accepted", "neighbor_class_conflict").astype(object)
    reasons[~valid_np] = "invalid_features"
    return {"valid": valid_np, "accepted": keep_np, "reasons": reasons,
            "neighbor_label": detected.cpu().numpy()}


def protected_ugo(X, y, seed, categorical_columns, max_iterations, device, model_fn,
                  targets, parameters, prepared=None, *, seed_score=None, h1_support=None,
                  flow_support=None, reject_neighbors=False, candidate_selector=None):
    parameters = dict(parameters)
    random_pool = isinstance(candidate_selector, RandomPoolSelector)
    geometry_pool = random_pool and candidate_selector.proposal_generator is not None
    if geometry_pool and (flow_support is not None or h1_support is not None or reject_neighbors or seed_score is not None or "f1_tau" in parameters):
        raise ValueError("Geometry proposals require native relocation without gates")
    pool_f1 = random_pool and isinstance(flow_support, FlowGate) and not isinstance(flow_support, CalibratedFlowGate)
    if candidate_selector is not None and (seed_score is not None or h1_support is not None
            or (flow_support is not None and not pool_f1) or reject_neighbors
            or any(key in parameters for key in ("stop", "policy", "budget_fraction", "budget_fractions"))):
        raise ValueError("Candidate selection requires plain U2 or native random with noncalibrated F1")
    if sum((h1_support is not None, flow_support is not None, bool(reject_neighbors))) > 1:
        raise ValueError("Post-generation handlers cannot be combined")
    rejecting = h1_support is not None or flow_support is not None or reject_neighbors
    f2 = isinstance(flow_support, CalibratedFlowGate)
    if not f2 and any(key in parameters for key in ("f2_q", "support_filter")):
        raise ValueError("F2 support parameters require a calibrated gate")
    if f2:
        q = _quantile(parameters.pop("f2_q", flow_support.q))
        if q != flow_support.q:
            raise ValueError("f2_q must agree with the calibrated gate's quantile, including explicit off")
        enabled = parameters.pop("support_filter", q is not None)
        if not isinstance(enabled, (bool, np.bool_)) or bool(enabled) != (q is not None):
            raise ValueError("support_filter must agree with the calibrated gate's explicit quantile/off state")
    post_policy = "f2" if f2 else "f1" if flow_support is not None else "n_reject" if reject_neighbors else "h1"
    capture = parameters.pop("capture_proposals", False) or rejecting
    tau = parameters.pop("f1_tau", .5)
    for key in ("h1_min_cluster_size", "h1_min_samples", "flow_epochs", "flow_layers",
                "flow_hidden", "flow_batch_size", "flow_learning_rate"):
        parameters.pop(key, None)
    grouping = group_configuration(parameters)
    allocation = AllocationConfig(**{field.name: parameters[field.name] for field in fields(AllocationConfig)
                                     if field.name in parameters})
    specialized = {field.name for cls in (GroupConfig, AllocationConfig) for field in fields(cls)}
    base = {key: value for key, value in parameters.items() if key not in specialized}
    base.update(random_state=seed, categorical_columns=tuple(categorical_columns),
                max_iterations=max_iterations, device=device)
    config = UGOConfig(**base)
    prepared = prepared if prepared is not None else prepare_protected_groups(X, y, seed, targets, parameters)
    if prepared["configuration"] != asdict(grouping):
        raise ValueError("Prepared real groups do not match the requested grouping configuration")
    allocator = GroupAllocator(prepared, y, targets, allocation)
    accepted_seed = np.zeros(len(y), dtype=np.int64) if rejecting else None
    candidate_log = []
    proposal_round = 0
    neighbor_seconds = 0.
    proposal_cap = sum(allocator.target_counts[c] - int((np.asarray(y) == c).sum())
                       for c in allocator.target_counts)
    conflict = None
    supported = [group for group in prepared["groups"] if group["real_support"] >= 2
                 and group["distinct_coordinates"] >= 2 and group["stability"] > 0]
    supported_ids = {group["id"] for group in supported}
    protected_indices = np.concatenate([group["indices"] for group in supported]) if supported else np.empty(0, dtype=int)
    target_labels = set(int(label) for label in np.unique(y)
                        if allocator.target_counts[int(label)] > int((y == label).sum()))

    def support(features, labels, cfg):
        nonlocal conflict
        classes = torch.unique(labels, sorted=True)
        conflict = noise_pre_detect(features, labels, cfg.k_pre, cfg.distance_batch_size)
        eligible = ~conflict
        # Cross-class conflict remains a diagnostic, not a minority corruption label.
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
            neighbors = nearest_indices(features[indices], features[indices], min(cfg.k_pre, len(indices)-1),
                                        cfg.distance_batch_size, torch.arange(len(indices), device=features.device),
                                        exclude_zero=True)
            partners[label] = indices[neighbors]
        return RealSupport(features, labels, classes, eligible, partners, indices_by_class, excluded)

    def budget(real, scores, remaining, cfg):
        available = [label for label in real.group_indices if label in target_labels
                     and int(allocator.used_seed[np.asarray(y) == label].sum())
                     < allocator.target_counts[label] - int((np.asarray(y) == label).sum())]
        if rejecting:
            remaining = min(remaining, proposal_cap - int(allocator.used_seed.sum()))
        if remaining <= 0:
            return None
        count = min(remaining, max(1, round(len(real.y) * cfg.lambda_step)))
        score_values = scores.detach().cpu().numpy()
        for target in sorted(available, key=lambda label: float(scores[real.group_indices[label]].mean()), reverse=True):
            plan = allocator.allocate(score_values, target, count, cfg.lambda_select)
            if plan is not None:
                seeds = torch.as_tensor(plan["seed_indices"], device=real.X.device, dtype=torch.long)
                return GroupBudget(target, seeds, plan["count"])
        return None

    def propose(real, planned, rng, cfg):
        if candidate_selector is not None:
            return candidate_selector.propose(real, planned, rng, cfg)
        return propose_interpolation(real, planned, rng, cfg, seed_indices=planned.seed_indices)

    def accept(real, candidates, cfg):
        nonlocal proposal_round, neighbor_seconds
        proposal_round += 1
        original = candidates.X.detach().cpu().numpy().copy() if capture else None
        # Flow gates replace both neighbor detection and relocation; N-reject replaces relocation.
        if geometry_pool:
            mask = torch.as_tensor(candidate_selector._pending["geometry_valid"], device=candidates.X.device)
            valid = type(candidates)(*(getattr(candidates, key)[mask] for key in
                ("X", "y", "seed_indices", "partner_indices", "relocation_indices")))
            valid = accept_relocation(real, valid, cfg)
            candidates.X[mask] = valid.X
            candidates.relocation_indices[mask] = valid.relocation_indices
            accepted = candidates
        else:
            accepted = candidates if flow_support is not None or reject_neighbors else accept_relocation(real, candidates, cfg)
        if candidate_selector is not None:
            accepted = candidate_selector.select(real, accepted, cfg, allocator.records[-1])
            if capture:
                pool = candidate_selector.rounds[-1]
                original = pool["original_coordinates"][pool["attempted_selected"] if random_pool else pool["selected"]]
        seeds = accepted.seed_indices.detach().cpu().numpy()
        allocator.commit(seeds)
        scores = None
        keep = np.ones(len(seeds), dtype=bool) if capture else None
        if flow_support is not None:
            scores = flow_support.predict(original, accepted.y.detach().cpu().numpy(), tau)
            keep = scores["accepted"]
        elif reject_neighbors:
            tick = perf_counter()
            scores = neighbor_rejection_scores(real.X, real.y, accepted.X, accepted.y,
                                                cfg.k_after, cfg.distance_batch_size)
            neighbor_seconds += perf_counter() - tick
            keep = scores["accepted"]
        elif h1_support is not None:
            scores = h1_support.predict(accepted.X.detach().cpu().numpy())
            keep = scores["valid"] & (scores["labels"] != -1)
        if random_pool:
            candidate_selector.finalize_acceptance(np.ones(len(seeds), dtype=bool) if keep is None else keep, scores)
        if rejecting:
            np.add.at(accepted_seed, seeds[keep], 1)
            if allocator.records:
                allocator.records[-1].update(attempted=len(seeds), accepted=int(keep.sum()))
        if capture:
            final = accepted.X.detach().cpu().numpy()
            partners = accepted.partner_indices.detach().cpu().numpy()
            relocations = accepted.relocation_indices.detach().cpu().numpy()
            labels = accepted.y.detach().cpu().numpy()
            row_ids = prepared.get("real_row_ids")
            proposal_ids = pool["proposal_ids"][pool["attempted_selected"]] if random_pool else None
            for index, source in enumerate(seeds):
                proposal_id = int(proposal_ids[index]) if random_pool else len(candidate_log)
                entry = {"proposal_id": proposal_id, "round": proposal_round,
                         "seed_index": int(source), "partner_index": int(partners[index]),
                         "relocation_index": int(relocations[index]),
                         "group_id": int(prepared["row_group_ids"][source]),
                         "class": int(labels[index]), "original_coordinates": original[index].tolist(),
                         "final_coordinates": final[index].tolist(), "accepted": bool(keep[index]),
                         "reason": "accepted" if scores is None else str(scores["reasons"][index]),
                         "hdbscan_label": int(scores["labels"][index]) if h1_support is not None else None,
                         "membership_strength": float(scores["strengths"][index]) if h1_support is not None else None,
                         "threshold": tau if flow_support is not None else
                             "neighbor label == intended class" if reject_neighbors else
                             "predicted_label != -1" if h1_support is not None else None}
                if flow_support is not None:
                    entry["log_ratio_threshold"] = float(np.log(tau) - np.log1p(-tau))
                    for key in ("log_likelihood_intended", "log_likelihood_opposing", "log_ratio",
                                "compatibility_score", "log_determinant"):
                        value = float(scores[key][index])
                        entry[key] = value if np.isfinite(value) else None
                    if f2:
                        entry.update(compatibility_accepted=bool(scores["compatibility_accepted"][index]),
                                     likelihood_accepted=bool(scores["likelihood_accepted"][index]),
                                     support_filter=q is not None, support_quantile=q)
                        value = float(scores["likelihood_threshold"][index])
                        entry["likelihood_threshold"] = _audit_score(value)
                        entry["likelihood_threshold_kind"] = ("off" if q is None else
                            "negative_infinity" if np.isneginf(value) else "finite")
                elif reject_neighbors:
                    entry["neighbor_label"] = int(scores["neighbor_label"][index])
                if row_ids is not None:
                    entry.update(seed_row_id=str(row_ids[source]), partner_row_id=str(row_ids[partners[index]]))
                candidate_log.append(entry)
        if rejecting:
            mask = torch.as_tensor(keep, device=accepted.X.device)
            accepted = type(accepted)(*(getattr(accepted, key)[mask] for key in
                ("X", "y", "seed_indices", "partner_indices", "relocation_indices")))
        return accepted

    scoring = {} if seed_score is None else {"seed_score": seed_score}
    if candidate_selector is not None:
        scoring = {"seed_score": candidate_selector.score_seeds}
    result = UGOPipeline(config, support=support, group_budget=budget, propose=propose,
                         accept=accept, **scoring).run(torch.as_tensor(X), torch.as_tensor(y), model_fn)
    used = allocator.used_seed if accepted_seed is None else accepted_seed
    groups = []
    conflict_numpy = conflict.detach().cpu().numpy()
    for group in prepared["groups"]:
        indices = group["indices"]
        groups.append({**group, "supported": group["id"] in supported_ids,
                       "cross_class_conflict_count": int(conflict_numpy[indices].sum()),
                       "actual_generated": int(used[indices].sum()),
                       "row_ids": None if "real_row_ids" not in prepared else prepared["real_row_ids"][indices]})
    original = {int(label): int((np.asarray(y) == label).sum()) for label in np.unique(y)}
    actual = {label: int((result.y == label).sum()) for label in original}
    result.support_summary.update(
        target_counts=allocator.target_counts, actual_counts=actual,
        protocol_deviations=["matched_per_class_target_caps"] if targets is not None else [],
        group_support={"configuration": prepared["configuration"], "allocation": asdict(allocation),
                       "row_group_ids": prepared["row_group_ids"], "groups": groups,
                       "real_row_ids": prepared.get("real_row_ids"), "synthetic_training_count": 0,
                       "cross_class_conflict_indices": np.flatnonzero(conflict_numpy),
                       "protected_conflict_indices": protected_indices[conflict_numpy[protected_indices]],
                       "seed_multiplicity": {str(index): int(used[index]) for index in np.flatnonzero(used)},
                       "allocation_records": allocator.records,
                       "partner_scope": "eligible_same_class_real_neighbors_not_group_restricted"})
    for entry in result.provenance:
        entry["group_id"] = int(prepared["row_group_ids"][entry["seed_index"]])
    if capture:
        for ancestry, proposal in zip(result.provenance, (row for row in candidate_log if row["accepted"])):
            ancestry["proposal_id"] = proposal["proposal_id"]
        result.support_summary["candidate_log"] = candidate_log
    if rejecting:
        if h1_support is not None:
            result.support_summary["h1_support"] = h1_support.audit
        elif flow_support is not None:
            result.support_summary["flow_support"] = flow_support.audit
        else:
            result.support_summary["neighbor_rejection"] = {"coordinate_space": "original proposal",
                "k_after": config.k_after, "relocation": False, "reference_source": "real training",
                "training_seconds": 0., "inference_seconds": neighbor_seconds}
        result.support_summary["proposal_budget"] = {"cap": proposal_cap,
            "attempted": int(allocator.used_seed.sum()), "accepted": int(used.sum()),
            "unspent_proposals": proposal_cap - int(allocator.used_seed.sum()),
            "rejected": int(allocator.used_seed.sum() - used.sum()),
            "allocation_accounting": "planned proposals; rejected quotas never refilled",
            "attempted_seed_multiplicity": allocator.used_seed.tolist()}
        if result.stop_reason == "no_valid_partner" and candidate_log:
            result.stop_reason = post_policy + "_no_candidates_accepted"
        elif result.stop_reason == "no_eligible_group" and int(allocator.used_seed.sum()) >= proposal_cap:
            result.stop_reason = "proposal_cap_reached"
    if all(actual[label] == allocator.target_counts[label] for label in original):
        result.stop_reason = "target_counts_reached"
    if candidate_selector is not None:
        result.support_summary[getattr(candidate_selector, "summary_key", "learning_benefit")] = candidate_selector.summary()
        for ancestry, proposal_id in zip(result.provenance,
                (int(proposal_id) for pool in candidate_selector.rounds
                 for proposal_id in pool["proposal_ids"][pool["selected"]])):
            ancestry["proposal_id"] = proposal_id
    return result
