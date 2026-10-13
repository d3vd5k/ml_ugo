"""U3 geometry-only ablation: real references, unchanged UGO scoring/allocation/stop."""
from copy import copy
from dataclasses import asdict, dataclass, fields
from numbers import Integral, Real

import numpy as np
import torch

from samplers.geometry import prepare_constraint_geometry
from samplers.ugo import (Candidates, RealSupport, UGOConfig, UGOPipeline,
                          allocate_group_budget, nearest_indices, noise_pre_detect)


@dataclass(frozen=True)
class ConstraintConfig:
    eta: float = .5
    rho: float = 1.
    repair: str = "shorten"
    max_shortening: int = 4
    epsilon: float = 1e-12
    graph_neighbors: int = 5
    graph_radius_quantile: float = .9

    def __post_init__(self):
        for name in ("eta", "rho", "epsilon"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.repair not in ("relocate", "reject", "shorten"):
            raise ValueError("repair must be relocate, reject or shorten")
        for name, minimum in (("max_shortening", 0), ("graph_neighbors", 1)):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (isinstance(self.graph_radius_quantile, (bool, np.bool_))
                or not isinstance(self.graph_radius_quantile, Real)
                or not np.isfinite(self.graph_radius_quantile)
                or not 0 <= self.graph_radius_quantile <= 1):
            raise ValueError("graph_radius_quantile must be finite and in [0, 1]")


def _metrics_row(metrics, index):
    return {key: value[index].item() for key, value in metrics.items()}


def accept_constraints(support, candidates, interpolation, geometry, config):
    """Repair a failed proposal, then recheck the same bounds on emitted Float32 points."""
    points = candidates.X.detach().cpu().numpy().copy()
    seeds = candidates.seed_indices.detach().cpu().numpy()
    partners = candidates.partner_indices.detach().cpu().numpy()
    lambdas = np.asarray(interpolation, dtype=float).copy()
    eligible = support.eligible.detach().cpu().numpy()
    initial = geometry.candidate_metrics(points, seeds, config.eta, config.rho, config.epsilon)
    final = {key: value.copy() for key, value in initial.items()}
    steps = np.zeros(len(seeds), dtype=int)
    relocation = np.full(len(seeds), -1, dtype=np.int64)
    failed = np.flatnonzero(~final["passes"])
    if config.repair == "relocate" and len(failed):
        indices = geometry.nearest_in_pool(points[failed], seeds[failed], eligible)
        repaired = failed[indices >= 0]
        relocation[repaired] = indices[indices >= 0]
        points[repaired] = geometry.X[relocation[repaired]]
        checked = geometry.candidate_metrics(points[repaired], seeds[repaired], config.eta, config.rho, config.epsilon)
        for key in final:
            final[key][repaired] = checked[key]
    elif config.repair == "shorten":
        for _ in range(config.max_shortening):
            failed = np.flatnonzero(~final["passes"])
            if not len(failed):
                break
            lambdas[failed] *= .5
            steps[failed] += 1
            direction = geometry.X[partners[failed]].astype(float) - geometry.X[seeds[failed]]
            points[failed] = geometry.X[seeds[failed]] + lambdas[failed, None] * direction
            checked = geometry.candidate_metrics(points[failed], seeds[failed], config.eta, config.rho, config.epsilon)
            for key in final:
                final[key][failed] = checked[key]
    keep = final["passes"]
    records = []
    for index, seed in enumerate(seeds):
        records.append({"seed_index": int(seed), "partner_index": int(partners[index]),
                        "class": int(geometry.y[seed]), "group_id": int(geometry.row_group_ids[seed]),
                        "initial_lambda": float(interpolation[index]),
                        "final_lambda": None if relocation[index] >= 0 else float(lambdas[index]),
                        "relocation_index": int(relocation[index]), "shortening_steps": int(steps[index]),
                        "repair": config.repair if not initial["passes"][index] else "none",
                        "accepted": bool(keep[index]),
                        "reason": "accepted" if keep[index] else "constraint_failed",
                        "initial_metrics": _metrics_row(initial, index), "final_metrics": _metrics_row(final, index)})
    device = support.X.device
    mask = torch.as_tensor(keep, device=device)
    accepted = Candidates(torch.as_tensor(points[keep], device=device), candidates.y[mask],
                          candidates.seed_indices[mask], candidates.partner_indices[mask],
                          torch.as_tensor(relocation[keep], device=device))
    return accepted, records


def constrained_ugo(X, y, seed, categorical_columns, max_iterations, device, model_fn,
                    targets, parameters, categorical_groups=(), prepared=None):
    parameters = dict(parameters)
    names = {field.name for field in fields(ConstraintConfig)}
    constraints = ConstraintConfig(**{key: value for key, value in parameters.items() if key in names})
    geometry = prepared if prepared is not None else prepare_constraint_geometry(
        X, y, categorical_columns, categorical_groups, parameters, targets)
    expected_classes = set(np.unique(y).tolist()) if targets is None else {
        int(label) for label in np.unique(y) if targets[int(label)] > int((np.asarray(y) == label).sum())}
    if (not np.array_equal(np.asarray(X, dtype=np.float32), geometry.X)
            or not np.array_equal(y, geometry.y)
            or set(categorical_columns or geometry.categorical_columns) != set(geometry.categorical_columns)
            or tuple(tuple(group) for group in categorical_groups) != tuple(geometry.categorical_groups)
            or {group["class"] for group in geometry.groups} != expected_classes
            or geometry.configuration["graph_neighbors"] != constraints.graph_neighbors
            or geometry.configuration["graph_radius_quantile"] != constraints.graph_radius_quantile):
        raise ValueError("Prepared geometry does not match this real training partition/configuration")
    base = {key: value for key, value in parameters.items() if key not in names}
    base.update(random_state=seed, categorical_columns=tuple(geometry.categorical_columns),
                max_iterations=max_iterations, device=device)
    config = UGOConfig(**base)
    original = {int(label): int((np.asarray(y) == label).sum()) for label in np.unique(y)}
    used = {label: 0 for label in original}
    attempts, accepted_details, round_details = [], [], []
    options = None
    pending = None

    def support(features, labels, cfg):
        nonlocal options
        classes = torch.unique(labels, sorted=True)
        eligible = ~noise_pre_detect(features, labels, cfg.k_pre, cfg.distance_batch_size)
        indices_by_class, excluded = {}, []
        for label in classes.tolist():
            indices = torch.nonzero((labels == label) & eligible, as_tuple=True)[0]
            if len(indices) < 2 or len(torch.unique(features[indices], dim=0)) < 2:
                eligible[indices] = False
                excluded.append(label)
            else:
                indices_by_class[label] = indices
        # Partner restrictions do not restore conflict flags or alter the baseline
        # class-level score/support. Unsupported selected seeds simply underspend.
        options = torch.full((len(labels), cfg.k_pre), -1, dtype=torch.long, device=features.device)
        for pool in geometry.pools:
            indices = torch.as_tensor(pool, device=features.device)
            indices = indices[eligible[indices]]
            if len(indices) < 2:
                continue
            count = min(cfg.k_pre, len(indices) - 1)
            nearest = nearest_indices(features[indices], features[indices], count, cfg.distance_batch_size,
                                      torch.arange(len(indices), device=features.device), exclude_zero=True)
            options[indices, :count] = indices[nearest]
        return RealSupport(features, labels, classes, eligible, {}, indices_by_class, excluded)

    def budget(real, scores, remaining, cfg):
        bounded = copy(real)
        if targets is not None:
            bounded.group_indices = {label: indices for label, indices in real.group_indices.items()
                                     if used[label] < targets[label] - original[label]}
        planned = allocate_group_budget(bounded, scores, remaining, cfg)
        if planned is not None and targets is not None:
            label = planned.target_class
            planned.count = min(planned.count, targets[label] - original[label] - used[label])
        return planned

    def propose(real, planned, rng, cfg):
        nonlocal pending
        chosen = planned.seed_indices[torch.as_tensor(rng.integers(len(planned.seed_indices), size=planned.count),
                                                     device=real.X.device)]
        partners = options[chosen]
        valid = (partners >= 0) & (real.X[partners.clamp_min(0)] != real.X[chosen, None]).any(dim=2)
        counts = valid.sum(dim=1)
        keep = counts > 0
        missing = chosen[~keep].tolist()
        seeds, partners, valid, counts = chosen[keep], partners[keep], valid[keep], counts[keep]
        choice = torch.as_tensor(rng.random(len(seeds)), device=real.X.device) * counts
        positions = (valid.cumsum(dim=1) > choice[:, None]).long().argmax(dim=1)
        selected = partners[torch.arange(len(seeds), device=real.X.device), positions]
        interpolation = rng.random(len(seeds))
        seed_indices, partner_indices = seeds.cpu().numpy(), selected.cpu().numpy()
        direction = geometry.X[partner_indices].astype(float) - geometry.X[seed_indices]
        points = (geometry.X[seed_indices] + interpolation[:, None] * direction).astype(np.float32)
        pending = {"interpolation": interpolation, "missing": missing, "attempted": len(chosen)}
        return Candidates(torch.as_tensor(points, device=real.X.device),
                          torch.full_like(seeds, planned.target_class), seeds, selected, torch.full_like(seeds, -1))

    def accept(real, candidates, cfg):
        accepted, records = accept_constraints(real, candidates, pending["interpolation"], geometry, constraints)
        round_number = len(round_details) + 1
        for record in records:
            record["round"] = round_number
        for source in pending["missing"]:
            records.append({"round": round_number, "seed_index": source, "partner_index": -1,
                            "class": int(geometry.y[source]), "group_id": int(geometry.row_group_ids[source]),
                            "accepted": False, "reason": "no_compatible_distinct_partner", "repair": "none",
                            "initial_metrics": None, "final_metrics": None, "shortening_steps": 0,
                            "initial_lambda": None, "final_lambda": None, "relocation_index": -1})
        attempts.extend(records)
        successful = [record for record in records if record["accepted"]]
        for record, point in zip(successful, accepted.X.detach().cpu().numpy()):
            accepted_details.append({**record, "point": point.tolist()})
        round_details.append({"attempted": pending["attempted"], "no_compatible_partner": len(pending["missing"]),
                              "rejected": len(candidates.X) - len(accepted.X),
                              "initial_constraint_failures": sum(record["initial_metrics"] is not None
                                  and not record["initial_metrics"]["passes"] for record in records),
                              "shortening_attempted": sum(record["shortening_steps"] > 0 for record in records),
                              "accepted_after_shortening": sum(record["accepted"] and record["shortening_steps"] > 0 for record in records),
                              "relocation_attempted": sum(record["relocation_index"] >= 0 for record in records),
                              "accepted_after_relocation": sum(record["accepted"] and record["relocation_index"] >= 0 for record in records)})
        for label in used:
            used[label] += int((accepted.y == label).sum())
        return accepted

    result = UGOPipeline(config, support=support, group_budget=budget, propose=propose, accept=accept).run(
        torch.as_tensor(X), torch.as_tensor(y), model_fn)
    for row, detail in zip(result.rounds, round_details):
        row.update(detail)
    for entry, detail in zip(result.provenance, accepted_details):
        entry.update({key: value for key, value in detail.items() if key != "round"})
    actual = {label: int((result.y == label).sum()) for label in original}
    result.support_summary.update(target_counts=targets, actual_counts=actual,
                                  protocol_deviations=["matched_per_class_target_caps"] if targets is not None else [],
                                  constraint_support={**geometry.snapshot(), "configuration": asdict(constraints),
                                                      "attempts": attempts, "partner_scope": "eligible_compatible_connected_group_real_neighbors"})
    if targets is not None and all(actual[label] == targets[label] for label in original):
        result.stop_reason = "target_counts_reached"
    elif result.stop_reason == "no_valid_partner" and attempts and any(record["reason"] == "constraint_failed" for record in attempts[-pending["attempted"]:]):
        result.stop_reason = "no_candidate_satisfies_constraints"
    return result
