"""Native U2 random pools, optionally thinned by F1 after random selection."""
from numbers import Integral
from time import perf_counter

import numpy as np
import torch

from samplers.learning_benefit import _array, _coordinate_key, model_state_hash, select_pool_mask
from samplers.ugo import propose_interpolation, score_seeds


class RandomPoolSelector:
    """Expand distinct planned anchors; choose once, then finalize actual acceptance.

    Only accepted synthetic coordinates enter cross-round deduplication. A rejected
    random choice never promotes another candidate or returns its allocation quota.
    """
    summary_key = "candidate_pool"

    def __init__(self, seed, k=5, checkpoint_callback=None, proposal_generator=None):
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
            raise ValueError("k must be an integer >= 1")
        self.seed, self.k = int(seed), int(k)
        self.selector = "random"
        self.checkpoint_callback = checkpoint_callback
        self.proposal_generator = proposal_generator
        self.rounds, self.checkpoints = [], []
        self._accepted_coordinates = set()
        self._next_proposal_id = 0
        self._pending = None
        self._awaiting_acceptance = False

    def _state_hash(self):
        after = model_state_hash(self.live_model)
        if after != self._checkpoint["state_before_hash"]:
            raise RuntimeError("Random pool handling mutated the native downstream model")
        return after

    def score_seeds(self, model, X, y):
        if self._pending is not None or self._awaiting_acceptance:
            raise RuntimeError("Finalize the previous pool before scoring a new checkpoint")
        before = model_state_hash(model)
        tick = perf_counter()
        values = score_seeds(model, X, y)
        seconds = perf_counter() - tick
        after = model_state_hash(model)
        if before != after:
            raise RuntimeError("Seed scoring mutated the native downstream model")
        self.live_model = model
        self._checkpoint = {"round": len(self.checkpoints) + 1,
            "state_before_hash": before, "state_after_hash": after,
            "downstream_state_before_hash": before, "downstream_state_after_hash": after,
            "native_anchor_scoring_seconds": seconds}
        self.checkpoints.append(self._checkpoint)
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(model, self._checkpoint)
            self._state_hash()
        return values

    def propose(self, real, budget, rng, cfg):
        if not hasattr(self, "_checkpoint"):
            raise RuntimeError("score_seeds must freeze the native fitted model first")
        if self._pending is not None or self._awaiting_acceptance:
            raise RuntimeError("Finalize the previous pool before proposing again")
        self._state_hash()
        tick = perf_counter()
        planned = _array(budget.seed_indices).astype(np.int64, copy=True)
        _, first = np.unique(planned, return_index=True)
        anchors = planned[np.sort(first)]
        seeds = torch.as_tensor(np.repeat(anchors, self.k), device=real.X.device, dtype=torch.long)
        propose = propose_interpolation if self.proposal_generator is None else self.proposal_generator.propose
        candidates = propose(real, budget, rng, cfg, seed_indices=seeds)
        n = len(candidates.X)
        self._pending = {**self._checkpoint, "original_coordinates": _array(candidates.X).copy(),
            "planned_anchor_indices": planned, "planned_budget": int(budget.count),
            "proposal_ids": np.arange(self._next_proposal_id, self._next_proposal_id + n, dtype=np.int64),
            "generation_seconds": perf_counter() - tick}
        if self.proposal_generator is not None:
            self._pending.update(self.proposal_generator.last_proposal)
        self._next_proposal_id += n
        return candidates

    def select(self, real, candidates, cfg, allocation_record):
        if self._pending is None:
            raise RuntimeError("propose must capture the original pool first")
        tick = perf_counter()
        record = self._pending
        final = np.asarray(_array(candidates.X), dtype=np.float32).copy()
        labels, anchors = _array(candidates.y).copy(), _array(candidates.seed_indices).copy()
        groups_by_anchor = {int(anchor): int(trace["group_id"])
            for trace in allocation_record["group_trace"] for anchor in trace["used_seed"]}
        groups = np.asarray([groups_by_anchor[int(anchor)] for anchor in anchors], dtype=np.int64)
        quotas = {int(trace["group_id"]): int(trace["quota"]) for trace in allocation_record["group_trace"]}
        real_keys = {_coordinate_key(row) for row in _array(real.X)}
        pool_seen, keep = set(), np.ones(len(final), dtype=bool)
        reasons = np.full(len(final), "not_selected", dtype=object)
        real_matches = np.zeros(len(final), dtype=bool)
        for index, row in enumerate(final):
            if self.proposal_generator is not None and not record["geometry_valid"][index]:
                keep[index], reasons[index] = False, record["geometry_reasons"][index]
                continue
            if not np.isfinite(row).all():
                keep[index], reasons[index] = False, "invalid_features"
                continue
            key = _coordinate_key(row)
            real_matches[index] = key in real_keys
            if key in pool_seen:
                keep[index], reasons[index] = False, "pool_duplicate"
            elif key in self._accepted_coordinates:
                keep[index], reasons[index] = False, "accumulated_duplicate"
            pool_seen.add(key)
        attempted = select_pool_mask(final, anchors, groups, None, keep, quotas,
                                     "random", self.seed, record["round"])
        reasons[attempted] = "selected"
        coverage = {}
        for group, quota in quotas.items():
            planned = [a for a in record["planned_anchor_indices"] if groups_by_anchor[int(a)] == group]
            coverage[group] = {"planned_anchors": int(len(np.unique(planned))),
                "viable_anchors": int(len(np.unique(anchors[keep & (groups == group)]))),
                "attempted_anchors": int(np.sum(attempted & (groups == group))), "quota": quota}
        after = self._state_hash()
        record.update(final_coordinates=final, labels=labels, anchor_indices=anchors,
            partner_indices=_array(candidates.partner_indices).copy(),
            relocation_indices=_array(candidates.relocation_indices).copy(), group_ids=groups,
            keep=keep, attempted_selected=attempted, selected=np.zeros(len(final), dtype=bool),
            rejection_reasons=reasons, requested_budget=int(allocation_record["requested"]),
            group_quotas=quotas, group_coverage=coverage, attempted=int(attempted.sum()),
            attempted_group_shortfalls={g: max(0, q - int(np.sum(attempted & (groups == g)))) for g, q in quotas.items()},
            anchor_coverage={"planned": int(len(np.unique(record["planned_anchor_indices"]))),
                "viable": int(len(np.unique(anchors[keep]))), "attempted": int(attempted.sum())},
            real_matches=real_matches, real_match_count=int(real_matches.sum()),
            pool_duplicate_count=int(np.sum(reasons == "pool_duplicate")),
            accumulated_duplicate_count=int(np.sum(reasons == "accumulated_duplicate")),
            selection_seconds=perf_counter() - tick, state_after_hash=after,
            downstream_state_after_hash=after)
        if self.proposal_generator is not None:
            displacement = record["nonlinearity_displacement"][record["geometry_valid"]]
            record.update(invalid_geometry_count=int((~record["geometry_valid"]).sum()),
                invalid_geometry_reasons={str(reason): int(np.sum(record["geometry_reasons"] == reason))
                    for reason in np.unique(record["geometry_reasons"][~record["geometry_valid"]])},
                raw_decode_negative_count=int(record["raw_decode_negative"].sum()),
                raw_decode_rate_failure_count=int(record["raw_decode_rate_failure"].sum()),
                relocation_count=int(np.sum(record["relocation_indices"] >= 0)),
                nonlinearity_displacement_mean=float(displacement.mean()) if len(displacement) else None,
                nonlinearity_displacement_max=float(displacement.max()) if len(displacement) else None)
        self.rounds.append(record)
        self._pending = None
        self._awaiting_acceptance = True
        mask = torch.as_tensor(attempted, device=candidates.X.device)
        return type(candidates)(*(getattr(candidates, key)[mask] for key in
            ("X", "y", "seed_indices", "partner_indices", "relocation_indices")))

    def finalize_acceptance(self, keep, scores=None):
        """Map chosen-candidate decisions to the full pool; never rerank or refill."""
        if not self._awaiting_acceptance:
            raise RuntimeError("Select a pool before finalizing acceptance")
        tick = perf_counter()
        row = self.rounds[-1]
        keep = np.asarray(_array(keep), dtype=bool)
        positions = np.flatnonzero(row["attempted_selected"])
        if keep.shape != (len(positions),):
            raise ValueError("Acceptance must align with the attempted random subset")
        if scores is not None:
            for key, values in scores.items():
                values = np.asarray(_array(values))
                if values.shape != (len(positions),):
                    raise ValueError("F1 scores must align with the attempted random subset")
            if not np.array_equal(keep, scores["accepted"]):
                raise ValueError("Acceptance mask must match F1 decisions")
            for key, values in scores.items():
                row["f1_" + key] = np.asarray(_array(values)).copy()
        elif not keep.all():
            raise ValueError("Plain random must accept every attempted candidate")
        after = self._state_hash()
        selected = row["selected"]
        selected[positions] = keep
        row["rejection_reasons"][positions] = (np.asarray(scores["reasons"]) if scores is not None else "accepted")
        self._accepted_coordinates.update(_coordinate_key(point) for point in row["final_coordinates"][selected])
        row.update(accepted=int(selected.sum()), rejected=int(len(keep) - keep.sum()),
            shortfall=int(row["requested_budget"] - selected.sum()),
            group_shortfalls={g: max(0, q - int(np.sum(selected & (row["group_ids"] == g))))
                for g, q in row["group_quotas"].items()},
            acceptance_seconds=perf_counter() - tick, state_after_hash=after,
            downstream_state_after_hash=after)
        row["anchor_coverage"]["selected"] = int(selected.sum())
        for group, coverage in row["group_coverage"].items():
            coverage["selected_anchors"] = int(np.sum(selected & (row["group_ids"] == group)))
        self._awaiting_acceptance = False

    def summary(self):
        """JSON-safe support metadata; full pool arrays remain in rounds."""
        keys = ("round", "requested_budget", "planned_budget", "attempted", "accepted", "rejected", "shortfall",
            "group_quotas", "attempted_group_shortfalls", "group_shortfalls", "anchor_coverage", "group_coverage",
            "real_match_count", "pool_duplicate_count", "accumulated_duplicate_count",
            "generation_seconds", "selection_seconds", "acceptance_seconds", "native_anchor_scoring_seconds",
            "state_before_hash", "state_after_hash", "downstream_state_before_hash", "downstream_state_after_hash")
        if self.proposal_generator is not None:
            keys += ("generation_geometry", "invalid_geometry_count", "invalid_geometry_reasons",
                "raw_decode_negative_count", "raw_decode_rate_failure_count", "relocation_count",
                "nonlinearity_displacement_mean", "nonlinearity_displacement_max", "inverse_seconds", "decode_seconds", "geometry_generation_seconds")
        return {"selector": "random", "k": self.k,
            "dedup": "exact float32 final coordinates; signed zero equal; first pool owner; accepted synthetics only",
            "real_matches_policy": "eligible", "acceptance_order": "choose random subset then optional F1; no rerank or refill",
            "native_anchor_scoring_seconds_total": sum(row["native_anchor_scoring_seconds"] for row in self.checkpoints),
            "rounds": [{key: row[key] for key in keys} for row in self.rounds]}
