"""Experimental full-parameter logistic CE selection over unchanged U2 proposals."""
from copy import deepcopy
from hashlib import sha256
from numbers import Integral
import pickle
from time import perf_counter


import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
import torch

from samplers.ugo import propose_interpolation, score_seeds


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def model_state_hash(model):
    """Include every live estimator attribute, learned or otherwise."""
    return sha256(pickle.dumps(model.__dict__, protocol=5)).hexdigest()


def balanced_weights(y):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if not np.array_equal(classes, [0, 1]):
        raise ValueError("Balanced guidance requires both binary classes 0 and 1")
    return 1. / (2. * counts[y.astype(int)])


def logistic_ce_gradients(model, X, y):
    """Per-observation gradients of unregularized CE, all weights AND bias."""
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    residual = expit(X @ model.coef_[0] + model.intercept_[0]) - y
    return residual[:, None] * X, residual


def balanced_ce_loss(model, X, y):
    z = np.asarray(X, dtype=np.float64) @ model.coef_[0] + model.intercept_[0]
    return float(balanced_weights(y) @ (np.logaddexp(0., z) - np.asarray(y) * z))


def _coordinate_key(row):
    # Equality is coordinate-only, irrespective of intended class; signed zero is equal.
    row = np.asarray(row, dtype=np.float32).copy()
    row[row == 0] = 0
    return row.tobytes()


def _correlation(function, predicted, observed):
    if len(predicted) < 2 or np.ptp(predicted) == 0 or np.ptp(observed) == 0:
        return None
    value = float(function(predicted, observed).statistic)
    return value if np.isfinite(value) else None


def select_pool_mask(final_coordinates, anchor_indices, group_ids, scores, keep,
                     group_quotas, selector, seed, round=1):
    """Rank only pre-deduplicated owners: every selector has the same attainable count."""
    if selector not in ("random", "utility"):
        raise ValueError("selector must be random or utility")
    anchors, groups, scores, keep = map(np.asarray, (anchor_indices, group_ids, scores, keep))
    if len(final_coordinates) != len(anchors):
        raise ValueError("Pool coordinates and anchors must have equal length")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1, int(round)]))
    selected = np.zeros(len(anchors), dtype=bool)
    for group, quota in group_quotas.items():
        positions = np.flatnonzero(keep & (groups == int(group)))
        representatives = []
        for anchor in np.unique(anchors[positions]):
            choices = positions[anchors[positions] == anchor]
            representative = (choices[np.argmax(scores[choices])] if selector == "utility"
                              else rng.choice(choices))
            representatives.append(int(representative))
        representatives = np.asarray(representatives, dtype=np.int64)
        order = (np.argsort(-scores[representatives], kind="stable") if selector == "utility"
                 else rng.permutation(len(representatives)))
        selected[representatives[order[:int(quota)]]] = True
    return selected


class CandidatePoolSelector:
    """At most one final-coordinate proposal per planned real anchor per round.

    Guidance is read-only, real held-out data, never interpolation support. The
    dot product predicts a hypothetical plain CE SGD step, not regularized lbfgs
    retraining. sklearn LogisticRegression has no persistent optimizer or
    normalization state: both are not applicable. All live attributes are hashed.
    """
    def __init__(self, selector, guidance_X, guidance_y, seed, *, k=5,
                 batch_size=512, verification_size=64, eta=1e-4):
        if selector not in ("random", "utility"):
            raise ValueError("selector must be random or utility")
        for name, value, minimum in (("k", k, 1), ("batch_size", batch_size, 1),
                                     ("verification_size", verification_size, 0)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError("eta must be finite and positive")
        self.selector, self.k = selector, int(k)
        self.batch_size, self.verification_size, self.eta = int(batch_size), int(verification_size), float(eta)
        self.guidance_X = np.asarray(_array(guidance_X), dtype=np.float64).copy()
        self.guidance_y = np.asarray(_array(guidance_y), dtype=np.int64).copy()
        if (self.guidance_X.ndim != 2 or self.guidance_y.ndim != 1
                or len(self.guidance_X) != len(self.guidance_y)
                or not np.isfinite(self.guidance_X).all()):
            raise ValueError("Guidance must be matching finite real feature and label arrays")
        self._guidance_weights = balanced_weights(self.guidance_y)
        self.guidance_X.flags.writeable = self.guidance_y.flags.writeable = False
        self.seed = int(seed)
        self.verification_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 2]))
        self.rounds, self.checkpoints = [], []
        self._accepted_coordinates = set()
        self._next_proposal_id = 0
        self._pending = None

    def score_seeds(self, model, X, y):
        if not isinstance(model, LogisticRegression):
            raise TypeError("Learning benefit requires the actual sklearn LogisticRegression")
        check_is_fitted(model)
        if (not np.array_equal(model.classes_, [0, 1]) or model.coef_.shape[0] != 1
                or not model.fit_intercept or model.solver != "lbfgs"):
            raise ValueError("Learning benefit requires binary lbfgs logistic regression with bias")
        if model.coef_.shape[1] != self.guidance_X.shape[1]:
            raise ValueError("Guidance and fitted model feature dimensions differ")
        diagnostic_tick = perf_counter()
        before = model_state_hash(model)
        self.live_model = model
        self.model = deepcopy(model)
        tick = perf_counter()
        # One balanced-G gradient per checkpoint, bounded memory over real guidance.
        self.vG_coefficient = np.zeros(model.coef_.shape[1], dtype=np.float64)
        self.vG_bias = 0.
        for start in range(0, len(self.guidance_y), self.batch_size):
            end = start + self.batch_size
            coefficient, bias = logistic_ce_gradients(self.model, self.guidance_X[start:end], self.guidance_y[start:end])
            weights = self._guidance_weights[start:end]
            self.vG_coefficient += weights @ coefficient
            self.vG_bias += float(weights @ bias)
        guidance_seconds = perf_counter() - tick
        baseline_tick = perf_counter()
        original_scores = score_seeds(model, X, y)
        baseline_seconds = perf_counter() - baseline_tick
        after = model_state_hash(model)
        if before != after:
            raise RuntimeError("Seed scoring mutated live logistic regression state")
        self._checkpoint = {"round": len(self.checkpoints) + 1,
            "coefficient": self.model.coef_.copy(), "intercept": self.model.intercept_.copy(),
            "vG_coefficient": self.vG_coefficient.copy(), "vG_bias": self.vG_bias,
            "state_before_hash": before, "state_after_hash": after,
            "guidance_loss": balanced_ce_loss(self.model, self.guidance_X, self.guidance_y),
            "guidance_scoring_seconds": guidance_seconds}
        self._checkpoint["guidance_scoring_seconds"] = perf_counter() - diagnostic_tick - baseline_seconds
        self.checkpoints.append(self._checkpoint)
        return original_scores

    def propose(self, real, budget, rng, cfg):
        if not hasattr(self, "model"):
            raise RuntimeError("score_seeds must freeze the current fitted model first")
        tick = perf_counter()
        planned = _array(budget.seed_indices).astype(np.int64, copy=True)
        # Keep first planned occurrence; allocator repeats are recorded, not expanded.
        _, first = np.unique(planned, return_index=True)
        anchors = planned[np.sort(first)]
        seeds = torch.as_tensor(np.repeat(anchors, self.k), device=real.X.device, dtype=torch.long)
        candidates = propose_interpolation(real, budget, rng, cfg, seed_indices=seeds)
        n = len(candidates.X)
        self._pending = {**self._checkpoint, "original_coordinates": _array(candidates.X).copy(),
            "planned_anchor_indices": planned, "planned_budget": int(budget.count),
            "proposal_ids": np.arange(self._next_proposal_id, self._next_proposal_id + n, dtype=np.int64),
            "generation_seconds": perf_counter() - tick}
        self._next_proposal_id += n
        return candidates

    def candidate_scores(self, X, y):
        X, y = _array(X), _array(y)
        values = np.empty(len(y), dtype=np.float64)
        for start in range(0, len(y), self.batch_size):
            end = start + self.batch_size
            coefficient, bias = logistic_ce_gradients(self.model, X[start:end], y[start:end])
            values[start:end] = coefficient @ self.vG_coefficient + bias * self.vG_bias
        return values

    def _verify(self, final, labels, scores):
        finite = np.flatnonzero(np.isfinite(scores) & np.isfinite(final).all(axis=1))
        sample = self.verification_rng.choice(finite, min(len(finite), self.verification_size, 64), replace=False)
        sample.sort()
        before = balanced_ce_loss(self.model, self.guidance_X, self.guidance_y)
        coefficient, bias = logistic_ce_gradients(self.model, final[sample], labels[sample])

        def diagnostic(eta):
            reductions = []
            for gw, gb in zip(coefficient, bias):
                stepped = deepcopy(self.model)
                stepped.coef_ = stepped.coef_ - eta * gw[None, :]
                stepped.intercept_ = stepped.intercept_ - eta * np.asarray([gb])
                reductions.append(before - balanced_ce_loss(stepped, self.guidance_X, self.guidance_y))
            observed = np.asarray(reductions)
            predicted = eta * scores[sample]
            error = observed - predicted
            return {"eta": eta, "sample_indices": sample.copy(), "predicted_reductions": predicted,
                "actual_reductions": observed, "spearman": _correlation(spearmanr, predicted, observed),
                "pearson": _correlation(pearsonr, predicted, observed),
                "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(observed))) if len(sample) else None,
                "mean_absolute_error": float(np.mean(np.abs(error))) if len(sample) else None,
                "max_absolute_error": float(np.max(np.abs(error))) if len(sample) else None}

        results = [diagnostic(self.eta)]
        first = results[0]
        failed = ((first["spearman"] is not None and first["spearman"] < .95)
                  or (first["sign_agreement"] is not None and first["sign_agreement"] < .95))
        if failed and self.eta != 1e-5:
            results.append(diagnostic(1e-5))
        return results

    def select(self, real, candidates, cfg, allocation_record):
        if self._pending is None:
            raise RuntimeError("propose must capture the original pool first")
        record = self._pending
        final = np.asarray(_array(candidates.X), dtype=np.float32).copy()
        labels, anchors = _array(candidates.y).copy(), _array(candidates.seed_indices).copy()
        groups_by_anchor = {int(anchor): int(trace["group_id"])
            for trace in allocation_record["group_trace"] for anchor in trace["used_seed"]}
        group_ids = np.asarray([groups_by_anchor[int(anchor)] for anchor in anchors], dtype=np.int64)
        quotas = {int(trace["group_id"]): int(trace["quota"]) for trace in allocation_record["group_trace"]}
        tick = perf_counter()
        scores = self.candidate_scores(final, labels)
        scoring_seconds = perf_counter() - tick
        # Score relocated final coordinates BEFORE applying global dedup for audit.
        tick = perf_counter()
        real_keys = {_coordinate_key(row) for row in _array(real.X)}
        pool_seen, keep = set(), np.ones(len(final), dtype=bool)
        reasons = np.full(len(final), "not_selected", dtype=object)
        real_matches = np.zeros(len(final), dtype=bool)
        for index, row in enumerate(final):
            if not np.isfinite(row).all() or not np.isfinite(scores[index]):
                keep[index], reasons[index] = False, "invalid_features"
                continue
            key = _coordinate_key(row)
            real_matches[index] = key in real_keys
            if key in pool_seen:
                keep[index], reasons[index] = False, "pool_duplicate"
            elif key in self._accepted_coordinates:
                keep[index], reasons[index] = False, "accumulated_duplicate"
            pool_seen.add(key)
        selected = select_pool_mask(final, anchors, group_ids, scores, keep, quotas,
                                    self.selector, self.seed, record["round"])
        alternative = "random" if self.selector == "utility" else "utility"
        counterfactual_tick = perf_counter()
        alternative_mask = select_pool_mask(final, anchors, group_ids, scores, keep, quotas,
                                            alternative, self.seed, record["round"])
        counterfactual_seconds = perf_counter() - counterfactual_tick
        masks = {self.selector: selected, alternative: alternative_mask}
        group_shortfalls, group_coverage = {}, {}
        for group, quota in quotas.items():
            viable = np.unique(anchors[keep & (group_ids == group)])
            chosen = np.flatnonzero(selected & (group_ids == group))
            group_shortfalls[group] = max(0, quota - len(chosen))
            group_coverage[group] = {"planned_anchors": int(len(np.unique(record["planned_anchor_indices"][
                np.asarray([groups_by_anchor[int(a)] == group for a in record["planned_anchor_indices"]], dtype=bool)]))),
                "viable_anchors": len(viable), "selected_anchors": len(chosen), "quota": quota}
        reasons[selected] = "selected"
        self._accepted_coordinates.update(_coordinate_key(row) for row in final[selected])
        selection_seconds = perf_counter() - tick - counterfactual_seconds
        tick = perf_counter()
        verification = self._verify(final, labels, scores)
        verification_seconds = perf_counter() - tick
        after = model_state_hash(self.live_model)
        if after != record["state_before_hash"]:
            raise RuntimeError("Candidate scoring/verification mutated live logistic regression state")
        record.update(final_coordinates=final, labels=labels, anchor_indices=anchors,
            partner_indices=_array(candidates.partner_indices).copy(),
            relocation_indices=_array(candidates.relocation_indices).copy(), group_ids=group_ids,
            scores=scores, keep=keep, selected=selected, rejection_reasons=reasons,
            random_selected=masks["random"], utility_selected=masks["utility"],
            requested_budget=int(allocation_record["requested"]), group_quotas=quotas,
            group_shortfalls=group_shortfalls, accepted=int(selected.sum()),
            shortfall=int(allocation_record["requested"]) - int(selected.sum()),
            anchor_coverage={"planned": len(np.unique(record["planned_anchor_indices"])),
                "viable": len(np.unique(anchors[keep])), "selected": len(np.unique(anchors[selected]))},
            group_coverage=group_coverage, real_matches=real_matches,
            real_match_count=int(real_matches.sum()), pool_duplicate_count=int(np.sum(reasons == "pool_duplicate")),
            accumulated_duplicate_count=int(np.sum(reasons == "accumulated_duplicate")),
            positive_selected_count=int(np.sum(scores[selected] > 0)),
            negative_selected_count=int(np.sum(scores[selected] < 0)),
            zero_selected_count=int(np.sum(scores[selected] == 0)), scoring_seconds=scoring_seconds,
            selection_seconds=selection_seconds, verification_seconds=verification_seconds,
            counterfactual_selection_seconds=counterfactual_seconds,
            diagnostic_seconds=verification_seconds + counterfactual_seconds +
                (scoring_seconds + record["guidance_scoring_seconds"] if self.selector == "random" else 0.),
            diagnostic_only_random_seconds=(scoring_seconds + record["guidance_scoring_seconds"] + verification_seconds
                                            + counterfactual_seconds if self.selector == "random" else 0.),
            verification=verification, state_after_hash=after)
        self.rounds.append(record)
        self._pending = None
        mask = torch.as_tensor(selected, device=candidates.X.device)
        return type(candidates)(*(getattr(candidates, key)[mask] for key in
            ("X", "y", "seed_indices", "partner_indices", "relocation_indices")))

    def summary(self):
        """JSON-safe small support metadata; full audit pools live only on rounds."""
        scalar_keys = ("round", "requested_budget", "planned_budget", "accepted", "shortfall",
            "group_quotas", "group_shortfalls", "anchor_coverage", "group_coverage", "real_match_count",
            "pool_duplicate_count", "accumulated_duplicate_count", "positive_selected_count",
            "negative_selected_count", "zero_selected_count", "generation_seconds", "selection_seconds",
            "scoring_seconds", "guidance_scoring_seconds", "verification_seconds", "diagnostic_only_random_seconds",
            "counterfactual_selection_seconds", "diagnostic_seconds",
            "state_before_hash", "state_after_hash", "guidance_loss")
        return {"selector": self.selector, "k": self.k, "eta": self.eta,
            "guidance_count": len(self.guidance_y), "guidance_class_counts": np.bincount(self.guidance_y).tolist(),
            "guidance_scoring_seconds_total": sum(row["guidance_scoring_seconds"] for row in self.checkpoints),
            "gradient": "full coefficient and bias unregularized binary CE; balanced real guidance mean",
            "score": "vG dot v_candidate, neither normalized nor influence",
            "optimizer_state": "not applicable: sklearn lbfgs has no persistent optimizer state",
            "normalization_state": "not applicable: preprocessing is external and fixed",
            "verification_claim": "hypothetical plain CE SGD only; no guarantee after regularized lbfgs retraining",
            "random_scoring": "diagnostic-only overhead, excluded from operational random runtime",
            "dedup": "exact float32 final coordinates; first proposal owns pool duplicates; accumulated synthetic excluded",
            "real_matches_policy": "diagnostic only; baseline relocation matches remain eligible",
            "rounds": [{key: row[key] for key in scalar_keys} for row in self.rounds]}
