from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from time import perf_counter

import numpy as np
import torch
from scipy.stats import f_oneway


@dataclass(frozen=True)
class UGOConfig:
    lambda_select: float = 0.5
    lambda_step: float = 0.1
    k_pre: int = 5
    k_after: int = 1
    phi: float = 2.5
    alpha: float = 0.05
    max_iterations: int = 20
    random_state: int = 42
    device: str = "auto"
    categorical_columns: tuple[int, ...] = ()
    distance_batch_size: int = 512

    def __post_init__(self):
        if not 0 < self.lambda_select <= 1 or self.lambda_step <= 0 or self.phi <= 0:
            raise ValueError("Selection must be in (0, 1]; step and phi must be positive.")
        if min(self.k_pre, self.k_after, self.max_iterations, self.distance_batch_size) < 1:
            raise ValueError("Neighbor counts, iteration cap and distance batch size must be positive.")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1].")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda.")


@dataclass
class RealSupport:
    X: torch.Tensor
    y: torch.Tensor
    classes: torch.Tensor
    eligible: torch.Tensor
    partners: dict[int, torch.Tensor]
    group_indices: dict[int, torch.Tensor]
    excluded_groups: list[int]


@dataclass
class GroupBudget:
    target_class: int
    seed_indices: torch.Tensor
    count: int


@dataclass
class Candidates:
    X: torch.Tensor
    y: torch.Tensor
    seed_indices: torch.Tensor
    partner_indices: torch.Tensor
    relocation_indices: torch.Tensor


@dataclass
class UGOResult:
    X: torch.Tensor
    y: torch.Tensor
    rounds: list[dict]
    provenance: list[dict]
    stop_reason: str
    config: UGOConfig
    support_summary: dict = field(default_factory=dict)


def nearest_indices(query, reference, k, batch_size=512, exclude_indices=None, exclude_zero=False):
    """Exact, bounded-memory Euclidean neighbors on the tensors' CPU/CUDA device."""
    if reference.device != query.device:
        raise ValueError("Query and reference must use the same device.")
    k = min(k, len(reference) - (exclude_indices is not None))
    if k <= 0:
        return torch.empty((len(query), 0), dtype=torch.long, device=query.device)
    result = torch.empty((len(query), k), dtype=torch.long, device=query.device)
    # Bound the temporary distance matrix even for a large reference partition.
    chunk = min(batch_size, max(1, 1_000_000 // len(reference)))
    for start in range(0, len(query), chunk):
        end = min(start + chunk, len(query))
        distances = torch.cdist(query[start:end], reference, compute_mode="donot_use_mm_for_euclid_dist")
        if exclude_zero:
            distances.masked_fill_(distances == 0, torch.inf)
        if exclude_indices is not None:
            distances[torch.arange(end - start, device=query.device), exclude_indices[start:end]] = torch.inf
        result[start:end] = distances.topk(k, dim=1, largest=False, sorted=True).indices
    return result


def _majority(neighbor_labels, class_count):
    counts = torch.zeros((len(neighbor_labels), class_count), dtype=torch.long, device=neighbor_labels.device)
    counts.scatter_add_(1, neighbor_labels, torch.ones_like(neighbor_labels))
    return counts.argmax(dim=1)


def noise_pre_detect(X: torch.Tensor, y: torch.Tensor, k: int = 5, batch_size=512) -> torch.Tensor:
    if len(X) < 2:
        return torch.zeros(len(X), dtype=torch.bool, device=X.device)
    indices = nearest_indices(X, X, k, batch_size, torch.arange(len(X), device=X.device))
    return _majority(y[indices], int(y.max()) + 1) != y


def calculate_umma(probabilities: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """True-label modified margin difficulty, not an epistemic uncertainty estimate."""
    true_probabilities = probabilities.gather(1, y.unsqueeze(1)).squeeze(1)
    competing = probabilities.clone()
    competing.scatter_(1, y.unsqueeze(1), -torch.inf)
    return competing.max(dim=1).values - true_probabilities


def _welch_anova_p_value(uncertainty_groups: list[np.ndarray]) -> float:
    if any(len(group) < 2 for group in uncertainty_groups):
        return 0.0
    constant = [np.allclose(group, group[0]) for group in uncertainty_groups]
    if all(constant):
        means = np.array([group[0] for group in uncertainty_groups])
        return 1.0 if np.allclose(means, means[0]) else 0.0
    # Welch's statistic has no defined inverse-variance weights for a constant group.
    # This is not evidence of equal difficulty; continue under the fixed cap.
    if any(constant):
        return 0.0
    _, p_value = f_oneway(*uncertainty_groups, equal_var=False)
    return 0.0 if not np.isfinite(p_value) else float(p_value)


def predict_probabilities(model, X: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "predict_proba_tensor"):
        return model.predict_proba_tensor(X).to(device=X.device, dtype=X.dtype)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X.detach().cpu().numpy())
        return torch.as_tensor(probabilities, dtype=X.dtype, device=X.device)
    model.eval()
    parameter = next(model.parameters())
    with torch.inference_mode():
        return torch.softmax(model(X.to(device=parameter.device, dtype=parameter.dtype)), dim=1).to(X.device)


def build_support(X, y, config):
    """Class-level support baseline; references and interpolation partners are real only."""
    classes = torch.unique(y, sorted=True)
    eligible = ~noise_pre_detect(X, y, config.k_pre, config.distance_batch_size)
    partners, groups, excluded_groups = {}, {}, []
    for label in classes.tolist():
        indices = torch.nonzero((y == label) & eligible, as_tuple=True)[0]
        if len(indices) < 2 or len(torch.unique(X[indices], dim=0)) < 2:
            eligible[indices] = False
            excluded_groups.append(label)
            continue
        groups[label] = indices
        local_neighbors = nearest_indices(X[indices], X[indices], min(config.k_pre, len(indices) - 1),
                                          config.distance_batch_size, torch.arange(len(indices), device=X.device), exclude_zero=True)
        partners[label] = indices[local_neighbors]
    return RealSupport(X, y, classes, eligible, partners, groups, excluded_groups)


def score_seeds(model, X, y):
    return calculate_umma(predict_probabilities(model, X), y)


def allocate_group_budget(support, scores, remaining, config):
    if not support.group_indices or remaining <= 0:
        return None
    target = max(support.group_indices, key=lambda label: float(scores[support.group_indices[label]].mean()))
    indices = support.group_indices[target]
    count = max(1, int(len(indices) * config.lambda_select))
    seeds = indices[torch.argsort(scores[indices], descending=True, stable=True)[:count]]
    return GroupBudget(target, seeds, min(remaining, max(1, round(len(support.y) * config.lambda_step))))


def propose_interpolation(support, budget, rng, config, *, seed_indices=None):
    device = support.X.device
    seeds = (budget.seed_indices[torch.as_tensor(rng.integers(len(budget.seed_indices), size=budget.count), device=device)]
             if seed_indices is None else seed_indices)
    group = support.group_indices[budget.target_class]
    local_seed = torch.searchsorted(group, seeds)
    options = support.partners[budget.target_class][local_seed]
    # Distinct real observations with identical coordinates are not interpolation partners.
    valid = (support.X[options] != support.X[seeds, None]).any(dim=2)
    valid_count = valid.sum(dim=1)
    keep = valid_count > 0
    seeds, options, valid, valid_count = seeds[keep], options[keep], valid[keep], valid_count[keep]
    if len(seeds):
        choice = torch.as_tensor(rng.random(len(seeds)), device=device) * valid_count
        positions = (valid.cumsum(dim=1) > choice[:, None]).long().argmax(dim=1)
        partners = options[torch.arange(len(seeds), device=device), positions]
        interpolation = torch.as_tensor(rng.random((len(seeds), 1)), dtype=support.X.dtype, device=device)
        generated = support.X[seeds] + interpolation * (support.X[partners] - support.X[seeds])
        if config.categorical_columns:
            columns = list(config.categorical_columns)
            generated[:, columns] = support.X[partners][:, columns]
    else:
        partners = seeds.clone()
        generated = support.X[:0]
    return Candidates(generated, torch.full((len(seeds),), budget.target_class, dtype=support.y.dtype, device=device),
                      seeds, partners, torch.full_like(seeds, -1))


def accept_relocation(support, candidates, config):
    if not len(candidates.X):
        return candidates
    nearest = nearest_indices(candidates.X, support.X, config.k_after, config.distance_batch_size)
    conflict = _majority(support.y[nearest], len(support.classes)) != candidates.y
    if conflict.any():
        target_indices = support.group_indices[int(candidates.y[0])]
        closest = nearest_indices(candidates.X[conflict], support.X[target_indices], 1, config.distance_batch_size)[:, 0]
        relocated = target_indices[closest]
        candidates.X[conflict] = support.X[relocated]
        candidates.relocation_indices[conflict] = relocated
    return candidates


def retrain_model(model_fn, X, y):
    return model_fn(X, y)


def stop_generation(p_value, generated, cap, config):
    if p_value > config.alpha:
        return "class_difficulty_balance"
    if generated >= cap:
        return "generation_cap"
    return None


class UGOPipeline:
    """Independent support → score → budget → propose → accept → retrain → stop stages.

    Callbacks use the public functions' contracts. Class-level allocation and original
    relocation are baselines, not the later group-protection/constraint experiments.
    """
    def __init__(self, config=None, *, support=build_support, seed_score=score_seeds,
                 group_budget=allocate_group_budget, propose=propose_interpolation,
                 accept=accept_relocation, retrain=retrain_model, stop=stop_generation):
        self.config = config or UGOConfig()
        self.support = support
        self.seed_score = seed_score
        self.group_budget = group_budget
        self.propose = propose
        self.accept = accept
        self.retrain = retrain
        self.stop = stop

    def run(self, X, y, model_fn: Callable, *, generation_cap=None):
        config = self.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
        X = X.detach().to(device=device, dtype=torch.float32)
        y = y.detach().to(device=device, dtype=torch.long)
        classes = torch.unique(y, sorted=True)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(classes) < 2:
            raise ValueError("UGO requires a feature matrix and at least two classes.")
        if not torch.equal(classes, torch.arange(len(classes), device=device)):
            raise ValueError("Labels must be consecutive integers starting at zero.")
        if not torch.isfinite(X).all():
            raise ValueError("UGO requires finite, training-preprocessed features.")
        support = self.support(X, y, config)
        cap = max(0, int(config.phi * int(torch.bincount(y).max()) * len(classes)) - len(y))
        if generation_cap is not None:
            cap = min(cap, generation_cap)
        summary = {"real_count": len(y), "eligible_seeds": int(support.eligible.sum()),
                   "excluded_seeds": int((~support.eligible).sum()), "excluded_groups": support.excluded_groups,
                   "generation_cap": cap, "device": str(device), "configuration": asdict(config)}
        X_current, y_current = X, y
        rounds, provenance = [], []
        rng = np.random.default_rng(config.random_state)
        descendants = torch.zeros(len(y), dtype=torch.long, device=device)
        generated = 0
        stop_reason = "iteration_cap"
        if not support.group_indices:
            return UGOResult(X, y, rounds, provenance, "no_eligible_group", config, summary)
        for iteration in range(1, config.max_iterations + 1):
            if generated >= cap:
                stop_reason = "generation_cap"
                break
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = perf_counter()
            model = self.retrain(model_fn, X_current, y_current)
            scores = self.seed_score(model, X, y)
            groups = [scores[(y == label) & support.eligible].detach().cpu().numpy() for label in classes]
            p_value = _welch_anova_p_value(groups)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            fit_seconds = perf_counter() - start
            row = {"round": iteration, "p_value": p_value, "excluded_seeds": summary["excluded_seeds"],
                   "excluded_groups": summary["excluded_groups"], "proposed": 0, "accepted": 0, "relocated": 0,
                   "unique_seeds": 0, "duplicates": 0, "retrains": iteration, "fit_score_seconds": fit_seconds}
            row["class_mean_difficulty"] = {str(int(label)): float(group.mean()) if len(group) else None
                                             for label, group in zip(classes.tolist(), groups)}
            row["real_group_support"] = {str(int(label)): len(group)
                                         for label, group in zip(classes.tolist(), groups)}
            stop_reason_now = self.stop(p_value, generated, cap, config)
            if stop_reason_now:
                row.update(stop_reason=stop_reason_now, total_generated=generated, unspent_budget=cap-generated)
                rounds.append(row)
                stop_reason = stop_reason_now
                break
            budget = self.group_budget(support, scores, cap - generated, config)
            if budget is None:
                stop_reason = "no_eligible_group"
                row["stop_reason"] = stop_reason
                rounds.append(row)
                break
            start = perf_counter()
            candidates = self.propose(support, budget, rng, config)
            proposed = len(candidates.X)
            candidates = self.accept(support, candidates, config)
            accepted = len(candidates.X)
            if not accepted:
                stop_reason = "no_valid_partner"
                row.update(proposed=proposed, stop_reason=stop_reason)
                rounds.append(row)
                break
            old_unique = len(torch.unique(X_current, dim=0))
            offset = len(y_current)
            X_current = torch.cat((X_current, candidates.X))
            y_current = torch.cat((y_current, candidates.y))
            duplicates = accepted - (len(torch.unique(X_current, dim=0)) - old_unique)
            descendants.scatter_add_(0, candidates.seed_indices, torch.ones_like(candidates.seed_indices))
            for index, (seed, partner, relocation) in enumerate(zip(candidates.seed_indices.tolist(),
                    candidates.partner_indices.tolist(), candidates.relocation_indices.tolist())):
                provenance.append({"row_index": offset + index, "round": iteration, "seed_index": seed,
                                   "partner_index": partner, "relocation_index": relocation, "class": budget.target_class})
            generated += accepted
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row.update(target_class=budget.target_class, proposed=proposed, accepted=accepted,
                       relocated=int((candidates.relocation_indices >= 0).sum()), unique_seeds=len(torch.unique(candidates.seed_indices)),
                       duplicates=duplicates, total_generated=generated, unspent_budget=cap-generated,
                       descendants_per_seed={str(i): int(descendants[i]) for i in torch.nonzero(descendants, as_tuple=True)[0].tolist()},
                       generation_seconds=perf_counter()-start)
            rounds.append(row)
        if generated >= cap:
            stop_reason = "generation_cap"
        return UGOResult(X_current, y_current, rounds, provenance, stop_reason, config, summary)
