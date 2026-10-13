"""Paper-derived numeric UGOSMOTE reference, NOT an executable author reproduction.

Audit: local 1-s2.0-S095070512600612X-main.pdf, Sağlam & Cengiz,
Knowledge-Based Systems 342 (2026) 115886, §§3.1–3.4, Tables 2–4.
Author code was not located in the paper or targeted public search.

Resolved ambiguities (not claims of exact author behavior):
* §3.1 explicitly retrains on augmented observations, whereas Table 4 leaves
  X,Y unchanged within the loop. Use the prose: descendants participate in
  in-sample scoring, neighbor support, interpolation and relocation.
* §3.3 cap is phi*max(initial class count)*K-n; Table 4 instead says
  phi*max(n_l). Use §3.3 with initial counts and clip the last batch.
* NoisePreDetect runs once (Table 4); generated/relocated rows are admitted
  as non-noisy. No real-only support restriction is imposed.
* R is a pluggable sampler; this implementation is UGOSMOTE: top-ranked
  seeds interpolate to k nearest current same-class observations. The paper
  does not specify neighbor-pool restriction or integer rounding: floor
  n_select and n_step, minimum one, exclude self, stable lower-class ties.
* Relocation copies the nearest non-noisy same-class observation, as §3.4
  specifies (not halfway interpolation). Undefined Welch tests halt with an
  explicit reason rather than treating an undefined statistic as evidence.

Operational max_iterations and optional target_counts are protocol deviations.
Optional categorical_projection copies the partner's categorical coordinates:
a controlled adapter deviation, disabled in faithful numeric mode. Numeric
mode can produce invalid categories; it must never be presented as type-safe.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.stats import f_oneway

from samplers.ugo import UGOResult, calculate_umma, nearest_indices, predict_probabilities


@dataclass(frozen=True)
class OriginalUGOConfig:
    lambda_select: float = .5
    lambda_step: float = .1
    k_pre: int = 5
    k_after: int = 1
    k_neighbors: int = 5
    phi: float = 2.5
    alpha: float = .05
    max_iterations: int = 20
    random_state: int = 42
    device: str = "auto"
    categorical_columns: tuple = ()
    categorical_projection: bool = False
    distance_batch_size: int = 512

    def __post_init__(self):
        if not 0 < self.lambda_select <= 1 or self.lambda_step <= 0 or self.phi <= 0:
            raise ValueError("Invalid UGO selection, step or phi")
        if min(self.k_pre, self.k_after, self.k_neighbors, self.max_iterations, self.distance_batch_size) < 1:
            raise ValueError("Neighbor counts and iteration cap must be positive")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0,1]")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Invalid UGO device")


def original_ugo(X, y, model_fn, config=None, target_counts=None):
    config = config or OriginalUGOConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
    X = torch.as_tensor(X, dtype=torch.float32, device=device).clone()
    y = torch.as_tensor(y, dtype=torch.long, device=device).clone()
    classes, counts = torch.unique(y, sorted=True, return_counts=True)
    if X.ndim != 2 or y.ndim != 1 or len(y) != len(X) or len(classes) < 2 or not bool(torch.isfinite(X).all()):
        raise ValueError("Original UGO requires finite features and at least two classes")
    if not torch.equal(classes, torch.arange(len(classes), device=device)):
        raise ValueError("Original UGO requires consecutive integer labels starting at zero")
    if len(y) <= config.k_pre or len(y) < config.k_after:
        raise ValueError("Original UGO has insufficient observations for requested noise neighbors")
    initial_n = len(y)
    cap = max(0, int(np.floor(config.phi * int(counts.max()) * len(classes) - initial_n)))
    targets = None if target_counts is None else {int(k): int(v) for k, v in target_counts.items()}
    if targets is not None:
        if set(targets) - set(classes.tolist()) or any(targets.get(int(c), int(n)) < int(n) for c, n in zip(classes, counts)):
            raise ValueError("target_counts must refer to observed classes and cannot remove rows")
        cap = min(cap, sum(targets.get(int(c), int(n)) - int(n) for c, n in zip(classes, counts)))
    own = torch.arange(initial_n, device=device)
    neighbors = nearest_indices(X, X, config.k_pre, config.distance_batch_size, own)
    eligible = torch.stack([(y[row] == torch.bincount(y[ids], minlength=len(classes)).argmax()) for row, ids in enumerate(neighbors)])
    rounds, provenance = [], []
    rng = np.random.default_rng(config.random_state)
    step = max(1, int(np.floor(initial_n * config.lambda_step)))
    generated = 0
    stop = "iteration_cap"
    for iteration in range(1, config.max_iterations + 1):
        if generated >= cap:
            stop = "generation_cap"
            break
        model = model_fn(X, y)
        scores = calculate_umma(predict_probabilities(model, X), y)
        groups = [torch.where(eligible & (y == c))[0] for c in classes]
        values = [scores[g].detach().cpu().numpy() for g in groups]
        row = {"round": iteration, "training_count": len(y), "eligible_count": int(eligible.sum()), "synthetic_support_count": int(eligible[initial_n:].sum()), "class_mean_difficulty": {int(c): float(v.mean()) if len(v) else None for c, v in zip(classes, values)}, "generated": 0}
        if any(len(v) < 2 for v in values) or any(float(np.var(v)) == 0 for v in values):
            stop = "undefined_welch_test"
            row["stop_reason"] = stop
            rounds.append(row)
            break
        p_value = float(f_oneway(*values, equal_var=False).pvalue)
        row["p_value"] = p_value if np.isfinite(p_value) else None
        if not np.isfinite(p_value) or p_value > config.alpha:
            stop = "undefined_welch_test" if not np.isfinite(p_value) else "class_difficulty_balance"
            row["stop_reason"] = stop
            rounds.append(row)
            break
        available = [int(c) for c in classes if targets is None or int((y == c).sum()) < targets.get(int(c), int((y == c).sum()))]
        if not available:
            stop = "target_counts_reached"
            break
        c = max(available, key=lambda label: float(scores[groups[label]].mean()))
        group = groups[c]
        selected_n = max(1, int(np.floor(len(group) * config.lambda_select)))
        selected = group[torch.argsort(scores[group], descending=True, stable=True)[:selected_n]]
        all_class = torch.where(y == c)[0]
        if len(all_class) <= config.k_neighbors:
            raise ValueError(f"Original UGOSMOTE class {c} needs more than {config.k_neighbors} observations")
        # The current pool includes accepted descendants; self is excluded.
        options = nearest_indices(X[selected], X[all_class], config.k_neighbors, config.distance_batch_size, torch.searchsorted(all_class, selected))
        count = min(step, cap - generated)
        if targets is not None:
            count = min(count, targets.get(c, int((y == c).sum())) - int((y == c).sum()))
        seed_local = rng.integers(len(selected), size=count)
        seeds = selected[torch.as_tensor(seed_local, device=device)]
        partner_local = rng.integers(config.k_neighbors, size=count)
        partners = all_class[options[torch.as_tensor(seed_local, device=device), torch.as_tensor(partner_local, device=device)]]
        fraction = torch.as_tensor(rng.random((count, 1)), dtype=X.dtype, device=device)
        candidates = X[seeds] + fraction * (X[partners] - X[seeds])
        if config.categorical_projection and config.categorical_columns:
            columns = list(config.categorical_columns)
            candidates[:, columns] = X[partners][:, columns]
        nearby = nearest_indices(candidates, X, config.k_after, config.distance_batch_size)
        majority = torch.stack([torch.bincount(y[ids], minlength=len(classes)).argmax() for ids in nearby])
        conflict = majority != c
        relocation = torch.full((count,), -1, device=device, dtype=torch.long)
        if bool(conflict.any()):
            nearest = nearest_indices(candidates[conflict], X[group], 1, config.distance_batch_size)[:, 0]
            relocation[conflict] = group[nearest]
            candidates[conflict] = X[relocation[conflict]]
        offset = len(y)
        for index in range(count):
            provenance.append({"row_index": offset + index, "round": iteration, "class": c, "seed_index": int(seeds[index]), "partner_index": int(partners[index]), "relocation_index": int(relocation[index]), "seed_is_synthetic": bool(seeds[index] >= initial_n), "partner_is_synthetic": bool(partners[index] >= initial_n), "categorical_projection": config.categorical_projection})
        X = torch.cat((X, candidates))
        y = torch.cat((y, torch.full((count,), c, device=device, dtype=y.dtype)))
        eligible = torch.cat((eligible, torch.ones(count, device=device, dtype=torch.bool)))
        generated += count
        row.update(target_class=c, selected_count=selected_n, generated=count, relocated=int(conflict.sum()), total_generated=generated)
        rounds.append(row)
    if generated >= cap:
        stop = "generation_cap"
    deviations = ["operational_iteration_cap", "paper_prose_augmented_retraining_resolution", "section_3_3_initial_count_cap_resolution", "floor_rounding_and_self_exclusion_resolution"]
    if targets is not None:
        deviations.append("matched_per_class_target_caps")
    if config.categorical_projection:
        deviations.append("categorical_partner_projection")
    summary = {"implementation": "paper_derived_numeric_UGOSMOTE", "author_reproduction_status": "unresolved", "audit_source": "1-s2.0-S095070512600612X-main.pdf §§3.1–3.4 Tables 2–4", "real_count": initial_n, "generation_cap": cap, "target_counts": targets, "actual_counts": {int(c): int((y == c).sum()) for c in classes}, "protocol_deviations": deviations, "configuration": asdict(config)}
    return UGOResult(X, y, rounds, provenance, stop, config, summary)
