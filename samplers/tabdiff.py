"""Frozen, train-only official TabDiff with joint-label feature inpainting.

Loads only the four pinned upstream architecture/diffusion modules, never the
upstream dataset protocol, trainer, metrics, main entrypoint or warning filters.
The RePaint x_t adaptation below corrects the anchor-centered missing-coordinate
prior and conditions known coordinates at each actual denoiser evaluation time.
Numeric inverse-quantile outputs are raw floats: no count rounding, bounds repair,
TAP policy, classifier preprocessing, CFG model or checkpoint selection.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
from numbers import Integral
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.preprocessing import QuantileTransformer
import torch


REFERENCE_COMMIT = "5ecdb3356261aea72716cc9a779f31d7ad083bf4"
IMPLEMENTATION_VERSION = "official_tabdiff_frozen_xt_v1"
REFERENCE_ROOT = Path(__file__).resolve().parents[1] / ".cache/references/TabDiff"
SOURCE_CHECKSUMS = {
    "tabdiff/modules/transformer.py": "f3197ebb0af083c25b5714ec81520b141d8cec5d7e2bc3fcb9d1c577e3748382",
    "tabdiff/modules/main_modules.py": "9d9ee6fb344cffdbf5f20c632b535c9df82e9537a429d1eb5fd7152a3e07f29d",
    "tabdiff/models/noise_schedule.py": "50006962741c2922bb6671aaceb0d743e0dfb93545f1c953b2fd7c0f6b718ee7",
    "tabdiff/models/unified_ctime_diffusion.py": "aa341c5c31e02fbe0d878257fdf8e05f820f2eff7ef41a9b6334ad2929a89721",
}


@dataclass(frozen=True)
class TabDiffConfig:
    epochs: int = 8000
    batch_size: int = 4096
    learning_rate: float = .001
    ema_decay: float = .997
    num_steps: int = 50
    hidden_dim: int = 1024
    num_layers: int = 2
    d_token: int = 4
    factor: int = 32
    lr_factor: float = .90
    lr_patience: int = 50

    def __post_init__(self):
        for name in ("epochs", "batch_size", "num_steps", "hidden_dim", "num_layers", "d_token", "factor", "lr_patience"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size < 2 or self.num_steps < 2 or self.hidden_dim < 4 or self.hidden_dim % 2:
            raise ValueError("batch_size and num_steps must be >=2; hidden_dim must be even and >=4")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0 < self.ema_decay < 1 or not 0 < self.lr_factor < 1:
            raise ValueError("ema_decay and lr_factor must be between zero and one")


def _official_modules():
    for relative, expected in SOURCE_CHECKSUMS.items():
        source = REFERENCE_ROOT / relative
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Missing or modified official TabDiff source: {source}; require commit {REFERENCE_COMMIT}")
    if str(REFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(REFERENCE_ROOT))
    modules = importlib.import_module("tabdiff.modules.main_modules")
    diffusion = importlib.import_module("tabdiff.models.unified_ctime_diffusion")
    for module in (modules, diffusion):
        if not Path(module.__file__).resolve().is_relative_to(REFERENCE_ROOT.resolve()):
            raise RuntimeError("A different tabdiff package shadows the pinned official source")
    return modules, diffusion


def _device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    result = torch.device(value)
    if result.type not in ("cpu", "cuda"):
        raise ValueError("TabDiff device must be auto, cpu or cuda")
    return result


def _seed(seed, purpose):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    return int.from_bytes(hashlib.sha256(f"tabdiff:{int(seed)}:{purpose}".encode()).digest()[:8], "little") % (2**63 - 1)


@contextmanager
def _rng(seed, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(seed)
        yield


def _scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _key(value):
    value = _scalar(value)
    return (type(value).__name__, value)


def _row_key(row):
    return tuple(_key(value) for value in row)


def _checksum(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _file_checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Preprocessing:
    def __init__(self, frame, labels, seed):
        self.columns = tuple(frame.columns)
        self.numeric = tuple(c for c in frame if is_numeric_dtype(frame[c].dtype) and not is_bool_dtype(frame[c].dtype))
        self.categorical = tuple(c for c in frame if c not in self.numeric)
        # Only observed training values, never unused pandas category levels.
        self.categories = [tuple(_scalar(v) for v in pd.unique(frame[c])) for c in self.categorical]
        self.dtypes = {c: (pd.CategoricalDtype(values, ordered=frame[c].dtype.ordered)
                           if isinstance(frame[c].dtype, pd.CategoricalDtype) else frame[c].dtype)
                       for c, values in zip(self.categorical, self.categories)}
        self.label_categories = tuple(_scalar(v) for v in pd.unique(labels))
        self.category_maps = [{_key(v): i for i, v in enumerate(values)} for values in self.categories]
        self.label_map = {_key(v): i for i, v in enumerate(self.label_categories)}
        self.quantile = None
        if self.numeric:
            self.quantile = QuantileTransformer(
                output_distribution="normal", n_quantiles=min(len(frame), max(min(len(frame) // 30, 1000), 10)),
                subsample=int(1e9), random_state=seed)
            self.quantile.fit(frame.loc[:, self.numeric].to_numpy(dtype=float))

    def encode(self, frame, labels):
        numeric = (self.quantile.transform(frame.loc[:, self.numeric].to_numpy(dtype=float))
                   if self.numeric else np.empty((len(frame), 0)))
        categories = np.empty((len(frame), len(self.categorical) + 1), dtype=np.int64)
        for j, column in enumerate(self.categorical):
            categories[:, j] = [self.category_maps[j][_key(v)] for v in frame[column]]
        categories[:, -1] = [self.label_map[_key(v)] for v in labels]
        return numeric.astype(np.float32), categories

    def decode(self, numeric, categories, index):
        frame = pd.DataFrame(index=index)
        if self.numeric:
            decoded = self.quantile.inverse_transform(numeric)
            for j, column in enumerate(self.numeric):
                frame[column] = decoded[:, j]
        for j, column in enumerate(self.categorical):
            values = [self.categories[j][int(i)] for i in categories[:, j]]
            # Restrict categorical dtype to observed training categories, preserving
            # actual bool/integer/string scalars instead of converting to strings.
            dtype = self.dtypes[column]
            if isinstance(dtype, pd.CategoricalDtype):
                dtype = pd.CategoricalDtype(self.categories[j], ordered=dtype.ordered)
            frame[column] = pd.Series(values, index=index, dtype=dtype)
        return frame.loc[:, self.columns]


def _make_diffusion(preprocessing, config, device):
    modules, upstream = _official_modules()
    counts = np.array([len(v) for v in preprocessing.categories] + [len(preprocessing.label_categories)])
    backbone = modules.UniModMLP(
        d_numerical=len(preprocessing.numeric), categories=counts + 1,
        num_layers=config.num_layers, d_token=config.d_token, n_head=1,
        factor=config.factor, bias=True, dim_t=config.hidden_dim, use_mlp=True)
    model = modules.Model(backbone, precond=True, sigma_data=1., net_conditioning="sigma")
    return upstream.UnifiedCtimeDiffusion(
        num_classes=counts, num_numerical_features=len(preprocessing.numeric),
        denoise_fn=model, y_only_model=None, num_timesteps=config.num_steps,
        scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column", noise_dist="uniform_t",
        edm_params={"precond": True, "sigma_data": 1., "net_conditioning": "sigma"},
        noise_schedule_params={"sigma_min": .002, "sigma_max": 80., "rho_init": 7., "rho_offset": 5.,
                               "eps_max": 1e-3, "eps_min": 1e-5, "k_init": -6., "k_offset": 1.},
        sampler_params={"stochastic_sampler": True, "second_order_correction": True}, device=device).to(device)


def _forward_known(diffusion, anchor_num, anchor_cat, sigma_num, sigma_cat):
    numeric = anchor_num + torch.randn_like(anchor_num) * sigma_num
    categorical, _ = diffusion.q_xt(anchor_cat, -torch.expm1(-sigma_cat))
    return numeric, categorical


@torch.no_grad()
def _sample_inpaint(diffusion, anchor_num, anchor_cat, num_missing, cat_missing):
    """Official EDM/MDLM reverse equations with actual-time q(x_t|known).

    One RePaint pass per timestep. Generated categorical coordinates use only the
    official forward churn and absorbing reverse update, not TAP remasking or
    argmax repair. At Heun's second evaluation known fields are re-noised at the
    *next* time, including the categorical label (never clean per-step clamping).
    """
    steps = diffusion.num_timesteps
    times = torch.linspace(0, 1, steps, device=diffusion.device)[:, None]
    num_sigmas = diffusion.num_schedule.total_noise(times)
    cat_sigmas = diffusion.cat_schedule.total_noise(times)
    # No anchor coordinates in the missing numerical prior.
    numeric = torch.randn_like(anchor_num) * num_sigmas[-1]
    categorical = diffusion._sample_masked_prior(len(anchor_cat), len(diffusion.num_classes))
    for i in reversed(range(steps)):
        sigma_cur, cat_cur = num_sigmas[i], cat_sigmas[i]
        sigma_next = num_sigmas[i - 1] if i else torch.zeros_like(sigma_cur)
        cat_next = cat_sigmas[i - 1] if i else torch.zeros_like(cat_cur)
        t_next = times[i - 1] if i else torch.zeros_like(times[i])
        # Same stochastic EDM churn as upstream, rejecting t_hat outside [0,1].
        if anchor_num.shape[1]:
            sigma_hat = sigma_cur * (1 + min(1 / steps, np.sqrt(2) - 1))
            t_hat = diffusion.num_schedule.inverse_to_t(sigma_hat[None]).min(dim=-1).values
            if t_hat.item() > 1:
                t_hat, sigma_hat = times[i], sigma_cur
            cat_hat = diffusion.cat_schedule.total_noise(t_hat[:, None])[0]
        else:
            t_hat, sigma_hat, cat_hat = times[i], sigma_cur, cat_cur
        numeric_hat = numeric + torch.randn_like(numeric) * (sigma_hat.square() - sigma_cur.square()).clamp_min(0).sqrt()
        categorical_hat, _ = diffusion.q_xt(categorical, -torch.expm1(cat_cur - cat_hat))
        known_num, known_cat = _forward_known(diffusion, anchor_num, anchor_cat, sigma_hat, cat_hat)
        numeric_hat = torch.where(num_missing, numeric_hat, known_num)
        categorical_hat = torch.where(cat_missing, categorical_hat, known_cat)
        batch = len(anchor_cat)
        denoised, raw_logits = diffusion._denoise_fn(
            numeric_hat, diffusion.to_one_hot(categorical_hat).float(), t_hat.reshape(1).repeat(batch),
            sigma=sigma_hat[None].expand(batch, -1))
        derivative = (numeric_hat - denoised) / sigma_hat
        numeric_next = numeric_hat + (sigma_next - sigma_hat) * derivative
        logits = diffusion._subs_parameterization(raw_logits, categorical_hat)
        categorical_next, _ = diffusion._mdlm_update(
            logits, categorical_hat, torch.exp(-cat_hat)[None].expand(batch, -1),
            torch.exp(-cat_next)[None].expand(batch, -1))
        if i and anchor_num.shape[1]:
            known_num, known_cat = _forward_known(diffusion, anchor_num, anchor_cat, sigma_next, cat_next)
            second_num = torch.where(num_missing, numeric_next, known_num)
            second_cat = torch.where(cat_missing, categorical_next, known_cat)
            denoised_next, _ = diffusion._denoise_fn(
                second_num, diffusion.to_one_hot(second_cat).float(), t_next.reshape(1).repeat(batch),
                sigma=sigma_next[None].expand(batch, -1))
            derivative_next = (second_num - denoised_next) / sigma_next
            numeric_next = numeric_hat + (sigma_next - sigma_hat) * .5 * (derivative + derivative_next)
        numeric, categorical = numeric_next, categorical_next
    # Exact known restoration is exclusively at the end; all tokens are checked
    # by the caller before categorical decoding, with no mask-token repair.
    return torch.where(num_missing, numeric, anchor_num), torch.where(cat_missing, categorical, anchor_cat)


class FrozenTabDiff:
    def __init__(self, diffusion, preprocessing, minority_label, anchor_keys, config, audit, checkpoint_path):
        self._diffusion = diffusion.eval().requires_grad_(False)
        self._preprocessing = preprocessing
        self._minority_label = minority_label
        self._anchor_keys = frozenset(anchor_keys)
        self.config = config
        self._audit = deepcopy(audit)
        self.checkpoint_path = Path(checkpoint_path)
        self._checkpoint_checksum = _file_checksum(self.checkpoint_path)

    @property
    def training_audit(self):
        audit = deepcopy(self._audit)
        audit["checkpoint_sha256"] = self._checkpoint_checksum
        audit["checkpoint_path"] = str(self.checkpoint_path)
        return audit

    @property
    def preprocessing(self):
        return deepcopy(self._preprocessing)

    def _condition_reasons(self, frame):
        if tuple(frame.columns) != self._preprocessing.columns:
            raise ValueError("Anchor frame must have the training feature columns in order")
        reasons = []
        for row in frame.itertuples(index=False, name=None):
            values = dict(zip(frame.columns, row))
            reason = None
            for column in self._preprocessing.numeric:
                if pd.isna(values[column]) or not np.isfinite(float(values[column])):
                    reason = f"nonfinite_numeric:{column}"
                    break
            if reason is None:
                for j, column in enumerate(self._preprocessing.categorical):
                    if pd.isna(values[column]) or _key(values[column]) not in self._preprocessing.category_maps[j]:
                        reason = f"unknown_category:{column}"
                        break
            if reason is None and _row_key(row) not in self._anchor_keys:
                reason = "not_real_minority_training_anchor"
            reasons.append(reason)
        return reasons

    def encode_valid_mask(self, frame):
        return np.array([reason is None for reason in self._condition_reasons(frame)], dtype=bool)

    def inpaint(self, anchor_frame, masked_columns, labels, seed):
        start = perf_counter()
        masked = tuple(masked_columns)
        if not masked or len(set(masked)) != len(masked) or not set(masked) <= set(self._preprocessing.columns):
            raise ValueError("masked_columns must be distinct original feature columns; label is never maskable")
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) != len(anchor_frame):
            raise ValueError("labels must have one minority condition per anchor")
        reasons = self._condition_reasons(anchor_frame)
        for i, label in enumerate(labels):
            if pd.isna(label) or _key(label) != _key(self._minority_label):
                reasons[i] = "label_not_training_minority"
        valid = np.array([reason is None for reason in reasons], dtype=bool)
        output = anchor_frame.copy(deep=True)
        for column in masked:
            if column in self._preprocessing.numeric:
                output[column] = np.nan
            else:
                output[column] = output[column].astype(object)
                output[column] = pd.NA
        sample_seed = _seed(seed, "inpaint")
        cat_tokens_valid = np.zeros(len(anchor_frame), dtype=bool)
        if valid.any():
            selected = anchor_frame.iloc[np.flatnonzero(valid)]
            encoded_num, encoded_cat = self._preprocessing.encode(selected, labels[valid])
            num_missing = torch.tensor([c in masked for c in self._preprocessing.numeric], device=self._diffusion.device)
            cat_missing = torch.tensor([c in masked for c in self._preprocessing.categorical] + [False], device=self._diffusion.device)
            # Zero missing coordinates before sampling, in addition to the
            # independent prior: even internal known-noise draws cannot read them.
            encoded_num[:, num_missing.cpu().numpy()] = 0
            encoded_cat[:, cat_missing.cpu().numpy()] = 0
            with _rng(sample_seed, self._diffusion.device):
                raw_num, raw_cat = _sample_inpaint(
                    self._diffusion, torch.as_tensor(encoded_num, device=self._diffusion.device),
                    torch.as_tensor(encoded_cat, device=self._diffusion.device), num_missing, cat_missing)
            raw_num, raw_cat = raw_num.cpu().numpy(), raw_cat.cpu().numpy()
            token_valid = ((raw_cat >= 0) & (raw_cat < self._diffusion.num_classes[None])).all(axis=1)
            finite = np.isfinite(raw_num).all(axis=1)
            positions = np.flatnonzero(valid)
            cat_tokens_valid[positions] = token_valid
            for position, good_cat, good_num in zip(positions, token_valid, finite):
                if not good_cat:
                    reasons[position] = "invalid_end_categorical_token"
                elif not good_num:
                    reasons[position] = "nonfinite_generated_numeric"
            decoded_valid = token_valid & finite
            if decoded_valid.any():
                positions = positions[decoded_valid]
                # Decode only valid tokens; no argmax or categorical fallback.
                decoded = self._preprocessing.decode(raw_num[decoded_valid], raw_cat[decoded_valid], pd.RangeIndex(len(positions)))
                for column in masked:
                    output.iloc[positions, output.columns.get_loc(column)] = decoded[column].to_numpy()
        # Preserve full original dtypes and values for unmasked fields, avoiding
        # even a numeric quantile round trip or categorical re-encoding there.
        for column in anchor_frame:
            if column not in masked:
                output[column] = anchor_frame[column].copy(deep=True)
            elif column in self._preprocessing.categorical:
                dtype = self._preprocessing.dtypes[column]
                if isinstance(dtype, pd.CategoricalDtype):
                    dtype = pd.CategoricalDtype(self._preprocessing.categories[self._preprocessing.categorical.index(column)], ordered=dtype.ordered)
                if all(reason is None for reason in reasons) or isinstance(dtype, pd.CategoricalDtype):
                    output[column] = output[column].astype(dtype)
        accepted = [reason is None for reason in reasons]
        audit = {"implementation": IMPLEMENTATION_VERSION, "seed": int(seed), "sampling_seed": sample_seed,
                 "checkpoint_sha256": self._checkpoint_checksum, "model_checksum": self._audit["model_checksum"],
                 "masked_columns": list(masked), "label_unmasked": True, "minority_label": _scalar(self._minority_label),
                 "valid_mask": accepted, "row_rejection_reasons": reasons, "categorical_end_tokens_valid": cat_tokens_valid.tolist(),
                 "proposed_rows": len(anchor_frame), "generated_rows": int(sum(accepted)),
                 "sampling_steps": self.config.num_steps, "seconds": perf_counter() - start,
                 "denoiser_evaluations": ((2 * self.config.num_steps - 1 if self._preprocessing.numeric else self.config.num_steps) if valid.any() else 0),
                 "sampling_stream": "purpose_derived_inpaint_seed_torch_fork_rng",
                 "known_condition_time": "Euler:t_hat;Heun:t_next;categorical_schedule_at_t;numeric_supplied_per_column_sigma",
                 "numeric_missing_prior": "independent_zero_mean_sigma_max_gaussian",
                 "numeric_known_noise": "anchor_plus_sigma_times_independent_standard_normal",
                 "categorical_known_noise": "absorbing_q_xt_including_unmasked_label",
                 "numeric_decode": "raw_inverse_quantile_float_no_rounding_or_bounds_repair",
                 "conditioning": "joint_label_q_xt_at_actual_denoiser_time_final_exact_restore"}
        return output, audit

    @classmethod
    def load(cls, folder, device="auto"):
        path = Path(folder)
        if path.is_dir():
            path = path / "generator.pt"
        # Local generator checkpoints contain the fitted sklearn transform and
        # original typed categories: load only trusted locally produced artifacts.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["implementation"] != IMPLEMENTATION_VERSION or payload["reference_commit"] != REFERENCE_COMMIT:
            raise ValueError("Unsupported TabDiff generator checkpoint provenance")
        config = TabDiffConfig(**payload["config"])
        target = _device(device)
        with _rng(_seed(payload["audit"]["seed"], "load_architecture"), target):
            diffusion = _make_diffusion(payload["preprocessing"], config, target)
        diffusion.load_state_dict(payload["final_ema_state"])
        if _checksum(diffusion) != payload["audit"]["model_checksum"]:
            raise ValueError("TabDiff checkpoint model checksum mismatch")
        return cls(diffusion, payload["preprocessing"], payload["minority_label"], payload["anchor_keys"], config, payload["audit"], path)


def _epoch_batches(permutation, batch_size):
    # Official mixed_loss squeezes t, so singleton batches cannot enter the
    # denoiser. Merge a final singleton into the previous batch; never drop a row.
    end = len(permutation)
    start = 0
    while start < end:
        stop = min(start + batch_size, end)
        if end - stop == 1:
            stop = end
        yield permutation[start:stop]
        start = stop


def fit_tabdiff(frame, y, row_ids, seed, folder, config=None, device="auto"):
    config = config if isinstance(config, TabDiffConfig) else TabDiffConfig(**(config or {}))
    target = _device(device)
    folder = Path(folder)
    path = folder / "generator.pt"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite a frozen generator checkpoint: {path}")
    y, row_ids = np.asarray(y), np.asarray(row_ids)
    if not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates or len(frame) < 2 or not len(frame.columns):
        raise ValueError("TabDiff requires a nonempty training dataframe with unique feature columns and >=2 rows")
    if y.ndim != 1 or row_ids.ndim != 1 or len(y) != len(frame) or len(row_ids) != len(frame):
        raise ValueError("Training frame, labels and row_ids must align")
    if len(set(map(_key, row_ids))) != len(row_ids):
        raise ValueError("Training row_ids must be unique")
    if frame.isna().any().any() or pd.isna(y).any():
        raise ValueError("TabDiff training requires observed finite fields, without missing values")
    class_values = tuple(_scalar(v) for v in pd.unique(y))
    if len(class_values) != 2:
        raise ValueError("This experiment requires real training examples from both classes")
    counts = [int(sum(_key(value) == _key(label) for value in y)) for label in class_values]
    minority = class_values[int(np.argmin(counts))]
    preprocessing_seed = int(_seed(seed, "preprocessing") % (2**32 - 1))
    training_seed = _seed(seed, "training")
    start = perf_counter()
    preprocessing = _Preprocessing(frame, y, preprocessing_seed)
    numeric, categorical = preprocessing.encode(frame, y)
    if not np.isfinite(numeric).all() or (preprocessing.numeric and not np.isfinite(frame.loc[:, preprocessing.numeric].to_numpy(dtype=float)).all()):
        raise ValueError("TabDiff numeric training fields must be finite")
    anchor_keys = [_row_key(row) for row, label in zip(frame.itertuples(index=False, name=None), y) if _key(label) == _key(minority)]
    history, updates = [], 0
    with _rng(training_seed, target):
        diffusion = _make_diffusion(preprocessing, config, target)
        initial_checksum = _checksum(diffusion)
        ema = deepcopy(diffusion).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(diffusion.parameters(), lr=config.learning_rate, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience)
        data = torch.as_tensor(np.concatenate([numeric, categorical], axis=1), dtype=torch.float32, device=target)
        for epoch in range(config.epochs):
            diffusion.train()
            d_total = c_total = 0.
            closs_weight = 1 - epoch / config.epochs
            lr = optimizer.param_groups[0]["lr"]
            for indices in _epoch_batches(torch.randperm(len(data), device=target), config.batch_size):
                optimizer.zero_grad(set_to_none=True)
                dloss, closs = diffusion.mixed_loss(data[indices])
                loss = dloss + closs_weight * closs
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite official TabDiff training loss at epoch {epoch + 1}")
                loss.backward()
                optimizer.step()
                d_total += dloss.item() * len(indices)
                c_total += closs.item() * len(indices)
                updates += 1
            d_mean, c_mean = d_total / len(data), c_total / len(data)
            # Upstream scheduler consumes rounded training losses, not held-out
            # metrics; EMA updates once per epoch, including both learned schedules.
            scheduler.step(float(np.around(d_mean, 4) + np.around(c_mean, 4)))
            with torch.no_grad():
                for ema_parameter, parameter in zip(ema.parameters(), diffusion.parameters()):
                    ema_parameter.mul_(config.ema_decay).add_(parameter, alpha=1 - config.ema_decay)
            history.append({"epoch": epoch + 1, "updates": updates, "categorical_loss": d_mean,
                            "numeric_loss": c_mean, "continuous_loss_weight": closs_weight, "learning_rate": lr})
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        training_seconds = perf_counter() - start
        audit = {"implementation": IMPLEMENTATION_VERSION, "reference_commit": REFERENCE_COMMIT,
                 "source_checksums": SOURCE_CHECKSUMS.copy(), "seed": int(seed), "training_seed": training_seed,
                 "preprocessing_seed": preprocessing_seed, "config": asdict(config), "device": str(target),
                 "training_row_ids": [_scalar(v) for v in row_ids], "training_rows": len(frame),
                 "training_row_ids_sha256": hashlib.sha256(repr([_key(v) for v in row_ids]).encode()).hexdigest(),
                 "training_data_sha256": hashlib.sha256(repr([_row_key(row) + (_key(label),) for row, label in zip(frame.itertuples(index=False, name=None), y)]).encode()).hexdigest(),
                 "class_counts": [{"label": label, "count": count} for label, count in zip(class_values, counts)],
                 "minority_label": minority, "numeric_columns": list(preprocessing.numeric),
                 "categorical_columns": list(preprocessing.categorical), "label_modeled_as": "additional_categorical_column",
                 "actual_epochs": config.epochs, "actual_updates": updates, "ema_update_frequency": "once_per_epoch",
                 "checkpoint_selection": "final_epoch_ema_no_validation_or_test_selection", "active_validation": False,
                 "initial_model_checksum": initial_checksum, "model_checksum": _checksum(ema),
                 "training_seconds": training_seconds, "loss_history": history,
                 "quantile_fit_rows": len(frame), "categorical_cardinalities": [len(v) for v in preprocessing.categories],
                 "numeric_counts_policy": "raw_float_external_common_codec"}
        payload = {"implementation": IMPLEMENTATION_VERSION, "reference_commit": REFERENCE_COMMIT,
                   "config": asdict(config), "preprocessing": preprocessing, "minority_label": minority,
                   "anchor_keys": anchor_keys, "audit": audit, "final_ema_state": ema.cpu().state_dict(),
                   "last_training_state": {k: v.detach().cpu() for k, v in diffusion.state_dict().items()},
                   "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                   "training_rng_state": torch.random.get_rng_state(),
                   "training_cuda_rng_state": torch.cuda.get_rng_state(target) if target.type == "cuda" else None}
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    frozen = FrozenTabDiff(ema.to(target), preprocessing, minority, anchor_keys, config, audit, path)
    (folder / "training_audit.json").write_text(json.dumps(frozen.training_audit, indent=2) + "\n")
    return frozen
