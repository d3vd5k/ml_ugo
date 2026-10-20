"""Independently written, pinned ``flowgmm_tabular_v1`` F1 compatibility model.

Algorithm reference only: Izmailov et al., *Semi-Supervised Learning with
Normalizing Flows*, ICML 2020, https://proceedings.mlr.press/v119/izmailov20a.html;
https://github.com/izmailovpavel/flowgmm commit
071285be739defd548ed0c90eec9e56e1112b773. No upstream code is vendored (no
license was detected). This implementation is supervised, real-only and binary.

Full one-hot coordinates receive independent U[0,1) dequantization. Training
jitter is fresh every epoch; evaluation jitter is content-keyed, label-independent
and invariant to row order, duplicates and batching. Numeric features, including
counts, remain continuous, as in U2 interpolation. Reported density is a
continuous dequantized proxy in the detector's standardized/one-hot coordinates,
NOT an exact discrete probability, support guarantee or calibrated correctness
probability. F1 uses equal priors and no likelihood support filter.
"""
from dataclasses import asdict, dataclass
import hashlib
import math
from numbers import Integral
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn

from datasets.scripts.climb import make_preprocessor


IMPLEMENTATION_VERSION = "flowgmm_tabular_v1"
REFERENCE_COMMIT = "071285be739defd548ed0c90eec9e56e1112b773"


@dataclass(frozen=True)
class FlowGMMConfig:
    epochs: int = 60
    layers: int = 6
    hidden: int = 64
    batch_size: int = 256
    learning_rate: float = .001
    device: str = "auto"

    def __post_init__(self):
        for name in ("epochs", "layers", "hidden", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.learning_rate, (bool, np.bool_))
                or not np.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError("learning_rate must be positive and finite")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")


def _layout(preprocessor):
    columns = preprocessor.named_steps["columns"]
    numeric = list(columns.transformers_[0][2])
    categorical = list(columns.transformers_[1][2])
    encoder = (columns.named_transformers_["categorical"].named_steps["encode"]
               if categorical else None)
    start = columns.output_indices_["categorical"].start
    groups = []
    if encoder is not None:
        for categories in encoder.categories_:
            groups.append(tuple(range(start, start + len(categories))))
            start += len(categories)
    return numeric, categorical, encoder, groups


class AffineCoupling(nn.Module):
    """Alternating affine coupling, x -> z, with log scale 2*tanh(raw)."""

    def __init__(self, width, hidden, parity):
        super().__init__()
        self.register_buffer("mask", (torch.arange(width) % 2 == parity).double())
        self.network = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 2 * width))
        # Identity initialization lets empirical class moments initialize latent components.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _parameters_at(self, fixed):
        raw_scale, shift = self.network(fixed).chunk(2, dim=-1)
        active = 1 - self.mask
        return 2 * torch.tanh(raw_scale) * active, shift * active

    def forward(self, x):
        fixed = x * self.mask
        scale, shift = self._parameters_at(fixed)
        return fixed + (1 - self.mask) * (x * torch.exp(scale) + shift), scale.sum(-1)

    def inverse(self, z):
        fixed = z * self.mask
        scale, shift = self._parameters_at(fixed)
        return fixed + (1 - self.mask) * ((z - shift) * torch.exp(-scale)), -scale.sum(-1)


class SharedRealNVP(nn.Module):
    """One invertible map and two learned diagonal Gaussian latent components."""

    def __init__(self, width, config, means, stds):
        super().__init__()
        self.couplings = nn.ModuleList(AffineCoupling(width, config.hidden, i % 2)
                                      for i in range(config.layers))
        self.means = nn.Parameter(torch.as_tensor(means, dtype=torch.float64).clone())
        self.log_stds = nn.Parameter(torch.as_tensor(stds, dtype=torch.float64).log())
        self.double()

    def forward(self, x):
        determinant = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
        for coupling in self.couplings:
            x, increment = coupling(x)
            determinant = determinant + increment
        return x, determinant

    def inverse(self, z):
        determinant = torch.zeros(z.shape[:-1], dtype=z.dtype, device=z.device)
        for coupling in reversed(self.couplings):
            z, increment = coupling.inverse(z)
            determinant = determinant + increment
        return z, determinant

    def gaussian_log_density(self, z):
        z = z.double()
        standardized = (z[:, None, :] - self.means.double()) * torch.exp(-self.log_stds.double())
        return (-.5 * standardized.square() - self.log_stds.double()
                - .5 * math.log(2 * math.pi)).sum(-1)

    def gaussian_log_ratio(self, z, intended):
        # Difference of squares, not subtraction of two huge likelihoods. The
        # shared flow Jacobian cancels analytically, including at extreme ratios.
        standardized = (z.double()[:, None, :] - self.means.double()) * torch.exp(-self.log_stds.double())
        rows = torch.arange(len(z), device=z.device)
        own, opposing = standardized[rows, intended], standardized[rows, 1 - intended]
        return (.5 * (opposing - own) * (opposing + own)
                + self.log_stds[1 - intended] - self.log_stds[intended]).sum(-1)


def _source_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _checksum(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class FlowGMMCore:
    """Frozen detector; forward/inverse operate on detector coordinates (no jitter)."""

    def __init__(self, model, preprocessor, classes, fit_row_ids, config, seed, audit):
        self.model, self.preprocessor = model, preprocessor
        self.classes = np.asarray(classes).copy()
        self.fit_row_ids = np.asarray(fit_row_ids).copy()
        self.classes.setflags(write=False)
        self.fit_row_ids.setflags(write=False)
        self.config, self.seed, self.audit = config, int(seed), audit
        _, _, _, groups = _layout(preprocessor)
        self.categorical_groups = groups
        self.categorical_columns = tuple(column for group in groups for column in group)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.zero_grad(set_to_none=True)

    @property
    def device(self):
        return self.model.means.device

    def _tensor(self, X):
        return torch.as_tensor(X, dtype=torch.float64, device=self.device)

    def forward(self, X):
        return self.model(self._tensor(X))

    def inverse(self, Z):
        return self.model.inverse(self._tensor(Z))

    def represent(self, frame):
        """Fit-only standardized/full-one-hot coordinates, then deterministic jitter.

        Unknown categories raise explicitly, rather than turning all-zero blocks
        into a fabricated numeric category. Gate isolates row-specific rejections.
        """
        X = np.asarray(self.preprocessor.transform(frame), dtype=np.float64)
        for group in self.categorical_groups:
            if not np.all(X[:, group].sum(axis=1) == 1):
                raise ValueError("unknown_flow_category")
        if not np.isfinite(X).all():
            raise ValueError("nonfinite_score")
        return self._dequantize(X)

    def _dequantize(self, X):
        """Jitter an owned, already-validated detector-coordinate array in place."""
        if self.categorical_columns:
            for i, row in enumerate(X):
                key = hashlib.sha256(int(self.seed).to_bytes(8, "little", signed=False)
                                     + row.astype("<f8", copy=False).tobytes()).digest()
                rng = np.random.default_rng(int.from_bytes(key[:16], "little"))
                X[i, self.categorical_columns] += rng.random(len(self.categorical_columns))
        return X

    def density(self, X, y):
        """Return log densities including Jacobian and its cancelled log ratio."""
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(X) or not np.isin(labels, self.classes).all():
            raise ValueError("Intended labels must match fitted binary classes")
        intended = torch.as_tensor((labels == self.classes[1]).astype(np.int64), device=self.device)
        points = self._tensor(X)
        values = torch.empty((len(points), 4), dtype=torch.float64, device=self.device)
        with torch.inference_mode():
            # Fixed-shape vectorized kernels bound memory and minimize batch-size
            # rounding differences. Scores retain float64 precision, not a promise
            # of theoretical bit identity for every BLAS backend/order; the exact
            # >= logit comparison below never uses a tolerance or epsilon.
            batch_size = self.config.batch_size
            batch = torch.zeros((batch_size, points.shape[1]), dtype=torch.float64, device=self.device)
            batch_labels = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
            rows = torch.arange(batch_size, device=self.device)
            for offset in range(0, len(points), batch_size):
                count = min(batch_size, len(points) - offset)
                batch[:count].copy_(points[offset:offset + count])
                batch_labels[:count].copy_(intended[offset:offset + count])
                if count < batch_size:
                    batch[count:].zero_()
                    batch_labels[count:].zero_()
                z, determinant = self.model(batch)
                gaussian = self.model.gaussian_log_density(z)
                ratio = self.model.gaussian_log_ratio(z, batch_labels)
                destination = values[offset:offset + count]
                destination[:, 0] = (gaussian[rows, batch_labels] + determinant)[:count]
                destination[:, 1] = (gaussian[rows, 1 - batch_labels] + determinant)[:count]
                destination[:, 2] = ratio[:count]
                destination[:, 3] = determinant[:count]
        values = values.cpu().numpy()
        return dict(zip(("log_likelihood_intended", "log_likelihood_opposing",
                         "log_ratio", "log_determinant"), values.T))

    def save(self, path):
        """Checkpoint complete frozen model, private preprocessing and provenance."""
        torch.save({"implementation_version": IMPLEMENTATION_VERSION,
                    "state_dict": self.model.state_dict(), "preprocessor": self.preprocessor,
                    "classes": self.classes, "fit_row_ids": self.fit_row_ids,
                    "config": asdict(self.config), "seed": self.seed, "audit": self.audit}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        """Load a trusted local checkpoint; sklearn preprocessing uses pickle."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint["implementation_version"] != IMPLEMENTATION_VERSION:
            raise ValueError("Unsupported FlowGMM implementation version")
        config = FlowGMMConfig(**checkpoint["config"])
        state = checkpoint["state_dict"]
        model = SharedRealNVP(state["means"].shape[1], config,
                              state["means"].cpu().numpy(),
                              state["log_stds"].exp().cpu().numpy()).to(device)
        model.load_state_dict(state)
        return cls(model, checkpoint["preprocessor"], checkpoint["classes"],
                   checkpoint["fit_row_ids"], config, checkpoint["seed"], checkpoint["audit"])


def fit_flow_gmm(frame, y, row_ids, seed, config=None):
    """Supervised class-balanced NLL, fixed epochs, no validation/retries/refits.

    Caller must supply real fit rows only. Calibration, validation, candidates
    and test rows are neither accepted by this API nor used to fit preprocessing.
    """
    config = FlowGMMConfig() if config is None else config
    if not isinstance(config, FlowGMMConfig):
        raise TypeError("config must be FlowGMMConfig")
    if not isinstance(frame, pd.DataFrame) or frame.empty or not len(frame.columns):
        raise ValueError("FlowGMM needs a nonempty raw feature DataFrame")
    labels, ids = np.asarray(y), np.asarray(row_ids)
    if labels.ndim != 1 or ids.ndim != 1 or len(frame) != len(labels) or len(ids) != len(frame):
        raise ValueError("Real frame, labels and row IDs must align")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Fit row IDs must be unique")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2 or pd.isna(labels).any():
        raise ValueError("FlowGMM requires both real binary classes")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative integer below 2**63")
    device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    start = perf_counter()
    preprocessor = make_preprocessor(frame)
    base = np.asarray(preprocessor.fit_transform(frame), dtype=np.float64)
    if not np.isfinite(base).all():
        raise ValueError("Nonfinite real fitting representation")
    _, _, _, groups = _layout(preprocessor)
    categorical = tuple(column for group in groups for column in group)
    encoded_y = (labels == classes[1]).astype(np.int64)
    rng = np.random.default_rng(seed)
    # E[X+U]=X+.5 and Var(U)=1/12 supply strictly positive categorical moments.
    initial = base.copy()
    initial[:, categorical] += .5
    means = np.stack([initial[encoded_y == i].mean(0) for i in range(2)])
    variances = np.stack([base[encoded_y == i].var(0) for i in range(2)])
    variances[:, categorical] += 1 / 12
    stds = np.sqrt(np.maximum(variances, 1e-4))
    # Torch initialization is local: fork_rng restores CPU and affected CUDA state.
    cuda_devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(int(seed))
        if cuda_devices:
            torch.cuda.manual_seed(int(seed))
        model = SharedRealNVP(base.shape[1], config, means, stds).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        targets = torch.as_tensor(encoded_y, device=device)
        # Weighted empirical NLL gives each class exactly half the total weight.
        weights = torch.as_tensor(len(labels) / (2 * counts[encoded_y]), dtype=torch.float64, device=device)
        losses = []
        model.train()
        for epoch in range(config.epochs):
            jittered = base.copy()
            jittered[:, categorical] += rng.random((len(base), len(categorical)))
            epoch_X = torch.as_tensor(jittered, dtype=torch.float64, device=device)
            order = rng.permutation(len(base))
            total = 0.
            for offset in range(0, len(base), config.batch_size):
                indices = torch.as_tensor(order[offset:offset + config.batch_size], device=device)
                z, determinant = model(epoch_X[indices])
                log_density = model.gaussian_log_density(z)
                batch_labels = targets[indices]
                own = log_density[torch.arange(len(indices), device=device), batch_labels]
                loss = (-(own + determinant) * weights[indices]).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite FlowGMM training loss at epoch {epoch + 1}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError(f"Nonfinite FlowGMM training gradient at epoch {epoch + 1}")
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise RuntimeError(f"Nonfinite FlowGMM training parameter at epoch {epoch + 1}")
                std = model.log_stds.detach().exp()
                if not torch.isfinite(std).all() or not (std > 0).all():
                    raise RuntimeError(f"Nonfinite FlowGMM training covariance at epoch {epoch + 1}")
                total += loss.item() * len(indices)
            losses.append(total / len(base))
    audit = {"implementation_version": IMPLEMENTATION_VERSION, "source_sha256": _source_sha(),
             "torch_version": str(torch.__version__), "reference_commit": REFERENCE_COMMIT,
             "reference_paper": "https://proceedings.mlr.press/v119/izmailov20a.html",
             "reference_repository": "https://github.com/izmailovpavel/flowgmm",
             "reference_only": True, "vendor_code": False,
             "model_config": asdict(config), "resolved_device": device,
             "model_checksum": _checksum(model), "fit_source": "real_fit_rows_only",
             "fit_row_ids": ids.tolist(), "fit_class_counts": {str(c): int(n) for c, n in zip(classes, counts)},
             "fit_rows": len(base), "synthetic_training_count": 0, "equal_priors": [.5, .5],
             "training_seconds": perf_counter() - start, "training_losses": losses,
             "objective": "supervised_class_balanced_negative_log_likelihood",
             "early_stopping": False, "retries": 0, "frozen": True,
             "density_measure": "dequantized_continuous_proxy_in_detector_coordinates_not_discrete_probability",
             "categorical_representation": "full_onehot_uniform_dequantization",
             "train_jitter": "fresh_each_epoch", "eval_jitter": "content_keyed_shared_between_classes",
             "numeric_counts": "continuous_as_U2_interpolation", "support_filter": False,
             "calibration_used": False}
    return FlowGMMCore(model, preprocessor, classes, ids, config, seed, audit)


class FlowGate:
    """Inverse classifier-coordinate bridge and frozen F1 log-space acceptance."""

    def __init__(self, core, classifier_preprocessor, *, calibration_row_ids, validation_row_ids):
        self.core, self.classifier_preprocessor = core, classifier_preprocessor
        self.calibration_row_ids = np.asarray(calibration_row_ids).copy()
        self.validation_row_ids = np.asarray(validation_row_ids).copy()
        for ids in (self.calibration_row_ids, self.validation_row_ids):
            if ids.ndim != 1 or len(set(ids.tolist())) != len(ids):
                raise ValueError("Detector role row IDs must be unique one-dimensional vectors")
            ids.setflags(write=False)
        roles = [set(ids.tolist()) for ids in (core.fit_row_ids, self.calibration_row_ids, self.validation_row_ids)]
        if any(roles[a] & roles[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Flow fit/calibration/validation row IDs must be disjoint")
        self.role, self.calibration_class_counts = None, {}
        self.inference_seconds, self.inference_rows = 0., 0
        self._numeric, self._categorical, self._encoder, self._groups = _layout(classifier_preprocessor)
        self._columns = classifier_preprocessor.named_steps["columns"]
        self._width = max(block.stop for block in self._columns.output_indices_.values())
        self._raw_columns = list(classifier_preprocessor.feature_names_in_)
        if self._raw_columns != list(core.preprocessor.feature_names_in_):
            raise ValueError("Classifier and detector raw feature schemas must match")

    @property
    def audit(self):
        return {**self.core.audit, "role": self.role,
                "calibration_row_ids": self.calibration_row_ids.tolist(),
                "validation_row_ids": self.validation_row_ids.tolist(), "roles_disjoint": True,
                "calibration_class_counts": dict(self.calibration_class_counts),
                "calibration_used": False, "support_filter": False,
                "inference_seconds": self.inference_seconds, "inference_rows": self.inference_rows}

    def inverse_classifier(self, X):
        """Decode valid standardized numeric/full-one-hot classifier coordinates."""
        raw = pd.DataFrame(index=np.arange(len(X)), columns=self._raw_columns)
        if self._numeric:
            scaler = self._columns.named_transformers_["numeric"].named_steps["scale"]
            numeric_block = self._columns.output_indices_["numeric"]
            values = scaler.inverse_transform(X[:, numeric_block])
            for i, name in enumerate(self._numeric):
                raw[name] = values[:, i]
        for name, categories, group in zip(self._categorical, self._encoder.categories_ if self._encoder is not None else (), self._groups):
            raw[name] = categories[np.argmax(X[:, group], axis=1)]
        return raw

    def predict(self, X, y, tau):
        if isinstance(tau, (bool, np.bool_)) or not np.isfinite(tau) or not 0 < tau < 1:
            raise ValueError("tau must be strictly between 0 and 1")
        points, labels = np.asarray(X), np.asarray(y)
        if (points.ndim != 2 or points.shape[1] != self._width
                or not np.issubdtype(points.dtype, np.number) or np.iscomplexobj(points)):
            raise ValueError("Candidate coordinates must have the classifier's numeric feature width")
        if labels.ndim != 1 or len(labels) != len(points) or not np.isin(labels, self.core.classes).all():
            raise ValueError("Candidate labels must match the fitted binary classes")
        start = perf_counter()
        n = len(points)
        valid = np.isfinite(points).all(axis=1)
        reasons = np.full(n, "accepted", dtype=object)
        reasons[~valid] = "nonfinite_domain"
        for group in self._groups:
            block = points[:, group]
            correct = np.isin(block, (0, 1)).all(axis=1) & (block.sum(axis=1) == 1)
            reasons[valid & ~correct] = "invalid_onehot"
            valid &= correct
        result = {key: np.full(n, np.nan, dtype=np.float64) for key in
                  ("log_likelihood_intended", "log_likelihood_opposing", "log_ratio",
                   "compatibility_score", "log_determinant")}
        indices = np.flatnonzero(valid)
        if len(indices):
            raw = self.inverse_classifier(points[indices].astype(np.float64))
            # Unknown to detector is separate from validity in classifier's domain.
            detector_X = np.asarray(self.core.preprocessor.transform(raw), dtype=np.float64)
            finite_representation = np.isfinite(detector_X).all(axis=1)
            if self._numeric:
                finite_representation &= np.isfinite(raw[self._numeric].to_numpy(dtype=np.float64)).all(axis=1)
            reasons[indices[~finite_representation]] = "nonfinite_score"
            valid[indices[~finite_representation]] = False
            known = finite_representation.copy()
            for group in self.core.categorical_groups:
                known &= detector_X[:, group].sum(axis=1) == 1
            reasons[indices[finite_representation & ~known]] = "unknown_flow_category"
            valid[indices[~known]] = False
            indices = indices[known]
            if len(indices):
                represented = self.core._dequantize(detector_X if known.all() else detector_X[known])
                scores = self.core.density(represented, labels[indices])
                finite = np.ones(len(indices), dtype=bool)
                for key, values in scores.items():
                    result[key][indices] = values
                    finite &= np.isfinite(values)
                reasons[indices[~finite]] = "nonfinite_score"
                valid[indices[~finite]] = False
                ratio = scores["log_ratio"]
                # Stable sigmoid is diagnostic only; thresholding never rounds it.
                result["compatibility_score"][indices] = np.exp(-np.logaddexp(0., -ratio))
        accepted = valid & (result["log_ratio"] >= math.log(tau) - math.log1p(-tau))
        reasons[valid & ~accepted] = "flow_incompatible"
        self.inference_seconds += perf_counter() - start
        self.inference_rows += n
        return {**result, "valid": valid, "accepted": accepted, "reasons": reasons}
