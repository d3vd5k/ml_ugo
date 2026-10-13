"""Matched within-U2-group input and mixed-data projected Flow interpolation."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from samplers.flow_gmm import _checksum, _layout
from samplers.ugo import Candidates

NUMERIC_COLUMNS = ("Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration", "BounceRates",
    "ExitRates", "PageValues", "SpecialDay")
RATE_COLUMNS = ("BounceRates", "ExitRates", "SpecialDay")


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _domain(values, names):
    finite = np.isfinite(values).all(axis=1)
    negative = (values < 0).any(axis=1)
    rate = (values[:, [names.index(name) for name in RATE_COLUMNS]] > 1).any(axis=1)
    reasons = np.full(len(values), "accepted", dtype=object)
    reasons[rate] = "numeric_rate_out_of_range"
    reasons[negative] = "numeric_negative"
    reasons[~finite] = "numeric_nonfinite"
    return finite & ~negative & ~rate, reasons, negative, rate


class FlowParentCache:
    """Represent original real rows once; never fit or mutate either preprocessor."""
    def __init__(self, core, classifier_preprocessor, raw_frame, real_coordinates, real_row_ids):
        tick = perf_counter()
        self.core, self.classifier_preprocessor = core, classifier_preprocessor
        self.raw_frame = raw_frame.copy()
        self.real_coordinates = _numpy(real_coordinates).copy()
        self.real_row_ids = np.asarray(real_row_ids).copy()
        if len(raw_frame) != len(self.real_coordinates) or len(raw_frame) != len(self.real_row_ids):
            raise ValueError("Parent provenance and coordinates must align")
        self.numeric_columns, categorical, encoder, _ = _layout(core.preprocessor)
        if set(self.numeric_columns) != set(NUMERIC_COLUMNS):
            raise ValueError("Flow geometry requires the Shoppers numeric schema")
        self.detector_scaler = core.preprocessor.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"]
        classifier_numeric, _, _, _ = _layout(classifier_preprocessor)
        if classifier_numeric != self.numeric_columns:
            raise ValueError("Classifier and detector numeric layouts must agree")
        self.known = np.ones(len(raw_frame), dtype=bool)
        for column, categories in zip(categorical, encoder.categories_ if encoder is not None else []):
            values = raw_frame[column]
            self.known &= values.notna().to_numpy() & values.astype(str).isin(categories).to_numpy()
        self.raw_numeric = raw_frame[self.numeric_columns].to_numpy(dtype=np.float64)
        domain, reasons, _, _ = _domain(self.raw_numeric, self.numeric_columns)
        self.finite = np.isfinite(self.real_coordinates).all(axis=1) & np.isfinite(self.raw_numeric).all(axis=1)
        self.parent_valid = self.known & self.finite & domain
        self.parent_reasons = reasons
        self.parent_reasons[~self.finite] = "parent_nonfinite"
        self.parent_reasons[~self.known] = "unknown_flow_category"
        dimension = core.model.means.shape[1]
        self.encoded = np.full((len(raw_frame), dimension), np.nan)
        self.latents = self.encoded.copy()
        self.roundtrip_coordinates = self.encoded.copy()
        positions = np.flatnonzero(self.parent_valid)
        self.frozen_checksum = _checksum(core.model)
        self.representation_seconds = 0.
        self.forward_seconds = 0.
        self.roundtrip_seconds = 0.
        if len(positions):
            start = perf_counter()
            self.encoded[positions] = core.represent(raw_frame.iloc[positions])
            self.representation_seconds = perf_counter() - start
            with torch.inference_mode():
                start = perf_counter()
                self.latents[positions] = _numpy(core.forward(self.encoded[positions])[0])
                self.forward_seconds = perf_counter() - start
                start = perf_counter()
                self.roundtrip_coordinates[positions] = _numpy(core.inverse(self.latents[positions])[0])
                self.roundtrip_seconds = perf_counter() - start
            valid = np.isfinite(self.encoded[positions]).all(axis=1) & np.isfinite(self.latents[positions]).all(axis=1) & np.isfinite(self.roundtrip_coordinates[positions]).all(axis=1)
            self.parent_valid[positions[~valid]] = False
            self.parent_reasons[positions[~valid]] = "parent_representation_nonfinite"
        self.cache_seconds = perf_counter() - tick
        self.roundtrip_max_absolute_error = float(np.max(np.abs(self.roundtrip_coordinates[positions] - self.encoded[positions]))) if len(positions) else None

    @property
    def real_ids(self):
        return self.real_row_ids

    @property
    def audit(self):
        return {"frozen_checksum": self.frozen_checksum, "numeric_columns": self.numeric_columns,
            "rows": len(self.real_row_ids), "valid_rows": int(self.parent_valid.sum()),
            "representation_seconds": self.representation_seconds, "forward_seconds": self.forward_seconds,
            "roundtrip_seconds": self.roundtrip_seconds, "cache_seconds": self.cache_seconds,
            "roundtrip_max_absolute_error": self.roundtrip_max_absolute_error}

    def state_checksum(self):
        return _checksum(self.core.model)

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "parents.npz", real_coordinates=self.real_coordinates,
            real_row_ids=self.real_row_ids, raw_numeric=self.raw_numeric, encoded=self.encoded,
            latents=self.latents, known=self.known, finite=self.finite, parent_valid=self.parent_valid,
            parent_reasons=self.parent_reasons.astype(str), roundtrip_coordinates=self.roundtrip_coordinates)
        metadata = {"frozen_checksum": self.frozen_checksum, "numeric_columns": self.numeric_columns,
            "fit_row_ids": self.core.fit_row_ids.tolist(), "rows": len(self.real_row_ids),
            "valid_rows": int(self.parent_valid.sum()), "representation_seconds": self.representation_seconds,
            "forward_seconds": self.forward_seconds, "roundtrip_seconds": self.roundtrip_seconds,
            "cache_seconds": self.cache_seconds, "roundtrip_max_absolute_error": self.roundtrip_max_absolute_error,
            "detector_scale_mean": self.detector_scaler.mean_.tolist(), "detector_scale_scale": self.detector_scaler.scale_.tolist()}
        (folder / "cache.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata


class GroupGeometryProposer:
    def __init__(self, cache, row_group_ids, mode="input"):
        if mode not in ("input", "latent"):
            raise ValueError("Geometry must be input or latent")
        self.cache, self.mode = cache, mode
        self.row_group_ids = np.asarray(row_group_ids)
        if self.row_group_ids.shape != (len(cache.real_coordinates),):
            raise ValueError("Group IDs must align with real rows")
        self.variant_name = "pool_random_group_flow" if mode == "latent" else "pool_random_group_input"
        self.last_proposal = {}
        self._options = {}

    def summary(self):
        return {"mode": self.mode, "variant_name": self.variant_name, "parent_cache": self.cache.audit,
            "parents": "same-class same-U2-group input-coordinate nearest distinct real observations",
            "categorical_projection": "copy original partner classifier categorical blocks",
            "domain": "numeric >=0; BounceRates, ExitRates, SpecialDay <=1; no rounding or clipping",
            "gate": None}

    def propose(self, real, budget, rng, cfg, *, seed_indices=None):
        tick = perf_counter()
        cache = self.cache
        X, y = _numpy(real.X), _numpy(real.y)
        if X.shape != cache.real_coordinates.shape or not np.array_equal(X, cache.real_coordinates, equal_nan=True):
            raise ValueError("Geometry cache must align with original real support")
        seeds = _numpy(seed_indices if seed_indices is not None else budget.seed_indices[rng.integers(len(budget.seed_indices), size=budget.count)]).astype(np.int64)
        options = []
        for anchor in seeds:
            key = (int(anchor), int(cfg.k_pre))
            if key not in self._options:
                eligible = np.flatnonzero((y == y[anchor]) & (self.row_group_ids == self.row_group_ids[anchor])) if self.row_group_ids[anchor] >= 0 else np.empty(0, dtype=int)
                distances = np.sum((X[eligible].astype(np.float64) - X[anchor]) ** 2, axis=1)
                order = np.argsort(distances, kind="stable")
                eligible = eligible[order][np.isfinite(distances[order]) & (distances[order] > 0)]
                self._options[key] = eligible[:cfg.k_pre]
            options.append(self._options[key])
        # Draw every tuple once; unsupported anchors retain an explicit invalid pool row.
        choices = rng.random(len(seeds))
        partners = np.asarray([rows[min(int(choice * len(rows)), len(rows)-1)] if len(rows) else -1 for rows, choice in zip(options, choices)], dtype=np.int64)
        coefficients = rng.random((len(seeds), 1)).astype(np.float32)
        supported = partners >= 0
        parent_valid = np.zeros(len(seeds), dtype=bool)
        parent_valid[supported] = cache.parent_valid[seeds[supported]] & cache.parent_valid[partners[supported]]
        counterfactual = np.full((len(seeds), X.shape[1]), np.nan, dtype=np.float32)
        numeric = np.full((len(seeds), len(cache.numeric_columns)), np.nan)
        inverse = np.full((len(seeds), cache.encoded.shape[1]), np.nan)
        positions = np.flatnonzero(parent_valid)
        counterfactual[positions] = X[seeds[positions]] + coefficients[positions] * (X[partners[positions]] - X[seeds[positions]])
        categorical = list(cfg.categorical_columns)
        if categorical:
            counterfactual[np.ix_(positions, categorical)] = X[np.ix_(partners[positions], categorical)]
        generated = counterfactual.copy()
        inverse_seconds = 0.
        decode_start = perf_counter()
        if self.mode == "latent" and len(positions):
            latent = cache.latents[seeds[positions]] + coefficients[positions] * (cache.latents[partners[positions]] - cache.latents[seeds[positions]])
            start = perf_counter()
            with torch.inference_mode():
                inverse[positions] = _numpy(cache.core.inverse(latent)[0])
            inverse_seconds = perf_counter() - start
            numeric[positions] = cache.detector_scaler.inverse_transform(inverse[positions, :len(cache.numeric_columns)])
            decoded = cache.raw_frame.iloc[partners[positions]].copy()
            for index, name in enumerate(cache.numeric_columns):
                decoded[name] = numeric[positions, index]
            finite = np.isfinite(numeric[positions]).all(axis=1)
            generated[positions] = np.nan
            if finite.any():
                generated[positions[finite]] = cache.classifier_preprocessor.transform(decoded.iloc[np.flatnonzero(finite)])
                if categorical:
                    generated[np.ix_(positions[finite], categorical)] = X[np.ix_(partners[positions[finite]], categorical)]
        elif len(positions):
            numeric[positions] = cache.raw_numeric[seeds[positions]] + coefficients[positions] * (cache.raw_numeric[partners[positions]] - cache.raw_numeric[seeds[positions]])
        geometry_valid, reasons, negative, rate = _domain(numeric, cache.numeric_columns)
        geometry_valid &= parent_valid & np.isfinite(generated).all(axis=1)
        representation_finite = np.isfinite(generated).all(axis=1)
        if self.mode == "latent":
            representation_finite &= np.isfinite(inverse).all(axis=1)
        geometry_valid &= representation_finite
        reasons[parent_valid & ~representation_finite] = "decoded_representation_nonfinite"
        reasons[~parent_valid] = "invalid_parent"
        for index in np.flatnonzero(supported & ~parent_valid):
            invalid = seeds[index] if not cache.parent_valid[seeds[index]] else partners[index]
            reasons[index] = cache.parent_reasons[invalid]
        reasons[~supported] = "no_distinct_within_group_partner"
        displacement = np.linalg.norm(generated.astype(np.float64) - counterfactual, axis=1)
        self.last_proposal = {"interpolation_coefficients": coefficients[:, 0], "parent_valid": parent_valid,
            "geometry_valid": geometry_valid, "geometry_reasons": reasons, "raw_numeric_decoded": numeric,
            "input_counterfactual_coordinates": counterfactual, "parent_group_ids": self.row_group_ids[seeds],
            "partner_group_ids": np.where(supported, self.row_group_ids[np.maximum(partners, 0)], -1),
            "raw_decode_negative": negative, "raw_decode_rate_failure": rate,
            "nonlinearity_displacement": displacement, "inverse_seconds": inverse_seconds,
            "geometry_generation_seconds": perf_counter() - tick, "decode_seconds": perf_counter() - decode_start - inverse_seconds, "generation_geometry": self.variant_name}
        if self.mode == "latent":
            self.last_proposal["inverse_detector_coordinates"] = inverse
        return Candidates(torch.as_tensor(generated, device=real.X.device, dtype=real.X.dtype),
            torch.full((len(seeds),), budget.target_class, device=real.y.device, dtype=real.y.dtype),
            torch.as_tensor(seeds, device=real.X.device), torch.as_tensor(partners, device=real.X.device),
            torch.full((len(seeds),), -1, dtype=torch.long, device=real.X.device))
