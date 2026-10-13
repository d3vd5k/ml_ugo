"""Real production-model adapters for the notebook's single-seed 2D examples."""
from functools import partial
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from samplers.controls import resample
from samplers.flow_gmm import FlowGMMConfig, FlowGate, fit_flow_gmm
from samplers.flow_support import CalibratedFlowGate, calibrate_flow
from samplers.learning_benefit import CandidatePoolSelector
from samplers.pool_random import RandomPoolSelector
from samplers.protected import protected_ugo
from samplers.tabdiff import TabDiffConfig, fit_tabdiff
from samplers.tabdiff_control import fixed_u2_schedule, match_group_counts
from samplers.ugo import Candidates
from samplers.validation_guided import ValidationGuidedSelector

EXTENDED_SECTIONS = {
    'UGO scoring, constraints, stopping and compositions': [
        ('Historical OOF + U2 + F1', 'ugo_combined_f1')],
    'Post-generation filters': [
        ('H1: real-minority HDBSCAN support', 'ugo_h1'),
        ('U2 + neighbor rejection', 'ugo_n_reject'),
        ('U2 + F1 (tau=0.5)', 'ugo_f1'),
        ('U2 + F1 (tau=0.6)', 'ugo_f1_tau06'),
        ('U2 + F2 (support off)', 'ugo_f2_off'),
        ('U2 + F2 (q=0.01)', 'ugo_f2_q001'),
        ('U2 + F2 (q=0.05)', 'ugo_f2_q005'),
        ('U2 + F2 (q=0.10)', 'ugo_f2_q010')],
    'Guidance-gradient candidate pools': [
        (f'{phase.title()} / {role} / {branch} K5', f'{phase}_{role}_{branch}')
        for phase in ('initial', 'iterative')
        for role in ('guidance', 'validation')
        for branch in ('random', 'utility')],
    'Random pools and Flow generation geometry': [
        ('Native random K5 pool', 'pool_random'),
        ('Native random K5 + F1', 'pool_random_f1'),
        ('Within-group input interpolation', 'group_input'),
        ('Within-group Flow latent interpolation', 'group_flow')],
    'TabDiff feature-block inpainting': [
        ('Fixed-round U2 proposals (no F1)', 'tabdiff_u2'),
        ('Fixed-round U2 proposals + F1', 'tabdiff_u2_f1'),
        ('TabDiff 25% / 50% nominal = 50% actual (no F1)', 'tabdiff25'),
        ('TabDiff 25% / 50% nominal = 50% actual + F1', 'tabdiff25_f1')],
    'Synthetic downweighting — classification metrics': [
        (f'{label} / synthetic weight {weight:g}', f'downweight_{family}_{suffix}')
        for family, label in (('u2', 'U2'), ('u2_f1', 'U2 + F1'), ('pool_random', 'Random K5'))
        for weight, suffix in ((1., '1'), (.5, '05'), (.25, '025'))
    ] + [
        (f'{label} / matched real-anchor duplication (weight 0.25)', f'parentdup_{family}')
        for family, label in (('u2', 'U2'), ('u2_f1', 'U2 + F1'), ('pool_random', 'Random K5'))],
}


def _array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


class _NumericFlowGate(FlowGate):
    """Empty sklearn categorical branches have slice(0, 0), not total width."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._categorical or self._groups:
            raise ValueError('These illustrative adapters require numeric-only raw features')
        self._width = self._columns.output_indices_['numeric'].stop


class _NumericGroupGeometry:
    """Production pool proposer protocol, without Shoppers-only physical bounds."""
    def __init__(self, context, core, mode):
        self.X, self.y = context['X'], context['y']
        self.groups = np.asarray(context['prepared']['row_group_ids'])
        self.raw, self.preprocessor = context['raw'], context['preprocessor']
        self.core, self.mode = core, mode
        self.variant_name = 'numeric_group_flow' if mode == 'latent' else 'numeric_group_input'
        self._options = {}
        self.last_proposal = {}
        self.encoded = core.represent(self.raw)
        with torch.inference_mode():
            self.latents = _array(core.forward(self.encoded)[0])
        self.detector_scaler = core.preprocessor.named_steps['columns'].named_transformers_['numeric'].named_steps['scale']

    def propose(self, real, budget, rng, cfg, *, seed_indices=None):
        tick = perf_counter()
        if not np.array_equal(_array(real.X), self.X):
            raise ValueError('Geometry must reference the original real-fit coordinates')
        seeds = _array(seed_indices if seed_indices is not None else budget.seed_indices).astype(np.int64)
        options = []
        for anchor in seeds:
            key = (int(anchor), int(cfg.k_pre))
            if key not in self._options:
                eligible = np.flatnonzero((self.y == self.y[anchor]) &
                                         (self.groups == self.groups[anchor]))
                if self.groups[anchor] < 0:
                    eligible = np.empty(0, dtype=np.int64)
                distances = np.sum((self.X[eligible].astype(np.float64) - self.X[anchor]) ** 2, axis=1)
                order = np.argsort(distances, kind='stable')
                self._options[key] = eligible[order][distances[order] > 0][:cfg.k_pre]
            options.append(self._options[key])
        choices = rng.random(len(seeds))
        partners = np.asarray([rows[min(int(choice * len(rows)), len(rows)-1)] if len(rows) else -1
                               for rows, choice in zip(options, choices)], dtype=np.int64)
        coefficients = rng.random((len(seeds), 1)).astype(np.float32)
        valid = partners >= 0
        positions = np.flatnonzero(valid)
        counterfactual = np.full((len(seeds), self.X.shape[1]), np.nan, dtype=np.float32)
        counterfactual[positions] = (self.X[seeds[positions]] + coefficients[positions] *
                                     (self.X[partners[positions]] - self.X[seeds[positions]]))
        generated = counterfactual.copy()
        inverse_seconds = decode_seconds = 0.
        if self.mode == 'latent' and len(positions):
            latent = (self.latents[seeds[positions]] + coefficients[positions] *
                      (self.latents[partners[positions]] - self.latents[seeds[positions]]))
            start = perf_counter()
            with torch.inference_mode():
                decoded = _array(self.core.inverse(latent)[0])
            inverse_seconds = perf_counter() - start
            start = perf_counter()
            raw_values = self.detector_scaler.inverse_transform(decoded)
            finite = np.isfinite(raw_values).all(axis=1)
            generated[positions] = np.nan
            if finite.any():
                raw = self.raw.iloc[partners[positions[finite]]].astype(np.float64)
                raw.loc[:, :] = raw_values[finite]
                generated[positions[finite]] = self.preprocessor.transform(raw)
            decode_seconds = perf_counter() - start
        valid &= np.isfinite(generated).all(axis=1)
        reasons = np.where(valid, 'accepted', 'decoded_representation_nonfinite').astype(object)
        reasons[partners < 0] = 'no_distinct_within_group_partner'
        self.last_proposal = {
            'geometry_valid': valid, 'geometry_reasons': reasons,
            'interpolation_coefficients': coefficients[:, 0],
            'parent_group_ids': self.groups[seeds],
            'partner_group_ids': np.where(partners >= 0, self.groups[np.maximum(partners, 0)], -1),
            'raw_decode_negative': np.zeros(len(seeds), dtype=bool),
            'raw_decode_rate_failure': np.zeros(len(seeds), dtype=bool),
            'nonlinearity_displacement': np.linalg.norm(generated.astype(np.float64) - counterfactual, axis=1),
            'inverse_seconds': inverse_seconds, 'decode_seconds': decode_seconds,
            'geometry_generation_seconds': perf_counter()-tick, 'generation_geometry': self.variant_name,
        }
        return Candidates(torch.as_tensor(generated, device=real.X.device),
                          torch.full((len(seeds),), budget.target_class, dtype=real.y.dtype, device=real.y.device),
                          torch.as_tensor(seeds, device=real.X.device),
                          torch.as_tensor(partners, device=real.X.device),
                          torch.full((len(seeds),), -1, dtype=torch.long, device=real.X.device))


def make_extended(context):
    """Return zero-argument samplers; every output retains the original real prefix.

    ``model_fn`` must return an actual binary lbfgs LogisticRegression for utility
    adapters and accept ``sample_weight`` for downweighted native trajectories.
    Callable ``audit`` attributes expose real counts, masks and model provenance.
    """
    c = context
    X, y = np.asarray(c['X'], dtype=np.float32), np.asarray(c['y'], dtype=np.int64)
    seed, parameters = int(c['seed']), dict(c['parameters'])
    raw, preprocessor = c['raw'], c['preprocessor']
    fit_ids = np.asarray(c.get('fit_ids', [f'fit-{i}' for i in range(len(y))]))
    validation_ids = np.asarray(c.get('validation_ids', [f'validation-{i}' for i in range(len(c['validation_y']))]))
    cached, adapters = {}, {}
    rounds = int(c.get('max_iterations', 20))

    def flow():
        if 'flow' not in cached:
            fitting, calibration = train_test_split(np.arange(len(y)), test_size=.2,
                                                    stratify=y, random_state=seed + 7919)
            core = fit_flow_gmm(raw.iloc[fitting], y[fitting], fit_ids[fitting], seed,
                                FlowGMMConfig(epochs=60, layers=4, hidden=32, batch_size=128, device='cpu'))
            gate = _NumericFlowGate(core, preprocessor, calibration_row_ids=fit_ids[calibration],
                                    validation_row_ids=validation_ids)
            scores = calibrate_flow(core, raw.iloc[calibration], y[calibration], fit_ids[calibration])
            cached['flow'] = (core, gate, scores)
        return cached['flow']

    def unpack(result, weight=1.):
        features, labels = _array(result.X).astype(np.float32), _array(result.y).astype(np.int64)
        weights = None if weight == 1 else np.r_[np.ones(len(y)), np.full(len(labels)-len(y), weight)]
        return features, labels, weights

    def trajectory(*, family='u2', weight=1., selector=None, gate=None, iterations=None):
        def fit(px, py):
            count = len(py)
            weights = None if weight == 1 or count == len(y) else np.r_[np.ones(len(y)), np.full(count-len(y), weight)]
            return c['model_fn'](px, py, sample_weight=weights)
        gate = flow()[1] if family == 'u2_f1' else gate
        selector = RandomPoolSelector(seed, k=5) if family == 'pool_random' else selector
        values = {**parameters, 'capture_proposals': True}
        if gate is not None:
            values['f1_tau'] = .5
        return protected_ugo(X, y, seed, (), rounds if iterations is None else iterations, 'cpu', fit,
                             c['targets'], values, c['prepared'], flow_support=gate, candidate_selector=selector)

    def filtered(method, tau=.5, q=None):
        kwargs = {}
        values = dict(parameters)
        if method in ('ugo_f1', 'ugo_f2', 'ugo_combined_f1'):
            _, gate, calibration = flow()
            kwargs['flow_support'] = CalibratedFlowGate(gate, calibration, q) if method == 'ugo_f2' else gate
            values['f1_tau'] = tau
        if method == 'ugo_combined_f1':
            kwargs['seed_score'] = lambda model, features, labels: torch.as_tensor(
                c['oof'], dtype=features.dtype, device=features.device)
        return resample(method, X, y, seed, device='cpu', model_fn=c['model_fn'],
                        max_iterations=rounds, target_counts=c['targets'], parameters=values,
                        group_partition=c['prepared'], **kwargs)

    def utility(phase, role, branch):
        guidance_X, guidance_y = c[f'{role}_X'], c[f'{role}_y']
        kwargs = dict(k=5, verification_size=0)
        if role == 'validation':
            selector = ValidationGuidedSelector(branch, guidance_X, guidance_y, seed,
                         algorithm='logistic_regression', scorer_for_model=lambda model: model, **kwargs)
        else:
            selector = CandidatePoolSelector(branch, guidance_X, guidance_y, seed, **kwargs)
        result = trajectory(selector=selector, iterations=1 if phase == 'initial' else rounds)
        result.support_summary['learning_benefit']['guidance_role'] = (
            'held-out validation; ranking only; disjoint from fit and separate guidance'
            if role == 'validation' else
            'separate held-out real guidance; ranking only; disjoint from fit and validation')
        return result

    def geometry(mode):
        proposer = _NumericGroupGeometry(c, flow()[0], mode)
        return trajectory(selector=RandomPoolSelector(seed, k=5, proposal_generator=proposer))

    def weighted(family, weight):
        key = (family, weight)
        if key not in cached:
            cached[key] = trajectory(family=family, weight=weight)
        return cached[key]

    def parentdup(family):
        source = weighted(family, .25)
        anchors = np.asarray([row['seed_index'] for row in source.provenance], dtype=np.int64)
        return (np.concatenate((X, X[anchors])), np.concatenate((y, y[anchors])),
                np.r_[np.ones(len(y)), np.full(len(anchors), .25)])

    def tabdiff():
        if 'tabdiff' in cached:
            return cached['tabdiff']
        model = c['model_fn'](X, y)
        cap = sum(c['targets'][int(label)] - int((y == label).sum()) for label in np.unique(y))
        candidates, schedule = fixed_u2_schedule(X, y, model, c['prepared'], parameters,
                                                 seed, (), c['targets'], cap)
        anchors, labels = _array(candidates.seed_indices), _array(candidates.y)
        groups = np.asarray(c['prepared']['row_group_ids'])[anchors]
        # A one-column block is the nearest nonempty feasible mask for both doses.
        mask = [raw.columns[np.random.default_rng(seed + 16201).integers(raw.shape[1])]]
        with TemporaryDirectory(prefix='synthetic-tabdiff-') as folder:
            generator = fit_tabdiff(raw, y, fit_ids, seed + 1889, folder,
                         TabDiffConfig(epochs=60, batch_size=128, hidden_dim=32, num_layers=1,
                                       d_token=4, factor=4, num_steps=10), device='cpu')
            generated, generation_audit = generator.inpaint(raw.iloc[anchors].reset_index(drop=True),
                                                           mask, labels, seed + 23021)
            generator_audit = generator.training_audit
        native = _array(candidates.X).astype(np.float32)
        emitted = np.full_like(native, np.nan)
        valid = np.asarray(generation_audit['valid_mask']) & np.isfinite(generated.to_numpy(dtype=float)).all(axis=1)
        emitted[valid] = preprocessor.transform(generated.iloc[np.flatnonzero(valid)])
        points = {'u2': native, '25': emitted}
        masks = {'u2': np.isfinite(native).all(axis=1), '25': valid}
        results = {}
        for use_f1 in (False, True):
            eligible = dict(masks)
            if use_f1:
                gate = flow()[1]
                eligible = {arm: keep & gate.predict(points[arm], labels, .5)['accepted']
                            for arm, keep in masks.items()}
            selected, counts = match_group_counts(eligible, groups,
                                schedule['allocation_record']['group_trace'], seed)
            for arm in points:
                key = f'tabdiff{arm}' + ('_f1' if use_f1 else '') if arm != 'u2' else 'tabdiff_u2' + ('_f1' if use_f1 else '')
                keep = selected[arm]
                results[key] = (np.concatenate((X, points[arm][keep])), np.concatenate((y, labels[keep])), None)
                adapters[key].audit = {'generator': generator_audit, 'sampling': generation_audit,
                    'masked_columns': [] if arm == 'u2' else mask,
                    'nominal_fractions': [] if arm == 'u2' else [.25, .5],
                    'actual_fraction': 0 if arm == 'u2' else len(mask)/raw.shape[1], 'matched_groups': counts,
                    'one_round': True, 'f1': use_f1}
        cached['tabdiff'] = results
        return results

    def register(key, run, weight=1., raw_output=False):
        def adapter():
            result = run()
            if raw_output:
                output = result
            else:
                adapter.audit = {'stop_reason': result.stop_reason, 'rounds': result.rounds,
                                 'support': result.support_summary, 'provenance': result.provenance}
                output = unpack(result, weight)
            return output
        adapter.audit = {}
        adapters[key] = adapter

    register('ugo_combined_f1', partial(filtered, 'ugo_combined_f1'))
    register('ugo_h1', partial(filtered, 'ugo_h1'))
    register('ugo_n_reject', partial(filtered, 'ugo_n_reject'))
    register('ugo_f1', partial(filtered, 'ugo_f1'))
    register('ugo_f1_tau06', partial(filtered, 'ugo_f1', tau=.6))
    for suffix, q in (('off', None), ('q001', .01), ('q005', .05), ('q010', .10)):
        register('ugo_f2_' + suffix, partial(filtered, 'ugo_f2', q=q))
    for phase in ('initial', 'iterative'):
        for role in ('guidance', 'validation'):
            for branch in ('random', 'utility'):
                register(f'{phase}_{role}_{branch}', partial(utility, phase, role, branch))
    register('pool_random', partial(trajectory, family='pool_random'))
    register('pool_random_f1', lambda: trajectory(family='pool_random', gate=flow()[1]))
    register('group_input', partial(geometry, 'input'))
    register('group_flow', partial(geometry, 'latent'))
    for key in ('tabdiff_u2', 'tabdiff_u2_f1', 'tabdiff25', 'tabdiff25_f1'):
        register(key, lambda key=key: tabdiff()[key], raw_output=True)
    for family in ('u2', 'u2_f1', 'pool_random'):
        for weight, suffix in ((1., '1'), (.5, '05'), (.25, '025')):
            register(f'downweight_{family}_{suffix}', partial(weighted, family, weight), weight)
        register(f'parentdup_{family}', partial(parentdup, family), raw_output=True)
    return adapters
