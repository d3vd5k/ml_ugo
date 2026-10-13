"""Independently verify synthetic-only downweighting artifacts and native refits."""
import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RECIPES = REPO / 'results/runs/20261011T154542-1b40c4b555c4/recipes'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RECIPES))
spec = importlib.util.spec_from_file_location('downweight_independent_recipe', RECIPES / 'tabdiff_verify.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

ALGORITHMS = ('logistic_regression', 'random_forest', 'lightgbm', 'mlp')
COEFFICIENTS = (.25, .5, 1.)


def metric_check(saved, y, probability, detail):
    """All 24 independent metrics, preserving native probability clipping dtype."""
    from sklearn.metrics import log_loss
    expected = v.metrics(y, probability)
    prevalence = np.asarray([(np.asarray(y) == label).mean() for label in (0, 1)])
    weights = np.where(np.asarray(y) == 0, 1 / (2 * prevalence[0]), 1 / (2 * prevalence[1]))
    expected['ordinary_cross_entropy'] = float(log_loss(y, probability, labels=[0, 1]))
    expected['balanced_cross_entropy'] = float(log_loss(y, probability, labels=[0, 1], sample_weight=weights))
    v.check(len(expected) == 24, 'independent_24_metrics', detail)
    numeric = {key: value for key, value in saved.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
    numeric.pop('primary_metric_value', None)
    v.check(set(numeric) == set(expected), 'saved_24_metric_fields', detail)
    for key, value in expected.items():
        if key in saved:
            v.close(saved[key], value, 'metric_recomputation', detail + ':' + key)
    if 'primary_metric_value' in saved:
        v.close(saved['primary_metric_value'], expected['average_precision'], 'primary_AP', detail)
    return expected



class IndependentMLP:
    """Native fixed-epoch torch fit, without producer training/factory calls."""

    def __init__(self, X, y, weight, parameters, seed, device):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        from torch import nn
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        self.classes_ = np.unique(y)
        x = torch.as_tensor(X, dtype=torch.float32, device=device)
        labels = torch.as_tensor(np.searchsorted(self.classes_, y), dtype=torch.long, device=device)
        w = None
        if not np.all(np.asarray(weight) == 1):
            w = torch.as_tensor(weight, dtype=torch.float32, device=device)
            w = w / w.mean()
        hidden = parameters['hidden_dim']
        self.model = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Dropout(.25),
                                   nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(.25),
                                   nn.Linear(hidden, len(self.classes_))).to(device)
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=parameters['lr'])
        self.best_epoch_ = parameters['epochs']
        for _ in range(parameters['epochs']):
            self.model.train()
            optimizer.zero_grad()
            logits = self.model(x)
            loss = nn.functional.cross_entropy(logits, labels) if w is None else (
                nn.functional.cross_entropy(logits, labels, reduction='none') * w).mean()
            loss.backward()
            optimizer.step()
        self.model.eval()

    def predict_proba(self, X):
        import torch
        with torch.no_grad():
            x = torch.as_tensor(X, dtype=torch.float32, device=next(self.model.parameters()).device)
            return torch.softmax(self.model(x), dim=1).cpu().numpy()



def weighted_refit(saved, X, y, weight, algorithm, seed, parameters, detail):
    if algorithm == 'mlp':
        import torch
        expected = dict(epochs=160, warmup_epochs=100, patience=20, device='cpu',
                        random_state=seed, lr=.001, hidden_dim=20)
        expected.update(parameters)
        v.check(saved.get_params() == expected, 'MLP_full_locked_parameters', detail)
        device = next(saved.model_.parameters()).device
        v.check(device.type == expected['device'], 'MLP_actual_device', detail)
        layers = list(saved.model_.net)
        v.check([type(x) for x in layers] == [torch.nn.Linear, torch.nn.ReLU, torch.nn.Dropout,
                torch.nn.Linear, torch.nn.ReLU, torch.nn.Dropout, torch.nn.Linear], 'MLP_native_architecture', detail)
        v.check(all(layers[i].p == .25 for i in (2, 5)), 'MLP_preserved_dropout', detail)
        fresh = IndependentMLP(X, y, weight, expected, seed, device)
        v.equal(saved.classes_, fresh.classes_, 'MLP_classes', detail)
        v.check(saved.model_.training_count_ == len(y), 'MLP_actual_training_count', detail)
        v.check(saved.model_.validation_count_ == 0, 'MLP_no_holdout_training', detail)
        v.check(saved.best_epoch_ == fresh.best_epoch_, 'MLP_independent_best_epoch', detail)
        for key, tensor in fresh.model.state_dict().items():
            v.equal(saved.model_.net.state_dict()[key].detach().cpu().numpy(), tensor.detach().cpu().numpy(),
                    'MLP_independent_learned_tensors', detail + ':' + key)
    else:
        fresh = v.native(algorithm, seed, parameters)
        v.check(type(saved) is type(fresh) and saved.get_params() == fresh.get_params(),
                'native_full_locked_parameters', detail)
        fresh.fit(X, y, sample_weight=None if np.all(np.asarray(weight) == 1) else weight)
        v.equal(saved.classes_, fresh.classes_, 'native_classes', detail)
        if algorithm == 'logistic_regression':
            for key in ('coef_', 'intercept_', 'n_iter_'):
                v.equal(getattr(saved, key), getattr(fresh, key), 'LR_weighted_learned_state', detail + ':' + key)
        elif algorithm == 'random_forest':
            v.check(len(saved.estimators_) == len(fresh.estimators_), 'RF_trees', detail)
            for a, b in zip(saved.estimators_, fresh.estimators_):
                for key in ('children_left', 'children_right', 'feature', 'threshold', 'value',
                            'n_node_samples', 'weighted_n_node_samples'):
                    v.equal(getattr(a.tree_, key), getattr(b.tree_, key), 'RF_weighted_learned_state', detail + ':' + key)
        elif algorithm == 'lightgbm':
            v.check(saved.booster_.dump_model() == fresh.booster_.dump_model(), 'LGB_weighted_learned_state', detail)
        else:
            raise ValueError('Unsupported native algorithm: ' + algorithm)
    v.COUNTS['independent_weighted_native_refits'] += 1
    v.close(saved.predict_proba(X), fresh.predict_proba(X), 'weighted_refit_training_probabilities', detail,
            atol=1e-13, rtol=1e-12)
    return fresh


def verify_weights(training, real, alpha, detail):
    n = len(real['y'])
    v.equal(training['X'][:n], real['X'], 'exact_real_training_prefix', detail)
    v.equal(training['y'][:n], real['y'], 'exact_real_label_prefix', detail)
    v.check(len(training['X']) == len(training['y']) == len(training['sample_weight']),
            'aligned_actual_training_arrays', detail)
    v.equal(training['sample_weight'], np.r_[np.ones(n), np.full(len(training['y']) - n, alpha)],
            'real_one_synthetic_alpha_weights', detail)
    v.check(np.isfinite(training['X']).all(), 'finite_actual_training_coordinates', detail)


FAMILIES = ('U2_F1', 'U2', 'pool_random')


def welch(scores, labels, eligible):
    from scipy.stats import f_oneway
    groups = [scores[(labels == label) & eligible] for label in np.unique(labels)]
    if any(len(g) < 2 for g in groups):
        return 0.
    constant = [np.allclose(g, g[0]) for g in groups]
    if all(constant):
        return float(np.allclose([g[0] for g in groups], groups[0][0]))
    if any(constant):
        return 0.
    result = float(f_oneway(*groups, equal_var=False).pvalue)
    return result if np.isfinite(result) else 0.


def eligible_real(real, groups):
    labels = real['y']
    eligible = np.ones(len(labels), dtype=bool)
    eligible[np.asarray(groups['cross_class_conflict_indices'], dtype=int)] = False
    target = min(np.unique(labels), key=lambda label: int((labels == label).sum()))
    eligible[labels == target] = False
    for group in groups['groups']:
        if group['real_support'] >= 2 and group['distinct_coordinates'] >= 2 and group['stability'] > 0:
            eligible[group['indices']] = True
    for label in np.unique(labels):
        ix = np.flatnonzero((labels == label) & eligible)
        if len(ix) < 2 or len(np.unique(real['X'][ix], axis=0)) < 2:
            eligible[ix] = False
    return eligible


def verify_pool(pool, seed, accepted_keys, detail):
    final = pool['final_coordinates']
    anchors = pool['anchor_indices']
    gids = pool['group_ids']
    keep = np.ones(len(final), dtype=bool)
    seen = set()
    for i, row in enumerate(final):
        key = tuple(row.tolist())
        if not np.isfinite(row).all() or key in seen or key in accepted_keys:
            keep[i] = False
        if np.isfinite(row).all():
            seen.add(key)
    v.equal(pool['keep'], keep, 'independent_pool_first_owner_dedup', detail)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1, int(pool['round'])]))
    selected = np.zeros(len(final), dtype=bool)
    for gid, quota in pool['group_quotas'].items():
        positions = np.flatnonzero(keep & (gids == int(gid)))
        representatives = np.asarray([rng.choice(positions[anchors[positions] == anchor])
                                      for anchor in np.unique(anchors[positions])], dtype=int)
        selected[representatives[rng.permutation(len(representatives))[:int(quota)]]] = True
    v.equal(pool['attempted_selected'], selected, 'independent_seeded_group_random_subset', detail)
    v.equal(pool['selected'], selected, 'K5_no_F1_no_refill_all_attempted_accepted', detail)
    v.check(not any(key.startswith('f1_') for key in pool), 'K5_no_F1_evidence', detail)
    planned = np.asarray(pool['planned_anchor_indices'])
    _, first = np.unique(planned, return_index=True)
    v.equal(anchors, np.repeat(planned[np.sort(first)], 5), 'K5_five_each_distinct_planned_anchor', detail)
    accepted_keys.update(tuple(row.tolist()) for row in final[selected])
    return final[selected], pool['labels'][selected]



def independent_fit(training, algorithm, seed, parameters):
    if algorithm == 'mlp':
        defaults = dict(epochs=160, warmup_epochs=100, patience=20, lr=.001, hidden_dim=20, device='cpu')
        defaults.update(parameters)
        model = IndependentMLP(training['X'], training['y'], training['sample_weight'], defaults,
                               seed, defaults['device'])
    else:
        model = v.native(algorithm, seed, parameters)
        model.fit(training['X'], training['y'], sample_weight=None if np.all(training['sample_weight'] == 1) else training['sample_weight'])
    v.COUNTS['independent_prefix_native_refits'] += 1
    return model


def verify_split(dataset, protocol, split, seed, smoke):
    from sklearn.model_selection import train_test_split
    y = dataset['y']
    train, test = train_test_split(np.arange(len(y)), test_size=.2, random_state=2026, stratify=y)
    canonical_test = test.copy()
    if smoke:
        chosen, _ = train_test_split(train, train_size=520, stratify=y[train], random_state=seed)
        train, test = train_test_split(chosen, train_size=400, stratify=y[chosen], random_state=seed)
        counts = np.bincount(y[train])
        ideal = counts * .2
        quotas = np.clip(np.floor(ideal).astype(int), 1, counts - 1)
        target = round(len(train) * .2)
        while quotas.sum() < target:
            quotas[np.argmax(np.where(quotas < counts - 1, ideal - quotas, -np.inf))] += 1
        while quotas.sum() > target:
            quotas[np.argmax(np.where(quotas > 1, quotas - ideal, -np.inf))] -= 1
        rng = np.random.default_rng(seed)
        fit, validation = [], []
        for label, quota in enumerate(quotas):
            ix = rng.permutation(train[y[train] == label])
            validation.extend(ix[:quota])
            fit.extend(ix[quota:])
        fit, validation = rng.permutation(fit), rng.permutation(validation)
    else:
        fit, validation = train_test_split(train, test_size=protocol['validation_fraction'],
                                           random_state=seed, stratify=y[train])
    for name, expected in [('train_indices', train), ('test_indices', test), ('fit_indices', fit),
                           ('validation_indices', validation), ('canonical_test_indices', canonical_test)]:
        v.equal(split[name], expected, 'independent_protocol_split', str(seed) + ':' + name)
    v.check(not set(fit) & set(validation) and not set(train) & set(test), 'train_evaluation_exclusion', seed)
    v.equal(np.sort(np.r_[fit, validation]), np.sort(train), 'inner_partition_exact_outer_training', seed)


def verify_role(folder, dataset, split, role, seed):
    detail = str(folder)
    real = v.arrays(folder / 'real.npz')
    metadata = v.js(folder / 'preprocessing.json')
    pp = v.obj(folder / 'preprocessor.joblib')
    ix = np.asarray(split['fit_indices' if role == 'inner' else 'train_indices'])
    raw = dataset['X'].iloc[ix].reset_index(drop=True)
    v.verify_preprocessing(raw, metadata, pp, detail)
    v.equal(real['X'], v.independently_transform(raw, metadata), 'independent_real_training_coordinates', detail)
    v.equal(real['y'], dataset['y'][ix], 'actual_real_training_labels', detail)
    v.equal(real['real_indices'], ix, 'exact_role_real_indices', detail)
    v.equal(real['row_ids'], dataset['row_ids'][ix], 'actual_real_role_row_ids', detail)
    validation = np.asarray(split['validation_indices'])
    v.equal(real['validation_X'], v.independently_transform(dataset['X'].iloc[validation], metadata),
            'independent_real_validation_coordinates', detail)
    v.equal(real['validation_y'], dataset['y'][validation], 'actual_real_validation_labels', detail)
    v.equal(real['validation_row_ids'], dataset['row_ids'][validation], 'actual_real_validation_row_ids', detail)
    groups = v.js(folder / 'groups.json')
    v.verify_groups(real['X'], real['y'], groups, seed, detail)
    v.js(folder / 'role.json')
    return real, metadata, groups


def verify_predictions(path, dataset, indices, metadata, model, metrics, detail):
    prediction = v.csv(path)
    v.equal(prediction.row_id, dataset['row_ids'][indices], 'prediction_actual_real_row_ids', detail)
    v.equal(prediction.y_true if 'y_true' in prediction else prediction.label, dataset['y'][indices],
            'prediction_actual_real_labels', detail)
    probability = model.predict_proba(v.independently_transform(dataset['X'].iloc[indices], metadata))
    serialized = prediction[['probability_0', 'probability_1']].to_numpy(dtype=probability.dtype)
    v.close(serialized, probability, 'independent_evaluation_probabilities', detail, atol=1e-13, rtol=1e-12)
    return metric_check(metrics, dataset['y'][indices], probability, detail)



def verify_numeric_allocations(audit, prefixes, real, control, family, smoke, detail):
    summary = audit['support_summary']
    group = summary['group_support']
    config = summary['configuration']
    v.check(config['lambda_select'] == .5 and config['lambda_step'] == .1 and config['k_pre'] == 5 and
            config['k_after'] == 1 and config['phi'] == 2.5 and config['alpha'] == .05 and
            config['max_iterations'] == (2 if smoke else 20), 'unchanged_native_U2_generation_and_stop_parameters', detail)
    allocation = group['allocation']
    v.check(allocation['allocation'] == 'stability' and allocation['seed_cap'] == 4 and
            allocation['group_cap_multiplier'] == 2 and allocation['group_alpha'] == control['group_alpha'],
            'locked_real_U2_quota_configuration', detail)
    y = real['y']
    counts = np.bincount(y)
    nominal = min(40 if smoke else 1000, int(counts.max() - counts.min()))
    v.check(summary['target_counts'] == {'0': int(counts[0]), '1': int(counts[1] + nominal)},
            'independent_real_deficit_matched_target', detail)
    committed = np.zeros(len(y), dtype=int)
    candidates = summary['candidate_log']
    ancestry = audit['provenance']
    for i, record in enumerate(group['allocation_records']):
        probabilities = prefixes[i]['real_probabilities'].astype(np.float32)
        scores = probabilities[np.arange(len(y)), 1 - y] - probabilities[np.arange(len(y)), y]
        generated = sum(row['round'] < i + 1 for row in ancestry)
        remaining = nominal - int(committed.sum())
        requested = min(summary['generation_cap'] - generated, max(1, round(len(y) * .1)))
        if family == 'U2_F1':
            requested = min(requested, remaining)
        used_group = np.asarray([committed[g['indices']].sum() for g in group['groups']], dtype=int)
        anchors, gids, quotas, traces = v.independent_plan(group['groups'], y, scores, control['group_alpha'],
                                                         requested, committed, used_group, remaining)
        v.check(record['requested'] == requested and record['allocated'] == len(anchors) and
                record['class_remaining'] == remaining, 'independent_round_nominal_budget', detail)
        for trace, (gid, n, difficulty, weight, capacity) in zip(record['group_trace'], traces):
            v.check(trace['group_id'] == gid and trace['n_real'] == n and trace['capacity'] == capacity and
                    trace['quota'] == int(quotas[gid]), 'independent_numeric_group_quota_plan', detail)
            v.close(trace['h'], difficulty, 'independent_refreshed_native_group_difficulty', detail)
            v.close(trace['weight'], weight, 'independent_locked_group_alpha_allocation_weight', detail)
        actual = [row for row in candidates if row['round'] == i + 1]
        if family != 'pool_random':
            v.equal([row['seed_index'] for row in actual], anchors, 'independent_native_ranked_real_anchor_slots', detail)
        else:
            # K5 commits only random selected candidates, never all expanded proposals.
            actual = [row for row in ancestry if row['round'] == i + 1]
        np.add.at(committed, [row['seed_index'] for row in actual], 1)



def verify_ancestry(audit, training, real, metadata, payload, family, tau, detail):
    n = len(real['y'])
    ancestry = audit['provenance']
    v.check(len(ancestry) == len(training['y']) - n, 'complete_synthetic_ancestry', detail)
    v.equal([a['row_index'] for a in ancestry], np.arange(n, len(training['y'])), 'ordered_synthetic_training_rows', detail)
    group = audit['support_summary']['group_support']
    supported = {g['id']: g for g in group['groups'] if g['real_support'] >= 2 and
                 g['distinct_coordinates'] >= 2 and g['stability'] > 0}
    for offset, row in enumerate(ancestry):
        a, b, relocation = row['seed_index'], row['partner_index'], row['relocation_index']
        v.check(0 <= a < n and 0 <= b < n and -1 <= relocation < n, 'all_parents_real_training_only', detail)
        v.check(real['y'][a] == real['y'][b] == row['class'] == training['y'][n + offset],
                'real_parents_intended_labels', detail)
        v.check(row['group_id'] in supported and a in supported[row['group_id']]['indices'],
                'supported_real_anchor_group', detail)
        if relocation >= 0:
            v.equal(training['X'][n + offset], real['X'][relocation], 'actual_real_relocation_coordinates', detail)
            v.check(real['y'][relocation] == row['class'], 'real_relocation_intended_label', detail)
        if family == 'U2_F1':
            v.check(relocation == -1, 'F1_no_relocation', detail)
    candidates = audit['support_summary'].get('candidate_log', [])
    if family != 'pool_random':
        accepted = [row for row in candidates if row['accepted']]
        v.check(len(accepted) == len(ancestry), 'complete_actual_candidate_acceptance', detail)
        if accepted:
            v.equal([row['final_coordinates'] for row in accepted], training['X'][n:],
                    'actual_accepted_candidate_training_tail', detail)
            v.equal([row['proposal_id'] for row in accepted], [row['proposal_id'] for row in ancestry],
                    'accepted_proposal_ancestry_ids', detail)
        if family == 'U2_F1' and candidates:
            original = np.asarray([row['original_coordinates'] for row in candidates], dtype=np.float32)
            intended = np.asarray([row['class'] for row in candidates])
            scores = v.independent_f1(original, metadata, payload, intended, tau)
            v.equal([row['accepted'] for row in candidates], scores['accepted'], 'independent_frozen_F1_acceptance', detail)
            v.equal(original, np.asarray([row['final_coordinates'] for row in candidates], dtype=np.float32),
                    'F1_unchanged_original_coordinates', detail)
            for key in ('log_ratio', 'log_likelihood_intended', 'log_likelihood_opposing', 'log_determinant'):
                stored = np.asarray([np.nan if row[key] is None else row[key] for row in candidates])
                v.close(stored, scores[key], 'independent_frozen_F1_numeric_score', detail + ':' + key, atol=1e-8, rtol=1e-8)
        elif family == 'U2':
            v.check(all(row['accepted'] for row in candidates), 'native_U2_no_post_gate', detail)
    committed = np.zeros(n, dtype=int)
    generated = 0
    for i, allocation in enumerate(group['allocation_records']):
        round_number = i + 1
        attempted = ([row for row in candidates if row['round'] == round_number] if family == 'U2_F1' else
                     [row for row in ancestry if row['round'] == round_number])
        for trace in allocation['group_trace']:
            ix = np.asarray(supported[trace['group_id']]['indices'], dtype=int) if trace['group_id'] in supported else np.asarray(
                next(g['indices'] for g in group['groups'] if g['id'] == trace['group_id']), dtype=int)
            v.equal([trace['used_seed'][str(j)] for j in ix], committed[ix], 'actual_consumed_seed_quota_history', detail)
            v.check(trace['used_group'] == int(committed[ix].sum()), 'actual_consumed_group_quota_history', detail)
            v.check(0 <= trace['quota'] <= trace['capacity'], 'round_quota_capacity_bound', detail)
            v.check(trace['used_group'] + trace['quota'] <= 2 * len(ix), 'group_double_real_support_cap', detail)
        for row in attempted:
            committed[row['seed_index']] += 1
        v.check(np.all(committed <= 4), 'cumulative_four_descendants_per_real_anchor', detail)
        for g in group['groups']:
            v.check(int(committed[g['indices']].sum()) <= 2 * g['real_support'], 'cumulative_group_real_support_cap', detail)
        generated += sum(row['round'] == round_number for row in ancestry)
        matching = [row for row in audit['rounds'] if row['round'] == round_number]
        v.check(len(matching) == 1 and matching[0]['accepted'] == sum(row['round'] == round_number for row in ancestry),
                'round_actual_accepted_ancestry_count', detail)
    v.check(generated == len(ancestry), 'all_accepted_rounds_accounted', detail)
    v.check(len(audit['rounds']) <= 20, 'native_twenty_round_limit', detail)
    if family == 'U2_F1':
        budget = audit['support_summary']['proposal_budget']
        v.check(budget['attempted'] == len(candidates) and budget['accepted'] == len(ancestry), 'F1_no_refill_actual_proposal_budget', detail)
        v.equal(budget['attempted_seed_multiplicity'], committed, 'F1_rejections_consume_source_caps', detail)



def verify_flow(seedroot, dataset, split, seed, smoke):
    import torch
    from sklearn.model_selection import train_test_split
    path = seedroot / 'flow-core.pt'
    digest = v.sha(path)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    flow = v.js(seedroot / 'flow.json')
    fit, calibration = train_test_split(split['fit_indices'], test_size=.2,
                                        stratify=dataset['y'][split['fit_indices']], random_state=seed + 7919)
    v.equal(payload['fit_row_ids'], dataset['row_ids'][fit], 'frozen_F1_real_inner_fit_80_percent', seed)
    v.equal(flow['calibration_row_ids'], dataset['row_ids'][calibration], 'F1_unused_real_calibration_20_percent', seed)
    v.equal(flow['validation_row_ids'], dataset['row_ids'][split['validation_indices']], 'F1_real_validation_roles', seed)
    v.check(payload['audit']['synthetic_training_count'] == 0 and payload['audit']['frozen'], 'F1_real_only_frozen_training', seed)
    v.check(payload['config']['epochs'] == 60, 'retained_F1_sixty_epoch_core', seed)
    v.check(flow['source_sha256'] == digest, 'actual_frozen_F1_checkpoint_hash', seed)
    v.verify_preprocessing(dataset['X'].iloc[fit].reset_index(drop=True),
                           v.preprocessing_state(payload['preprocessor']), payload['preprocessor'], str(seed) + ':flow')
    if not smoke:
        v.check(flow['imported_frozen_checkpoint'] and flow['fit_count_this_run'] == 0, 'no_full_run_F1_retraining', seed)
        v.check(v.sha(flow['source_checkpoint']) == digest, 'full_frozen_F1_exact_retained_source', seed)
    else:
        v.check(flow['fit_count_this_run'] == 1, 'actual_smoke_F1_training', seed)
    return payload


def actual_parameters(control, algorithm, smoke, device):
    parameters = dict(control['parameters'])
    if algorithm == 'mlp':
        parameters.update(epochs=20 if smoke else 160, device=device.split(':')[0])
    return parameters



def compare_initial(trajectories, detail, families=FAMILIES):
    for family in families:
        arms = [trajectories[family, weight] for weight in COEFFICIENTS]
        fits = [v.arrays(folder / 'fits/fit-01.npz') for folder in arms]
        for other in fits[1:]:
            for key in ('X', 'y', 'sample_weight', 'real_probabilities', 'validation_probabilities'):
                v.equal(fits[0][key], other[key], 'same_real_only_initial_native_fit', detail + ':' + family + ':' + key)
        audits = [v.js(folder / 'audit.json') for folder in arms]
        if family == 'pool_random':
            pools = [sorted((folder / 'pools').glob('*.npz')) for folder in arms]
            v.check(all(bool(pool) == bool(pools[0]) for pool in pools), 'same_initial_pool_presence', detail + family)
            if pools[0]:
                first = [v.arrays(pool[0]) for pool in pools]
                keys = ('original_coordinates', 'final_coordinates', 'labels', 'anchor_indices', 'partner_indices',
                        'relocation_indices', 'planned_anchor_indices', 'proposal_ids', 'group_ids', 'keep',
                        'attempted_selected', 'selected')
                for other in first[1:]:
                    for key in keys:
                        v.equal(first[0][key], other[key], 'same_initial_K5_pool_and_random_selection', detail + ':' + key)
        else:
            candidates = [[r for r in a['support_summary']['candidate_log'] if r['round'] == 1] for a in audits]
            for other in candidates[1:]:
                keys = ('proposal_id', 'seed_index', 'partner_index', 'relocation_index', 'group_id', 'class',
                        'original_coordinates', 'final_coordinates', 'accepted')
                for key in keys:
                    v.equal([r[key] for r in candidates[0]], [r[key] for r in other],
                            'same_initial_native_proposal_pool', detail + ':' + family + ':' + key)



def verify_trajectory(folder, record, dataset, split, role_data, control, payload, algorithm, seed, smoke):
    real, metadata, groups = role_data
    role, family, alpha = record['role'], record['family'], record['synthetic_weight']
    detail = str(folder)
    training = v.arrays(folder / 'training.npz')
    audit = v.js(folder / 'audit.json')
    saved = v.obj(folder / 'model.joblib')
    device = str(next(saved.model_.parameters()).device) if algorithm == 'mlp' else 'cpu'
    parameters = actual_parameters(control, algorithm, smoke, device)
    verify_weights(training, real, alpha, detail)
    v.check(audit['native_parameters'] == saved.get_params(), 'all_arms_locked_native_parameters', detail)
    v.check(record['config']['selected_parameters'] == control['parameters'], 'record_locked_native_parameters', detail)
    v.check(audit['validation_used_in_fit'] is False and audit['bagging_experiment'] is False and
            audit['no_refill'] is True and audit['no_utility_or_surrogate'] is True,
            'native_no_holdout_bagging_utility_or_refill', detail)
    n = len(real['y'])
    v.check(audit['real_count'] == n and audit['accepted_count'] == len(training['y']) - n,
            'actual_audit_training_counts', detail)
    v.close(audit['added_weight'], training['sample_weight'][n:].sum(), 'actual_added_synthetic_weight_mass', detail)
    v.close(audit['total_weight'], training['sample_weight'].sum(), 'actual_total_native_weight_mass', detail)
    for key in ('native_fit_count', 'native_fit_seconds', 'runtime_seconds', 'flow_inference_seconds'):
        v.check(record[key] == audit[key], 'record_actual_native_cost', detail + ':' + key)
    for key, value in [('actual_count', len(training['y']) - n), ('added_weight', float(training['sample_weight'][n:].sum()))]:
        v.check(record['augmentation'][key] == value, 'record_actual_augmentation_accounting', detail + ':' + key)

    files = sorted((folder / 'fits').glob('fit-*.npz'))
    v.check(bool(files) and len(files) == audit['native_fit_count'], 'complete_actual_native_fit_prefixes', detail)
    prefixes = []
    for number, path in enumerate(files, 1):
        v.check(path.name == f'fit-{number:02d}.npz', 'contiguous_fit_prefix_artifacts', detail)
        prefix = v.arrays(path)
        fit = v.js(path.with_suffix('.json'))
        v.check(fit['fit_number'] == number and fit['training_count'] == len(prefix['y']) and
                fit['real_count'] == len(real['y']) and fit['synthetic_count'] == len(prefix['y']) - len(real['y']),
                'actual_prefix_fit_counts', detail)
        v.check(fit['parameters'] == saved.get_params() and fit['device'] == device,
                'prefix_full_locked_parameters_and_device', detail)
        v.check(fit['validation_used_in_fit'] is False and fit['cold_refit'] is True,
                'native_cold_refit_no_validation_training', detail)
        v.check(fit['real_weight'] == 1 and fit['synthetic_weight'] == alpha and
                fit['added_weight'] == alpha * (len(prefix['y']) - len(real['y'])),
                'actual_prefix_weight_mass', detail)
        v.check(isinstance(fit['state_hash'], str) and len(fit['state_hash']) == 64, 'prefix_native_state_hash_present', detail)
        v.check(fit['actual_epochs'] == (20 if smoke else 160) if algorithm == 'mlp' else fit['actual_epochs'] is None,
                'all_prefixes_same_native_fixed_epoch_policy', detail)
        if algorithm == 'mlp':
            v.check(fit['MLP_loss'] == 'sum(w*CE)/sum(w)' and fit['architecture'] == str(saved.model_),
                    'MLP_actual_architecture_and_weighted_objective', detail)

        verify_weights(prefix, real, alpha, detail + ':fit' + str(number))
        v.equal(prefix['X'], training['X'][:len(prefix['y'])], 'prefix_exact_final_training_history', detail)
        v.equal(prefix['y'], training['y'][:len(prefix['y'])], 'prefix_exact_final_labels', detail)
        fresh = independent_fit(prefix, algorithm, seed, parameters)
        v.close(prefix['real_probabilities'], fresh.predict_proba(real['X']), 'independent_prefix_native_real_scores',
                detail, atol=1e-13, rtol=1e-12)
        v.close(prefix['validation_probabilities'], fresh.predict_proba(real['validation_X']),
                'independent_prefix_real_validation_probabilities', detail, atol=1e-13, rtol=1e-12)
        if algorithm == 'mlp':
            v.check(fresh.best_epoch_ == (20 if smoke else 160), 'same_fixed_MLP_epochs_all_native_fits', detail)
        prefixes.append(prefix)
    v.equal(prefixes[-1]['X'], training['X'], 'final_native_fit_all_actual_training_rows', detail)
    v.equal(prefixes[-1]['y'], training['y'], 'final_native_fit_all_actual_training_labels', detail)
    fresh = weighted_refit(saved, training['X'], training['y'], training['sample_weight'], algorithm,
                           seed, parameters, detail)
    validation = np.asarray(split['validation_indices'])
    verify_predictions(folder / 'validation_predictions.csv', dataset, validation, metadata, fresh,
                       record['validation_metrics'], detail + ':validation')
    evaluation = np.asarray(split['validation_indices' if role == 'inner' else 'test_indices'])
    verify_predictions(record['predictions_path'], dataset, evaluation, metadata, fresh, record['metrics'], detail)
    metrics_artifact = v.js(folder / 'metrics.json')
    metric_check(metrics_artifact['validation'], dataset['y'][validation], fresh.predict_proba(real['validation_X']), detail + ':metrics_artifact')
    if role == 'outer':
        metric_check(metrics_artifact['test'], dataset['y'][evaluation], fresh.predict_proba(v.independently_transform(dataset['X'].iloc[evaluation], metadata)), detail + ':test_metrics_artifact')
    if record['phase'] == 'iterative':
        verify_ancestry(audit, training, real, metadata, payload, family, control['f1_tau'] if family == 'U2_F1' else None, detail)
        verify_numeric_allocations(audit, prefixes, real, control, family, smoke, detail)
        group = audit['support_summary']['group_support']
        eligible = eligible_real(real, group)
        n = len(real['y'])
        accepted_so_far = 0
        v.check(len(prefixes) == len(audit['rounds']) + (0 if audit['final_fit_reused'] else 1), 'one_native_scoring_fit_per_round_and_final', detail)
        for i, round_record in enumerate(audit['rounds']):
            v.check(round_record['round'] == i + 1, 'contiguous_native_rounds', detail)
            prefix = prefixes[i]
            v.check(len(prefix['y']) == n + accepted_so_far, 'round_refit_exact_accepted_prefix_length', detail)
            p = prefix['real_probabilities'].astype(np.float32)
            scores = p[np.arange(n), 1 - real['y']] - p[np.arange(n), real['y']]
            expected = welch(scores, real['y'], eligible)
            v.close(round_record['p_value'], expected, 'independent_native_Welch_stopping_value', detail)
            if expected > .05:
                v.check(round_record['accepted'] == 0 and round_record['proposed'] == 0 and i == len(audit['rounds']) - 1,
                        'Welch_balance_stops_without_proposals', detail)
            accepted_so_far += round_record['accepted']
        v.check(accepted_so_far == len(training['y']) - n, 'rounds_all_actual_synthetics', detail)
        if family == 'pool_random':
            selector = audit['selector']
            v.check(selector['selector'] == 'random' and selector['k'] == 5, 'native_K5_random_no_utility', detail)
            accepted_keys, coordinates, labels = set(), [], []
            pools = sorted((folder / 'pools').glob('*.npz'))
            v.check(len(pools) == len(selector['rounds']), 'complete_actual_K5_pool_artifacts', detail)
            for path in pools:
                number = path.name.split('-')[1]
                meta = v.js(path.with_name(f'round-{number}.json'))
                v.check(meta['pool_sha256'] == v.sha(path), 'actual_K5_pool_hash_binding', detail)
                pool = {**meta, **v.arrays(path)}
                points, intended = verify_pool(pool, seed, accepted_keys, detail)
                coordinates.extend(points)
                labels.extend(intended)
            v.equal(np.asarray(coordinates, dtype=np.float32).reshape((-1, real['X'].shape[1])),
                    training['X'][n:], 'actual_K5_selected_training_tail', detail)
            v.equal(labels, training['y'][n:], 'actual_K5_selected_training_labels', detail)
    elif record['phase'] == 'real_only':
        v.equal(training['X'], real['X'], 'real_only_control_no_added_rows', detail)
        v.equal(training['y'], real['y'], 'real_only_control_all_real_labels', detail)
        v.check(len(prefixes) == 1, 'real_only_one_native_fit', detail)
    return audit, training



def trial_record(folder, parameters):
    audit = v.js(folder / 'audit.json')
    metrics = v.js(folder / 'metrics.json')['validation']
    return {'role': 'inner', 'family': audit['family'], 'synthetic_weight': audit['synthetic_weight'],
            'phase': 'iterative', 'metrics': metrics, 'validation_metrics': metrics,
            'predictions_path': str(folder / 'validation_predictions.csv'),
            'trajectory_path': str(folder), 'config': {'selected_parameters': parameters},
            'augmentation': {'actual_count': audit['accepted_count'], 'added_weight': audit['added_weight']},
            **{key: audit[key] for key in ('native_fit_count', 'native_fit_seconds', 'runtime_seconds', 'flow_inference_seconds')}}


def verify_mlp_search(folder, dataset, split, protocol, control, payload, seed, smoke, inner):
    from sklearn.model_selection import ParameterSampler
    selection = v.js(folder / 'canonical/selection.json')
    trials = selection['trials']
    budget = 1 if smoke else 10
    v.check(len(trials) == budget and selection['no_test_selection'] and selection['objective'] == 'inner_validation_AP',
            'MLP_complete_inner_only_joint_search_budget', folder)
    rng = np.random.RandomState(protocol['tuning_budget'].get('sampler_seed', seed))
    space = protocol.get('search_space', {}).get('mlp', {'hidden_dim': [20, 64, 128], 'lr': [.0003, .001, .003]})
    explicit = protocol.get('candidate_parameters', {}).get('mlp')
    candidates = explicit[:budget] if explicit is not None else [{}] + [next(iter(ParameterSampler(space, n_iter=1, random_state=rng))) for _ in range(budget - 1)]
    base = protocol.get('classifier_parameters', {}).get('mlp', {})
    scores = []
    for index, (trial, candidate) in enumerate(zip(trials, candidates)):
        parameters = {**base, **candidate}
        v.check(trial['candidate'] == index and trial['selected']['inner']['parameters'] == parameters and
                trial['selected']['inner']['sampler_parameters']['group_alpha'] == (0., .5, 1.)[index % 3],
                'independent_MLP_joint_candidate_parameter_stream', folder)
        role = verify_role(folder / 'canonical' / f'candidate-{index}' / 'real', dataset, split, 'inner', seed)
        path = Path(trial['trajectory_path']).resolve()
        v.check(path == (folder / 'canonical' / f'candidate-{index}' / 'trajectory').resolve(), 'MLP_canonical_trial_actual_path', folder)
        trial_control = {'parameters': parameters, 'group_alpha': (0., .5, 1.)[index % 3], 'f1_tau': control['f1_tau']}
        record = trial_record(path, parameters)
        verify_trajectory(path, record, dataset, split, role, trial_control, payload, 'mlp', seed, smoke)
        v.close(trial['AP'], record['metrics']['average_precision'], 'MLP_joint_search_actual_inner_AP', folder)
        scores.append(record['metrics']['average_precision'])
    choice = int(np.argmax(scores))
    v.check(selection['selected_candidate'] == choice and control['selected'] == trials[choice]['selected']['inner'],
            'MLP_native_control_locked_from_inner_AP_first_tie', folder)
    tau_selection = v.js(folder / 'tau/selection.json')
    v.check([row['tau'] for row in tau_selection['trials']] == [.5, .6, .7, .8] and
            tau_selection['no_test_selection'] and tau_selection['unit_weight_only'], 'MLP_unit_inner_only_tau_grid', folder)
    values = []
    for trial in tau_selection['trials']:
        path = folder / 'tau' / f"tau-{trial['tau']:g}"
        record = trial_record(path, control['parameters'])
        v.check(record['synthetic_weight'] == 1, 'MLP_tau_selection_only_unit_weight', path)
        trial_control = {**control, 'f1_tau': trial['tau']}
        verify_trajectory(path, record, dataset, split, inner, trial_control, payload, 'mlp', seed, smoke)
        v.close(trial['AP'], record['metrics']['average_precision'], 'MLP_tau_trial_actual_inner_AP', path)
        values.append(record['metrics']['average_precision'])
    expected = tau_selection['trials'][int(np.argmax(values))]['tau']
    v.check(control['f1_tau'] == tau_selection['selected_tau'] == expected, 'MLP_tau_locked_by_inner_unit_AP_first_tie', folder)


def verify_parent(folder, record, selected_record, dataset, split, role_data, control, algorithm, seed, smoke):
    real, metadata, _ = role_data
    audit = v.js(folder / 'audit.json')
    training = v.arrays(folder / 'training.npz')
    source = v.js(Path(selected_record['trajectory_path']) / 'audit.json')
    anchors = np.asarray([row['seed_index'] for row in source['provenance']], dtype=int)
    v.equal(training['anchor_indices'], anchors, 'parent_duplicates_exact_selected_synthetic_anchors', folder)
    v.equal(training['X'], np.concatenate([real['X'], real['X'][anchors]]), 'matched_real_anchor_duplicate_training', folder)
    v.equal(training['y'], np.concatenate([real['y'], real['y'][anchors]]), 'matched_parent_duplicate_labels', folder)
    v.check(record['synthetic_weight'] == selected_record['synthetic_weight'] and audit['parent_only'],
            'parent_control_selected_coefficient_and_no_synthetic_geometry', folder)
    verify_weights(training, real, record['synthetic_weight'], str(folder))
    saved = v.obj(folder / 'model.joblib')
    device = str(next(saved.model_.parameters()).device) if algorithm == 'mlp' else 'cpu'
    v.check(audit['native_parameters'] == saved.get_params(), 'parent_native_locked_parameters', folder)
    fresh = weighted_refit(saved, training['X'], training['y'], training['sample_weight'], algorithm, seed,
                           actual_parameters(control, algorithm, smoke, device), str(folder))
    verify_predictions(folder / 'validation_predictions.csv', dataset, np.asarray(split['validation_indices']), metadata,
                       fresh, record['validation_metrics'], str(folder) + ':validation')
    verify_predictions(record['predictions_path'], dataset, np.asarray(split['test_indices']), metadata,
                       fresh, record['metrics'], str(folder))
    v.check(audit['source_trajectory'] == record['source_trajectory'] == selected_record['trajectory_path'] and
            audit['accepted_count'] == len(anchors) and record['augmentation']['actual_count'] == len(anchors),
            'parent_actual_selected_trajectory_binding', folder)
    for key in ('native_fit_count', 'native_fit_seconds'):
        v.check(audit[key] == record[key], 'parent_actual_fit_cost', folder)
    v.close(audit['added_weight'], selected_record['augmentation']['added_weight'], 'matched_parent_added_weight_mass', folder)



def verify_selection(folder, records, algorithm, seed, control):
    selection = v.js(folder / 'selection.json')
    v.check(selection['no_test_selection'] and selection['all_inner_selections_before_outer'],
            'all_inner_choices_locked_before_outer_access', folder)
    for family in FAMILIES:
        entry = selection['families'][family]
        v.check(entry['objective'] == 'inner_validation_AP' and entry['tie_break'] == 'larger_weight' and
                entry['no_test_selection'], 'coefficient_selection_inner_AP_only', folder)
        v.check([r['weight'] for r in entry['trials']] == [1., .25, .5], 'complete_selection_unit_and_weighted_trials', folder)
        for trial in entry['trials']:
            row = records[algorithm, seed, f"inner_{family}_w{trial['weight']:g}"]
            v.check(trial['path'] == row['trajectory_path'], 'selection_bound_actual_inner_trajectory', folder)
            v.close(trial['AP'], row['metrics']['average_precision'], 'selection_actual_independent_inner_AP', folder)
        choice = max((.25, .5), key=lambda weight: (records[algorithm, seed, f'inner_{family}_w{weight:g}']['metrics']['average_precision'], weight))
        chosen = records[algorithm, seed, f'inner_{family}_w{choice:g}']
        outer = records[algorithm, seed, f'outer_{family}_selected']
        v.check(entry['selected_weight'] == choice == outer['synthetic_weight'] and
                entry['selected_inner_path'] == chosen['trajectory_path'], 'actual_inner_selected_coefficient_outer_lock', folder)
        v.check(records[algorithm, seed, f'outer_{family}_unit']['synthetic_weight'] == 1,
                'distinct_outer_unit_weight_control', folder)
    v.check(selection['canonical_control']['parameters'] == control['parameters'] and
            selection['canonical_control']['group_alpha'] == control['group_alpha'] and
            selection['canonical_control']['f1_tau'] == control['f1_tau'], 'selection_canonical_native_control_lock', folder)


def verify_comparisons(folder, records, algorithm, seed):
    comparison = v.js(folder / 'comparison.json')
    for role in ('inner', 'outer'):
        for family in FAMILIES:
            unit = records[algorithm, seed, f'inner_{family}_w1' if role == 'inner' else f'outer_{family}_unit']
            entries = comparison[role][family] if role == 'inner' else [comparison[role][family]]
            for entry in entries:
                weighted = records[algorithm, seed, f"inner_{family}_w{entry['weight']:g}" if role == 'inner' else f'outer_{family}_selected']
                a, b = [v.js(Path(row['trajectory_path']) / 'audit.json') for row in (unit, weighted)]
                v.check(entry['initial_selected_candidates_identical'] and entry['initial_pool_identity_checked'],
                        'logged_initial_pool_comparison_proof', folder)
                for key, expected in [('unit_accepted', a['accepted_count']), ('weighted_accepted', b['accepted_count']),
                                      ('unit_rounds', len(a['rounds'])), ('weighted_rounds', len(b['rounds']))]:
                    v.check(entry[key] == expected, 'actual_comparison_counts_not_forced_matching', folder)
                fields = ('seed_index', 'partner_index', 'group_id', 'original_coordinates', 'final_coordinates', 'accepted')
                stripped = [[{k: row[k] for k in fields} for row in audit['support_summary']['candidate_log']] for audit in (a, b)]
                v.check(entry['later_candidate_trajectory_diverged'] == (stripped[0] != stripped[1]),
                        'actual_logged_trajectory_divergence', folder)
                if role == 'outer':
                    paths = [Path(r['trajectory_path']) for r in (unit, weighted)]
                    # Reuse the three-arm initial checker with the actual weighted path twice.
                    compare_initial({(f, w): paths[0 if w == 1 else 1] for f in (family,) for w in COEFFICIENTS},
                                    str(folder) + ':outer', families=(family,))


def verify(root):
    import hashlib
    import importlib.metadata
    import zipfile
    import pandas as pd
    from benchmarks.protocols import load_dataset
    root = Path(root).resolve()
    v.COUNTS.clear()
    v.FAILURES.clear()
    v.HASHES.clear()
    for path in (__file__, RECIPES / 'tabdiff_verify.py', RECIPES / 'tabdiff_report.py'):
        v.sha(path)
    manifest = v.js(root / 'manifest.json')
    rows = v.js(root / 'benchmark_records.json')
    args = manifest['arguments']
    smoke, seeds = bool(args['smoke']), args['effective_seeds']
    v.check(manifest['status'] == 'completed', 'completed_actual_invocation', root)
    v.check(args['effective_classifiers'] == list(ALGORITHMS), 'all_four_native_classifiers_present', root)
    v.check(seeds == [0] if smoke else seeds == list(range(5)), 'complete_matched_seed_design', root)
    v.check(args['weights'] == [.25, .5] and args['control_weight'] == 1 and args['bagging_deferred'] and
            args['no_automatic_retention'], 'precommitted_synthetic_only_grid_and_exclusions', root)
    settings = args['effective_settings']
    v.check(settings == {'augmentation': {'mode': 'matched', 'total': 40 if smoke else 1000},
                         'mlp_epochs': 20 if smoke else 160, 'ugo_max_iterations': 2 if smoke else 20},
            'fixed_native_epoch_round_and_budget_design', root)
    variants = ['inner_real_only', 'outer_real_only']
    for family in FAMILIES:
        variants += [f'inner_{family}_w{weight:g}' for weight in COEFFICIENTS]
        variants += [f'outer_{family}_{suffix}' for suffix in ('unit', 'selected', 'parent')]
    expected = {(algorithm, seed, variant) for algorithm in ALGORITHMS for seed in seeds for variant in variants}
    records = {(r['algorithm'], r['seed'], r['variant_id']): r for r in rows}
    v.check(len(rows) == len(records) == len(expected) and set(records) == expected and
            all(row['status'] == 'completed' for row in rows), 'complete_unique_all_twenty_records_each_native_cell', root)
    for row in rows:
        v.check(Path(row['trajectory_path']).resolve().is_relative_to(root) and
                Path(row['predictions_path']).resolve().is_relative_to(root), 'all_new_trajectory_predictions_actual_invocation', row['variant_id'])
    for dependency in ('numpy', 'pandas', 'scikit-learn', 'scipy', 'torch', 'lightgbm'):
        v.check(importlib.metadata.version(dependency) == manifest['dependencies'][dependency], 'actual_native_dependency_version', dependency)
    v.sha(root / 'source.zip')
    with zipfile.ZipFile(root / 'source.zip') as snapshot:
        for filename, digest in manifest['source_sha256'].items():
            v.check(hashlib.sha256(snapshot.read(filename)).hexdigest() == digest, 'actual_saved_source_snapshot_binding', filename)
    for filename in ('all_metrics.csv', 'metrics.csv'):
        v.sha(root / filename)
    audit = v.js(root / 'control_audit.json')
    v.check(audit == v.js(args['controls']), 'exact_retained_canonical_control_audit', root)
    controls = {(row['seed'], row['algorithm']): row['control'] for row in audit['cells']}
    dataset = load_dataset(manifest['protocol_snapshot'], args['data_home'])
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(dataset['X'], index=True).values.tobytes())
    digest.update(str([(str(c), str(t)) for c, t in dataset['X'].dtypes.items()]).encode())
    digest.update(dataset['y'].tobytes())
    digest.update(json.dumps(dataset['row_ids'].tolist()).encode())
    v.check(digest.hexdigest() == manifest['datasets']['shoppers']['fingerprint'], 'actual_real_dataset_fingerprint', root)
    cells = 0
    for seed in seeds:
        seedroot = root / 'shoppers/fold-0' / f'seed-{seed}'
        split = v.js(seedroot / 'split.json')
        verify_split(dataset, manifest['protocol_snapshot'], split, seed, smoke)
        payload = verify_flow(seedroot, dataset, split, seed, smoke)
        for algorithm in ALGORITHMS:
            folder = seedroot / algorithm
            control = v.js(folder / 'control.json')
            v.check(control['native_parameters_locked'] and control['F1_tau_locked_across_weights'], 'explicit_native_and_F1_locks', folder)
            if algorithm != 'mlp':
                canonical = controls[seed, algorithm]
                v.check(all(control[key] == value for key, value in canonical.items()), 'exact_existing_native_canonical_control', folder)
                v.check(v.sha(control['source_selection']) == control['source_selection_sha256'] and
                        v.sha(control['f1_checkpoint']) == control['f1_checkpoint_sha256'], 'actual_retained_control_source_hashes', folder)
                original = v.js(control['source_selection'])
                v.check(control['selected'] == original['inner'], 'native_selected_control_exact_source', folder)
                tau_source = v.js(control['f1_tau_source'])
                v.check(tau_source['inner']['sampler_parameters']['f1_tau'] == control['f1_tau'], 'exact_historical_locked_tau_source_no_reselection', folder)
            roles = {role: verify_role(folder / role, dataset, split, role, seed) for role in ('inner', 'outer')}
            for variant in variants:
                row = records[algorithm, seed, variant]
                if row['phase'] == 'parent_duplication':
                    source = records[algorithm, seed, f"outer_{row['family']}_selected"]
                    verify_parent(Path(row['trajectory_path']), row, source, dataset, split, roles['outer'], control, algorithm, seed, smoke)
                else:
                    verify_trajectory(Path(row['trajectory_path']), row, dataset, split, roles[row['role']], control,
                                      payload, algorithm, seed, smoke)
            inner_paths = {(family, weight): Path(records[algorithm, seed, f'inner_{family}_w{weight:g}']['trajectory_path'])
                           for family in FAMILIES for weight in COEFFICIENTS}
            compare_initial(inner_paths, str(folder))
            verify_selection(folder, records, algorithm, seed, control)
            verify_comparisons(folder, records, algorithm, seed)
            if algorithm == 'mlp':
                verify_mlp_search(folder, dataset, split, manifest['protocol_snapshot'], control, payload, seed, smoke, roles['inner'])
            cells += 1
    return {'status': 'verified' if not v.FAILURES else 'failed', 'root': str(root), 'smoke': smoke,
            'cells': cells, 'records': len(rows), 'checks': dict(v.COUNTS), 'failures': v.FAILURES,
            'actual_file_sha256': v.HASHES,
            'independence': 'Manual weighted native prefix refits and final learned-state comparison, independent PyTorch RMSprop/CE/dropout loop, real-only split/preprocessing/group/F1 numeric audit, ancestry and K5 random thinning; no producer factory, scoring, sampler, generator, or trajectory regeneration.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        result = verify(args.root)
    except Exception:
        result = {'status': 'failed', 'root': str(Path(args.root).resolve()), 'error': traceback.format_exc(),
                  'checks': dict(v.COUNTS), 'failures': v.FAILURES, 'actual_file_sha256': v.HASHES}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'cells', 'records', 'error') if key in result}))
    return 0 if result['status'] == 'verified' else 1


if __name__ == '__main__':
    sys.exit(main())

