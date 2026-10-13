"""Reproducible, single-seed 2D illustrations for the experiment notebook."""
from functools import lru_cache

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

from datasets.scripts.synthetic import create_synthetic_split
from datasets.scripts.climb import make_preprocessor
from samplers.controls import resample
from samplers.protected import prepare_protected_groups

DATASETS = ('linear', 'moons', 'circles')
SEED = 42
PARAMETERS = {'k_pre': 5, 'k_after': 1, 'lambda_step': .1, 'alpha': .05,
              'n_clusters': 8, 'repeats': 5, 'grouping': 'kmeans',
              'allocation': 'stability', 'group_alpha': .5}
BASE_PARAMETERS = {k: PARAMETERS[k] for k in ('k_pre', 'k_after', 'lambda_step', 'alpha')}
BASIC_SECTIONS = {
    'Baseline and resampling controls': [
        ('Real-only', 'baseline'), ('Class weighting', 'class_weight'),
        ('Random oversampling', 'ros'), ('SMOTE (numeric features)', 'smote'),
        ('BorderlineSMOTE', 'borderline_smote'), ('KMeansSMOTE', 'kmeans_smote'),
        ('Balanced random forest (no generated rows)', 'brf')],
    'UGO scoring, constraints, stopping and compositions': [
        ('Original UGO', 'ugo_original'), ('Corrected real-only UGO', 'ugo_corrected'),
        ('U1: OOF margin', 'ugo_oof_mma'), ('U3: bounded interpolation + repair', 'u3_shorten'),
        ('U3: constrained relocation', 'u3_relocate'), ('U3: constraint rejection', 'u3_reject'),
        ('U4: trajectory budget checkpoints', 'u4_checkpoint'),
        ('U4: independent fixed-budget grid', 'u4_fixed_grid'),
        ('Historical OOF + U2 (U5)', 'ugo_combined')],
    'U2 grouping and quota variants': [('KMeans / stability U2', 'u2')] + [
        (f'{group} / {allocation}', f'u2_{grouping}_{allocation}')
        for group, grouping in [('KMeans', 'kmeans'), ('Mutual-kNN', 'graph')]
        for allocation in ('stability', 'size', 'uniform', 'difficulty')
        if (grouping, allocation) != ('kmeans', 'stability')],
}


def array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def fit_model(X, y, sample_weight=None, *, class_weight=None):
    return LogisticRegression(max_iter=1000, random_state=SEED, class_weight=class_weight).fit(
        array(X), array(y), sample_weight=None if sample_weight is None else array(sample_weight))


@lru_cache(None)
def context(dataset):
    torch.set_num_threads(1)
    data = create_synthetic_split(n_samples=600, dataset=dataset, minority_weight=.2,
                                  random_state=SEED, test_fraction=.2)
    raw_train = pd.DataFrame(array(data['X_train']), columns=['x1', 'x2'])
    y_train = array(data['y_train'])
    remaining, validation = train_test_split(np.arange(len(y_train)), test_size=.2,
                                             random_state=SEED, stratify=y_train)
    fit, guidance = train_test_split(remaining, test_size=.25, random_state=SEED,
                                     stratify=y_train[remaining])
    raw = raw_train.iloc[fit].reset_index(drop=True)
    y = y_train[fit]
    preprocessor = make_preprocessor(raw).fit(raw)
    X = preprocessor.transform(raw).astype(np.float32)
    oof = np.empty(len(y), dtype=np.float32)
    for training, heldout in StratifiedKFold(3, shuffle=True, random_state=SEED).split(raw, y):
        fold_preprocessor = make_preprocessor(raw.iloc[training]).fit(raw.iloc[training])
        model = fit_model(fold_preprocessor.transform(raw.iloc[training]), y[training])
        p = model.predict_proba(fold_preprocessor.transform(raw.iloc[heldout]))
        oof[heldout] = p[np.arange(len(heldout)), 1-y[heldout]] - p[np.arange(len(heldout)), y[heldout]]
    classes, counts = np.unique(y, return_counts=True)
    targets = {int(c): int(counts.max()) for c in classes}
    prepared = prepare_protected_groups(X, y, SEED, targets, PARAMETERS)
    prepared['real_row_ids'] = data['train_ids'][fit]
    return dict(X=X, y=y, raw=raw, preprocessor=preprocessor, seed=SEED, model_fn=fit_model,
                targets=targets, parameters=dict(PARAMETERS), prepared=prepared, oof=oof,
                guidance_X=preprocessor.transform(raw_train.iloc[guidance]).astype(np.float32),
                guidance_y=y_train[guidance],
                validation_X=preprocessor.transform(raw_train.iloc[validation]).astype(np.float32),
                validation_y=y_train[validation],
                test_X=preprocessor.transform(pd.DataFrame(array(data['X_test']), columns=['x1', 'x2'])).astype(np.float32),
                test_y=array(data['y_test']), fit_ids=data['train_ids'][fit],
                guidance_ids=data['train_ids'][guidance], validation_ids=data['train_ids'][validation])


def unpack(result):
    return array(result.X).astype(np.float32), array(result.y).astype(np.int64), None


def basic(context, key):
    c = context
    parameters = dict(BASE_PARAMETERS)
    method = key
    kwargs = {}
    if key == 'brf':
        return c['X'], c['y'], None
    if key.startswith('u2') or key == 'ugo_combined':
        method = 'ugo_combined' if key == 'ugo_combined' else 'ugo_groups_only'
        parameters = dict(PARAMETERS)
        if key.startswith('u2_'):
            _, grouping, allocation = key.split('_')
            parameters.update(grouping=grouping, allocation=allocation)
    if key in ('ugo_oof_mma', 'ugo_combined'):
        kwargs['seed_score'] = lambda model, X, y: torch.as_tensor(c['oof'], device=X.device, dtype=X.dtype)
    if key.startswith('u3_'):
        method = 'ugo_constraints_only'
        parameters['repair'] = key.removeprefix('u3_')
    if key.startswith('u4_'):
        candidates = []
        for fraction in (0., .25, .5, 1.):
            result = resample('ugo_stopping_only', c['X'], c['y'], SEED, device='cpu',
                              model_fn=fit_model, target_counts=c['targets'],
                              parameters={**BASE_PARAMETERS, 'policy': key.removeprefix('u4_'),
                                          'budget_fraction': fraction})
            X, y, weight = unpack(result)
            score = average_precision_score(c['validation_y'], fit_model(X, y).predict_proba(c['validation_X'])[:, 1])
            candidates.append((score, -len(y), X, y, weight))
        selected = max(candidates, key=lambda candidate: candidate[:2])
        return selected[2:]
    if method in ('baseline', 'class_weight', 'ros', 'smote', 'borderline_smote', 'kmeans_smote'):
        parameters = {'cluster_balance_threshold': 0.0} if method == 'kmeans_smote' else {}
    return unpack(resample(method, c['X'], c['y'], SEED, device='cpu', model_fn=fit_model,
                           target_counts=c['targets'], parameters=parameters, **kwargs))


@lru_cache(None)
def extended(dataset):
    from reports.synthetic_extended import make_extended
    return make_extended(context(dataset))


@lru_cache(None)
def result(dataset, key):
    c = context(dataset)
    basic_keys = {key for rows in BASIC_SECTIONS.values() for _, key in rows}
    X, y, weight = basic(c, key) if key in basic_keys else extended(dataset)[key]()
    X, y = array(X).astype(np.float32), array(y).astype(np.int64)
    n = len(c['y'])
    np.testing.assert_array_equal(X[:n], c['X'])
    np.testing.assert_array_equal(y[:n], c['y'])
    assert np.isfinite(X).all()
    if key == 'brf':
        model = BalancedRandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1).fit(X, y)
    else:
        model = fit_model(X, y, weight, class_weight='balanced' if key == 'class_weight' else None)
    p = model.predict_proba(c['test_X'])[:, 1]
    predicted = (p >= .5).astype(np.int64)
    truth = c['test_y']
    metrics = {'AP': average_precision_score(truth, p), 'ROC AUC': roc_auc_score(truth, p),
               'F1': f1_score(truth, predicted, zero_division=0),
               'Precision': precision_score(truth, predicted, zero_division=0),
               'Recall': recall_score(truth, predicted, zero_division=0),
               'Balanced accuracy': balanced_accuracy_score(truth, predicted),
               'MCC': matthews_corrcoef(truth, predicted), 'CE': log_loss(truth, p),
               'Added rows': len(y)-n}
    scaler = c['preprocessor'].named_steps['columns'].named_transformers_['numeric'].named_steps['scale']
    return {'original': c['raw'].to_numpy(), 'original_y': c['y'],
            'generated': scaler.inverse_transform(X[n:]) if len(y) > n else np.empty((0, 2)),
            'generated_y': y[n:],
            'weight': weight, 'metrics': metrics}


def sections():
    from reports.synthetic_extended import EXTENDED_SECTIONS
    output = {name: list(rows) for name, rows in BASIC_SECTIONS.items()}
    for name, rows in EXTENDED_SECTIONS.items():
        output.setdefault(name, []).extend(rows)
    return output


def plot_approach(label, key):
    figures = [result(dataset, key) for dataset in DATASETS]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), layout='constrained')
    colors = ('#4076ad', '#e58c38')
    for column, (dataset, data) in enumerate(zip(DATASETS, figures)):
        original, labels, generated = data['original'], data['original_y'], data['generated']
        all_points = np.vstack((original, generated))
        lower, upper = all_points.min(axis=0), all_points.max(axis=0)
        padding = np.maximum((upper-lower)*.08, .1)
        for row in range(2):
            ax = axes[row, column]
            for label_id, color in enumerate(colors):
                points = original[labels == label_id]
                ax.scatter(points[:, 0], points[:, 1], s=11, color=color,
                           alpha=.8 if row == 0 else .28, linewidths=0)
            ax.set(xlim=(lower[0]-padding[0], upper[0]+padding[0]),
                   ylim=(lower[1]-padding[1], upper[1]+padding[1]), xlabel='x1', ylabel='x2')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=.12)
        axes[0, column].set_title(f'{dataset.title()} · original (n={len(original)})', fontsize=10)
        for label_id, color in enumerate(colors):
            points = generated[data['generated_y'] == label_id]
            axes[1, column].scatter(points[:, 0], points[:, 1], s=24, color=color,
                                    marker='x', alpha=.9, linewidths=.9)
        axes[1, column].set_title(f'Original + added rows (n={len(generated)})', fontsize=10)
        if not len(generated):
            axes[1, column].text(.5, .94, 'No generated rows', transform=axes[1, column].transAxes,
                                ha='center', va='top', fontsize=9)
    fig.suptitle(label, fontsize=13)
    handles = [Line2D([], [], color=color, marker='o', linestyle='', markersize=5,
                      label=f'Real class {i}') for i, color in enumerate(colors)]
    handles.append(Line2D([], [], color='black', marker='x', linestyle='', label='Added rows'))
    fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False, fontsize=9)
    return fig


def show_section(name):
    from IPython.display import HTML, display
    approaches = sections()[name]
    rows = []
    for label, key in approaches:
        for dataset in DATASETS:
            metrics = result(dataset, key)['metrics']
            rows.append({'Approach': label, 'Dataset': dataset, **{
                metric: value if metric == 'Added rows' else f'{value:.4f}'
                for metric, value in metrics.items()}})
    display(HTML('<h3>Synthetic datasets — held-out metrics and generation geometry</h3>'))
    display(HTML(pd.DataFrame(rows).to_html(index=False, border=0)))
    for label, key in approaches:
        fig = plot_approach(label, key)
        display(fig)
        plt.close(fig)
