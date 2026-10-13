"""Independently audit saved schema-only U2 replay, refits and all 24 metrics."""
import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RECIPES = REPO / 'results/runs/20261011T154542-1b40c4b555c4/recipes'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RECIPES))
spec = importlib.util.spec_from_file_location('independent_parent_verifier', RECIPES / 'tabdiff_verify.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def bind(path, proof, detail):
    path = Path(path).resolve()
    digest = v.sha(path)
    v.check(proof['actual_file_sha256'].get(str(path)) == digest,
            'parent_independent_proof_file_binding', detail + ':' + str(path))
    return path


def verify_cell(folder, records, proofs):
    source = v.js(folder / 'source.json')
    parent = Path(source['parent_cell']).resolve()
    parentroot = next(p for p in parent.parents if (p / 'independent_verification.json').exists())
    if parentroot not in proofs:
        proof = v.js(parentroot / 'independent_verification.json')
        v.check(proof['status'] == 'verified' and not proof['failures'], 'verified_parent_proof', parentroot)
        proofs[parentroot] = proof
    proof = proofs[parentroot]
    role = folder.name
    algorithm = folder.parent.name
    seed = int(folder.parent.parent.name.split('-')[-1])
    detail = f'{seed}/{algorithm}/{role}'
    hashes = source['paths_sha256']
    for path, digest in hashes.items():
        actual = v.sha(path)
        v.check(actual == digest, 'source_pointer_actual_SHA256', path)
        v.check(proof['actual_file_sha256'].get(str(Path(path).resolve())) == actual,
                'source_pointer_bound_parent_proof', path)
    # Shared real rows, preprocessing, anchors and roles were independently audited.
    shared = ['real.npz', 'real_raw.joblib', 'preprocessor.joblib', 'preprocessing.json',
              'groups.json', 'schedule.npz', 'schedule.json', 'role.json', 'initial_native.joblib']
    for name in shared:
        bind(parent / name, proof, detail)
    seedroot = parent.parent.parent
    split = v.js(bind(seedroot / 'split.json', proof, detail))
    masks = v.js(bind(seedroot / 'masks.json', proof, detail))
    control = v.js(bind(parent.parent / 'control.json', proof, detail))
    generator = seedroot / role
    for name in ['generator/generator.pt', 'generator_role.json', 'generator/training_audit.json']:
        bind(generator / name, proof, detail)
    z = v.arrays(parent / 'real.npz')
    raw = v.obj(parent / 'real_raw.joblib')
    metadata = v.js(parent / 'preprocessing.json')
    pp = v.obj(parent / 'preprocessor.joblib')
    v.verify_preprocessing(raw, metadata, pp, detail)
    v.equal(v.independently_transform(raw, metadata), z['X'], 'real_only_training_transform', detail)
    fit = split['fit_indices' if role == 'inner' else 'train_indices']
    evaluation = split['validation_indices' if role == 'inner' else 'test_indices']
    v.equal(z['real_indices'], fit, 'role_exact_fit_indices', detail)
    v.check(not set(fit) & set(evaluation), 'fit_evaluation_exclusion', detail)
    schedule = v.arrays(parent / 'schedule.npz')
    plan = v.js(parent / 'schedule.json')
    pools = {}
    metrics = {}
    for arm in ('A', 'B', 'C'):
        d = folder / arm
        original = parent / arm
        for name in ['proposals.npz', 'generated_raw.joblib', 'decoded.joblib', 'generation.json']:
            bind(original / name, proof, detail + arm)
        old = v.arrays(original / 'proposals.npz')
        a = v.arrays(d / 'proposals.npz')
        forbidden = {'f1_valid', 'f1_accepted', 'log_ratio', 'compatibility_score',
                     'log_determinant', 'log_likelihood_intended', 'log_likelihood_opposing', 'f1_reasons'}
        v.check(not forbidden.intersection(a), 'no_old_F1_arrays', detail + arm)
        v.check(set(a) == set(old) - forbidden, 'complete_pool_without_F1_arrays', detail + arm)
        for key in a:
            if key not in ('eligible', 'selected'):
                v.equal(a[key], old[key], 'complete_frozen_pool_exact_array', detail + arm + ':' + key)
        v.equal(a['eligible'], a['schema_valid'], 'exclusively_schema_eligibility', detail + arm)
        generated = v.obj(original / 'generated_raw.joblib')
        decoded = v.obj(original / 'decoded.joblib')
        generation = v.js(original / 'generation.json')
        independently_decoded, valid, changes, _ = v.common_codec(generated, raw)
        valid &= np.asarray(generation['valid_mask'], dtype=bool)
        v.equal(valid, a['schema_valid'], 'independent_schema_codec_validity', detail + arm)
        for col in raw:
            v.equal(independently_decoded[col], decoded[col], 'raw_schema_count_rint_codec', detail + arm + ':' + col)
            if arm != 'A' and col not in masks['masks'][arm]:
                v.equal(generated[col], raw.iloc[a['anchor_indices']][col], 'unmasked_raw_preservation', detail + arm + ':' + col)
        v.equal(a['labels'], schedule['labels'], 'intended_unmasked_labels', detail + arm)
        v.equal(a['labels'], z['y'][a['anchor_indices']], 'anchor_source_labels', detail + arm)
        positions = np.flatnonzero(valid)
        coords = np.full_like(a['coordinates'], np.nan)
        coords[positions] = v.independently_transform(independently_decoded.iloc[positions], metadata)
        v.close(coords, a['coordinates'], 'raw_to_classifier_coordinates', detail + arm, atol=0, rtol=0)
        pools[arm] = a
        audit = v.js(d / 'audit.json')
        v.check(audit['F1_enabled'] is False, 'F1_disabled_audit', detail + arm)
        oldaudit = v.js(bind(original / 'audit.json', proof, detail + arm))
        v.check(audit['F1_scoring_calls'] == 0 and not audit['relocation'] and not audit['utility_ranking'], 'no_F1_utility_relocation', detail + arm)
        for key in ['schema_failure_counts', 'count_decode_changed_cells', 'exact_real_duplicates', 'exact_pool_duplicates', 'near_duplicates']:
            v.check(audit[key] == oldaudit[key], 'parent_audited_schema_duplicate_diagnostics', detail + arm + ':' + key)
        checkpoint = audit['generator_checkpoint']
        v.check(checkpoint['checkpoint_sha256'] == v.sha(generator / 'generator/generator.pt'), 'actual_generator_checkpoint_binding', detail + arm)
        v.check(all(checkpoint[k] == oldaudit['generator_checkpoint'][k] for k in checkpoint), 'frozen_generator_metadata_exact', detail + arm)
        for key in ['sampling_seconds', 'generator_training_seconds']:
            v.check(audit[key] == 0, 'fresh_replay_no_generation_or_training', detail + arm + ':' + key)
        training = v.arrays(d / 'training.npz')
        v.equal(training['X'], np.concatenate([z['X'], coords[a['selected']]]), 'actual_real_selected_training_prefix', detail + arm)
        v.equal(training['y'], np.concatenate([z['y'], a['labels'][a['selected']]]), 'actual_training_labels', detail + arm)
        v.equal(training['sample_weight'], np.ones(len(training['y'])), 'actual_unit_weights', detail + arm)
        fresh = v.native_refit(v.obj(d / 'model.joblib'), training['X'], training['y'], algorithm, seed, control['parameters'], detail + arm)
        pred = v.csv(d / ('validation_predictions.csv' if role == 'inner' else 'test_predictions.csv'))
        parentpred = v.csv(original / ('validation_predictions.csv' if role == 'inner' else 'test_predictions.csv'))
        v.equal(pred.row_id, parentpred.row_id, 'evaluation_parent_row_ids', detail + arm)
        v.equal(pred.label, parentpred.label, 'evaluation_parent_labels', detail + arm)
        # Parent probabilities never enter the new fit or metric recomputation.
        from benchmarks.protocols import load_dataset
        manifest = v.js(parentroot / 'manifest.json')
        dataset = load_dataset(manifest['protocol_snapshot'], manifest['arguments']['data_home'])
        evaluation_X = v.independently_transform(dataset['X'].iloc[evaluation], metadata)
        v.equal(pred.row_id, dataset['row_ids'][evaluation], 'actual_evaluation_row_ids', detail + arm)
        v.equal(pred.label, dataset['y'][evaluation], 'actual_evaluation_labels', detail + arm)
        p = fresh.predict_proba(evaluation_X)
        v.close(pred[['probability_0', 'probability_1']], p, 'independent_refit_evaluation_predictions', detail + arm, atol=1e-13, rtol=1e-12)
        record = records[(algorithm, seed, role + '_' + arm)]
        metrics[arm] = v.metric_check(record['metrics'], pred.label.to_numpy(), p, detail + arm)
        if (d / 'metrics.json').exists():
            v.metric_check(v.js(d / 'metrics.json'), pred.label.to_numpy(), p, detail + arm + ':artifact')
        v.check(record['config']['F1_enabled'] is False, 'record_no_F1', detail + arm)
        for key, value in [('proposal_count', len(valid)), ('accepted_count', int(a['selected'].sum())),
                           ('schema_valid_count', int(valid.sum())), ('quota_shortfall', len(valid) - int(a['selected'].sum()))]:
            v.check(audit[key] == value, 'actual_count_quota_accounting', detail + arm + ':' + key)
    matched = v.js(folder / 'matched_quotas.json')
    rng = {a: np.random.default_rng(seed + 23021 + i * 1009) for i, a in enumerate(('A', 'B', 'C'))}
    selected = {a: np.zeros(len(pools[a]['eligible']), bool) for a in pools}
    v.check(len(matched['groups']) == len(plan['allocation_record']['group_trace']), 'complete_group_quota_trace', detail)
    for entry, quota in zip(matched['groups'], plan['allocation_record']['group_trace']):
        gid = quota['group_id']
        possible = {a: np.flatnonzero(p['schema_valid'] & (p['group_ids'] == gid)) for a, p in pools.items()}
        common = min(quota['quota'], *(len(ix) for ix in possible.values()))
        expected = {'group_id': gid, 'nominal_quota': quota['quota'], 'available': {a: len(ix) for a, ix in possible.items()}, 'matched_accepted': common, 'quota_shortfall': quota['quota'] - common}
        v.check(entry == expected, 'manual_group_min_quota_and_shortfall', detail)
        for a, ix in possible.items():
            selected[a][rng[a].permutation(ix)[:common]] = True
    for a in pools:
        v.equal(selected[a], pools[a]['selected'], 'manual_seeded_uniform_thinning', detail + a)
    v.check(len({int(s.sum()) for s in selected.values()}) == 1, 'all_arms_equal_counts', detail)
    return metrics


def verify(root):
    root = Path(root).resolve()
    rows = v.js(root / 'benchmark_records.json')
    manifest = v.js(root / 'manifest.json')
    v.check(manifest['status'] == 'completed', 'completed_invocation')
    records = {(r['algorithm'], r['seed'], r['variant_id']): r for r in rows}
    v.check(len(records) == len(rows), 'unique_records')
    smoke = manifest['arguments']['smoke']
    seeds = manifest['arguments']['effective_seeds']
    algorithms = ('logistic_regression', 'random_forest', 'lightgbm')
    variants = ['inner_A', 'inner_B', 'inner_C', 'outer_A', 'outer_B', 'outer_C', 'outer_selected_TabDiff', 'prior_f1_A', 'prior_f1_B', 'prior_f1_C', 'prior_f1_selected_TabDiff']
    if not smoke:
        variants += ['historical_U0', 'historical_U2', 'historical_F1', 'historical_U5_F1']
    expected = {(a, s, variant) for a in algorithms for s in seeds for variant in variants}
    v.check(set(records) == expected and len(rows) == len(expected), 'complete_current_and_reference_records')
    v.check(smoke or seeds == list(range(5)), 'full_five_seeds')
    source_root = Path(manifest['arguments']['source_root']).resolve()
    v.check(v.sha(source_root / 'independent_verification.json') == manifest['arguments']['source_verification_sha256'], 'manifest_parent_proof_SHA_binding')
    costs = v.js(root / 'invocation_costs.json')
    v.check(costs['generator_fits'] == 0 and costs['F1_scoring_calls'] == 0, 'actual_no_new_generator_or_F1_invocations')
    proofs = {}
    folders = sorted(root.glob('shoppers/fold-0/seed-*/*/inner')) + sorted(root.glob('shoppers/fold-0/seed-*/*/outer'))
    v.check(bool(folders), 'actual_replay_cells_present')
    for folder in folders:
        verify_cell(folder, records, proofs)
    v.check(len(folders) == 6 * len(seeds), 'complete_six_roles_per_seed')
    for parentroot in proofs:
        v.check(parentroot == source_root, 'all_cells_same_locked_parent_root')
    for record in rows:
        if record['variant_id'].startswith(('historical_', 'prior_f1_')):
            pred = v.csv(record['predictions_path'])
            label_column = 'label' if 'label' in pred else 'y_true'
            v.metric_check(record['metrics'], pred[label_column].to_numpy(), pred[['probability_0', 'probability_1']].to_numpy(), record['variant_id'])
    for algorithm, seed in sorted({(k[0], k[1]) for k in records}):
        inner = {a: records[(algorithm, seed, 'inner_' + a)]['metrics']['average_precision'] for a in ('A', 'B', 'C')}
        choice = 'B' if inner['B'] >= inner['C'] else 'C'
        selected = records[(algorithm, seed, 'outer_selected_TabDiff')]
        base = records[(algorithm, seed, 'outer_' + choice)]
        v.check(selected['selected_arm'] == choice and selected['metrics'] == base['metrics'] and selected['predictions_path'] == base['predictions_path'] and selected['native_fit_count'] == 0,
                'fresh_inner_only_selection_no_extra_fit', f'{seed}/{algorithm}')
    return {'status': 'verified' if not v.FAILURES else 'failed', 'root': str(root), 'cells': len(folders), 'records': len(rows), 'checks': dict(v.COUNTS), 'failures': v.FAILURES, 'actual_file_sha256': v.HASHES,
            'independence': 'Durable audited-parent independent codec/native/24-metric recipe; parent proofs re-bound by actual SHA256; manual group-min seeded thinning; no producer matcher, factory, F1 scoring or initial refit.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    v.sha(__file__)
    v.sha(RECIPES / 'tabdiff_verify.py')
    v.sha(RECIPES / 'tabdiff_report.py')
    try:
        result = verify(args.root)
    except Exception:
        result = {'status': 'failed', 'root': str(Path(args.root).resolve()), 'error': traceback.format_exc(), 'checks': dict(v.COUNTS), 'failures': v.FAILURES, 'actual_file_sha256': v.HASHES}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'cells', 'records', 'error') if k in result}))
    return 0 if result['status'] == 'verified' else 1


if __name__ == '__main__':
    sys.exit(main())
