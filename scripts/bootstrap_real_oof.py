"""Paired group-bootstrap diagnostics of frozen DEV OOF predictions, no refits."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.validation.group_bootstrap import draw_group_multiplicities, weighted_ts_auc_replicates
from src.validation.splits import FrozenSplit, assert_development_ids
from src.utils.artifacts import write_json, sha256, utc_now
from src.scoring.ts_auc import ts_auc


def interval(array):
    return {'estimate': float(array[0]), 'bootstrap_mean': float(np.mean(array[1:])),
            'percentile_95_low': float(np.quantile(array[1:], .025)),
            'percentile_95_high': float(np.quantile(array[1:], .975)),
            'bootstrap_std': float(np.std(array[1:], ddof=1))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', nargs='+', required=True)
    parser.add_argument('--replicates', type=int, default=200)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum(): raise ValueError('Simple alphanumeric tag required')
    split = FrozenSplit(ROOT / 'artifacts/validation')
    ids = split.frame.loc[split.frame.role == 'DEVELOPMENT_POOL', 'dataset_id'].tolist()
    metadata = pd.read_parquet(ROOT / 'artifacts/audits/SERIES_METADATA.parquet', filters=[('dataset_id', 'in', ids)]).set_index('dataset_id')
    directory = ROOT / 'artifacts/real/bootstrap' / args.tag
    directory.mkdir(parents=True, exist_ok=True)
    policy = {'seed': 20260908, 'replicates': args.replicates, 'unit': 'frozen historical duplicate group',
              'strata': 'original development validation fold', 'comparison': 'paired, same multiplicities for all candidates',
              'candidates': args.candidates, 'split_sha256': split.digest,
              'slices': ['all_time', 'online_time_0_9', 'break_age_0_4_with_negative_controls'],
              'source_sha256': sha256(ROOT / 'src/validation/group_bootstrap.py'),
              'limitations': 'Conditional on fixed OOF predictions and adaptive candidate selection. No model refitting; not a selection-adjusted confidence interval.'}
    if (directory / 'POLICY.json').exists() and json.loads((directory / 'POLICY.json').read_text()) != policy:
        raise RuntimeError('Existing bootstrap policy cannot be modified')
    write_json(directory / 'POLICY.json', policy)
    started = time.perf_counter()
    scores = {candidate: {label: [] for label in policy['slices']} for candidate in args.candidates}
    reference_manifest = json.loads((ROOT / 'data/processed/oof' / args.candidates[0] / 'MANIFEST.json').read_text())
    for fold in range(5):
        reference = pd.read_parquet(ROOT / reference_manifest['cache_directory'] / f'fold_{fold}.parquet', columns=['dataset_id', 'time_online', 'target'])
        assert_development_ids(reference.dataset_id.unique(), split)
        if set(reference.dataset_id.unique()) != set(split.frame.loc[(split.frame.role == 'DEVELOPMENT_POOL') & (split.frame.fold == fold), 'dataset_id']):
            raise RuntimeError('Fold coverage mismatch')
        groups = np.array([split.groups[sid] for sid in reference.dataset_id])
        code, counts = draw_group_multiplicities(groups, args.replicates, 20260908 + fold)
        target = reference.target.to_numpy(); online_time = reference.time_online.to_numpy()
        age = online_time - metadata.tau_index.reindex(reference.dataset_id).to_numpy()
        masks = {'all_time': np.ones(len(reference), dtype=bool), 'online_time_0_9': online_time < 10,
                 'break_age_0_4_with_negative_controls': (target == 0) | ((age >= 0) & (age < 5))}
        for candidate in args.candidates:
            oof = ROOT / 'data/processed/oof' / candidate
            manifest = json.loads((oof / 'MANIFEST.json').read_text())
            if manifest['split_sha256'] != split.digest: raise RuntimeError('OOF split hash mismatch')
            if manifest['cache_directory'] != reference_manifest['cache_directory']:
                other = pd.read_parquet(ROOT / manifest['cache_directory'] / f'fold_{fold}.parquet', columns=list(reference.columns))
                pd.testing.assert_frame_equal(reference, other)
            prediction = np.load(oof / f'fold_{fold}.npy', allow_pickle=False)
            for label, mask in masks.items():
                samples = weighted_ts_auc_replicates(target[mask], prediction[mask], online_time[mask], code[mask], counts)
                np.testing.assert_allclose(samples[0], ts_auc(target[mask], prediction[mask], online_time[mask]), atol=1e-14, rtol=0)
                scores[candidate][label].append(samples)
            print('BOOTSTRAP', args.tag, 'fold', fold, candidate, flush=True)
    report = {'status': 'PASS', 'timestamp': utc_now(), 'policy': policy, 'candidates': {}, 'paired_differences': {},
              'LOCAL_FINAL_SEAL_OPENED': False, 'reduced_targets_used': False}
    reference_id = args.candidates[0]
    for candidate in args.candidates:
        report['candidates'][candidate] = {}
        report['paired_differences'][candidate] = {}
        for label in policy['slices']:
            matrix = np.array(scores[candidate][label]).T
            reference_matrix = np.array(scores[reference_id][label]).T
            report['candidates'][candidate][label] = {
                'mean_of_folds': interval(matrix.mean(axis=1)), 'median_of_folds': interval(np.median(matrix, axis=1)),
                'worst_fold': interval(matrix.min(axis=1)), 'folds': [interval(matrix[:, k]) for k in range(5)]}
            delta = matrix.mean(axis=1) - reference_matrix.mean(axis=1)
            report['paired_differences'][candidate][label] = {**interval(delta), 'reference': reference_id,
                'bootstrap_fraction_delta_above_zero': float(np.mean(delta[1:] > 0))}
    report['elapsed_seconds'] = time.perf_counter() - started
    write_json(directory / 'RESULTS.json', report)
    np.savez_compressed(directory / 'REPLICATE_SCORES.npz', **{f'{candidate}__{label}': np.array(values).T for candidate, slices in scores.items() for label, values in slices.items()})
    print('BOOTSTRAP COMPLETE', directory.relative_to(ROOT), round(report['elapsed_seconds'], 1), 'seconds', flush=True)


if __name__ == '__main__': main()
