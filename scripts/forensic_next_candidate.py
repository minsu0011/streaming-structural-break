"""Exact paired primary-OOF bootstrap and frozen fractures for one valid model."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from numba import set_num_threads
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next.candidate_io import load_full_oof
from src.next.references import load_reference_fold
from src.next.bootstrap import PreparedAUC, paired_summary
from src.next.bootstrap_integrity import load_reference_samples
from src.validation.group_bootstrap import draw_group_multiplicities
from src.utils.artifacts import write_json, sha256, utc_now
from src.next.progress import checkpoint


def rank_within_time(prediction, times):
    order = np.argsort(times, kind='stable')
    cuts = np.r_[0, np.flatnonzero(np.diff(times[order])) + 1, len(times)]
    result = np.empty(len(times), dtype=np.float32)
    for start, end in zip(cuts[:-1], cuts[1:]):
        ix = order[start:end]
        result[ix] = (rankdata(prediction[ix], method='average') - .5) / len(ix)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    set_num_threads(2)
    started = time.perf_counter()
    data = prepare_dev(ROOT)
    candidate, original, metadata, full_prediction, sources = load_full_oof(data, args.candidate)
    policy = json.loads((ROOT/'ROBUSTNESS_SPLIT_LOCK.json').read_text(encoding='utf-8'))
    meta_path = ROOT/'artifacts/next/current_forensic/DEV_SIGNATURES.parquet'
    regime_lock = json.loads((meta_path.parent/'REGIME_ASSIGNMENT_LOCK.json').read_text(encoding='utf-8'))
    if sha256(meta_path) != regime_lock['files_sha256'][meta_path.name]:
        raise RuntimeError('Frozen signatures changed')
    series_meta = pd.read_parquet(meta_path).set_index('dataset_id')
    data.guard.admit(series_meta.index.to_numpy(), purpose='candidate post-hoc signatures')
    reference_dir = ROOT/'artifacts/next/reference_fracture_bootstrap'
    reference_result = json.loads((reference_dir/'RESULTS.json').read_text(encoding='utf-8'))
    out = ROOT/'artifacts/next/forensic'/args.candidate
    out.mkdir(parents=True, exist_ok=True)
    identity = {'candidate': candidate, 'oof_sources': sources, 'replicates': policy['bootstrap']['final_replicates'],
        'policy_sha256': sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'), 'series_metadata_sha256': sha256(meta_path),
        'reference_fracture_result_sha256': sha256(reference_dir/'RESULTS.json'),
        'script_sha256': sha256(Path(__file__)), 'primary_gate_status_retained': original['status'],
        'sampling': 'Same whole-series multiplicities in each original fold before applying each fixed slice.',
        'support': 'Mechanism has at least 100 break series; H-regime has at least 100 break and 100 no-break series in full DEV.'}
    if (out/'POLICY.json').exists():
        if json.loads((out/'POLICY.json').read_text(encoding='utf-8')) != identity:
            raise RuntimeError('Candidate forensic policy changed')
    else:
        write_json(out/'POLICY.json', identity)
    if (out/'RESULTS.json').exists():
        saved = json.loads((out/'RESULTS.json').read_text(encoding='utf-8'))
        for name, digest in saved['files_sha256'].items():
            if sha256(out/name) != digest:
                raise RuntimeError('Completed forensic artifacts changed')
        print('CANDIDATE FORENSIC retained PASS', args.candidate)
        return
    samples_by_scope, references_by_scope = {}, {}
    folds, supports, correlations, disagreements = [], [], [], []
    for fold in range(5):
        rows = np.flatnonzero(metadata['fold'] == fold)
        ids, y, t = (metadata[k][rows] for k in ('dataset_id', 'target', 'time_online'))
        prediction = full_prediction[rows]
        frame, primary, _ = load_reference_fold(ROOT, data.guard, 'PRIMARY', fold)
        for key in ('dataset_id', 'target', 'time_online'):
            np.testing.assert_array_equal(frame[key].to_numpy(), metadata[key][rows])
        rmeta = series_meta.reindex(ids)
        age = t - rmeta.tau.to_numpy()
        signatures = rmeta.signature.to_numpy()
        masks = {'all_time': np.ones(len(rows), dtype=bool)}
        for name, a, b in [('0-9',0,10), ('10-24',10,25), ('25-49',25,50), ('50-99',50,100), ('100-199',100,200), ('200-399',200,400), ('400+',400,100000)]:
            masks['time_'+name] = (t >= a) & (t < b)
        for name, a, b in [('0-4',0,5), ('5-9',5,10), ('10-19',10,20), ('20-49',20,50), ('50+',50,100000)]:
            masks['age_'+name] = (y == 0) | ((y == 1) & (age >= a) & (age < b))
        for regime in range(5):
            masks['regime_'+str(regime)] = rmeta.historical_regime.to_numpy() == regime
        for signature in ['MEAN_DOMINANT','SCALE_DOMINANT','DEPENDENCE_DOMINANT','TAIL_DOMINANT','MIXED','WEAK_OR_AMBIGUOUS']:
            masks['mechanism_'+signature] = np.isin(signatures, [signature, 'NO_BREAK_CONTROL'])
        code, counts = draw_group_multiplicities(ids, policy['bootstrap']['final_replicates'], policy['bootstrap']['seed'] + fold)
        for scope, mask in masks.items():
            path = out/f'fold_{fold}_{scope}.npy'
            digest_path = out/f'fold_{fold}_{scope}.json'
            if path.exists() and digest_path.exists():
                record = json.loads(digest_path.read_text(encoding='utf-8'))
                if record['sha256'] != sha256(path):
                    raise RuntimeError('Partial candidate bootstrap array changed')
                samples = np.load(path, allow_pickle=False)
            else:
                samples = PreparedAUC.create(y[mask], prediction[mask], t[mask], code[mask], counts.shape[1]).evaluate(counts)
                np.save(path, samples, allow_pickle=False)
                write_json(digest_path, {'sha256': sha256(path), 'created_utc': utc_now()})
            if scope == 'all_time':
                ref_samples = load_reference_samples(ROOT, 'PRIMARY', fold)
            else:
                name = f'PRIMARY_{fold}_{scope}.npy'
                if sha256(reference_dir/name) != reference_result['files_sha256'][name]:
                    raise RuntimeError('Reference fracture bootstrap array changed')
                ref_samples = np.load(reference_dir/name, allow_pickle=False)
            samples_by_scope.setdefault(scope, []).append(samples)
            references_by_scope.setdefault(scope, []).append(ref_samples)
            break_count = len(np.unique(ids[mask & (rmeta.tau.to_numpy() >= 0)]))
            null_count = len(np.unique(ids[mask & (rmeta.tau.to_numpy() < 0)]))
            supports.append({'scope': scope, 'fold': fold, 'break_series': break_count, 'null_series': null_count, 'rows': int(mask.sum())})
            folds.append({'candidate': args.candidate, 'scope': scope, 'fold': fold, 'ts_auc': samples[0], 'primary_ts_auc': ref_samples[0], 'delta': samples[0]-ref_samples[0]})
        p_rank, n_rank = rank_within_time(primary, t), rank_within_time(prediction, t)
        correlations.append({'fold': fold, 'raw_correlation': float(np.corrcoef(primary, prediction)[0,1]),
            'within_time_rank_correlation': float(np.corrcoef(p_rank, n_rank)[0,1])})
        detail = pd.DataFrame({'dataset_id': ids, 'time_online': t, 'absolute_score_difference': np.abs(prediction-primary),
            'rank_delta': n_rank-p_rank, 'absolute_rank_difference': np.abs(n_rank-p_rank)})
        grouped = detail.groupby('dataset_id', sort=False)
        aggregate = grouped.agg(mean_absolute_score_difference=('absolute_score_difference','mean'),
            max_absolute_score_difference=('absolute_score_difference','max'), mean_signed_rank_delta=('rank_delta','mean'),
            mean_absolute_rank_difference=('absolute_rank_difference','mean'))
        aggregate['maximum_difference_online_time'] = detail.loc[grouped.absolute_score_difference.idxmax(), 'time_online'].to_numpy()
        aggregate['fold'] = fold
        aggregate['signature'] = series_meta.reindex(aggregate.index).signature
        aggregate['historical_regime'] = series_meta.reindex(aggregate.index).historical_regime
        disagreements.append(aggregate.reset_index())
        print('CANDIDATE FORENSIC', args.candidate, fold, round(time.perf_counter()-started,1), flush=True)
    support = pd.DataFrame(supports)
    totals = support.groupby('scope')[['break_series','null_series']].sum()
    rows = []
    for scope in samples_by_scope:
        new = np.asarray(samples_by_scope[scope]).T
        ref = np.asarray(references_by_scope[scope]).T
        supported = (int(totals.loc[scope,'break_series']) >= 100 and
            (not scope.startswith('regime_') or int(totals.loc[scope,'null_series']) >= 100))
        rows.append({'scope': scope, 'candidate': args.candidate, 'reference': 'PRIMARY',
            'candidate_mean': float(new[0].mean()), 'reference_mean': float(ref[0].mean()),
            'supported': supported, **totals.loc[scope].astype(int).to_dict(), **paired_summary(ref, new)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out/'PAIRED_FRACTURES.csv', index=False)
    support.to_csv(out/'SUPPORT.csv', index=False)
    pd.DataFrame(folds).to_csv(out/'FOLD_SCORES.csv', index=False)
    pd.DataFrame(correlations).to_csv(out/'RANK_CORRELATIONS.csv', index=False)
    pd.concat(disagreements, ignore_index=True).to_csv(out/'DISAGREEMENTS.csv', index=False)
    subgroup = summary.loc[summary.scope.str.startswith(('regime_','mechanism_')) & summary.supported]
    mean_corr = float(np.mean([row['within_time_rank_correlation'] for row in correlations]))
    complementarity = {'mean_within_time_rank_correlation': mean_corr,
        'supported_positive_delta_at_least_0_002': bool((subgroup.estimate_mean_delta >= .002).any()),
        'supported_negative_delta_at_most_minus_0_002': bool((subgroup.estimate_mean_delta <= -.002).any())}
    complementarity['admitted_for_predeclared_fixed_late_combination'] = bool(mean_corr < .98 and complementarity['supported_positive_delta_at_least_0_002'] and complementarity['supported_negative_delta_at_most_minus_0_002'])
    files = list(out.glob('*.npy')) + list(out.glob('*.csv'))
    result = {'status': 'PASS', 'candidate': args.candidate, 'timestamp': utc_now(), 'primary_gate_status_retained': original['status'],
        'all_time_paired_bootstrap': rows[0], 'comparisons': len(rows), 'complementarity': complementarity,
        'files_sha256': {p.name: sha256(p) for p in files}, 'elapsed_seconds': time.perf_counter()-started,
        'seal_rows': 0, 'reduced_new_usage': 0,
        'limitations': 'Conditional fixed-OOF inference; no correction for adaptive model selection, fitting uncertainty or multiple subgroup comparisons. Mechanisms are post-hoc and evaluation-only.'}
    write_json(out/'RESULTS.json', result)
    existing = pd.read_csv(ROOT/'PAIRED_BOOTSTRAP_RESULTS.csv')
    existing = existing.loc[existing.candidate != args.candidate]
    pd.concat([existing, pd.DataFrame([rows[0]])], ignore_index=True).to_csv(ROOT/'PAIRED_BOOTSTRAP_RESULTS.csv', index=False)
    checkpoint(ROOT, args.candidate+'/forensic', ['24 paired 2000-replicate candidate fracture comparisons'],
        ['candidate_robustness', 'new_sequential_hypotheses'], artifacts=[str((out/'RESULTS.json').relative_to(ROOT))], best=max(.611292686371357, original['full']['mean']))
    print('CANDIDATE FORENSIC COMPLETE', args.candidate, complementarity, flush=True)


if __name__ == '__main__':
    main()
