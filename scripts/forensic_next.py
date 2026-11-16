"""Freeze H-only regimes and produce current-reference DEV forensic evidence."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next.regimes import historical_descriptors, mechanism_signature
from src.next.references import REFERENCES, load_reference_fold, verify_reference_sources
from src.next.progress import checkpoint
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import write_json, sha256, utc_now


def prepare_groups(data, policy, directory):
    path = directory/'REGIME_ASSIGNMENT_LOCK.json'
    identity = {'split_sha256': data.guard.split.digest, 'dev_manifest_sha256': sha256(data.directory/'MANIFEST.json'),
                'policy_sha256': sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),
                'source_sha256': {name: sha256(ROOT/name) for name in ['src/next/regimes.py', 'src/next/whitening.py']},
                'historical_policy_sha256': sha256(ROOT/'HISTORICAL_REGIME_POLICY.md'),
                'mechanism_policy_sha256': sha256(ROOT/'MECHANISM_SIGNATURE_POLICY.md')}
    if path.exists():
        lock = json.loads(path.read_text(encoding='utf-8'))
        if lock['identity'] != identity:
            raise RuntimeError('Regime/diagnostic definition changed')
        for name, digest in lock['files_sha256'].items():
            if sha256(directory/name) != digest:
                raise RuntimeError('Frozen regime artifact changed')
        return pd.read_parquet(directory/'DEV_SIGNATURES.parquet').set_index('dataset_id')
    rows, signatures = [], []
    for j, sid in enumerate(data.ids):
        h, o, tau = data.get(sid)
        rows.append({'dataset_id': int(sid), **historical_descriptors(h)})
        signatures.append({'dataset_id': int(sid), 'tau': tau, 'online_length_diagnostic_only': len(o),
                           **mechanism_signature(h, o, tau, policy['mechanism'])})
        if (j+1) % 1000 == 0:
            print('H-ONLY REGIME DESCRIPTORS', j+1, flush=True)
    descriptors = pd.DataFrame(rows).set_index('dataset_id')
    x = descriptors.to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise RuntimeError('Nonfinite historical descriptors')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1.
    z = np.clip((x-mean)/scale, -8., 8.)
    hp = policy['historical_regime']
    with threadpool_limits(limits=1):
        model = KMeans(n_clusters=hp['k'], random_state=hp['seed'], n_init=hp['n_init'], algorithm='lloyd').fit(z)
    ordered = sorted(range(hp['k']), key=lambda i: tuple(model.cluster_centers_[i]))
    mapping = {old: new for new, old in enumerate(ordered)}
    regime = np.asarray([mapping[int(i)] for i in model.labels_], dtype=np.int8)
    metadata = pd.DataFrame(signatures).set_index('dataset_id')
    metadata['historical_regime'] = regime
    metadata['fold'] = data.series['fold']
    descriptors.reset_index().to_parquet(directory/'DEV_H_DESCRIPTORS.parquet', index=False)
    metadata.reset_index().to_parquet(directory/'DEV_SIGNATURES.parquet', index=False)
    write_json(directory/'HISTORICAL_TRANSFORM.json', {'columns': list(descriptors.columns),
        'mean': mean.tolist(), 'scale': scale.tolist(), 'clip': [-8,8],
        'centers': model.cluster_centers_[ordered].tolist(), 'canonical_order': ordered})
    files = ['DEV_H_DESCRIPTORS.parquet', 'DEV_SIGNATURES.parquet', 'HISTORICAL_TRANSFORM.json']
    write_json(path, {'status': 'LOCKED', 'created_utc': utc_now(), 'identity': identity,
        'files_sha256': {n: sha256(directory/n) for n in files},
        'regime_counts': metadata.historical_regime.value_counts().sort_index().to_dict(),
        'mechanism_counts': metadata.signature.value_counts().to_dict(),
        'regime_uses_historical_only': True, 'mechanism_is_evaluation_only': True, 'seal_rows': 0})
    return metadata


def ranked_within_time(prediction, times):
    order = np.argsort(times, kind='stable')
    cuts = np.r_[0, np.flatnonzero(np.diff(times[order]))+1, len(times)]
    result = np.empty(len(times), dtype=np.float32)
    for start, end in zip(cuts[:-1], cuts[1:]):
        ix = order[start:end]
        result[ix] = (rankdata(prediction[ix], method='average')-.5)/len(ix)
    return result


def main():
    start = time.perf_counter()
    data = prepare_dev(ROOT)
    verify_reference_sources(ROOT)
    out = ROOT/'artifacts/next/current_forensic'
    out.mkdir(exist_ok=True)
    policy = json.loads((ROOT/'ROBUSTNESS_SPLIT_LOCK.json').read_text(encoding='utf-8'))
    metadata = prepare_groups(data, policy, out)
    if (out/'RESULTS.json').exists():
        print('Forensic already complete; retained', flush=True)
        return
    regime_rows, mechanism_rows, age_rows, null_rows, correlations, disagreements = [], [], [], [], [], []
    for fold in range(5):
        ref, primary, _ = load_reference_fold(ROOT, data.guard, 'PRIMARY', fold)
        rmeta = metadata.reindex(ref.dataset_id)
        target, times, ids = ref.target.to_numpy(), ref.time_online.to_numpy(), ref.dataset_id.to_numpy()
        prank = ranked_within_time(primary, times)
        age = times-rmeta.tau.to_numpy()
        for role in REFERENCES:
            if role == 'PRIMARY':
                pred = primary
            else:
                other, pred, _ = load_reference_fold(ROOT, data.guard, role, fold)
                pd.testing.assert_frame_equal(ref, other)
            rank = prank if role == 'PRIMARY' else ranked_within_time(pred, times)
            correlations.append({'fold': fold, 'candidate': role,
                'raw_correlation': float(np.corrcoef(primary, pred)[0,1]),
                'within_time_rank_correlation': float(np.corrcoef(prank, rank)[0,1])})
            if role != 'PRIMARY':
                detail = pd.DataFrame({'dataset_id': ids, 'time': times,
                    'absolute_score_difference': np.abs(pred-primary), 'rank_delta': rank-prank})
                for sid, group in detail.groupby('dataset_id', sort=False):
                    maximum = group.absolute_score_difference.idxmax()
                    disagreements.append({'candidate': role, 'fold': fold, 'dataset_id': int(sid),
                        'mean_absolute_score_difference': float(group.absolute_score_difference.mean()),
                        'max_absolute_score_difference': float(group.loc[maximum, 'absolute_score_difference']),
                        'maximum_difference_online_time': int(group.loc[maximum, 'time']),
                        'mean_absolute_within_time_rank_difference': float(group.rank_delta.abs().mean()),
                        'mean_signed_rank_delta': float(group.rank_delta.mean()),
                        'signature': metadata.loc[sid, 'signature'], 'historical_regime': int(metadata.loc[sid, 'historical_regime'])})
            for regime in range(5):
                mask = rmeta.historical_regime.to_numpy() == regime
                score, details = ts_auc(target[mask], pred[mask], times[mask], return_details=True)
                regime_rows.append({'candidate': role, 'evaluation': 'primary_OOF_within_regime_diagnostic',
                    'fold': fold, 'regime': regime, 'series': len(np.unique(ids[mask])),
                    'pair_count': sum(d['weight'] for d in details), 'ts_auc': score})
            for signature in ['MEAN_DOMINANT', 'SCALE_DOMINANT', 'DEPENDENCE_DOMINANT', 'TAIL_DOMINANT', 'MIXED', 'WEAK_OR_AMBIGUOUS']:
                signatures = rmeta.signature.to_numpy()
                mask = (signatures == signature) | (signatures == 'NO_BREAK_CONTROL')
                score, details = ts_auc(target[mask], pred[mask], times[mask], return_details=True)
                mechanism_rows.append({'candidate': role, 'evaluation': 'primary_OOF_mechanism_plus_null_controls',
                    'fold': fold, 'signature': signature,
                    'break_series': len(np.unique(ids[signatures == signature])),
                    'control_series': len(np.unique(ids[signatures == 'NO_BREAK_CONTROL'])),
                    'pair_count': sum(d['weight'] for d in details), 'ts_auc': score})
            for label, a, b in [('0-4',0,5), ('5-9',5,10), ('10-19',10,20), ('20-49',20,50), ('50+',50,100000)]:
                mask = (target == 0) | ((age >= a) & (age < b) & (target == 1))
                age_rows.append({'candidate': role, 'fold': fold, 'break_age': label,
                                 'ts_auc': ts_auc(target[mask], pred[mask], times[mask])})
            null = rmeta.tau.to_numpy() < 0
            per_series = pd.DataFrame({'id': ids[null], 'prediction': pred[null]}).groupby('id').prediction.max()
            null_rows.append({'candidate': role, 'fold': fold, 'null_series': len(per_series),
                'mean_score': float(pred[null].mean()), 'mean_maximum': float(per_series.mean()),
                'p95_maximum': float(per_series.quantile(.95)),
                'fraction_any_score_over_fixed_0_5': float((per_series > .5).mean()),
                'interpretation': 'Uncalibrated fixed-score diagnostic, not train-calibrated alarm rate'})
        print('CURRENT FORENSIC fold', fold, 'complete', flush=True)
    pd.DataFrame(disagreements).to_csv(ROOT/'CURRENT_TOP_DISAGREEMENT.csv', index=False)
    pd.DataFrame(regime_rows).to_csv(ROOT/'HISTORICAL_REGIME_RESULTS.csv', index=False)
    pd.DataFrame(mechanism_rows).to_csv(ROOT/'MECHANISM_FRACTURE_RESULTS.csv', index=False)
    pd.DataFrame(age_rows).to_csv(out/'REFERENCE_BREAK_AGE.csv', index=False)
    pd.DataFrame(null_rows).to_csv(out/'REFERENCE_NULL_SCORES.csv', index=False)
    pd.DataFrame(correlations).to_csv(out/'REFERENCE_RANK_CORRELATIONS.csv', index=False)
    regime_summary = pd.DataFrame(regime_rows).groupby(['candidate', 'regime']).ts_auc.mean().unstack(0)
    mechanism_summary = pd.DataFrame(mechanism_rows).groupby(['candidate', 'signature']).ts_auc.mean().unstack(0)
    regime_summary.to_csv(out/'REGIME_SUMMARY.csv')
    mechanism_summary.to_csv(out/'MECHANISM_SUMMARY.csv')
    md = '# Current top DEV forensic\n\n'
    md += 'Frozen PRIMARY, HIGHEST_MEAN and ROBUST_BACKUP OOFs were re-scored at float32 and matched their five original scores. Two thousand paired whole-series bootstrap replicates give an HIGHEST_MEAN − PRIMARY delta of +0.001242 with a 95% interval crossing zero; the mean ranking alone is insufficient to replace the primary.\n\n'
    md += 'Historical clusters are H-only and fixed at K=5; this initial table is primary OOF subgroup evaluation. Leave-one-regime-out refits will be separately labelled in the stress table. Mechanism signatures use post-hoc tau windows and are evaluation-only.\n\n'
    md += 'Regime mean-of-fold TS-AUC:\n\n```text\n'+regime_summary.to_string(float_format=lambda x:f'{x:.6f}')+'\n```\n\n'
    md += 'Mechanism plus no-break control mean-of-fold TS-AUC:\n\n```text\n'+mechanism_summary.to_string(float_format=lambda x:f'{x:.6f}')+'\n```\n\n'
    md += 'Each signature includes all no-break controls. Categories and regimes have different pair sets, so neither table is an additive decomposition. Fixed score >0.5 false-positive diagnostics are uncalibrated and must not be interpreted as operating alarm probabilities. Disagreement CSV contains all 8,000 DEV series for each alternative, including the largest-difference online time.\n'
    (ROOT/'CURRENT_TOP_FORENSIC.md').write_text(md, encoding='utf-8')
    write_json(out/'RESULTS.json', {'status': 'PASS', 'timestamp': utc_now(), 'seal_rows': 0,
        'reduced_new_usage': 0, 'series': len(metadata), 'elapsed_seconds': time.perf_counter()-start,
        'signature_counts': metadata.signature.value_counts().to_dict(),
        'regime_counts': metadata.historical_regime.value_counts().sort_index().to_dict(),
        'fold_rank_correlations': correlations})
    checkpoint(ROOT, 'AR-whitening', ['reference_2000_bootstrap', 'historical_regime_lock', 'current_top_forensic'],
               ['AR1/2/4/8/12', 'innovation_normalization', 'sequential_evidence'],
               artifacts=['CURRENT_TOP_FORENSIC.md', 'CURRENT_TOP_DISAGREEMENT.csv'], best=.611292686371357)
    print(regime_summary.to_string(), flush=True)
    print(mechanism_summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
