"""Conditional paired alarm-rate uncertainty, preserving every original fold.

This is a post-hoc diagnostic of already fixed outer predictions and thresholds.
It neither recalibrates alarms nor chooses a production operating point.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.guard import SealGuard
from src.validation.group_bootstrap import draw_group_multiplicities
from src.utils.artifacts import write_json, sha256, utc_now


def paired_rate(weights, mask, left, right):
    mask = np.asarray(mask, dtype=bool)
    left, right = np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)
    denominator = weights[:, mask].sum(axis=1)
    if not mask.any():
        return {'eligible_series': 0, 'status': 'UNSUPPORTED_EMPTY', 'replicates': weights.shape[0]-1}
    a = weights[:, mask & left].sum(axis=1)
    b = weights[:, mask & right].sum(axis=1)
    valid = denominator > 0
    delta = np.full(len(weights), np.nan)
    delta[valid] = (b[valid]-a[valid])/denominator[valid]
    finite = delta[1:][valid[1:]]
    return {'status': 'PASS' if valid.all() else 'UNDEFINED_BOOTSTRAP_DENOMINATORS',
        'eligible_series': int(mask.sum()), 'primary_events': int(a[0]), 'candidate_events': int(b[0]),
        'primary_fraction': float(a[0]/denominator[0]), 'candidate_fraction': float(b[0]/denominator[0]),
        'estimate_candidate_minus_primary': float(delta[0]),
        'ci_2_5': float(np.quantile(finite, .025)) if valid.all() else None,
        'ci_97_5': float(np.quantile(finite, .975)) if valid.all() else None,
        'conditional_probability_delta_positive': float(np.mean(finite > 0)) if valid.all() else None,
        'replicates': len(weights)-1, 'undefined_replicates': int((~valid[1:]).sum())}


def main():
    guard = SealGuard(ROOT)
    started = time.perf_counter()
    base = ROOT/'artifacts/next/alarm_calibration'
    out = base/'paired_forensic'
    out.mkdir(exist_ok=True)
    if (out/'RESULTS.json').exists():
        saved = json.loads((out/'RESULTS.json').read_text(encoding='utf-8'))
        if any(sha256(out/name) != digest for name, digest in saved['files_sha256'].items()):
            raise RuntimeError('Completed paired alarm diagnostic changed')
        print('PAIRED ALARM FORENSIC retained PASS')
        return
    sources = {}
    frames = []
    for eid in ('PRIMARY', 'V007_VARIANCE_FORECAST'):
        for fold in range(5):
            folder = base/eid/f'fold_{fold}'
            receipt = json.loads((folder/'RESULTS.json').read_text(encoding='utf-8'))
            if receipt['status'] != 'PASS':
                raise RuntimeError('Alarm diagnostic fold incomplete')
            for name, digest in receipt['files_sha256'].items():
                if sha256(folder/name) != digest:
                    raise RuntimeError('Fixed alarm predictions or thresholds changed')
                sources[str((folder/name).relative_to(ROOT))] = digest
            frame = pd.read_csv(folder/'SERIES.csv')
            guard.admit(frame.dataset_id.to_numpy(), purpose='paired alarm post-hoc diagnostics')
            if set(frame.fold) != {fold} or set(frame.candidate) != {eid}:
                raise RuntimeError('Alarm fold/candidate identity changed')
            frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    signatures_path = ROOT/'artifacts/next/current_forensic/DEV_SIGNATURES.parquet'
    lock = json.loads((signatures_path.parent/'REGIME_ASSIGNMENT_LOCK.json').read_text(encoding='utf-8'))
    if sha256(signatures_path) != lock['files_sha256'][signatures_path.name]:
        raise RuntimeError('Frozen signatures changed')
    signatures = pd.read_parquet(signatures_path).set_index('dataset_id')
    sources[str(signatures_path.relative_to(ROOT))] = sha256(signatures_path)
    policy_path = ROOT/'ROBUSTNESS_SPLIT_LOCK.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    identity = {'status': 'LOCKED', 'source_files_sha256': sources, 'created_utc': utc_now(),
        'replicates': policy['bootstrap']['final_replicates'], 'seed': policy['bootstrap']['seed'],
        'script_sha256': sha256(Path(__file__)), 'policy_sha256': sha256(policy_path),
        'selection': 'Post-hoc uncertainty and descriptive subgroup fractures after viewing the aggregate alarm summary; not independent confirmation.',
        'estimand': 'Difference in pooled per-series event rates with identical eligibility denominators, not TS-AUC or an unweighted mean of fold rates.',
        'resampling': 'Resample whole series independently within each original fold, with the same draws for both fixed models and all scopes.',
        'support_rule': 'At least 100 eligible series; sparse results stay visible and cannot establish a subgroup claim.'}
    write_json(out/'POLICY.json', identity)
    rows = []
    for alpha in (.01, .05):
        primary = frame.loc[(frame.candidate == 'PRIMARY') & (frame.nominal_alpha == alpha)].sort_values('dataset_id').reset_index(drop=True)
        candidate = frame.loc[(frame.candidate == 'V007_VARIANCE_FORECAST') & (frame.nominal_alpha == alpha)].sort_values('dataset_id').reset_index(drop=True)
        for key in ('dataset_id', 'fold', 'online_length', 'tau', 'has_break', 'post_break_observed_points'):
            np.testing.assert_array_equal(primary[key].to_numpy(), candidate[key].to_numpy())
        if len(primary) != len(guard.dev_ids) or set(primary.dataset_id) != set(guard.dev_ids):
            raise RuntimeError('Incomplete or duplicated alarm DEV coverage')
        weights = np.zeros((policy['bootstrap']['final_replicates']+1, len(primary)), dtype=np.int32)
        for fold in range(5):
            positions = np.flatnonzero(primary.fold.to_numpy() == fold)
            ids = primary.dataset_id.to_numpy()[positions]
            expected = {sid for sid, (role, f) in guard.split.mapping.items() if role == 'DEVELOPMENT_POOL' and f == fold}
            if set(ids) != expected:
                raise RuntimeError('Alarm original fold changed')
            code, counts = draw_group_multiplicities(ids, policy['bootstrap']['final_replicates'], policy['bootstrap']['seed']+fold)
            weights[:, positions] = counts[:, code]
        sig = signatures.reindex(primary.dataset_id)
        scopes = {'all_series': np.ones(len(primary), dtype=bool)}
        scopes.update({f'regime_{j}': sig.historical_regime.to_numpy() == j for j in range(5)})
        scopes.update({'mechanism_'+name: np.isin(sig.signature, [name, 'NO_BREAK_CONTROL']) for name in ('MEAN_DOMINANT', 'SCALE_DOMINANT', 'DEPENDENCE_DOMINANT', 'TAIL_DOMINANT', 'MIXED', 'WEAK_OR_AMBIGUOUS')})
        for scope, scope_mask in scopes.items():
            metrics = [('null_false_alarm', ~primary.has_break.to_numpy(), primary.null_false_alarm, candidate.null_false_alarm, 'lower'),
                ('premature_break_alarm', primary.has_break.to_numpy(), primary.premature_break_alarm, candidate.premature_break_alarm, 'lower'),
                ('clean_detection', primary.has_break.to_numpy(), primary.clean_post_break_detection, candidate.clean_post_break_detection, 'higher')]
            for horizon in (5, 10, 20, 50):
                eligible = primary.has_break.to_numpy() & (primary.post_break_observed_points.to_numpy() >= horizon)
                metrics.append((f'clean_detection_within_{horizon}', eligible,
                    primary.clean_post_break_detection & (primary.delay < horizon), candidate.clean_post_break_detection & (candidate.delay < horizon), 'higher'))
            for metric, eligible, a, b, direction in metrics:
                result = paired_rate(weights, scope_mask & eligible, a, b)
                rows.append({'scope': scope, 'nominal_alpha': alpha, 'metric': metric,
                    'favorable_direction': direction, 'supported': result['eligible_series'] >= 100, **result})
        print('PAIRED ALARM FORENSIC', alpha, len(rows), flush=True)
    pd.DataFrame(rows).to_csv(out/'PAIRED_EVENT_RATES.csv', index=False)
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'comparisons': len(rows),
        'files_sha256': {'PAIRED_EVENT_RATES.csv': sha256(out/'PAIRED_EVENT_RATES.csv')},
        'policy_sha256': sha256(out/'POLICY.json'), 'elapsed_seconds': time.perf_counter()-started,
        'seal_rows': 0, 'reduced_new_uses': 0,
        'limitations': 'Conditional fixed-model/threshold intervals, no fitting uncertainty, multiplicity or adaptive selection adjustment. Mechanism null controls are repeated by design, so rows are not independent tests. Detection delay is censored and no production threshold was selected.'})
    print('PAIRED ALARM FORENSIC COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
