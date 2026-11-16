"""Fixed 1024-point calibration tested on new 16384-point stationary streams.

The aim is horizon sensitivity and state stability, not a new operating point.
Known horizons are used only after causal predictions for reporting.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import copy
import gc
import json
import sys
import time
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.synthetic_variance_stress import generate_series
from src.next.alarm_calibration import null_maximum_threshold, wilson_interval
from src.next.candidate_io import model_class
from src.next.guard import SealGuard
from src.next.fast_variance_initialization import PreparedFastVariancePredictor, source_hash as fast_hash
from src.models.cross_order import CrossOrderBlendBundle
from src.streaming.prepared import prepare_model, source_hashes
from src.validation.group_bootstrap import draw_group_multiplicities
from src.utils.artifacts import write_json, sha256, utc_now


def predict_complete(prepared, historical, points):
    state = prepared.new_state(historical)
    before = state.state_array_bytes
    prediction = np.asarray([state.predict_one(point) for point in points], dtype=np.float32)
    if state.state_array_bytes != before:
        raise RuntimeError('Long-null detector retained growing online state')
    if not np.isfinite(prediction).all() or np.any((prediction < 0)|(prediction > 1)):
        raise RuntimeError('Long-null detector emitted an invalid score')
    return prediction, before


def main():
    start = time.perf_counter()
    policy_path = ROOT/'configs/next_long_null_policy.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    for relative, digest in policy['source_files_sha256'].items():
        if sha256(ROOT/relative) != digest:
            raise RuntimeError('Frozen long-null implementation changed: '+relative)
    guard = SealGuard(ROOT)
    parent_path = ROOT/'configs/next_synthetic_stress_policy.json'
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    if sha256(parent_path) != policy['parent_policy_sha256']:
        raise RuntimeError('Original null calibration policy changed')
    prepared = {}
    for eid, relative in policy['models'].items():
        if sha256(ROOT/relative) != policy['model_sha256'][eid]:
            raise RuntimeError('Long-null model changed')
        if eid == 'PRIMARY':
            model = CrossOrderBlendBundle.load(ROOT/relative)
            if source_hashes(model, fast_initialization=True) != policy['primary_runtime_receipt']:
                raise RuntimeError('Frozen primary runtime changed')
            prepared[eid] = prepare_model(model, fast_initialization=True)
        else:
            parity_path = ROOT/'artifacts/next/runtime_parity'/eid/'pruned_variance_fast_h/RESULTS.json'
            parity = json.loads(parity_path.read_text(encoding='utf-8'))
            if parity['status'] != 'PASS' or parity['points'] != 4014405 or not parity['all_float32_predictions_bitwise_equal']:
                raise RuntimeError('Full real DEV runtime parity is required before long-null stress')
            if parity['identity']['runtime_sha256'] != fast_hash() or fast_hash() != policy['variance_runtime_sha256']:
                raise RuntimeError('Verified variance runtime changed')
            candidate = json.loads((ROOT/'artifacts/next/experiments'/eid/'RESULTS.json').read_text(encoding='utf-8'))['candidate']
            model = model_class(candidate).load(ROOT/relative)
            prepared[eid] = PreparedFastVariancePredictor(model)
    roles = list(prepared)
    parent_roles = list(parent['models'])
    reused = ['PRIMARY', 'V007_VARIANCE_FORECAST']
    added = [eid for eid in roles if eid not in reused]
    parent_out = ROOT/'artifacts/next/synthetic_variance_stress'
    parent_cache = ROOT/'data/next/synthetic_variance_stress_v1'
    parent_result = json.loads((parent_out/'RESULTS.json').read_text(encoding='utf-8'))
    if parent_result['status'] != 'PASS':
        raise RuntimeError('Original controlled diagnostic must finish first')
    out = ROOT/'artifacts/next/long_null_stress'
    out.mkdir(parents=True, exist_ok=True)
    identity = {'policy_sha256': sha256(policy_path), 'parent_result_sha256': sha256(parent_out/'RESULTS.json'),
        'split_sha256': guard.split.digest, 'seal_lock_sha256': guard.lock_sha,
        'purpose': 'All evaluated input values are synthetic; the application guard verifies unchanged seal/quota state.'}
    if (out/'IDENTITY.json').exists() and json.loads((out/'IDENTITY.json').read_text(encoding='utf-8')) != identity:
        raise RuntimeError('Long-null resume identity changed')
    if not (out/'IDENTITY.json').exists():
        write_json(out/'IDENTITY.json', identity)
    if (out/'RESULTS.json').exists():
        report = json.loads((out/'RESULTS.json').read_text(encoding='utf-8'))
        if any(sha256(out/name) != digest for name, digest in report['files_sha256'].items()):
            raise RuntimeError('Completed long-null report changed')
        print('LONG NULL retained PASS')
        return
    maxima = {eid: [] for eid in roles}
    calibration_receipts = []
    for regime in range(len(parent['regimes'])):
        for first in range(0, parent['calibration_series_per_regime'], 100):
            name = f'partition_0_regime_{regime}_block_{first:04d}'
            prior = json.loads((parent_out/(name+'.json')).read_text(encoding='utf-8'))
            if sha256(parent_cache/(name+'.npy')) != prior['prediction_sha256']:
                raise RuntimeError('Reused null calibration predictions changed')
            values = np.load(parent_cache/(name+'.npy'), allow_pickle=False)
            for eid in reused:
                maxima[eid].extend(values[parent_roles.index(eid), 0].max(axis=1).tolist())
            folder = out/'calibration'
            folder.mkdir(exist_ok=True)
            path = folder/(name+'.json')
            data_path = folder/(name+'.npy')
            if path.exists():
                receipt = json.loads(path.read_text(encoding='utf-8'))
                if sha256(data_path) != receipt['maxima_sha256']:
                    raise RuntimeError('Additional calibration maxima changed')
                block = np.load(data_path, allow_pickle=False)
            else:
                tick = time.perf_counter()
                block = np.empty((len(added), 100), dtype=np.float32)
                for j, index in enumerate(range(first, first+100)):
                    historical, online, _, _ = generate_series(parent, 0, regime, index)
                    for r, eid in enumerate(added):
                        prediction, _ = predict_complete(prepared[eid], historical, online['NO_CHANGE'])
                        block[r, j] = prediction.max()
                np.save(data_path, block, allow_pickle=False)
                receipt = {'status': 'PASS', 'created_utc': utc_now(), 'roles': added,
                    'maxima_sha256': sha256(data_path), 'elapsed_seconds': time.perf_counter()-tick}
                write_json(path, receipt)
                print('LONG NULL CALIBRATION', regime, first+100, round(receipt['elapsed_seconds'], 1), flush=True)
            for r, eid in enumerate(added):
                maxima[eid].extend(block[r].tolist())
            calibration_receipts.append({'file': str(path.relative_to(out)), 'sha256': sha256(path),
                'reused_parent_prediction_sha256': prior['prediction_sha256']})
    thresholds = {eid: [null_maximum_threshold(maxima[eid], alpha) for alpha in policy['alphas']] for eid in roles}
    parent_thresholds_path = parent_out/'THRESHOLDS.json'
    if sha256(parent_thresholds_path) != parent_result['files_sha256']['THRESHOLDS.json']:
        raise RuntimeError('Original synthetic thresholds changed')
    prior_thresholds = json.loads(parent_thresholds_path.read_text(encoding='utf-8'))['thresholds']
    for eid in reused:
        if thresholds[eid] != prior_thresholds[eid]:
            raise RuntimeError('Reused threshold reconstruction differs from the original')
    path = out/'THRESHOLDS.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))['thresholds'] != thresholds:
        raise RuntimeError('Long-null thresholds changed')
    if not path.exists():
        write_json(path, {'status': 'LOCKED', 'created_utc': utc_now(), 'thresholds': thresholds,
            'calibration_horizon': parent['online_points'], 'long_evaluation_computed_yet': False,
            'calibration_receipts': calibration_receipts, 'new_calibration_model_series': 10000,
            'reused_calibration_model_series': 10000})
    extended = copy.deepcopy(parent)
    extended['online_points'] = policy['online_points']
    horizons = policy['report_horizons']
    summary, paired, block_receipts = [], [], []
    for regime, regime_spec in enumerate(parent['regimes']):
        maxima_blocks, endpoint_blocks = [], []
        for first in range(0, policy['evaluation_series_per_regime'], 50):
            last = min(first+50, policy['evaluation_series_per_regime'])
            folder = out/'evaluation'
            folder.mkdir(exist_ok=True)
            prefix = f'regime_{regime}_block_{first:04d}'
            path = folder/(prefix+'.json')
            max_path, end_path = folder/(prefix+'_maxima.npy'), folder/(prefix+'_endpoints.npy')
            if path.exists():
                record = json.loads(path.read_text(encoding='utf-8'))
                if any(sha256(folder/name) != digest for name, digest in record['files_sha256'].items()):
                    raise RuntimeError('Completed long-null evaluation block changed')
                peak = np.load(max_path, allow_pickle=False)
                endpoint = np.load(end_path, allow_pickle=False)
            else:
                tick = time.perf_counter()
                peak = np.empty((len(roles), last-first, len(horizons)), dtype=np.float32)
                endpoint = np.empty_like(peak)
                max_bytes, prefix_checks = {}, 0
                for j, index in enumerate(range(first, last)):
                    historical, online, _, _ = generate_series(extended, policy['evaluation_partition'], regime, index)
                    points = online['NO_CHANGE']
                    for r, eid in enumerate(roles):
                        prediction, size = predict_complete(prepared[eid], historical, points)
                        max_bytes[eid] = max(max_bytes.get(eid, 0), size)
                        for k, horizon in enumerate(horizons):
                            peak[r, j, k] = prediction[:horizon].max()
                            endpoint[r, j, k] = prediction[horizon-1]
                        if index == 0:
                            reference, _ = predict_complete(prepared[eid], historical, points[:1024])
                            np.testing.assert_array_equal(prediction[:1024], reference)
                            prefix_checks += 1
                np.save(max_path, peak, allow_pickle=False)
                np.save(end_path, endpoint, allow_pickle=False)
                record = {'status': 'PASS', 'created_utc': utc_now(), 'regime': regime, 'first': first, 'last': last,
                    'maximum_fixed_state_bytes': max_bytes, 'prefix_reinitialization_checks': prefix_checks,
                    'all_scalar_predictions_finite_and_bounded': True, 'online_state_size_constant': True,
                    'elapsed_seconds': time.perf_counter()-tick,
                    'files_sha256': {item.name: sha256(item) for item in (max_path, end_path)}}
                write_json(path, record)
                print('LONG NULL EVALUATION', regime, last, policy['evaluation_series_per_regime'], round(record['elapsed_seconds'], 1), flush=True)
            maxima_blocks.append(peak)
            endpoint_blocks.append(endpoint)
            block_receipts.append({'file': str(path.relative_to(out)), 'sha256': sha256(path)})
            guard.check_state()
        peaks = np.concatenate(maxima_blocks, axis=1)
        endpoints = np.concatenate(endpoint_blocks, axis=1)
        count = peaks.shape[1]
        _, weights = draw_group_multiplicities(np.arange(count), policy['bootstrap_replicates'], policy['bootstrap_seed']+regime)
        for r, eid in enumerate(roles):
            for a, alpha in enumerate(policy['alphas']):
                threshold = thresholds[eid][a]['threshold']
                reference_threshold = thresholds['PRIMARY'][a]['threshold']
                for k, horizon in enumerate(horizons):
                    alarms = peaks[r, :, k] > threshold if threshold is not None else np.zeros(count, dtype=bool)
                    reference_alarms = peaks[roles.index('PRIMARY'), :, k] > reference_threshold if reference_threshold is not None else np.zeros(count, dtype=bool)
                    events = int(alarms.sum())
                    low, high = wilson_interval(events, count)
                    summary.append({'model': eid, 'regime': regime_spec['name'], 'horizon': horizon, 'nominal_alpha_at_1024': alpha,
                        'threshold': threshold, 'series': count, 'false_alarms': events, 'false_alarm_fraction': events/count,
                        'wilson_95_low': low, 'wilson_95_high': high, 'endpoint_score_median': float(np.median(endpoints[r, :, k])),
                        'running_peak_score_median': float(np.median(peaks[r, :, k])),
                        'endpoint_score_p99': float(np.quantile(endpoints[r, :, k], .99))})
                    delta = (weights@(alarms.astype(np.int32)-reference_alarms.astype(np.int32)))/count
                    paired.append({'model': eid, 'regime': regime_spec['name'], 'horizon': horizon, 'nominal_alpha_at_1024': alpha,
                        'estimate_fpr_delta_vs_primary': float(delta[0]), 'ci_2_5': float(np.quantile(delta[1:], .025)),
                        'ci_97_5': float(np.quantile(delta[1:], .975)), 'conditional_probability_delta_positive': float(np.mean(delta[1:] > 0)),
                        'replicates': policy['bootstrap_replicates'], 'lower_is_better': True})
        pd.DataFrame(summary).to_csv(out/'HORIZON_FALSE_ALARMS.csv', index=False)
        pd.DataFrame(paired).to_csv(out/'PAIRED_HORIZON_RATES.csv', index=False)
        del maxima_blocks, endpoint_blocks, peaks, endpoints, weights
        gc.collect()
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'policy_sha256': sha256(policy_path),
        'identity_sha256': sha256(out/'IDENTITY.json'), 'blocks': block_receipts,
        'new_long_null_series': 5000, 'model_long_stream_evaluations': 20000, 'long_scalar_predictions': 327680000,
        'new_calibration_scalar_predictions': 10240000, 'prefix_verification_scalar_predictions': 20480,
        'model_refits': 0, 'official_evaluation_rows': 0, 'seal_rows': 0, 'reduced_new_uses': 0,
        'elapsed_this_invocation_seconds': time.perf_counter()-start,
        'files_sha256': {name: sha256(out/name) for name in ['THRESHOLDS.json', 'HORIZON_FALSE_ALARMS.csv', 'PAIRED_HORIZON_RATES.csv']},
        'authority': 'Horizon-sensitivity diagnostic on a fixed synthetic null family; the 1024-point threshold is never re-estimated from the longer futures. No competition score, promotion gate or deployment threshold changes.',
        'limitations': 'Intervals condition on H-trained models and separately fixed finite calibration thresholds. AR/ARCH/Student simulations do not identify the competition generator or guarantee real-world false-alarm control.'})
    print('LONG NULL STRESS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
