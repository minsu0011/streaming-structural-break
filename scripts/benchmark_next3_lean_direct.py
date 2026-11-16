"""Three fixed paired direct 10M trials of E003 against selected E002."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[key] = '1'
from pathlib import Path
import gc
import hashlib
import sys
import time
import numpy as np
import pandas as pd
import psutil
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from src.next3.lean_model import load as lean_load
from src.next3.lean_shared_runtime import PreparedLeanSharedPredictor, accounting as lean_accounting
from src.next3.packed_shared_runtime import PreparedPackedSharedPredictor, accounting as packed_accounting
from src.next2.canonical_model import load as canonical_load
from src.next.data import prepare_dev
from scripts.benchmark_next2_runtime import assert_isolated, distribution
from scripts.verify_next3_lean_shared import checked_implementation


def trial(engine, variant, repeat, examples, buffer, expected, out):
    assert_isolated(); folder = out/f'trial_{repeat}_{variant}'; folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder/'START.json', {'created_utc': utc_now(), 'variant': variant, 'trial': repeat}, immutable=True)
    _, h0, o0 = examples[0]; tick = time.perf_counter(); engine.new_state(h0).predict_one(o0[0]); warm = time.perf_counter()-tick
    points = []; histories = []; records = []; samples = {}
    for inner in range(3):
        assert_isolated()
        for sid, h, o in examples:
            tick = time.perf_counter_ns(); state = engine.new_state(h); initialization = time.perf_counter_ns()-tick
            p = np.empty(len(o), dtype=np.float32); duration = np.empty(len(o), dtype=np.int64)
            for j, value in enumerate(o):
                tick = time.perf_counter_ns(); p[j] = state.predict_one(value); duration[j] = time.perf_counter_ns()-tick
            if inner == 0: samples[sid] = p
            else: np.testing.assert_array_equal(samples[sid], p)
            points.extend(duration); histories.append(initialization)
            records.append({'inner_repeat': inner, 'dataset_id': sid, 'historical_init_ns': initialization,
                'historical_points': len(h), 'online_points': len(o), 'point_median_ns': float(np.median(duration))})
        progress('E003 actual scalar timing', f'{repeat} {variant} prefix repetition {inner+1}', experiment='E003_LEAN_SHARED')
    sample_hash = hashlib.sha256(b''.join(samples[sid].tobytes() for sid, _, _ in examples)).hexdigest()
    assert sample_hash == expected['sample_prediction_sha256']
    assert_isolated(); state = engine.new_state(h0)
    accounting = lean_accounting if variant == 'LEAN_PACKED_SHARED' else packed_accounting
    memory, _, _ = accounting(state); before = state.state_array_bytes
    gc.collect(); process = psutil.Process(); rss = process.memory_info().rss; peak = rss
    elapsed = 0.; count = 0; fingerprint = hashlib.sha256(); blocks = []; wall = time.perf_counter(); first_million = None
    while count < 10_000_000:
        tick = time.perf_counter_ns(); prediction = np.asarray([state.predict_one(v) for v in buffer], dtype=np.float32)
        duration = (time.perf_counter_ns()-tick)/1e9; elapsed += duration; count += len(buffer)
        assert np.isfinite(prediction).all(); fingerprint.update(prediction.tobytes())
        blocks.append({'points': count, 'seconds': duration, 'cumulative_seconds': elapsed})
        if count == 1_000_000: first_million = elapsed
        if count % 1_000_000 == 0:
            assert_isolated(); peak = max(peak, process.memory_info().rss)
            progress('E003 direct 10M measurement', f'{repeat} {variant} {count} points, {elapsed:.3f}s', experiment='E003_LEAN_SHARED')
            print('NEXT3 LEAN DIRECT', repeat, variant, count, round(elapsed, 3), flush=True)
    assert count == 10_000_000 and state.state_array_bytes == before
    assert fingerprint.hexdigest() == expected['continuous_stream']['output_sha256']
    pd.DataFrame(records).to_csv(folder/'SERIES_TIMING.csv', index=False)
    pd.DataFrame(blocks).to_csv(folder/'CONTINUOUS_BLOCKS.csv', index=False)
    result = {'status': 'PASS', 'created_utc': utc_now(), 'variant': variant, 'trial': repeat,
        'point_distribution': distribution(points), 'historical_initialization': distribution(histories),
        'warm_H_plus_possible_JIT_seconds': warm, 'numeric_memory': memory,
        'continuous_stream': {'points': count, 'measured_prediction_seconds': elapsed,
            'measured_first_million_seconds': first_million, 'wall_seconds_including_checks': time.perf_counter()-wall,
            'output_sha256': fingerprint.hexdigest(), 'state_array_bytes_before': before, 'state_array_bytes_after': state.state_array_bytes,
            'rss_before_bytes': rss, 'rss_peak_bytes': peak, 'rss_after_bytes': process.memory_info().rss},
        'sample_prediction_sha256': sample_hash, 'new_fits': 0, 'seal_rows': 0,
        'files_sha256': {name: digest(folder/name) for name in ('SERIES_TIMING.csv', 'CONTINUOUS_BLOCKS.csv')}}
    atomic_json(folder/'RESULTS.json', result, immutable=True)
    return result


def main():
    assert_isolated(); lock_sha = checked_implementation(); proofs = {}
    for stage in ('all_dev', 'oof', 'alternate', 'cold'):
        path = BASE/'engineering/lean_shared'/stage/'RESULTS.json'; report = read_json(path)
        assert report['status'] == 'PASS'
        if stage == 'cold': assert report['cold_cost_gate_pass']
        else: assert report['identity']['implementation_lock_sha256'] == lock_sha
        proofs[stage] = digest(path)
    api = BASE/'lean_public_api_all_dev/RESULTS.json'; assert read_json(api)['status'] == 'PASS'; proofs['public_api'] = digest(api)
    policy = read_json(ROOT/'configs/next2/RUNTIME_MEASUREMENT_LOCK.json')
    expected = read_json(ROOT/'artifacts/next2/latency/BL80_SHARED_V1/RESULTS.json')
    data = prepare_dev(ROOT); data.guard.admit(policy['series_ids'], purpose='Fixed E003 direct runtime probes')
    examples = [(sid, *data.get(sid)[:2]) for sid in policy['series_ids']]
    buffer = np.concatenate([o for _, _, o in examples])[:policy['block_points']]; assert len(buffer) == 10000
    path = ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1/model.sbnext2'
    engines = {'PACKED_SHARED': PreparedPackedSharedPredictor(canonical_load(path), variant='PACKED_SHARED'),
        'LEAN_PACKED_SHARED': PreparedLeanSharedPredictor(lean_load(path))}
    out = BASE/'engineering/lean_shared/direct'; out.mkdir(parents=True, exist_ok=False)
    orders = [('PACKED_SHARED', 'LEAN_PACKED_SHARED'), ('LEAN_PACKED_SHARED', 'PACKED_SHARED'), ('PACKED_SHARED', 'LEAN_PACKED_SHARED')]
    atomic_json(out/'POLICY.json', {'created_utc': utc_now(), 'orders': orders, 'implementation_lock_sha256': lock_sha,
        'proofs_sha256': proofs, 'timing_probe_lock_sha256': digest(ROOT/'configs/next2/RUNTIME_MEASUREMENT_LOCK.json')}, immutable=True)
    pairs = []; records = []
    for repeat, order in enumerate(orders):
        results = {}
        for variant in order:
            results[variant] = trial(engines[variant], variant, repeat, examples, buffer, expected, out); records.append(results[variant])
        a, b = results['PACKED_SHARED'], results['LEAN_PACKED_SHARED']
        for key in ('private_per_series_numeric_bytes', 'model_and_prepared_shared_numeric_bytes', 'total_one_stream_plus_one_model_numeric_bytes'):
            assert a['numeric_memory'][key] == b['numeric_memory'][key]
        pairs.append({'trial': repeat, 'direct_10M_ratio': b['continuous_stream']['measured_prediction_seconds']/a['continuous_stream']['measured_prediction_seconds'],
            'packed_seconds': a['continuous_stream']['measured_prediction_seconds'], 'lean_seconds': b['continuous_stream']['measured_prediction_seconds'],
            'point_median_ratio': b['point_distribution']['median_us']/a['point_distribution']['median_us'],
            'H_median_ratio': b['historical_initialization']['median_us']/a['historical_initialization']['median_us']})
    ratio = float(np.median([p['direct_10M_ratio'] for p in pairs]))
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'pairs': pairs,
        'median_direct_10M_ratio': ratio, 'direct_cost_gate_pass': ratio <= 1.05,
        'constant_state': True, 'private_and_prepared_numeric_bytes_unchanged': True,
        'total_direct_points': 60_000_000, 'new_fits': 0, 'seal_rows': 0,
        'files_sha256': {p.relative_to(out).as_posix(): digest(p) for p in out.rglob('RESULTS.json')},
        'limitation': 'Host-conditional timing; cold process memory is measured separately in six fresh subprocesses'}, immutable=True)
    progress('E003 direct timing complete', f'All six 10M trials retained; median paired ratio {ratio:.6f}', experiment='E003_LEAN_SHARED')


if __name__ == '__main__': main()
