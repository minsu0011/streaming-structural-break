"""Measure exact complete-DEV artifact bytes under the fixed scalar protocol."""
import os
import sys
sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import argparse
import gc
import hashlib
import json
import time
import numpy as np
import psutil
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next.inference import load_artifact
from src.next.fast_variance_initialization import PreparedFastVariancePredictor, source_hash
from src.next.references import verify_reference_sources
from src.utils.artifacts import write_json, sha256, utc_now


def assert_isolated():
    excluded = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    active = []
    for process in psutil.process_iter(['pid', 'name']):
        if process.pid in excluded or not (process.info['name'] or '').lower().startswith('python'):
            continue
        try:
            cwd = Path(process.cwd()).resolve()
            if cwd == ROOT or ROOT in cwd.parents:
                active.append(process.pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    if active:
        raise RuntimeError('Actual-artifact isolated timing refused while project Python workers are active: '+str(active))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--refit-tag', default='research_v1')
    parser.add_argument('--tag', default='all_dev_v1')
    args = parser.parse_args()
    if not all(value.replace('_', '').isalnum() for value in vars(args).values()):
        raise ValueError('Invalid artifact benchmark identifier')
    assert_isolated()
    verify_reference_sources(ROOT)
    policy_path = ROOT/'configs/next_all_dev_latency_policy.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    if sha256(Path(__file__)) != policy['source_sha256'] or sha256(ROOT/'configs/next_latency_policy.json') != policy['inherited_policy_sha256']:
        raise RuntimeError('Frozen artifact benchmark protocol changed')
    data = prepare_dev(ROOT)
    selected = policy['series_ids']
    data.guard.admit(selected, purpose='all-DEV artifact latency; no quality score')
    examples = [(sid, *data.get(sid)[:2]) for sid in selected]
    directory = ROOT/policy['baseline_model_directory'] if args.candidate == 'PRIMARY' else ROOT/'artifacts/next/refits'/args.candidate/('cache_'+args.refit_tag)
    tick = time.perf_counter()
    if args.candidate == 'PRIMARY':
        from src.models.cross_order import CrossOrderBlendBundle
        from src.streaming.prepared import prepare_model, source_hashes
        descriptor = json.loads((directory/'model.json').read_text(encoding='utf-8'))
        if sha256(directory/'model.joblib') != policy['baseline_model_sha256'] or sha256(directory/'model.json') != policy['baseline_descriptor_sha256']:
            raise RuntimeError('Frozen old PRIMARY artifact changed')
        model = CrossOrderBlendBundle.load(directory/'model.joblib')
        receipt = source_hashes(model, fast_initialization=True)
        if any(descriptor[key] != value for key, value in receipt.items()):
            raise RuntimeError('Frozen old PRIMARY runtime identity changed')
        prepare = lambda item: prepare_model(item, fast_initialization=True)
        contract_sha = None
    else:
        model, descriptor = load_artifact(directory)
        if descriptor['candidate']['experiment_id'] != args.candidate:
            raise RuntimeError('Artifact candidate mismatch')
        contract_path = ROOT/'artifacts/next/all_dev_artifact_contract'/args.candidate/args.refit_tag/'RESULTS.json'
        contract = json.loads(contract_path.read_text(encoding='utf-8'))
        if contract['status'] != 'PASS' or contract['all_dev_full_api_points'] != 4014405 or contract['identity']['model_sha256'] != sha256(directory/'model.joblib'):
            raise RuntimeError('A complete actual-artifact API and causality audit is required first')
        receipt, contract_sha = {'runtime_source_sha256': source_hash()}, sha256(contract_path)
        prepare = PreparedFastVariancePredictor
    load_seconds = time.perf_counter()-tick
    identity = {'candidate': args.candidate, 'model_kind': 'ALL_DEV_REFIT_ARTIFACT',
        'model_directory': str(directory.relative_to(ROOT)), 'model_sha256': sha256(directory/'model.joblib'),
        'descriptor_sha256': sha256(directory/'model.json'), 'policy_sha256': sha256(policy_path),
        'runtime_receipt': receipt, 'driver_sha256': sha256(Path(__file__)), 'contract_report_sha256': contract_sha}
    out = ROOT/'artifacts/next/latency'/args.tag
    out.mkdir(parents=True, exist_ok=True)
    report_path = out/(args.candidate+'.json')
    if report_path.exists():
        old = json.loads(report_path.read_text(encoding='utf-8'))
        if old['status'] != 'PASS' or old['identity'] != identity:
            raise RuntimeError('Existing actual-artifact measurement changed')
        print('ALL-DEV LATENCY retained PASS', args.candidate, flush=True)
        return
    identity_path = out/(args.candidate+'_IDENTITY.json')
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding='utf-8')) != identity:
            raise RuntimeError('Partial actual-artifact measurement identity changed')
        raise RuntimeError('An interrupted actual-artifact measurement requires an explicit documented revision')
    write_json(identity_path, identity)
    prepared = prepare(model)
    _, first_h, first_o = examples[0]
    tick = time.perf_counter()
    warm = prepared.new_state(first_h)
    warm.predict_one(first_o[0])
    cold_seconds = time.perf_counter()-tick
    del warm
    assert_isolated()
    initialization, loops, point_times, sizes, sampled_predictions = [], [], [], [], []
    for sid, h, points in examples:
        before = time.perf_counter_ns()
        detector = prepared.new_state(h)
        initialized = time.perf_counter_ns()
        prediction = np.empty(len(points), dtype=np.float32)
        durations = np.empty(len(points), dtype=np.int64)
        start_loop = time.perf_counter_ns()
        for index, point in enumerate(points):
            tick = time.perf_counter_ns()
            prediction[index] = detector.predict_one(point)
            durations[index] = time.perf_counter_ns()-tick
        end_loop = time.perf_counter_ns()
        initialization.append((initialized-before)/1e9)
        loops.append((end_loop-start_loop)/1e9)
        point_times.append(durations)
        sizes.append(detector.state_array_bytes)
        sampled_predictions.append(prediction)
    durations = np.concatenate(point_times)/1000.
    total_points = sum(len(o) for _, h, o in examples)
    amortized = (sum(initialization)+sum(loops))/total_points*1e6
    buffer = np.concatenate([o for _, h, o in examples])[:policy['block_points']]
    if len(buffer) != policy['block_points']:
        raise RuntimeError('Insufficient fixed repeated-buffer input')
    detector = prepared.new_state(first_h)
    detector.predict_one(buffer[0])
    gc.collect()
    before_size = detector.state_array_bytes
    process = psutil.Process()
    rss_before = peak_rss = process.memory_info().rss
    processed, elapsed, million = 0, 0., None
    digest = hashlib.sha256()
    wall = time.perf_counter()
    while processed < policy['continuous_points']:
        block = buffer[:min(len(buffer), policy['continuous_points']-processed)]
        tick = time.perf_counter_ns()
        prediction = np.asarray([detector.predict_one(point) for point in block], dtype=np.float32)
        elapsed += (time.perf_counter_ns()-tick)/1e9
        if not np.isfinite(prediction).all() or np.any((prediction < 0)|(prediction > 1)):
            raise RuntimeError('Long-stream prediction range failure')
        digest.update(prediction.tobytes())
        processed += len(block)
        if processed == policy['measured_checkpoint_points']:
            million = elapsed
        if processed % 1000000 == 0:
            assert_isolated()
            peak_rss = max(peak_rss, process.memory_info().rss)
            print('ALL-DEV LATENCY', args.candidate, processed, round(elapsed, 3), flush=True)
    long_wall = time.perf_counter()-wall
    if detector.state_array_bytes != before_size:
        raise RuntimeError('Long-stream state arrays grew')
    # Validate after timing so native reference compilation cannot contaminate
    # the cold-H/JIT measurement. No labels or AUC enter this comparison.
    sample_digest = hashlib.sha256()
    for (sid, h, points), actual in zip(examples, sampled_predictions):
        state = model.make_state(h)
        update = state.update_and_get if args.candidate == 'PRIMARY' else state.update
        expected = np.asarray([model.predict_one(update(point)) for point in points], dtype=np.float32)
        np.testing.assert_array_equal(actual, expected)
        sample_digest.update(actual.tobytes())
    report = {'status': 'PASS', 'created_utc': utc_now(), 'identity': identity,
        'isolated_from_workspace_research_workers': True, 'model_load_seconds': load_seconds,
        'first_history_and_jit_seconds': cold_seconds,
        'point_distribution_us': {'samples': len(durations), 'median': float(np.median(durations)),
            'p95': float(np.quantile(durations, .95)), 'p99': float(np.quantile(durations, .99)), 'mean': float(np.mean(durations)),
            'includes_per_point_timer_and_float32_store': True},
        'real_history_initialization': {'series': len(examples), 'mean_seconds': float(np.mean(initialization)),
            'median_seconds': float(np.median(initialization)), 'p95_seconds': float(np.quantile(initialization, .95)),
            'p99_seconds': float(np.quantile(initialization, .99)), 'historical_length_min': min(len(h) for _, h, o in examples),
            'historical_length_max': max(len(h) for _, h, o in examples), 'actual_online_points': total_points,
            'aggregate_h_and_online_seconds': sum(initialization)+sum(loops), 'amortized_us_per_point': amortized,
            'one_million_amortized_seconds': amortized, 'ten_million_amortized_seconds': 10*amortized},
        'continuous_stream': {'points': processed, 'measured_prediction_seconds': elapsed,
            'measured_first_million_seconds': million, 'microseconds_per_point': elapsed/processed*1e6,
            'wall_seconds_including_checks': long_wall, 'output_sha256': digest.hexdigest(),
            'state_array_bytes_before': before_size, 'state_array_bytes_after': detector.state_array_bytes,
            'rss_before_bytes': rss_before, 'rss_peak_bytes': peak_rss, 'rss_after_bytes': process.memory_info().rss},
        'real_prefix_state_array_bytes_maximum': max(sizes), 'all_sampled_predictions_bitwise_equal_to_scalar_bundle': True,
        'sample_prediction_sha256': sample_digest.hexdigest(), 'quality_score_computed': False,
        'seal_rows': 0, 'reduced_new_uses': 0,
        'limitations': 'Same-policy complete-DEV artifact timings, not CV predictions. Excludes file/socket/cloud overhead and cannot control ordinary OS load or power state. Continuous timing repeats a bounded real-value buffer; quality is not evaluated. Amortized extrapolations assume the sampled H/online-length mixture.'}
    write_json(report_path, report)
    print('ALL-DEV LATENCY PASS', args.candidate, report['point_distribution_us']['median'], amortized, flush=True)


if __name__ == '__main__':
    main()
