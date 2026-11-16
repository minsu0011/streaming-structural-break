"""Measure real-history initialization, point inference and long-stream memory."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import psutil
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.models.learners import ModelBundle
from src.models.calibrated import CalibratedBundle
from src.features.streaming import StreamingFeatureState
from src.streaming.fused import PreparedFusedPredictor, implementation_hash as fused_hash
from src.data.loader import load_training
from src.validation.splits import FrozenSplit
from src.utils.artifacts import write_json, sha256, utc_now
from scripts.verify_engineering import benchmark
from scripts.stress_streaming import deep_size


class NativeDetector:
    def __init__(self, model, historical):
        self.model = model
        self.state = model.make_state(historical) if hasattr(model, 'make_state') else StreamingFeatureState(historical, config=model.feature_config)
    def predict_one(self, point): return self.model.predict_one(self.state.update_and_get(point))
    @property
    def state_array_bytes(self): return self.state.state_array_bytes


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--candidate', required=True); parser.add_argument('--tag', required=True)
    parser.add_argument('--mode', choices=['native', 'fused'], default='fused'); parser.add_argument('--points', type=int, default=10_000_000)
    parser.add_argument('--series', type=int, default=128); parser.add_argument('--isolated', action='store_true'); parser.add_argument('--fast-initialization', action='store_true')
    parser.add_argument('--model-directory',help='Measure a specific deployment artifact instead of its CV fold-0 model');args = parser.parse_args()
    if args.fast_initialization and args.mode != 'fused': raise ValueError('Fast initialization requires fused mode')
    if args.points < 1 or args.series < 1 or not args.tag.replace('_', '').isalnum(): raise ValueError('Invalid benchmark configuration')
    active = []
    ancestors={p.pid for p in psutil.Process().parents()}
    worker_names={'run_research.py','run_real_research.py','run_rank_gaussian_research.py','run_pairwise_research.py','reproduce_selected_cv.py',
        'run_ar_order_research.py','run_ar_residual_research.py','run_complementary_research.py','train_final_artifacts.py','verify_fused_real.py',
        'benchmark_real_candidate.py','compare_historical_initialization.py','profile_ar_initialization.py','audit_fitted_features.py','bootstrap_real_oof.py','reproduce_snapshot.py',
        'run_cross_order_blend.py','run_historical_context_research.py','integrate_cross_order.py','run_arch_research.py','audit_nonlinear_residual_dependence.py','diagnose_first_alarms.py','compare_cross_order_dispatch.py',
        'run_magnitude_research.py','run_historical_quantile_research.py','verify_real_protocol.py','diagnose_historical_cohorts.py','probe_state_dispatch.py','compare_resident_dispatch.py',
        'official_socket_worker.py','verify_selected_official_scores.py','audit_arch_historical_residuals.py','profile_full_dev_runtime.py'}
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            command=process.info['cmdline'] or []
            if process.pid != os.getpid() and process.pid not in ancestors and (any(Path(v).name in worker_names for v in command) or 'pytest' in command):active.append(process.pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess): pass
    if args.isolated and active: raise RuntimeError('Research/refit workers are active; isolated timing refused')
    registry = json.loads((ROOT / 'artifacts/real/RESEARCH_STATE.json').read_text())
    result = next((r for r in registry['results'] if r['experiment_id'] == args.candidate), None)
    if result is None: result = json.loads((ROOT / 'artifacts/real' / f'{args.candidate}.json').read_text())
    representation = result.get('representation', result.get('candidate', {}).get('representation', 'base'))
    if representation.startswith('rank_gaussian'):
        from src.models.rank_calibrated import RankCalibratedBundle
        loader = RankCalibratedBundle
        if args.mode == 'fused': raise RuntimeError('Rank-input fused mapping must be implemented and verified before benchmarking it')
    else:
        from src.models.representations import bundle_class
        loader=bundle_class(representation)
    model_path = ROOT / 'artifacts/models/real' / args.candidate / 'fold_0.joblib';deployment_descriptor=None
    if args.model_directory:
        directory=Path(args.model_directory).resolve();model_path=directory/'model.joblib';deployment_descriptor=json.loads((directory/'model.json').read_text())
        if deployment_descriptor.get('candidate')!=result['candidate'] or deployment_descriptor.get('model_sha256')!=sha256(model_path):raise RuntimeError('Explicit deployment artifact identity differs from the CV candidate')
        if bool(deployment_descriptor.get('use_fused_inference'))!=(args.mode=='fused') or bool(deployment_descriptor.get('use_fast_historical_init'))!=args.fast_initialization:raise RuntimeError('Benchmark mode differs from the deployed mode')
    started = time.perf_counter(); model = loader.load(model_path); model_load_seconds = time.perf_counter() - started
    from src.streaming.prepared import prepare_model,source_hashes
    prepared=prepare_model(model,fast_initialization=args.fast_initialization) if args.mode=='fused' else None
    receipt=source_hashes(model,fast_initialization=args.fast_initialization) if prepared else {}
    if deployment_descriptor and prepared and any(deployment_descriptor.get(k)!=v for k,v in receipt.items()):raise RuntimeError('Deployment production source identity changed')
    factory = prepared.new_state if prepared else lambda h: NativeDetector(model, h)
    split = FrozenSplit(ROOT / 'artifacts/validation'); examples = []
    for series in load_training(ROOT / 'data/raw', split=split):
        examples.append((series.historical.astype(np.float32), series.online.astype(np.float32)))
        if len(examples) >= args.series: break
    h, online = examples[0]
    started = time.perf_counter(); detector = factory(h); detector.predict_one(online[0]); cold_start_seconds = time.perf_counter() - started
    initialization, point_times, lengths = [], [], []
    for historical, points in examples:
        started = time.perf_counter_ns(); detector = factory(historical); initialization.append((time.perf_counter_ns() - started) / 1e9)
        started = time.perf_counter_ns(); prediction = np.array([detector.predict_one(p) for p in points], dtype=np.float32); point_times.append((time.perf_counter_ns() - started) / 1e9)
        if not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 1)): raise RuntimeError('Real-prefix prediction range failure')
        lengths.append(len(points))
    probe = np.tile(online, max(1, 10000 // len(online) + 1))[:10000]
    point_benchmark = benchmark(lambda: factory(h).predict_one, probe, repeats=7)
    detector = factory(h); detector.predict_one(probe[0]); gc.collect()
    process = psutil.Process(); rss_start = process.memory_info().rss; rss_peak = rss_start
    before = detector.state_array_bytes; deep_before = deep_size(detector)
    elapsed = 0.0; processed = 0; million_checkpoint = None; last_value = None
    wall_start = time.perf_counter()
    while processed < args.points:
        block = probe[:min(len(probe), args.points - processed)]
        started = time.perf_counter_ns(); prediction = np.array([detector.predict_one(p) for p in block], dtype=np.float32); elapsed += (time.perf_counter_ns() - started) / 1e9
        processed += len(block)
        if not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 1)): raise RuntimeError('Long-stream prediction range failure')
        rss_peak = max(rss_peak, process.memory_info().rss); last_value = float(prediction[-1])
        if processed == 1_000_000: million_checkpoint = elapsed
        if processed % 1_000_000 == 0: print('LONG STREAM', args.candidate, args.mode, processed, 'points', round(elapsed, 3), 'seconds', flush=True)
    after = detector.state_array_bytes; deep_after = deep_size(detector)
    if before != after or deep_before != deep_after: raise RuntimeError('Detector retains growing online state')
    init_mean = float(np.mean(initialization)); mean_length = float(np.mean(lengths)); throughput = elapsed / processed * 1e6
    report = {'status': 'PASS', 'timestamp': utc_now(), 'candidate': args.candidate, 'representation': representation, 'mode': args.mode,
              'model_fold': None if args.model_directory else 0, 'model_sha256': sha256(model_path), 'model_bytes': model_path.stat().st_size,
              'artifact_kind':deployment_descriptor['status'] if deployment_descriptor else 'CV_FOLD_0_MODEL',
              'fused_source_hash': fused_hash() if prepared else None, 'split_sha256': split.digest,
              'fast_historical_initialization': args.fast_initialization, **receipt,
              'isolated_from_research_workers': not active, 'other_research_worker_pids': active,
              'scope': 'Local Windows CPU; no socket IPC, cloud runtime, or input-file I/O in timed intervals',
              'benchmark_script_sha256':sha256(__file__),'memory_accounting_source_sha256':sha256(ROOT/'scripts/stress_streaming.py'),
              'memory_accounting_scope':'Reachable Python objects, nested containers and NumPy owners count once. Native extension allocations hidden from Python are reflected only in RSS; explicit online array bytes are reported separately.',
              'model_load_seconds': model_load_seconds, 'first_history_plus_jit_seconds': cold_start_seconds,
              'point_benchmark': point_benchmark, 'real_history_initialization': {
                  'series': len(examples), 'historical_length_range': [min(len(h) for h, _ in examples), max(len(h) for h, _ in examples)],
                  'mean_seconds': init_mean, 'median_seconds': float(np.median(initialization)),
                  'p95_seconds': float(np.quantile(initialization, .95)), 'mean_online_length': mean_length,
                  'observed_initialization_plus_prediction_seconds': sum(initialization) + sum(point_times),
                  'observed_points': sum(lengths), 'observed_amortized_us_per_point': (sum(initialization) + sum(point_times)) / sum(lengths) * 1e6,
                  'ten_million_amortized_seconds_at_observed_mean_length': 10_000_000 * (init_mean / mean_length + throughput / 1e6)},
              'long_stream': {'points': processed, 'measured_prediction_seconds': elapsed, 'total_wall_seconds': time.perf_counter() - wall_start,
                  'measured_us_per_point': throughput, 'measured_first_million_seconds': million_checkpoint,
                  'ten_million_point_extrapolation_seconds': throughput * 10, 'last_prediction': last_value,
                  'state_array_bytes_before': before, 'state_array_bytes_after': after, 'deep_bytes_before': deep_before, 'deep_bytes_after': deep_after,
                  'rss_start_bytes': rss_start, 'rss_peak_bytes': rss_peak, 'rss_end_bytes': process.memory_info().rss,
                  'input': 'Repeated bounded buffer of real DEV values; one continuous synthetic-duration stream, no performance score'},
              'LOCAL_FINAL_SEAL_OPENED': False, 'reduced_targets_used': False}
    output = ROOT / 'artifacts/real/latency' / args.tag / f'{args.candidate}_{args.mode}{"_fast_init" if args.fast_initialization else ""}.json'
    write_json(output, report); print('BENCHMARK PASS', output.relative_to(ROOT), throughput, 'us per point', flush=True)


if __name__ == '__main__': main()
