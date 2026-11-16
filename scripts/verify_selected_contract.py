"""Real-prefix model contract: no length, indexing, replay, read-ahead or writes."""
import argparse, builtins, io, json, os, shutil, sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.models.learners import ModelBundle
from src.models.calibrated import CalibratedBundle
from src.streaming.inference import infer
from src.streaming.fused import implementation_hash as fused_hash
from src.features.streaming import StreamingFeatureState
from src.validation.splits import FrozenSplit
from src.data.loader import load_training
from src.utils.artifacts import write_json, sha256, utc_now


class GuardedOnline:
    def __init__(self, values): self.values = iter(values); self.started = False; self.pending = False; self.count = 0
    def __iter__(self):
        if self.started: raise AssertionError('Online iterator was replayed')
        self.started = True; return self
    def __len__(self): raise AssertionError('Future online length queried')
    def __getitem__(self, key): raise AssertionError('Online values indexed')
    def __array__(self, *args, **kwargs): raise AssertionError('Online iterable materialized')
    def __next__(self):
        if self.pending: raise AssertionError('Read-ahead before yielding the current prediction')
        value = next(self.values); self.pending = True; self.count += 1; return value
    def acknowledge(self):
        if not self.pending: raise AssertionError('Prediction without a new input')
        self.pending = False


def guarded_prediction(historical, points, directory):
    online = GuardedOnline(points)
    generator = infer(iter([(historical, online)]), directory)
    if next(generator) is not None or online.count != 0: raise AssertionError('Ready yield consumed an online point')
    values = []
    for _ in points:
        value = next(generator)
        if not isinstance(value, float) or not np.isfinite(value) or not 0 <= value <= 1: raise AssertionError('Prediction type, range or finiteness failed')
        values.append(value); online.acknowledge()
    try: next(generator)
    except StopIteration: pass
    else: raise AssertionError('More predictions than online points')
    return np.asarray(values, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--candidate', required=True); parser.add_argument('--mode', choices=['native', 'fused'], default='fused')
    parser.add_argument('--series', type=int, default=32); parser.add_argument('--fast-initialization', action='store_true')
    parser.add_argument('--model-directory',help='Verify an already fitted deployment artifact without copying or modifying it');args = parser.parse_args()
    if args.fast_initialization and args.mode != 'fused': raise ValueError('Fast initialization requires fused mode')
    registry = json.loads((ROOT / 'artifacts/real/RESEARCH_STATE.json').read_text())
    result = next((r for r in registry['results'] if r['experiment_id'] == args.candidate),None)
    if result is None:result=json.loads((ROOT/'artifacts/real'/f'{args.candidate}.json').read_text())
    representation = result.get('representation', result.get('candidate', {}).get('representation', 'base'))
    if representation.startswith('rank_gaussian'): raise RuntimeError('Rank candidate needs an explicit deployment kind before contract verification')
    from src.models.representations import bundle_class,deployment_kind
    loader=bundle_class(representation)
    mode_label=('final_' if args.model_directory else '')+args.mode+('_fast_init' if args.fast_initialization else '')
    if args.model_directory:
        directory=Path(args.model_directory).resolve();model_path=directory/'model.joblib';model=loader.load(model_path)
        descriptor=json.loads((directory/'model.json').read_text())
        if bool(descriptor.get('use_fused_inference'))!=(args.mode=='fused') or bool(descriptor.get('use_fast_historical_init'))!=args.fast_initialization:raise RuntimeError('Requested contract mode does not match the immutable deployment descriptor')
        if descriptor.get('candidate')!=result['candidate']:raise RuntimeError('Deployment candidate differs from the selected CV specification')
    else:
        model_path = ROOT / 'artifacts/models/real' / args.candidate / 'fold_0.joblib'; model = loader.load(model_path)
        directory = ROOT / 'artifacts/deployment/contracts' / args.candidate / mode_label; directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, directory / 'model.joblib')
        descriptor = {'kind': deployment_kind(representation), 'name': model.name,
                      'feature_hash': model.feature_hash, 'model_sha256': sha256(model_path), 'use_fused_inference': args.mode == 'fused',
                      'status': 'REAL_DEV_API_VERIFICATION_ONLY'}
        if args.mode == 'fused':
            from src.streaming.prepared import source_hashes
            descriptor.update(source_hashes(model,fast_initialization=args.fast_initialization))
    if args.fast_initialization:
        from src.streaming.fast_initialization import implementation_hash as initialization_hash
        if not args.model_directory:descriptor['use_fast_historical_init']=True;descriptor['initialization_source_hash']=initialization_hash()
    if not args.model_directory:write_json(directory / 'model.json', descriptor)
    split = FrozenSplit(ROOT / 'artifacts/validation'); examples = []
    for series in load_training(ROOT / 'data/raw', split=split):
        examples.append((series.historical.astype(np.float32), series.online.astype(np.float32)))
        if len(examples) >= args.series: break
    # Exclude third-party first-import setup; every guarded call still reloads
    # the model, validates hashes and executes the official infer generator.
    guarded_prediction(examples[0][0], examples[0][1][:1], directory)
    original_open, original_io_open, original_os_open = builtins.open, io.open, os.open
    def guarded_open(original):
        def open_read_only(file, mode='r', *a, **kw):
            if any(flag in mode for flag in 'wax+'): raise AssertionError('Inference attempted a file write')
            return original(file, mode, *a, **kw)
        return open_read_only
    def os_open_read_only(file, flags, *a, **kw):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND): raise AssertionError('Inference attempted an OS-level file write')
        return original_os_open(file, flags, *a, **kw)
    count = 0; max_native_diff = 0.; suffix_cases = 0
    with patch('builtins.open', guarded_open(original_open)), patch('io.open', guarded_open(original_io_open)), patch('os.open', os_open_read_only):
        for historical, online in examples:
            first = guarded_prediction(historical, online, directory)
            repeated = guarded_prediction(historical, online, directory); np.testing.assert_array_equal(first, repeated)
            state = model.make_state(historical) if hasattr(model, 'make_state') else StreamingFeatureState(historical, config=model.feature_config)
            native = np.array([model.predict_one(state.update_and_get(p)) for p in online], dtype=np.float32)
            np.testing.assert_array_equal(first, native); max_native_diff = max(max_native_diff, float(np.max(abs(first - native))))
            prefix = min(64, len(online))
            for suffix in (np.full(5, -1e5, dtype=np.float32), np.full(17, np.nan, dtype=np.float32), np.linspace(-500, 500, 257, dtype=np.float32)):
                altered = guarded_prediction(historical, np.r_[online[:prefix], suffix], directory)
                np.testing.assert_array_equal(first[:prefix], altered[:prefix]); suffix_cases += 1
            count += len(online)
    report = {'status': 'PASS', 'timestamp': utc_now(), 'candidate': args.candidate, 'mode': args.mode, 'real_development_series': len(examples),
              'original_online_points': count, 'modified_suffix_cases': suffix_cases, 'native_live_max_abs_diff': max_native_diff,
              'repeat_predictions_bitwise_equal': True, 'prefix_predictions_bitwise_equal': True,
              'guards': ['no online length', 'no online indexing', 'no online array materialization', 'no second iteration',
                         'one input consumed per yielded prediction', 'ready yield consumes zero online values', 'no Python or OS file writes'],
              'model_descriptor_sha256': sha256(directory / 'model.json'), 'inference_source_sha256': sha256(ROOT / 'src/streaming/inference.py'),
              'feature_hash': model.feature_hash, 'fused_source_hash': fused_hash() if args.mode == 'fused' else None,
              'fast_historical_initialization': args.fast_initialization, 'initialization_source_hash': initialization_hash() if args.fast_initialization else None,
              'ar_fused_source_hash':descriptor.get('ar_fused_source_hash'),
              'cross_order_fused_source_hash':descriptor.get('cross_order_fused_source_hash'),
              'arch_fused_source_hash':descriptor.get('arch_fused_source_hash'),
              'split_sha256': split.digest, 'LOCAL_FINAL_SEAL_OPENED': False, 'reduced_targets_used': False,
              'artifact_kind':'final DEV refit' if args.model_directory else 'fold-0 CV model',
              'scope': 'API and numerical behavior only; no scores for these prefix probes'}
    path = ROOT / 'artifacts/real/contracts' / f'{args.candidate}_{mode_label}.json'; write_json(path, report)
    print('REAL CONTRACT PASS', args.candidate, args.mode, count, 'points', suffix_cases, 'suffix cases', flush=True)


if __name__ == '__main__': main()
