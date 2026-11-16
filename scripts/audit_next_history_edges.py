"""Fixed synthetic H edge cases for actual artifacts, without quality scores."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import importlib
import json
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.candidate_io import model_class
from src.next.guard import SealGuard
from src.utils.artifacts import sha256, write_json, utc_now


def histories(length, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, length]))
    noise = rng.normal(size=length)
    return {'constant_zero': np.zeros(length, dtype=np.float32),
        'constant_one': np.ones(length, dtype=np.float32),
        'nearly_constant': (1.+1e-6*noise).astype(np.float32),
        'large_location_small_scale': (1e6+noise).astype(np.float32),
        'tiny_scale': (1e-9*noise).astype(np.float32),
        'alternating_large': (np.where(np.arange(length)%2, 1., -1.)*1e4).astype(np.float32),
        'isolated_historical_spike': np.where(np.arange(length) == length//2, 1e6, noise).astype(np.float32)}


def main():
    policy_path = ROOT/'configs/next_history_edge_policy.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    out = ROOT/'artifacts/next/history_edge_contract'
    if (out/'RESULTS.json').exists():
        raise RuntimeError('Preserve the first fixed history-edge audit')
    out.mkdir(parents=True, exist_ok=True)
    guard = SealGuard(ROOT)
    rows, sources = [], {}
    start = time.perf_counter()
    prepared_classes = {
        'conditional_variance': ('src.next.fast_variance_initialization', 'PreparedFastVariancePredictor'),
        'log_energy': ('src.next.log_energy_runtime', 'PreparedLogEnergyPredictor'),
        'serial_variance': ('src.next.serial_variance_runtime', 'PreparedSerialVariancePredictor')}
    for eid in policy['candidates']:
        directory = ROOT/'artifacts/next/refits'/eid/'cache_research_v1'
        descriptor = json.loads((directory/'model.json').read_text(encoding='utf-8'))
        if sha256(directory/'model.joblib') != descriptor['model_sha256']:
            raise RuntimeError('Edge audit artifact bytes changed')
        candidate = descriptor['candidate']
        model = model_class(candidate).load(directory/'model.joblib')
        module_name, class_name = prepared_classes[candidate['extension']['kind']]
        module = importlib.import_module(module_name)
        if module.source_hash() != descriptor['runtime_sha256']:
            raise RuntimeError('Edge audit runtime source changed')
        prepared = getattr(module, class_name)(model)
        sources[eid] = {'model_sha256': descriptor['model_sha256'], 'descriptor_sha256': sha256(directory/'model.json'), 'runtime_sha256': module.source_hash()}
        for length in policy['historical_lengths']:
            for name, historical in histories(length, policy['seed']).items():
                rng = np.random.default_rng(np.random.SeedSequence([policy['seed'], length, list(histories(length, policy['seed'])).index(name)]))
                normal = rng.normal(size=policy['online_points']).astype(np.float32)
                cases = {'zero': np.zeros_like(normal), 'unit_normal': normal,
                    'extreme_suffix': np.r_[normal[:257], np.where(np.arange(len(normal)-257)%2, 1e9, -1e9)].astype(np.float32)}
                for online_name, online in cases.items():
                    original, runtime = model.make_state(historical), prepared.new_state(historical)
                    size = runtime.state_array_bytes
                    expected = np.array([model.predict_one(original.update(point)) for point in online], dtype=np.float32)
                    actual = np.array([runtime.predict_one(point) for point in online], dtype=np.float32)
                    equal = bool(np.array_equal(expected, actual))
                    finite = bool(np.isfinite(actual).all() and ((actual >= 0)&(actual <= 1)).all())
                    row = {'candidate': eid, 'history_length': length, 'history_case': name, 'online_case': online_name,
                        'points': len(online), 'bitwise_scalar_bundle_parity': equal, 'finite_bounded_predictions': finite,
                        'fixed_state_array_bytes': runtime.state_array_bytes == size, 'max_abs_difference': float(np.max(np.abs(actual-expected)))}
                    rows.append(row)
                    if not (equal and finite and row['fixed_state_array_bytes']):
                        write_json(out/'FAILURE.json', {'status': 'FAIL', 'created_utc': utc_now(), 'policy_sha256': sha256(policy_path), 'row': row})
                        raise RuntimeError('A fixed historical edge case differs; retain and investigate')
        print('HISTORY EDGE AUDIT PASS', eid, flush=True)
    guard.check_state()
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'policy_sha256': sha256(policy_path),
        'driver_sha256': sha256(Path(__file__)), 'sources': sources, 'cases': rows, 'streams': len(rows),
        'scalar_predictions_per_implementation': sum(row['points'] for row in rows), 'elapsed_seconds': time.perf_counter()-start,
        'quality_score_computed': False, 'model_refits': 0, 'seal_rows': 0, 'reduced_new_uses': 0,
        'scope': 'Predefined finite synthetic H and O edge cases compared with the original scalar model, on actual complete-DEV artifacts. This is an engineering contract, not a simulation of the competition generator.'})


if __name__ == '__main__':
    main()
