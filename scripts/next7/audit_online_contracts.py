"""Final causal replay and serialization checks for every frozen research fold."""
from pathlib import Path
import hashlib
import inspect
import os
import pickle
import sys
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.worker_environment import environment
os.environ.update(environment())
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from src.next7.common import BASE, save, read_json, digest, utc_now, activity, note

PREFIXES = (1, 16, 64, 257)
BANNED_ARGUMENTS = ('tau', 'future_total_length', 'dataset_id', 'future_suffix')


def stream(prepared, h, online):
    state = prepared.new_state(h)
    return np.array([state.predict_one(value) for value in online]), state


def state_bytes(method, state):
    return int(state.state_array_bytes if method == 'M0' else state.instance_array_bytes_excluding_models)


def assert_rejected(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except TypeError:
        return
    raise AssertionError('Forbidden future/identity keyword was accepted')


def main():
    from scripts.next7.benchmark_runtime import audit_gate
    audit_gate()
    from src.next7.bundle import load, RECIPES
    from src.next7.benchmark_adapter import checkpoint, restore
    from src.next6_extension.synthetic import generated
    suite = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    fixture_dir = BASE / 'reproduction_fixture'
    fixture = read_json(fixture_dir / 'FIXTURE.json')
    assert digest(fixture_dir / 'PREDICTIONS.npz') == fixture['prediction_sha256']
    methods = ('M0', *RECIPES)
    cases = (('N0', 11), ('V4', 1), ('RETURN_NORMAL', 0))
    assert list(methods) == fixture['methods']
    forbidden_reads = []
    def deny_data(event, arguments):
        if event != 'open' or not arguments or not isinstance(arguments[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(arguments[0])).resolve()
        if path.is_relative_to((ROOT / 'data').resolve()):
            forbidden_reads.append(str(path))
            raise PermissionError('Final online contract tests cannot read competition data')
    sys.addaudithook(deny_data)
    identity = {'driver_sha256': digest(__file__), 'fixture_sha256': digest(fixture_dir / 'FIXTURE.json'),
        'prediction_sha256': fixture['prediction_sha256'],
        'audit_admission_sha256': digest(BASE / 'runtime/FINAL_AUDIT_ADMISSION.json')}
    relative = 'audit/online_contracts'
    directory = BASE / relative
    checks, reverse_checks = [], []
    note('Final online contract audit started', 'All five frozen research folds, exact fixture replay and causal input traps', 'FINAL_AUDIT', phase='FINAL_AUDIT')
    with np.load(fixture_dir / 'PREDICTIONS.npz', allow_pickle=False) as expected_file:
        for fold in range(5):
            bundle = load(BASE / f'research_bundles/fold_{fold}')
            assert digest(BASE / f'research_bundles/fold_{fold}/MANIFEST.json') == fixture['bundle_manifest_sha256'][str(fold)]
            for case, seed in cases:
                h, online, _, _ = generated(case, seed, partition=31, horizon=1024, control=False, lock=suite)
                input_key = f'{case}_seed{seed}'
                hashes = {'H': hashlib.sha256(h.tobytes()).hexdigest(), 'O': hashlib.sha256(online.tobytes()).hexdigest()}
                assert fixture['synthetic_input_sha256'][input_key] == hashes
                for method in methods:
                    key = f'fold{fold}_{input_key}_{method}'
                    output_path = directory / (key + '.json')
                    if output_path.exists():
                        previous = read_json(output_path)
                        assert previous['status'] == 'PASS' and previous['identity'] == identity
                        checks.append(previous)
                        continue
                    with activity('Final causal contract ' + key, 'FINAL_AUDIT', 'FINAL_AUDIT'):
                        prepared = bundle.prepare(method)
                        expected = expected_file[key]
                        actual, last = stream(prepared, h, online)
                        np.testing.assert_array_equal(actual, expected)
                        for prefix in PREFIXES:
                            short, _ = stream(prepared, h, online[:prefix])
                            np.testing.assert_array_equal(short, expected[:prefix])
                        changed_suffix = online.copy()
                        changed_suffix[257:] = np.where(np.arange(len(online) - 257) % 2, 10000., -10000.)
                        longer = np.r_[online[:257], np.full(1280, 10000., dtype=np.float32)]
                        for changed in (changed_suffix, longer):
                            altered, _ = stream(prepared, h, changed)
                            np.testing.assert_array_equal(altered[:257], expected[:257])
                        for name in BANNED_ARGUMENTS:
                            assert name not in inspect.signature(prepared.new_state).parameters
                            assert name not in inspect.signature(last.predict_one).parameters
                            assert_rejected(prepared.new_state, h, **{name: 17})
                            assert_rejected(last.predict_one, 0., **{name: 17})
                        a, b = prepared.new_state(h), prepared.new_state(h)
                        initial_bytes = state_bytes(method, a)
                        prefix = np.array([a.predict_one(value) for value in online[:257]])
                        np.testing.assert_array_equal(prefix, expected[:257])
                        snapshot = pickle.dumps(checkpoint(method, a), protocol=pickle.HIGHEST_PROTOCOL)
                        resumed = restore(SimpleNamespace(method=method, prepared=prepared), pickle.loads(snapshot))
                        for i in range(257, len(online)):
                            left = a.predict_one(online[i])
                            right = resumed.predict_one(online[i])
                            assert left == right == expected[i]
                            b.predict_one(-online[i] * 3.)
                        assert state_bytes(method, a) == state_bytes(method, resumed) == initial_bytes
                        batch_checked = method != 'M0'
                        batch_error = None
                        batch_bitwise = None
                        if batch_checked:
                            batch = prepared.new_state(h).replay(online)
                            reference_base = expected_file[f'fold{fold}_{input_key}_M0']
                            batch_error = float(max(np.max(np.abs(batch[:, 1] - expected)),
                                                    np.max(np.abs(batch[:, 0] - reference_base))))
                            batch_bitwise = bool(np.array_equal(batch[:, 1], expected)
                                                 and np.array_equal(batch[:, 0], reference_base))
                            assert batch_error <= 1e-12
                        result = {'status': 'PASS', 'created_utc': utc_now(), 'identity': identity,
                            'fold': fold, 'case': case, 'seed': seed, 'method': method, 'points': len(online),
                            'scalar_frozen_fixture_bitwise': True, 'prefixes_checked': PREFIXES,
                            'changed_suffix_prefix_bitwise': True, 'changed_future_length_prefix_bitwise': True,
                            'forbidden_arguments_rejected': BANNED_ARGUMENTS,
                            'independent_interleaved_state_bitwise': True, 'save_load_continuation_bitwise': True,
                            'offline_batch_scalar_checked': batch_checked,
                            'offline_batch_scalar_parity_tolerance': 1e-12,
                            'offline_batch_scalar_max_absolute_error': batch_error,
                            'offline_batch_scalar_bitwise_equal': batch_bitwise,
                            'array_bytes_unchanged_across_checked_path': initial_bytes,
                            'long_run_bounded_memory_scope': 'Short path check here; separate16384 unit tests and direct10M audit workload verify longer runs.'}
                        save(relative + '/' + key + '.json', result)
                        checks.append(result)
        # Reset independence is evaluated in the opposite fold/case/method order,
        # against the same fixtures rather than against a newly regenerated oracle.
        for fold in reversed(range(5)):
            bundle = load(BASE / f'research_bundles/fold_{fold}')
            with activity(f'Final reverse-order reset fold{fold}', 'FINAL_AUDIT', 'FINAL_AUDIT'):
                for case, seed in reversed(cases):
                    h, online, _, _ = generated(case, seed, partition=31, horizon=1024, control=False, lock=suite)
                    for method in reversed(methods):
                        key = f'fold{fold}_{case}_seed{seed}_{method}'
                        actual, _ = stream(bundle.prepare(method), h, online)
                        np.testing.assert_array_equal(actual, expected_file[key])
                        reverse_checks.append(key)
    assert len(checks) == len(reverse_checks) == 90 and not forbidden_reads
    result = {'status': 'PASS', 'created_utc': utc_now(), 'identity': identity,
        'fitted_folds': 5, 'methods': methods, 'fixed_synthetic_paths': 90,
        'reverse_order_paths_exact': len(reverse_checks), 'competition_data_read_attempts': forbidden_reads,
        'all_scientific_prediction_fingerprints_unchanged': True,
        'offline_batch_scope': 'NEXT7 recipes expose batch replay; M0 scalar fixture and original legacy runtime tests cover M0.',
        'per_path_receipts_sha256': {path.name: digest(path) for path in sorted(directory.glob('fold*.json'))},
        'no_new_fits_or_scoring_or_selection': True}
    save('audit/FINAL_ONLINE_CONTRACTS.json', result)
    note('Final online contract audit passed', '90 original paths plus opposite order replay; exact predictions and causal input traps', 'FINAL_AUDIT', phase='FINAL_AUDIT')


if __name__ == '__main__':
    main()
