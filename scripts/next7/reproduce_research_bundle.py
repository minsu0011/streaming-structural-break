"""Raw-data-free streaming reproduction of the frozen five research bundles."""
from pathlib import Path
import argparse
import os
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(output_directory, *, create_fixture=False, deny_original_root=None):
    # Install the optional audit before scientific imports or model loading.
    denied = []
    opened_payload_files = set()
    if deny_original_root is not None:
        original = Path(deny_original_root).resolve()
        payload = ROOT.resolve()
        venv = (original / '.venv').resolve()
        assert payload != original
        def audit(event, arguments):
            if event != 'open' or not arguments or not isinstance(arguments[0], (str, bytes, os.PathLike)):
                return
            raw = os.fsdecode(arguments[0])
            path = Path(raw).resolve()
            if path.is_relative_to(payload):
                opened_payload_files.add(str(path.relative_to(payload)))
                return
            if path.is_relative_to(original) and not path.is_relative_to(venv):
                denied.append(str(path))
                raise PermissionError('Portable reproduction attempted original workspace access: ' + str(path))
        sys.addaudithook(audit)
    os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
    from src.next7.worker_environment import environment
    os.environ.update(environment())
    import numpy as np
    from src.next7.common import BASE, read_json, atomic, digest, utc_now
    from src.next7.bundle import load, RECIPES
    from src.next6_extension.synthetic import generated
    output = Path(output_directory).resolve()
    assert not output.exists(), 'Use a new output directory; preserve previous reproduction'
    output.mkdir(parents=True)
    fixture = BASE / 'reproduction_fixture'
    methods = ('M0', *RECIPES)
    cases = (('N0', 11), ('V4', 1), ('RETURN_NORMAL', 0))
    suite = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    if not create_fixture:
        authority = read_json(fixture / 'FIXTURE.json')
        assert authority['status'] == 'FROZEN' and authority['source_sha256'] == digest(__file__)
        assert digest(fixture / 'PREDICTIONS.npz') == authority['prediction_sha256']
        expected_file = np.load(fixture / 'PREDICTIONS.npz', allow_pickle=False)
    else:
        expected_file = None
    checks, predictions, input_hashes, bundle_hashes = [], {}, {}, {}
    start = time.perf_counter()
    import hashlib
    for fold in range(5):
        directory = BASE / f'research_bundles/fold_{fold}'
        bundle = load(directory)
        bundle_hashes[str(fold)] = digest(directory / 'MANIFEST.json')
        for case, seed in cases:
            h, o, _, _ = generated(case, seed, partition=31, horizon=1024, control=False, lock=suite)
            key = f'{case}_seed{seed}'
            hashes = {'H': hashlib.sha256(h.tobytes()).hexdigest(), 'O': hashlib.sha256(o.tobytes()).hexdigest()}
            input_hashes[key] = hashes
            if not create_fixture:
                assert authority['synthetic_input_sha256'][key] == hashes
                assert authority['bundle_manifest_sha256'][str(fold)] == bundle_hashes[str(fold)]
            for method in methods:
                state = bundle.new_state(h, method)
                # Only H and the present scalar enter inference. Truth and the
                # known synthetic duration remain outside the state API.
                result = np.array([state.predict_one(float(value)) for value in o])
                assert np.isfinite(result).all() and np.all((result >= 0) & (result <= 1))
                prediction_key = f'fold{fold}_{key}_{method}'
                predictions[prediction_key] = result
                error = None
                exact = None
                if expected_file is not None:
                    expected = expected_file[prediction_key]
                    error = float(np.max(np.abs(result - expected)))
                    exact = bool(np.array_equal(result, expected))
                    assert error <= 1e-12
                checks.append({'fold': fold, 'case': case, 'seed': seed, 'method': method,
                    'points': len(result), 'max_absolute_error': error, 'bitwise_equal': exact,
                    'last_score': float(result[-1]), 'new_state_receives_no_labels_or_ID': True})
        print('Reproduced research fold', fold, flush=True)
    np.savez_compressed(output / 'PREDICTIONS.npz', **predictions)
    if expected_file is not None:
        assert set(expected_file.files) == set(predictions)
        expected_file.close()
    assert not denied
    meta = {'status': 'FROZEN' if create_fixture else 'PASS', 'created_utc': utc_now(),
        'source_sha256': digest(__file__), 'methods': methods, 'fitted_folds': 5,
        'synthetic_cases': [{'case': case, 'seed': seed, 'partition': 31, 'horizon': 1024} for case, seed in cases],
        'checks': checks, 'prediction_sha256': digest(output / 'PREDICTIONS.npz'),
        'synthetic_input_sha256': input_hashes, 'bundle_manifest_sha256': bundle_hashes,
        'elapsed_seconds': time.perf_counter() - start, 'python_version': sys.version, 'numpy_version': np.__version__,
        'role': 'RESEARCH_FOLD_ONLY_NOT_PROMOTED', 'primary_role_changed': False, 'new_fits': 0,
        'raw_competition_data_used': False, 'original_workspace_access_blocked': deny_original_root is not None,
        'denied_access_attempts': len(denied), 'payload_files_opened': len(opened_payload_files),
        'scope': 'Fresh process load and scalar streaming reproduction; not a raw-RP training refit or evidence of predictive improvement.'}
    atomic(output / ('FIXTURE.json' if create_fixture else 'REPRODUCTION_DONE.json'), meta)
    print('PASS: research reproduction,90 paths,92160 scalar outputs; original roles retained', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-directory', required=True)
    parser.add_argument('--create-fixture', action='store_true')
    parser.add_argument('--deny-original-root')
    args = parser.parse_args()
    main(args.output_directory, create_fixture=args.create_fixture, deny_original_root=args.deny_original_root)
