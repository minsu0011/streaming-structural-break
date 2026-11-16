"""Fresh replay v2: same frozen model computation, short payload-local cache."""
from pathlib import Path
import argparse
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(output_directory, create_fixture=False, deny_original_root=None):
    output = Path(output_directory).resolve()
    assert not output.exists(), 'Use a fresh output directory; existing evidence is preserved'
    # Load the boundary file directly so no package initialization precedes it.
    import importlib.util
    spec = importlib.util.spec_from_file_location('next8_rawfree_boundary', ROOT / 'src/next8/rawfree_boundary.py')
    boundary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(boundary)
    denied, opened = boundary.install(ROOT, original=deny_original_root, output=output)
    import src.next8
    # The prior nested run cache produced an actual 282-character Windows
    # temporary filename. Keep the fresh cache inside the same allowed payload.
    import os
    cache = ROOT / 'nc'
    assert not cache.exists(), 'Use a fresh extraction; do not reuse a compiled cache'
    assert 'numba' not in sys.modules, 'Cache must be selected before Numba import'
    os.environ['NUMBA_CACHE_DIR'] = str(cache)
    import numpy as np
    from src.next8.common import BASE, atomic, read_json, digest, utc_now, activity
    from src.next8.portable_replay import calculate_research, calculate_canonical
    repair_lock = read_json(BASE / 'PORTABLE_CACHE_PATH_IMPLEMENTATION_LOCK_NEXT8.json')
    for name, sha in repair_lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    lock = read_json(BASE / 'PORTABLE_REPLAY_IMPLEMENTATION_LOCK_NEXT8.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    manifest_path = BASE / 'PORTABLE_RUNTIME_MANIFEST_NEXT8.json'
    manifest = read_json(manifest_path)
    for name, sha in manifest['files_sha256'].items():
        assert digest(ROOT / name) == sha, name
    if create_fixture:
        from src.next8.runtime_accounting import audit_admitted
        admission = read_json(BASE / 'runtime/FINAL_AUDIT_ADMISSION.json')
        hb = read_json(BASE / 'runtime/HEARTBEAT.json')
        assert audit_admitted(admission, hb['productive_seconds'], digest(BASE / 'SESSION_NEXT8.json'), 36000.)
        assert output == (BASE / 'portable_fixture').resolve()
    else:
        authority = read_json(BASE / 'portable_fixture/FIXTURE.json')
        assert authority['status'] == 'FROZEN'
        assert authority['runtime_manifest_sha256'] == digest(manifest_path)
        assert authority['implementation_lock_sha256'] == digest(BASE / 'PORTABLE_REPLAY_IMPLEMENTATION_LOCK_NEXT8.json')
        assert digest(BASE / 'portable_fixture/PREDICTIONS.npz') == authority['predictions_sha256']
    started = time.monotonic()
    with activity('Portable raw-free actual-model replay', 'FINAL_AUDIT', 'B7_NORMALIZATION_AND_LINEAGE'):
        prediction, checks, inputs = calculate_research(manifest)
        canonical, canonical_checks = calculate_canonical()
        prediction.update(canonical)
        if not create_fixture:
            assert inputs == authority['synthetic_inputs']
            with np.load(BASE / 'portable_fixture/PREDICTIONS.npz', allow_pickle=False) as expected:
                assert set(expected.files) == set(prediction)
                for name, values in prediction.items():
                    np.testing.assert_array_equal(values.view(np.uint64), expected[name].view(np.uint64))
        assert not denied
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output / 'PREDICTIONS.npz', **prediction)
        result = {'status': 'FROZEN' if create_fixture else 'PASS', 'timestamp': utc_now(),
                  'runtime_manifest_sha256': digest(manifest_path),
                  'implementation_lock_sha256': digest(BASE / 'PORTABLE_REPLAY_IMPLEMENTATION_LOCK_NEXT8.json'),
                  'predictions_sha256': digest(output / 'PREDICTIONS.npz'), 'synthetic_inputs': inputs,
                  'checks': checks, 'canonical_public_entry_checks': canonical_checks,
                  'research_paths': len(checks), 'canonical_paths': len(canonical_checks),
                  'scalar_prediction_points': sum(len(v) for v in prediction.values()),
                  'replay_new_process_fixture_bitwise': not create_fixture,
                  'elapsed_seconds': time.monotonic() - started, 'new_fits': 0, 'new_quality_scores': 0,
                  'competition_or_RP_cache_access': 0, 'denied_read_attempts': len(denied),
                  'original_workspace_access_blocked': deny_original_root is not None,
                  'payload_opened_files': sorted(opened), 'boundary_not_an_OS_sandbox': True,
                  'interpreter_environment_reused': 'Existing .venv; no installation or package mutation',
                  'fresh_compiler_cache_relative_to_payload': 'nc',
                  'original_read_boundary_and_numerical_sources_unchanged': True,
                  'portable_cache_path_implementation_lock_sha256': digest(BASE / 'PORTABLE_CACHE_PATH_IMPLEMENTATION_LOCK_NEXT8.json'),
                  'scope': 'Actual frozen inference and state reproduction on synthetic observations; not a raw-data retraining or held-out quality claim'}
        atomic(output / ('FIXTURE.json' if create_fixture else 'REPRODUCTION_DONE.json'), result)
    print('Portable NEXT8 replay complete', len(prediction), 'paths', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-directory', required=True)
    parser.add_argument('--create-fixture', action='store_true')
    parser.add_argument('--deny-original-root')
    args = parser.parse_args()
    main(args.output_directory, args.create_fixture, args.deny_original_root)
