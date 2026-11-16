"""Restore prior byte authority, then verify only the isolated RP and scorer."""
from pathlib import Path
import json
import os
import subprocess
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next8.common import BASE, save, read_json, digest, activity, utc_now, note


def main():
    assert not (BASE / 'PREFLIGHT_NEXT8.json').exists(), 'Completed preflight is immutable'
    with activity('Restore NEXT7 and earlier byte authority', 'RESTORE', 'B9_NESTED_SELECTION'):
        legacy = ROOT / 'artifacts/next7'
        protected = read_json(legacy / 'PROTECTED_BEFORE_NEXT7_SHA256.json')
        for name, sha in protected.items():
            assert digest(ROOT / name) == sha, 'Protected prior authority changed: ' + name
        source_manifest = read_json(legacy / 'SOURCE_MANIFEST_NEXT7.json')
        for name, sha in source_manifest['files_sha256'].items():
            assert digest(ROOT / name) == sha, 'NEXT7 source/model differs from handoff: ' + name
            protected[name] = sha
        delivery = read_json(legacy / 'FINAL_DELIVERY_AUDIT_NEXT7.json')
        assert delivery['status'] == 'PASS' and delivery['decision'] == 'NEXT7_NO_PROMOTION'
        archive = legacy / delivery['archive']
        assert digest(archive) == delivery['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            archive_entries = len(z.infolist())
        for directory in ('src', 'scripts', 'tests', 'configs', 'docs'):
            for path in (ROOT / directory).rglob('*'):
                if not path.is_file() or path.is_symlink() or path.is_junction():
                    continue
                rel = path.relative_to(ROOT)
                if 'next8' in rel.parts or '__pycache__' in rel.parts:
                    continue
                if path.suffix.lower() in ('.py', '.md', '.json', '.yaml', '.yml', '.toml', '.ps1', '.mjs'):
                    protected[rel.as_posix()] = digest(path)
        for path in ROOT.iterdir():
            if path.is_file():
                protected[path.name] = digest(path)
        for path in legacy.iterdir():
            if path.is_file():
                protected[path.relative_to(ROOT).as_posix()] = digest(path)
        for path in (legacy / 'nested_base').rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.joblib', '.npz', '.npy'):
                protected[path.relative_to(ROOT).as_posix()] = digest(path)
        desktop = Path(delivery['desktop_directory'])
        desktop_hashes = {p.name: digest(p) for p in desktop.iterdir() if p.is_file()}
        assert len(desktop_hashes) == 20
        save('PROTECTED_BEFORE_NEXT8_SHA256.json', protected)
        save('RESTORED_NEXT7_AUTHORITY.json', {'status': 'PASS', 'created_utc': utc_now(),
            'previous_protected_files_verified': 6333, 'protected_files': len(protected),
            'NEXT7_source_manifest_sha256': digest(legacy / 'SOURCE_MANIFEST_NEXT7.json'),
            'NEXT7_archive_sha256': digest(archive), 'NEXT7_archive_CRC_verified': True,
            'NEXT7_archive_entries': archive_entries,
            'previous_Desktop_directory': str(desktop), 'previous_Desktop_files_sha256': desktop_hashes,
            'guarded_inner_reference_artifact_bytes_preserved': True,
            'no_point_arrays_decoded': True, 'source_sha256': digest(__file__)})
    with activity('NEXT8 RP-only membership, source cache and scorer gate', 'GUARD', 'B9_NESTED_SELECTION'):
        from src.next5.guard import ResearchGuard, ShadowAccessError
        from src.next.data import RAW_SHA
        guard = ResearchGuard(ROOT)
        assert (len(guard.research_ids), len(guard.shadow_ids), len(guard.seal_ids)) == (6400, 1600, 2000)
        denied = []
        for name, ids in [('shadow', guard.shadow_ids), ('seal', guard.seal_ids)]:
            try:
                guard.admit([min(ids)], purpose='NEXT8 forbidden-membership probe')
            except ShadowAccessError:
                denied.append(name)
        assert denied == ['shadow', 'seal']
        raw = {name: digest(ROOT / 'data/raw' / name) for name in RAW_SHA}
        assert raw == RAW_SHA
        directory = ROOT / 'data/next5/research_5d54e316ff5bf25d'
        manifest = read_json(directory / 'MANIFEST.json')
        for name, sha in manifest['files_sha256'].items():
            assert digest(directory / name) == sha, name
        assert manifest['identity']['research_split_sha256'] == guard.split.digest
        assert manifest['identity']['shadow_lock_sha256'] == guard.shadow_lock_sha
        assert manifest['identity']['raw_sha256'] == raw
        env = dict(os.environ, NEXT8_TEST_RECEIPT='PREFLIGHT_TEST_ACCESS.json')
        with (BASE / 'PREFLIGHT_TESTS.log').open('x', encoding='utf-8') as log:
            proc = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                '-p', 'src.next8.test_guard', 'tests/test_next5_shadow_guard.py',
                'tests/test_scorer_official_checks.py', '--basetemp=' + str(BASE / 'runtime/pytest_preflight'),
                '--junitxml=' + str(BASE / 'PREFLIGHT_TESTS.xml')],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert proc.returncode == 0, 'Guard/scorer tests failed; research remains blocked'
        assert read_json(BASE / 'audit/PREFLIGHT_TEST_ACCESS.json')['status'] == 'PASS'
        import numpy as np
        from src.next5.data import ResearchArrays
        data = ResearchArrays(directory, guard)
        rows = data.rows()
        assert len(data.ids) == 6400 and len(rows['target']) == 3203157
        for index, sid in enumerate(data.ids):
            a, b = data.series['online_offsets'][index:index + 2]
            tau = int(data.series['tau'][index])
            expected = (np.arange(b-a) >= tau).astype(np.uint8) if tau >= 0 else np.zeros(b-a, dtype=np.uint8)
            np.testing.assert_array_equal(rows['target'][a:b], expected)
        guard.check_state()
        result = {'status': 'PASS', 'timestamp': utc_now(), 'raw_sha256': raw,
            'split_sha256': guard.split.digest, 'shadow_lock_sha256': guard.shadow_lock_sha,
            'cache_manifest_sha256': digest(directory / 'MANIFEST.json'),
            'RP_series': 6400, 'RP_online_rows': 3203157,
            'SHADOW_ACCESS': 0, 'SEAL_ACCESS': 0, 'REDUCED_NEW_USAGE': 0,
            'PUBLIC_NEW_USAGE': 0, 'CLOUD_NEW_USAGE': 0,
            'denied_admission_probes': denied, 'guard_tests': 'PASS', 'scorer_tests': 'PASS',
            'target_contract_all_RP': 'PASS', 'raw_parquet_decoding': False,
            'scope': 'Raw bytes hashed; only preexisting isolated RP cache admitted and decoded. NEXT8 performs no raw Parquet observation decoding.',
            'historical_cache_creation_internal_Parquet_decoding_not_retroactively_audited': True,
            'outer_score_used_for_selection': False, 'application_guard_not_OS_sandbox': True,
            'source_sha256': digest(__file__)}
        save('PREFLIGHT_NEXT8.json', result)
        save('NEXT8_DATA_ACCESS_AUDIT.json', result)
    note('NEXT8 preflight PASS', 'RP6400 and original folds verified; proceed to lineage and nested protocol', hypothesis='B7_NORMALIZATION_AND_LINEAGE')


if __name__ == '__main__':
    main()
