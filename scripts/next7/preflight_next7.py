"""NEXT7 byte hashes, admission denial and scorer parity before RP experiments."""
from pathlib import Path
import os
import sys
import subprocess
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
from src.next7.common import BASE, save, read_json, digest, activity, utc_now, note


def main():
    if (BASE / 'PREFLIGHT_NEXT7.json').exists():
        raise RuntimeError('Preserve completed guard audit')
    assert read_json(BASE / 'RESTORED_NEXT6_AUTHORITY.json')['status'] == 'PASS'
    with activity('NEXT7 data boundary and scorer gate', 'GUARD'):
        from src.next5.guard import ResearchGuard, ShadowAccessError
        from src.next.data import RAW_SHA
        guard = ResearchGuard(ROOT)
        assert (len(guard.research_ids), len(guard.shadow_ids), len(guard.seal_ids)) == (6400, 1600, 2000)
        denied = []
        for name, ids in [('shadow', guard.shadow_ids), ('seal', guard.seal_ids)]:
            try:
                guard.admit([min(ids)], purpose='NEXT7 negative admission probe')
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
        env = os.environ.copy()
        env['NEXT7_TEST_RECEIPT'] = 'PREFLIGHT_TEST_ACCESS.json'
        with (BASE / 'PREFLIGHT_TESTS.log').open('x', encoding='utf-8') as log:
            proc = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                '-p', 'src.next7.test_guard', 'tests/test_next5_shadow_guard.py',
                'tests/test_scorer_official_checks.py', '--basetemp=' + str(BASE / 'runtime/pytest_preflight'),
                '--junitxml=' + str(BASE / 'PREFLIGHT_TESTS.xml')],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        assert proc.returncode == 0, 'Scientific experiments blocked: guard/scorer tests failed'
        assert read_json(BASE / 'audit/PREFLIGHT_TEST_ACCESS.json')['status'] == 'PASS'
        # Admission precedes decoding the already isolated RP cache.
        import numpy as np
        from src.next5.data import ResearchArrays
        data = ResearchArrays(directory, guard)
        rows = data.rows()
        assert len(data.ids) == 6400 and len(rows['target']) == 3203157
        for index, sid in enumerate(data.ids):
            a, b = data.series['online_offsets'][index:index + 2]
            tau = int(data.series['tau'][index])
            expected = (np.arange(b - a) >= tau).astype(np.uint8) if tau >= 0 else np.zeros(b - a, dtype=np.uint8)
            np.testing.assert_array_equal(rows['target'][a:b], expected)
        guard.check_state()
        save('PREFLIGHT_NEXT7.json', {'status': 'PASS', 'timestamp': utc_now(),
            'raw_sha256': raw, 'split_sha256': guard.split.digest,
            'shadow_lock_sha256': guard.shadow_lock_sha, 'cache_manifest_sha256': digest(directory / 'MANIFEST.json'),
            'RP_series': 6400, 'RP_online_rows': 3203157,
            'SHADOW_ACCESS': 0, 'SEAL_ACCESS': 0, 'REDUCED_NEW_USAGE': 0,
            'PUBLIC_NEW_USAGE': 0, 'CLOUD_NEW_USAGE': 0,
            'denied_admission_probes': denied, 'guard_tests': 'PASS', 'scorer_tests': 'PASS',
            'target_contract_all_RP': 'PASS', 'raw_parquet_decoding': False,
            'scope': 'Raw byte hashes only; existing isolated RP arrays admitted before decoding',
            'application_guard_not_OS_sandbox': True})
    note('NEXT7 guard PASS', 'Proceed to fixed scientific protocol and failure deduplication')


if __name__ == '__main__':
    main()
