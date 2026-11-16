"""Final raw byte and admitted ResearchPool membership/target audit."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, save, read_json, digest, utc_now, activity, note


def main():
    from scripts.next7.benchmark_runtime import audit_gate
    audit_gate()
    from src.next5.guard import ResearchGuard, ShadowAccessError
    from src.next.data import RAW_SHA
    import numpy as np
    from src.next5.data import ResearchArrays
    original = read_json(BASE / 'PREFLIGHT_NEXT7.json')
    guard = ResearchGuard(ROOT)
    note('Final data boundary audit', 'Raw bytes and original RP membership; no hidden observation decoding', 'FINAL_AUDIT', phase='FINAL_AUDIT')
    with activity('Final data identity and RP target parity', 'FINAL_AUDIT', 'FINAL_AUDIT'):
        denied = []
        for name, ids in (('SHADOW', guard.shadow_ids), ('SEAL', guard.seal_ids)):
            try:
                guard.admit([min(ids)], purpose='NEXT7 final negative admission probe')
            except ShadowAccessError:
                denied.append(name)
        assert denied == ['SHADOW', 'SEAL']
        raw = {name: digest(ROOT / 'data/raw' / name) for name in RAW_SHA}
        assert raw == RAW_SHA == original['raw_sha256']
        assert guard.split.digest == original['split_sha256']
        assert guard.shadow_lock_sha == original['shadow_lock_sha256']
        directory = ROOT / 'data/next5/research_5d54e316ff5bf25d'
        manifest_path = directory / 'MANIFEST.json'
        assert digest(manifest_path) == original['cache_manifest_sha256']
        manifest = read_json(manifest_path)
        assert manifest['identity']['raw_sha256'] == raw
        assert manifest['identity']['research_split_sha256'] == guard.split.digest
        for name, sha in manifest['files_sha256'].items():
            assert digest(directory / name) == sha
        guard.admit(guard.research_ids, purpose='NEXT7 final RP cache membership verification')
        data = ResearchArrays(directory, guard)
        rows = data.rows()
        assert set(map(int, data.ids)) == guard.research_ids and len(data.ids) == 6400
        assert len(rows['target']) == original['RP_online_rows'] == 3203157
        folds = []
        for fold in range(5):
            valid = {sid for sid in guard.research_ids if guard.split.assignment(sid)[1] == fold}
            train = guard.research_ids - valid
            guard.partition(train, valid)
            folds.append({'fold': fold, 'train_series': len(train), 'validation_series': len(valid),
                'train_validation_group_overlap': False})
        for index, sid in enumerate(data.ids):
            a, b = data.series['online_offsets'][index:index + 2]
            tau = int(data.series['tau'][index])
            expected = (np.arange(b - a) >= tau).astype(np.uint8) if tau >= 0 else np.zeros(b - a, dtype=np.uint8)
            np.testing.assert_array_equal(rows['target'][a:b], expected)
        guard.check_state()
        save('audit/FINAL_DATA_BOUNDARY.json', {'status': 'PASS', 'created_utc': utc_now(),
            'source_sha256': digest(__file__), 'original_preflight_sha256': digest(BASE / 'PREFLIGHT_NEXT7.json'),
            'raw_sha256': raw, 'raw_parquet_decoded': False,
            'RP_manifest_sha256': digest(manifest_path), 'RP_files_sha256': manifest['files_sha256'],
            'research_split_sha256': guard.split.digest, 'shadow_lock_sha256': guard.shadow_lock_sha,
            'RP_series': 6400, 'RP_online_rows': 3203157, 'all_RP_target_rows_match_tau_contract': True,
            'truth_use': 'Audit of evaluation labels only; tau never supplied to a detector.',
            'folds': folds, 'denied_hidden_admission_probes': denied,
            'SHADOW_observation_rows_decoded': 0, 'SEAL_observation_rows_decoded': 0,
            'new_reduced_or_public_or_cloud_use': 0, 'application_guard_not_OS_sandbox': True})
    note('Final data boundary audit passed', 'Original hashes and all RP target rows match; hidden admission denied', 'FINAL_AUDIT', phase='FINAL_AUDIT')


if __name__ == '__main__':
    main()
