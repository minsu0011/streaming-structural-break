"""Final-phase byte/source/data/role integrity; no competition value decoding."""
from pathlib import Path
import sys
import subprocess
import hashlib
import importlib.metadata
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, atomic, utc_now, activity, note
from src.next8.runtime_accounting import audit_admitted


def main():
    admission = read_json(BASE / 'runtime/FINAL_AUDIT_ADMISSION.json')
    heartbeat = read_json(BASE / 'runtime/HEARTBEAT.json')
    assert audit_admitted(admission, heartbeat['productive_seconds'], digest(BASE / 'SESSION_NEXT8.json'), 36000.)
    closure_path = BASE / 'runtime/FINAL_SOURCE_CLOSURE_NEXT8.json'
    closure = read_json(closure_path)
    assert closure['status'] == 'FROZEN'
    out = BASE / 'final_integrity'
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError('Existing or partial final integrity evidence retained')
    with activity('Final protected models, sources and prior Desktop bytes', 'FINAL_AUDIT', 'B9_NESTED_SELECTION'):
        protected = read_json(BASE / 'PROTECTED_BEFORE_NEXT8_SHA256.json')
        for index, (name, sha) in enumerate(protected.items()):
            path = (ROOT / name).resolve()
            assert path.is_relative_to(ROOT), name
            assert digest(path) == sha, 'Protected authority changed: ' + name
            if index % 500 == 0:
                print('Verified protected files', index, len(protected), flush=True)
        for name, sha in closure['source_sha256'].items():
            assert digest(ROOT / name) == sha, 'Final source closure changed: ' + name
        previous = read_json(BASE / 'RESTORED_NEXT7_AUTHORITY.json')
        desktop = Path(previous['previous_Desktop_directory'])
        for name, sha in previous['previous_Desktop_files_sha256'].items():
            assert digest(desktop / name) == sha, 'Previous Desktop result changed: ' + name
        staged = subprocess.check_output(['git', 'diff', '--cached', '--binary'], cwd=ROOT)
        worktree = subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT)
        assert staged == (BASE / 'STAGED_PATCH_BEFORE.diff').read_bytes(), 'Existing staged patch changed'
        assert worktree == (BASE / 'WORKTREE_PATCH_BEFORE.diff').read_bytes(), 'Existing tracked worktree patch changed'
        atomic(out / 'PROTECTED_INTEGRITY.json', {'status': 'PASS', 'timestamp': utc_now(),
               'protected_files_verified': len(protected), 'prior_Desktop_file_hashes_verified': len(previous['previous_Desktop_files_sha256']),
               'original_staged_and_tracked_worktree_patches_bitwise_equal': True,
               'staged_patch_sha256': hashlib.sha256(staged).hexdigest(),
               'tracked_worktree_patch_sha256': hashlib.sha256(worktree).hexdigest(),
               'protected_manifest_sha256': digest(BASE / 'PROTECTED_BEFORE_NEXT8_SHA256.json'),
               'source_closure_sha256': digest(closure_path), 'source_files_verified': len(closure['source_sha256']),
               'canonical_BL80_D2_J102_roles_changed': False})
    with activity('Final raw bytes, isolated RP cache and closed membership guards', 'FINAL_AUDIT', 'B7_NORMALIZATION_AND_LINEAGE'):
        from src.next5.guard import ResearchGuard, ShadowAccessError
        guard = ResearchGuard(ROOT)
        preflight = read_json(BASE / 'PREFLIGHT_NEXT8.json')
        assert guard.split.digest == preflight['split_sha256']
        assert guard.shadow_lock_sha == preflight['shadow_lock_sha256']
        raw = {name: digest(ROOT / 'data/raw' / name) for name in preflight['raw_sha256']}
        assert raw == preflight['raw_sha256']
        cache = ROOT / 'data/next5/research_5d54e316ff5bf25d'
        assert digest(cache / 'MANIFEST.json') == preflight['cache_manifest_sha256']
        manifest = read_json(cache / 'MANIFEST.json')
        for name, sha in manifest['files_sha256'].items():
            assert digest(cache / name) == sha, name
        rejected = []
        for family, ids in [('SHADOW', guard.shadow_ids), ('SEAL', guard.seal_ids)]:
            try:
                guard.admit([min(ids)], purpose='NEXT8 final forbidden-membership denial test')
            except ShadowAccessError:
                rejected.append(family)
        assert rejected == ['SHADOW', 'SEAL']
        assert (len(guard.research_ids), len(guard.shadow_ids), len(guard.seal_ids)) == (6400, 1600, 2000)
        guard.check_state()
        atomic(out / 'DATA_BOUNDARY_INTEGRITY.json', {'status': 'PASS', 'timestamp': utc_now(),
               'raw_sha256': raw, 'research_split_sha256': guard.split.digest, 'shadow_lock_sha256': guard.shadow_lock_sha,
               'isolated_RP_cache_all_file_bytes_verified': True, 'cache_manifest_sha256': digest(cache / 'MANIFEST.json'),
               'permitted_RP_series': 6400, 'Shadow_series_still_closed': 1600, 'Seal_series_still_closed': 2000,
               'forbidden_membership_admission_probes_rejected': rejected, 'probes_decode_no_forbidden_values': True,
               'audit_Parquet_value_decoding': 0, 'audit_RP_value_decoding': 0,
               'Shadow_Seal_reduced_public_cloud_new_value_usage': 0,
               'scope': 'Original bytes and application-level admission verified; memberships and source metadata may be inspected',
               'historical_cache_creation_Parquet_internal_decoding_not_retroactively_audited': True,
               'not_an_OS_information_flow_proof': True})
    with activity('Final frozen prediction and numerical environment integrity', 'FINAL_AUDIT', 'B7_NORMALIZATION_AND_LINEAGE'):
        release = read_json(BASE / 'PREDICTION_FREEZE_NEXT8.json')
        assert release['status'] == 'FROZEN'
        for index, (name, sha) in enumerate(release['files_sha256'].items()):
            assert digest(BASE / name) == sha, 'Frozen prediction/model bytes changed: ' + name
            if index % 500 == 0:
                print('Verified frozen prediction artifacts', index, len(release['files_sha256']), flush=True)
        versions = {}
        for name in ('numpy', 'numba', 'llvmlite', 'scipy', 'scikit-learn', 'lightgbm', 'xgboost',
                     'joblib', 'threadpoolctl', 'pandas', 'pyarrow', 'psutil', 'pytest'):
            versions[name] = importlib.metadata.version(name)
        assert versions == closure['package_versions'], 'Numerical environment changed after final source closure'
        atomic(out / 'FROZEN_PREDICTION_INTEGRITY.json', {'status': 'PASS', 'timestamp': utc_now(),
               'prediction_freeze_sha256': digest(BASE / 'PREDICTION_FREEZE_NEXT8.json'),
               'frozen_artifact_count': len(release['files_sha256']), 'all_frozen_bytes_unchanged': True,
               'python': sys.version, 'package_versions': versions, 'existing_environment_modified': False,
               'model_quality_recomputed_by_this_audit': False})
    atomic(out / 'DONE.json', {'status': 'COMPLETE', 'timestamp': utc_now(),
           'files_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file()},
           'final_full_replay_tests_and_delivery_are_separate_required_receipts': True})
    note('Final byte/data/source integrity PASS', 'Full replay, complete tests and delivery receipts remain separate',
         'B7_NORMALIZATION_AND_LINEAGE', phase='FINAL_AUDIT')


if __name__ == '__main__':
    main()
