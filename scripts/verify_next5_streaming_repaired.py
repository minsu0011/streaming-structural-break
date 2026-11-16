"""Run the reviewed import-only API repair and publish hash-bound aliases."""
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress


def main():
    policy_path = BASE / 'maintenance/READONLY_IMPORT_REPAIR/POLICY_LOCK.json'
    policy = read_json(policy_path)
    for name, sha in policy['sources_sha256'].items():
        assert digest(ROOT / name) == sha
    from scripts.verify_next5_streaming_readonly_core import main as verify
    # The generated verifier imports the public adapter inside its original
    # active write audit. No model implementation is imported above this call.
    verify()
    index = sys.argv.index('--candidate')
    cid = sys.argv[index + 1]
    assert cid in policy['candidates']
    fresh = BASE / 'runtime_validation/api_readonly_v1' / cid
    result = read_json(fresh / 'RESULTS.json')
    assert result['status'] == 'PASS' and result['candidate'] == cid
    assert result['series'] == 6400 and result['points'] == 3203157
    assert result['persistent_filesystem_write_attempts'] == 0 and result['prefix_cases'] >= 300
    canonical = BASE / 'runtime_validation/api' / cid
    canonical.mkdir(parents=True, exist_ok=True)
    original_start = digest(canonical / 'START.json') if (canonical / 'START.json').exists() else None
    for name in ['RESULTS.json'] + ['fold_' + str(fold) + '.json' for fold in range(5)]:
        assert not (canonical / name).exists(), 'Never overwrite an earlier completed API result'
        shutil.copyfile(fresh / name, canonical / name)
        assert digest(fresh / name) == digest(canonical / name)
    if original_start is None:
        shutil.copyfile(fresh / 'START.json', canonical / 'START.json')
    atomic_json(canonical / 'API_IMPORT_REPAIR.json', {'created_utc': utc_now(), 'status': 'PASS',
        'candidate': cid, 'actual_entry': 'src.next5.readonly_streaming',
        'unchanged_mathematical_core': 'src.next5.streaming',
        'policy_sha256': digest(policy_path), 'fresh_result': (fresh / 'RESULTS.json').relative_to(ROOT).as_posix(),
        'fresh_result_sha256': digest(fresh / 'RESULTS.json'),
        'preserved_original_start_sha256': original_start, 'new_model_fits': 0,
        'canonical_result_is_byte_equal_copy': True,
        'original_failed_or_unstarted_jobs_preserved': True,
        'scope': 'Only import-time disk caching and fresh audit output location changed; original models, source math, scoring and prefix assertions are unchanged'}, immutable=True)
    progress('Read-only API import repair verified', cid + '; complete fresh all-RP audit published with original failure evidence preserved')


if __name__ == '__main__':
    main()
