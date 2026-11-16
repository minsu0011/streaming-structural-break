"""Resolve original API import failures only after every fresh audit passes."""
from pathlib import Path
import csv
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress


def main():
    parent = BASE / 'maintenance/READONLY_IMPORT_REPAIR'
    policy_path = parent / 'POLICY_LOCK.json'
    policy = read_json(policy_path)
    for name, sha in policy['sources_sha256'].items():
        assert digest(ROOT / name) == sha
    repair = read_json(parent / 'REPAIR.json')
    rows = []
    for old_id, new_id in repair['replacement_jobs'].items():
        cid = old_id.removeprefix('API_')
        new_state_path = BASE / 'jobs' / new_id / 'STATE.json'
        new_state = read_json(new_state_path)
        assert new_state['status'] == 'COMPLETE' and new_state['exit_code'] == 0
        api_path = BASE / 'runtime_validation/api' / cid / 'RESULTS.json'
        api = read_json(api_path)
        adapter_path = api_path.parent / 'API_IMPORT_REPAIR.json'
        adapter = read_json(adapter_path)
        assert api['status'] == adapter['status'] == 'PASS'
        assert api['persistent_filesystem_write_attempts'] == 0
        assert adapter['fresh_result_sha256'] == digest(api_path)
        old_path = BASE / 'jobs' / old_id / 'STATE.json'
        old = read_json(old_path)
        assert old['status'] == 'QUARANTINED'
        actual_failure = old_id in policy['actual_failed_jobs']
        if actual_failure:
            assert old['exit_code'] == 1 and old.get('started_utc')
            assert digest(ROOT / old['attempt_directory'] / 'RUN.log') == old['log_sha256']
        else:
            assert not old.get('started_utc') and old['resolved_by'] == new_id
        evidence_paths = [policy_path, parent / 'REPAIR.json', parent / 'DERIVATION.json',
            parent / 'COLD_PROBE.json', new_state_path, api_path, adapter_path]
        resolution = {'created_utc': utc_now(), 'status': 'RESOLVED_WITH_PRESERVED_FAILURE',
            'job': old_id, 'replacement_job': new_id, 'classification': 'BUG' if actual_failure else 'QUARANTINED',
            'actual_failed_execution': actual_failure, 'original_state_remains_QUARANTINED': True,
            'quarantined_state_sha256': digest(old_path), 'reason': 'Import-time Numba cache initialization was blocked before inference' if actual_failure else 'Unstarted audit superseded by reviewed read-only import adapter',
            'resolution': 'Existing read-only bootstrap before project kernel imports; original model/core/math and complete API assertions preserved',
            'new_fits_from_recovery': 0, 'evidence_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in evidence_paths},
            'source_sha256': digest(Path(__file__))}
        atomic_json(BASE / 'failure_resolutions' / (old_id + '.json'), resolution, immutable=True)
        rows.append({'record_type': 'API_import_infrastructure', 'id': old_id,
            'classification': resolution['classification'], 'actual_failed_execution': actual_failure,
            'replacement_job': new_id, 'replacement_status': 'COMPLETE',
            'quarantined_state_sha256': digest(old_path), 'verified_API_result_sha256': digest(api_path)})
    assert len(rows) == 7 and sum(r['actual_failed_execution'] for r in rows) == 2
    with (parent / 'FAILURE_CLASSIFICATION.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(parent / 'RESOLVED.json', {'created_utc': utc_now(), 'status': 'PASS',
        'actual_failed_API_imports': 2, 'superseded_unstarted_API_jobs': 5,
        'successful_complete_replacement_API_jobs': 7, 'new_fits': 0,
        'original_failures_and_start_receipts_preserved': True,
        'table_sha256': digest(parent / 'FAILURE_CLASSIFICATION.csv'),
        'policy_sha256': digest(policy_path), 'source_sha256': digest(Path(__file__))}, immutable=True)
    progress('Read-only API recovery fully resolved', 'Two blocked imports and five unstarted audits preserved; seven fresh full API verifications passed')


if __name__ == '__main__':
    main()
