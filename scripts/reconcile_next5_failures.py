"""Preserve infrastructure failures and distinguish completed negative science."""
from pathlib import Path
import csv
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress


def main():
    recovery_path = BASE / 'preflight/COLD_FAILURE_RECOVERY.json'
    recovery = read_json(recovery_path)
    preflight_path = BASE / 'preflight/NEXT5_PREFLIGHT.json'
    assert read_json(preflight_path)['status'] == 'PASS'
    repair_state = BASE / 'jobs/PF016_FINALIZE_RECOVERY/STATE.json'
    assert read_json(repair_state)['status'] == 'COMPLETE'
    assert recovery['failure_before_model_inference'] and recovery['new_fits'] == 0
    ids = ['PF003_BL80_COLD', 'PF004_B2_COLD', 'PF005_D2_COLD', 'PF006_FINALIZE']
    rows = []
    for identifier in ids:
        state_path = BASE / 'jobs' / identifier / 'STATE.json'
        state = read_json(state_path)
        assert state['status'] == 'QUARANTINED'
        failed_execution = identifier == 'PF003_BL80_COLD'
        if failed_execution:
            assert digest(state_path) == recovery['failed_state_sha256']
            assert state['exit_code'] == 1
            assert digest(ROOT / state['attempt_directory'] / 'RUN.log') == state['log_sha256']
        else:
            assert 'started_utc' not in state and state['resolved_by'] == 'PF016_FINALIZE_RECOVERY'
            assert digest(BASE / 'queue' / (identifier + '.json')) == recovery['superseded_queue_sha256'][identifier]
        resolution = {
            'created_utc': utc_now(), 'status': 'RESOLVED_WITH_PRESERVED_FAILURE', 'job': identifier,
            'classification': 'BUG' if failed_execution else 'QUARANTINED',
            'actual_failed_execution': failed_execution, 'original_state_remains_QUARANTINED': True,
            'quarantined_state_sha256': digest(state_path),
            'reason': recovery['reason'] if failed_execution else 'Unstarted dependent job superseded by documented -B cold verification chain',
            'resolution': recovery['recovery'], 'new_fits_from_recovery': 0,
            'evidence_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in (recovery_path, preflight_path, repair_state)},
            'source_sha256': digest(Path(__file__)),
        }
        atomic_json(BASE / 'failure_resolutions' / (identifier + '.json'), resolution, immutable=True)
        rows.append({'record_type': 'infrastructure', 'id': identifier,
            'classification': resolution['classification'], 'execution_status': 'QUARANTINED',
            'resolution': resolution['reason'], 'source_file': state_path.relative_to(ROOT).as_posix(),
            'source_sha256': digest(state_path)})
    for path in sorted((BASE / 'experiments').glob('*/screen/RESULTS.json')):
        result = read_json(path)
        if result['screen_pass'] is False:
            rows.append({'record_type': 'scientific_gate', 'id': result['candidate'],
                'classification': 'SCREEN_FAIL', 'execution_status': result['status'],
                'resolution': 'Screen score/standalone requirements failed; no full fit admitted',
                'source_file': path.relative_to(ROOT).as_posix(), 'source_sha256': digest(path)})
    for path in sorted((BASE / 'bootstrap').glob('*/RESULTS.json')):
        result = read_json(path)
        if result.get('performance_primary_and_bootstrap_admission') is False:
            rows.append({'record_type': 'scientific_gate', 'id': result['candidate'],
                'classification': 'FULL_FAIL', 'execution_status': result['status'],
                'resolution': 'Completed full evaluation fails frozen PRIMARY/bootstrap promotion requirements; not a computation crash',
                'source_file': path.relative_to(ROOT).as_posix(), 'source_sha256': digest(path)})
    assert len(rows) == 24
    out = BASE / 'failure_classification/research_v1'
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'FAILURE_CLASSIFICATION.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(out / 'RESULTS.json', {'created_utc': utc_now(), 'status': 'PASS',
        'actual_failed_preflight_executions': 1, 'superseded_unstarted_jobs': 3,
        'scientific_screen_failures': 1, 'scientific_full_promotion_failures': 19,
        'shadow_failures': 0, 'new_fits': 0,
        'table_sha256': digest(out / 'FAILURE_CLASSIFICATION.csv'),
        'scope': 'Current completed scientific menu and original four quarantined preflight records. Development-only corrections remain in maintenance receipts.',
        'source_sha256': digest(Path(__file__))}, immutable=True)
    progress('Failure classification reconciled', 'One execution bug, three unstarted superseded jobs, one screen failure and 19 full promotion failures')


if __name__ == '__main__':
    main()
