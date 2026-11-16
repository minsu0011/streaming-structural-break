"""Explicit evidence for the versioned final-suite infrastructure recovery."""
from collections import Counter
from src.next5.artifacts import BASE, ROOT, read_json, digest

REPAIR = BASE / 'maintenance/FINAL_TEST_ACCESS_REPAIR'


def verify_sources():
    policy = read_json(REPAIR / 'POLICY_LOCK.json')
    assert digest(BASE / 'final_audit/PLAN.json') == policy['original_plan_sha256']
    for name, sha in read_json(BASE / 'final_audit/PLAN.json')['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    for name, sha in policy['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    return policy


def verify_execution_changes(jobs, paths, allowed_late_dependencies):
    verify_sources()
    repair = read_json(REPAIR / 'REPAIR.json')
    assert read_json(REPAIR / 'RESOLVED.json')['status'] == 'PASS'
    for record in repair['changed_queues']:
        before = read_json(ROOT / record['before_path'])
        assert digest(ROOT / record['before_path']) == record['before_sha256']
        assert digest(paths[record['job']]) == record['after_sha256']
        actual = jobs[record['job']]
        expected = dict(before)
        expected.update(record['exact_updates'])
        assert actual == expected
    allowed_late_dependencies['FA003_FROZEN_STATISTICS'] = {'FA002R_TEST_ACCESS_RESOLUTION'}


def completed_failure_rows(read):
    from src.next5.failure_reports import completed_failure_rows as original_rows
    rows, previous = original_rows(read)
    resolution = read(REPAIR / 'RESOLVED.json', 'PASS')
    state_path = BASE / 'jobs/FA002_COMPLETE_TESTS/STATE.json'
    state = read(state_path, 'QUARANTINED')
    proof = read(BASE / 'failure_resolutions/FA002_COMPLETE_TESTS.json', 'RESOLVED_WITH_PRESERVED_FAILURE')
    assert state['exit_code'] == 1
    assert digest(state_path) == proof['quarantined_state_sha256']
    for name, sha in proof['evidence_sha256'].items():
        assert digest(ROOT / name) == sha
    rows.append({'record_type': 'final_suite_access_infrastructure',
        'id': 'FA002_COMPLETE_TESTS', 'classification': 'BUG',
        'execution_status': 'QUARANTINED',
        'resolution': 'Exact hash/schema-verified ID metadata admission and test-only identical legacy fold weights; all original tests rerun, no observation/label/old OOF access',
        'source_file': state_path.relative_to(ROOT).as_posix(), 'source_sha256': digest(state_path),
        'actual_failed_execution': True, 'successful_recovery_verified': True,
        'replacement_job': 'FA002R_COMPLETE_TESTS',
        'replacement_result_sha256': resolution['successful_suite_result_sha256']})
    counts = dict(Counter(row['classification'] for row in rows))
    assert len(rows) == len({row['id'] for row in rows}) == 33
    assert counts == {'BUG': 5, 'QUARANTINED': 8, 'SCREEN_FAIL': 1, 'FULL_FAIL': 19}
    return rows, {'status': 'PASS_COMPLETE_FAILURE_CLASSIFICATION', 'records': 33,
        'classification_counts': counts, 'actual_execution_failures': 5,
        'supervised_Python_execution_failures': 4, 'native_metadata_publication_failures': 1,
        'unstarted_superseded_jobs': 8, 'scientific_screen_failures': 1,
        'scientific_promotion_failures': 19, 'all_12_preserved_quarantines_resolved': True,
        'historical_32_row_classification': previous,
        'final_suite_original_tests': resolution['original_test_cases'],
        'final_suite_replacement_tests': resolution['replacement_test_cases'],
        'new_fits': 0, 'source_sha256': digest(ROOT / 'src/next5/final_test_repair_evidence.py'),
        'scope': 'Original 32 queued scientific/validation and native-publication records plus one failed final-suite job. Historical monitor heartbeat and development harness exceptions are separately preserved maintenance evidence.'}
