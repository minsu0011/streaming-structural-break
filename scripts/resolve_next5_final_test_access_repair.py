"""Resolve the failed suite only after a complete, guarded replacement run."""
from pathlib import Path
from collections import Counter
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, read_json, digest, atomic_json, utc_now, progress
from src.next5.final_test_repair_evidence import REPAIR, verify_sources


def cases(path):
    return Counter((row.get('classname'), row.get('name'))
        for row in ET.parse(path).findall('.//testcase'))


def main():
    verify_sources()
    old = BASE / 'final_audit/final_v1/full_tests'
    fresh = BASE / 'final_audit/final_tests_repair_v1/full_tests'
    failed = read_json(old / 'RESULTS.json')
    result = read_json(fresh / 'RESULTS.json')
    guard = read_json(fresh / 'ACCESS_GUARD.json')
    compatibility = read_json(fresh / 'TEST_COMPATIBILITY.json')
    assert failed['status'] == 'FAIL' and failed['tests'] == 1357
    assert failed['failures'] == 8 and failed['errors'] == 37 and failed['skipped'] == 0
    assert result['status'] == 'PASS' and result['exit_code'] == 0
    assert result['failures'] == result['errors'] == result['skipped'] == 0
    assert guard['status'] == 'PASS' and not guard['blocked_reads']
    before, after = cases(old / 'FULL_TESTS.xml'), cases(fresh / 'FULL_TESTS.xml')
    assert before <= after and sum(after.values()) == result['tests']
    assert result['tests'] == 1363 and len(compatibility['legacy_model_loads']) == 4
    assert compatibility['raw_and_old_array_guard_retained']
    assert compatibility['metadata_columns'] == ['dataset_id', 'fold', 'group_id', 'role']
    replacement_path = BASE / 'jobs/FA002R_COMPLETE_TESTS/STATE.json'
    replacement = read_json(replacement_path)
    assert replacement['status'] == 'COMPLETE' and replacement['exit_code'] == 0
    old_state = BASE / 'jobs/FA002_COMPLETE_TESTS/STATE.json'
    assert read_json(old_state)['status'] == 'QUARANTINED'
    paths = [old / name for name in ('RESULTS.json', 'FULL_TESTS.xml', 'ACCESS_GUARD.json')]
    paths += [fresh / name for name in ('RESULTS.json', 'FULL_TESTS.xml', 'ACCESS_GUARD.json', 'TEST_COMPATIBILITY.json')]
    paths += [REPAIR / 'POLICY_LOCK.json', REPAIR / 'REPAIR.json', replacement_path,
        BASE / 'final_audit/PLAN.json', BASE / 'final_audit/PLAN_AFTER_TEST_REPAIR.json']
    evidence = {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}
    resolution = {'created_utc': utc_now(), 'status': 'RESOLVED_WITH_PRESERVED_FAILURE',
        'job': 'FA002_COMPLETE_TESTS', 'actual_failed_execution': True,
        'quarantined_state_sha256': digest(old_state), 'replacement_job': replacement['id'],
        'evidence_sha256': evidence, 'new_fits': 0, 'new_predictions': 0,
        'resolution': 'All 1357 original tests and six new boundary contracts passed with the exact ID metadata exception and identical legacy model weights; no skips or old per-series array decoding.'}
    atomic_json(BASE / 'failure_resolutions/FA002_COMPLETE_TESTS.json', resolution, immutable=True)
    atomic_json(REPAIR / 'RESOLVED.json', {'created_utc': utc_now(), 'status': 'PASS',
        'original_test_cases': sum(before.values()), 'replacement_test_cases': sum(after.values()),
        'all_original_test_cases_preserved': True, 'original_test_sources_unchanged': True,
        'successful_suite_result_sha256': digest(fresh / 'RESULTS.json'),
        'quarantined_state_sha256': digest(old_state), 'evidence_sha256': evidence,
        'failure_resolution_sha256': digest(BASE / 'failure_resolutions/FA002_COMPLETE_TESTS.json'),
        'new_fits': 0, 'new_predictions': 0, 'new_draws': 0,
        'final_audit_start_preserved': read_json(BASE / 'SESSION.json')['audit_started_utc'],
        'source_sha256': digest(Path(__file__))}, immutable=True)
    progress('Complete final-suite recovery verified', '1363 tests including all original 1357, zero failures/errors/skips/forbidden reads; original failed attempt retained', phase='FINAL_AUDIT')
    print('NEXT5 FINAL TEST RECOVERY PASS', result['tests'], flush=True)


if __name__ == '__main__':
    main()
