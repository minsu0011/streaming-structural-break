"""Narrow policy-inventory repair; scientific and earlier audit sources stay frozen."""
from collections import Counter
from pathlib import Path
from src.next5.artifacts import BASE, ROOT, read_json, digest

REPAIR = BASE / 'maintenance/FINAL_GRAPH_POLICY_SCOPE_REPAIR'
FIXTURE_ROOTS = tuple(BASE / 'final_audit' / tag / 'full_tests/pytest_tmp'
                      for tag in ('final_v1', 'final_tests_repair_v1'))


def is_completed_suite_fixture(path):
    path = Path(path).absolute()
    return any(path.is_relative_to(root.absolute()) for root in FIXTURE_ROOTS)


def verify_fixture_inventory():
    review = read_json(REPAIR / 'REVIEW_BEFORE_CHANGES.json')
    current = []
    for root in FIXTURE_ROOTS:
        for path in sorted(root.rglob('*LOCK*.json')):
            current.append({'path': path.relative_to(ROOT).as_posix(),
                            'bytes': path.stat().st_size, 'sha256': digest(path)})
    assert current == review['excluded_fixture_locks']
    assert len(current) == 44 and len(review['invalid_fixture_locks']) == 2
    return {'exact_fixture_roots': [p.relative_to(ROOT).as_posix() for p in FIXTURE_ROOTS],
            'fixture_lock_files_excluded_from_production_policy_scan': len(current),
            'all_fixture_bytes_preserved': True,
            'review_sha256': digest(REPAIR / 'REVIEW_BEFORE_CHANGES.json'),
            'all_other_original_policy_paths_retained': True}


def production_policy_paths():
    verify_fixture_inventory()
    return [p for p in BASE.rglob('*LOCK*.json')
            if 'jobs' not in p.parts and 'report_tables' not in p.parts
            and not is_completed_suite_fixture(p)]


def verify_sources():
    from src.next5.final_test_repair_evidence import verify_sources as previous_sources
    previous_sources()
    policy = read_json(REPAIR / 'POLICY_LOCK.json')
    assert digest(BASE / 'final_audit/PLAN_AFTER_TEST_REPAIR.json') == policy['previous_plan_sha256']
    for name, sha in policy['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    return policy


def verify_reviewed_graph_failure():
    review = read_json(REPAIR / 'REVIEW_BEFORE_CHANGES.json')
    path = BASE / 'jobs/FA008_EXECUTION_GRAPH/STATE.json'
    state = read_json(path)
    assert state['status'] == 'QUARANTINED' and state['exit_code'] == 1
    assert digest(path) == review['preserved_files'][path.relative_to(ROOT).as_posix()]['sha256']
    log = ROOT / state['attempt_directory'] / 'RUN.log'
    assert digest(log) == state['log_sha256']
    assert 'json.decoder.JSONDecodeError' in log.read_text(encoding='utf-8')
    return state


def verify_execution_changes(jobs, paths, allowed_late_dependencies):
    verify_sources()
    verify_reviewed_graph_failure()
    repair = read_json(REPAIR / 'REPAIR.json')
    record = repair['changed_queue']
    before = read_json(ROOT / record['before_path'])
    assert digest(ROOT / record['before_path']) == record['before_sha256']
    assert digest(paths[record['job']]) == record['after_sha256']
    expected = dict(before)
    expected.update(record['exact_updates'])
    assert jobs[record['job']] == expected
    allowed_late_dependencies[record['job']] = set(expected['depends_on_success'])


def completed_failure_rows(read):
    from src.next5.final_test_repair_evidence import completed_failure_rows as previous_rows
    rows, previous = previous_rows(read)
    resolution = read(REPAIR / 'RESOLVED.json', 'PASS')
    state_path = BASE / 'jobs/FA008_EXECUTION_GRAPH/STATE.json'
    state = read(state_path, 'QUARANTINED')
    proof = read(BASE / 'failure_resolutions/FA008_EXECUTION_GRAPH.json', 'RESOLVED_WITH_PRESERVED_FAILURE')
    assert state['exit_code'] == 1 and digest(state_path) == proof['quarantined_state_sha256']
    for name, sha in proof['evidence_sha256'].items():
        assert digest(ROOT / name) == sha
    rows.append({'record_type': 'final_graph_policy_inventory_infrastructure',
                 'id': 'FA008_EXECUTION_GRAPH', 'classification': 'BUG',
                 'execution_status': 'QUARANTINED',
                 'resolution': 'Exclude only two exact completed-suite synthetic fixture roots from production policy discovery; preserve all fixture bytes and replay all graph/source checks',
                 'source_file': state_path.relative_to(ROOT).as_posix(),
                 'source_sha256': digest(state_path), 'actual_failed_execution': True,
                 'successful_recovery_verified': True, 'replacement_job': 'FA008R_GRAPH_CHECK',
                 'replacement_result_sha256': resolution['final_graph_result_sha256']})
    counts = dict(Counter(row['classification'] for row in rows))
    assert len(rows) == len({row['id'] for row in rows}) == 34
    assert counts == {'BUG': 6, 'QUARANTINED': 8, 'SCREEN_FAIL': 1, 'FULL_FAIL': 19}
    return rows, {'status': 'PASS_COMPLETE_FAILURE_CLASSIFICATION', 'records': 34,
                  'classification_counts': counts, 'actual_execution_failures': 6,
                  'supervised_Python_execution_failures': 5, 'native_metadata_publication_failures': 1,
                  'unstarted_superseded_jobs': 8, 'scientific_screen_failures': 1,
                  'scientific_promotion_failures': 19, 'all_13_preserved_quarantines_resolved': True,
                  'historical_33_row_classification': previous, 'new_fits': 0,
                  'source_sha256': digest(ROOT / 'src/next5/final_graph_repair_evidence.py'),
                  'scope': 'Prior 33 execution/scientific records plus the failed graph inventory job. Monitor and development harness exceptions remain separately documented.'}
