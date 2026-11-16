"""Freeze the reviewed recovery and reconnect only two unstarted audit jobs."""
from pathlib import Path
import ast
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, read_json, digest, atomic_json, utc_now, progress
from src.next5.scheduler import register_job


def main():
    out = BASE / 'maintenance/FINAL_TEST_ACCESS_REPAIR'
    original_path = BASE / 'final_audit/PLAN.json'
    original = read_json(original_path)
    session = read_json(BASE / 'SESSION.json')
    assert session['status'] == 'ACTIVE_FINAL_AUDIT'
    assert read_json(BASE / 'jobs/FA002_COMPLETE_TESTS/STATE.json')['status'] == 'QUARANTINED'
    for name, sha in original['source_sha256'].items():
        assert digest(ROOT / name) == sha
    probe = BASE / 'final_audit/test_access_probe_v1'
    xml = ET.parse(probe / 'TESTS.xml')
    assert len(xml.findall('.//testcase')) == 53
    assert not any(xml.findall('.//' + tag) for tag in ('failure', 'error', 'skipped'))
    assert read_json(probe / 'ACCESS_GUARD.json')['status'] == 'PASS'
    assert not read_json(probe / 'ACCESS_GUARD.json')['blocked_reads']
    assert len(read_json(probe / 'TEST_COMPATIBILITY.json')['legacy_model_loads']) == 4
    for name, sha in read_json(out / 'PROBE_POLICY_LOCK.json')['source_sha256'].items():
        assert digest(ROOT / name) == sha
    names = ['src/next5/final_test_compatibility.py', 'src/next5/final_test_repair_evidence.py',
        'src/next5/korean_reports_after_test_repair.py', 'tests/test_next5_final_test_compatibility.py',
        'scripts/derive_next5_final_test_repair_sources.py', 'scripts/resolve_next5_final_test_access_repair.py',
        'scripts/register_next5_final_test_access_repair.py', 'scripts/run_next5_full_tests_after_access_repair.py',
        'scripts/audit_next5_execution_graph_after_test_repair.py', 'scripts/publish_next5_final_reports_after_test_repair.py']
    for name in names:
        ast.parse((ROOT / name).read_text(encoding='utf-8-sig'))
    source_hashes = {name: digest(ROOT / name) for name in names}
    for identifier in ('FA003_FROZEN_STATISTICS', 'FA008_EXECUTION_GRAPH'):
        assert not (BASE / 'jobs' / identifier / 'STATE.json').exists()
    atomic_json(out / 'POLICY_LOCK.json', {'created_utc': utc_now(),
        'status': 'FROZEN_REVIEWED_FINAL_TEST_RECOVERY',
        'original_plan_sha256': digest(original_path), 'source_sha256': source_hashes,
        'probe_sources_sha256': digest(out / 'PROBE_POLICY_LOCK.json'),
        'probe_xml_sha256': digest(probe / 'TESTS.xml'),
        'probe_access_guard_sha256': digest(probe / 'ACCESS_GUARD.json'),
        'derivation_sha256': digest(out / 'DERIVATION.json'),
        'review_before_changes_sha256': digest(out / 'REVIEW_BEFORE_CHANGES.json'),
        'new_fits': 0, 'new_predictions': 0, 'new_draws': 0,
        'all_original_audit_pinned_sources_preserved': True,
        'audit_started_utc_unchanged': session['audit_started_utc']}, immutable=True)
    replacement = register_job({'id': 'FA002R_COMPLETE_TESTS', 'kind': 'final_audit', 'priority': 201,
        'argv': ['scripts/run_next5_full_tests_after_access_repair.py', '--tag', 'final_tests_repair_v1'],
        'description': 'All original tests plus reviewed metadata-boundary contracts; old failed suite preserved',
        'retry_safe': False, 'depends_on_success': ['FA001_FIT_INVENTORY']})
    resolver = register_job({'id': 'FA002R_TEST_ACCESS_RESOLUTION', 'kind': 'final_audit', 'priority': 202,
        'argv': ['scripts/resolve_next5_final_test_access_repair.py'],
        'description': 'Verify complete replacement, exact original case coverage and zero forbidden reads before resolving the failed audit job',
        'retry_safe': False, 'depends_on_success': ['FA002R_COMPLETE_TESTS']})
    stamp = utc_now()
    changed = []
    changes = {
        'FA003_FROZEN_STATISTICS': {'depends_on_success': ['FA002R_TEST_ACCESS_RESOLUTION']},
        'FA008_EXECUTION_GRAPH': {'argv': ['scripts/audit_next5_execution_graph_after_test_repair.py', '--tag', 'final_v1', '--final-audit']}}
    for identifier, updates in changes.items():
        path = BASE / 'queue' / (identifier + '.json')
        before_path = out / 'before' / path.relative_to(ROOT)
        assert digest(path) == digest(before_path)
        assert not (BASE / 'jobs' / identifier / 'STATE.json').exists()
        updates.update(execution_repair_utc=stamp,
            execution_repair_reason='Reviewed final-suite access recovery; original queue and final plan preserved')
        updated = dict(read_json(path))
        updated.update(updates)
        atomic_json(path, updated)
        changed.append({'job': identifier, 'before_path': before_path.relative_to(ROOT).as_posix(),
            'before_sha256': digest(before_path), 'after_sha256': digest(path), 'exact_updates': updates})
    atomic_json(out / 'REPAIR.json', {'created_utc': stamp, 'status': 'REGISTERED_REVIEWED_RECOVERY',
        'policy_sha256': digest(out / 'POLICY_LOCK.json'),
        'original_plan_sha256': digest(original_path),
        'replacement_jobs': [replacement['id'], resolver['id']], 'changed_queues': changed,
        'new_fits': 0, 'new_predictions': 0, 'new_draws': 0,
        'failed_state_preserved': 'FA002_COMPLETE_TESTS',
        'successful_recovery_not_yet_claimed': True}, immutable=True)
    revised = dict(original)
    revised.update(created_utc=utc_now(), status='REGISTERED_SEPARATE_AUDIT_AFTER_REVIEWED_TEST_RECOVERY',
        original_plan_sha256=digest(original_path), recovery_record_sha256=digest(out / 'REPAIR.json'),
        recovery_policy_sha256=digest(out / 'POLICY_LOCK.json'),
        source_sha256={**original['source_sha256'], **source_hashes},
        source_script_sha256=digest(Path(__file__)),
        audit_started_utc_preserved=session['audit_started_utc'])
    revised['jobs'] = []
    for identifier in original['jobs']:
        revised['jobs'].extend([replacement['id'], resolver['id']] if identifier == 'FA002_COMPLETE_TESTS' else [identifier])
    atomic_json(BASE / 'final_audit/PLAN_AFTER_TEST_REPAIR.json', revised, immutable=True)
    progress('Reviewed complete-suite rerun registered', 'Original PLAN and failed FA002 remain immutable; replacement and resolution precede unchanged numerical audit', phase='FINAL_AUDIT')
    print('NEXT5 FINAL TEST RECOVERY REGISTERED', revised['jobs'], flush=True)


if __name__ == '__main__':
    main()
