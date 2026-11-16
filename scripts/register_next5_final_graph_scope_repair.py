"""Freeze the repair, then release a prospectively prepared dependency graph."""
from pathlib import Path
import ast
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, read_json, digest, atomic_json, utc_now, progress
from src.next5.final_test_repair_evidence import verify_sources as previous_sources
from src.next5.final_graph_repair_evidence import REPAIR, verify_reviewed_graph_failure


def main():
    previous_sources()
    verify_reviewed_graph_failure()
    session = read_json(BASE / 'SESSION.json')
    assert session['status'] == 'ACTIVE_FINAL_AUDIT'
    previous_path = BASE / 'final_audit/PLAN_AFTER_TEST_REPAIR.json'
    previous = read_json(previous_path)
    checks = read_json(REPAIR / 'CONTRACT_CHECKS.json')
    assert checks['status'] == 'PASS' and all(checks['checks'].values())
    for name, sha in checks['source_sha256'].items():
        assert digest(ROOT / name) == sha
    names = ['src/next5/final_graph_repair_evidence.py', 'src/next5/korean_reports_after_graph_repair.py',
        'scripts/verify_next5_graph_policy_scope_repair.py', 'scripts/resolve_next5_final_graph_scope_repair.py',
        'scripts/derive_next5_graph_scope_repair_sources.py', 'scripts/register_next5_final_graph_scope_repair.py',
        'scripts/audit_next5_execution_graph_after_scope_repair.py',
        'scripts/publish_next5_final_reports_after_graph_repair.py']
    for name in names:
        ast.parse((ROOT / name).read_text(encoding='utf-8'))
    source_hashes = {name: digest(ROOT / name) for name in names}
    atomic_json(REPAIR / 'POLICY_LOCK.json', {'created_utc': utc_now(),
        'status': 'FROZEN_REVIEWED_GRAPH_POLICY_SCOPE_RECOVERY',
        'previous_plan_sha256': digest(previous_path), 'source_sha256': source_hashes,
        'review_sha256': digest(REPAIR / 'REVIEW_BEFORE_CHANGES.json'),
        'contract_checks_sha256': digest(REPAIR / 'CONTRACT_CHECKS.json'),
        'derivation_sha256': digest(REPAIR / 'DERIVATION.json'),
        'audit_started_utc_unchanged': session['audit_started_utc'],
        'new_fits': 0, 'new_predictions': 0, 'new_draws': 0}, immutable=True)
    # Freeze prospective definitions outside the live queue. Publish the resolver
    # first and the runnable probe last, after the repair journal is complete.
    probe = {'id': 'FA008R_GRAPH_CHECK', 'kind': 'final_audit', 'priority': 207,
        'argv': ['scripts/audit_next5_execution_graph_after_scope_repair.py', '--tag',
                 'graph_scope_probe_v1', '--final-audit', '--reviewed-recovery-probe'],
        'description': 'Unchanged graph/source audit with exact synthetic fixture policy scope; preserved FA008 resolution explicitly pending',
        'retry_safe': False, 'depends_on_success': ['FA007_SCALAR_TABLES'],
        'registered_utc': utc_now(), 'status': 'PENDING'}
    resolver = {'id': 'FA008R_GRAPH_RESOLUTION', 'kind': 'final_audit', 'priority': 208,
        'argv': ['scripts/resolve_next5_final_graph_scope_repair.py'],
        'description': 'Resolve FA008 from completed graph replay, then verify all final graph checks and preserved resolutions',
        'retry_safe': False, 'depends_on_success': [probe['id']],
        'registered_utc': utc_now(), 'status': 'PENDING'}
    for job in (probe, resolver):
        assert not (BASE / 'queue' / (job['id'] + '.json')).exists()
        assert not (BASE / 'jobs' / job['id'] / 'STATE.json').exists()
        assert (ROOT / job['argv'][0]).is_file()
        atomic_json(REPAIR / 'prepared_queue' / (job['id'] + '.json'), job, immutable=True)
    identifier = 'FA009_PORTABLE_REPLAY'
    path = BASE / 'queue' / (identifier + '.json')
    before_path = REPAIR / 'before' / path.relative_to(ROOT)
    assert digest(path) == digest(before_path)
    assert not (BASE / 'jobs' / identifier / 'STATE.json').exists()
    updates = {'depends_on_success': [resolver['id']], 'execution_repair_utc': utc_now(),
        'execution_repair_reason': 'Reviewed policy-inventory fixture scope repair; original failed graph and prior plans preserved'}
    updated = {**read_json(path), **updates}
    after_path = REPAIR / 'PREPARED_FA009_QUEUE.json'
    atomic_json(after_path, updated, immutable=True)
    record = {'job': identifier, 'before_path': before_path.relative_to(ROOT).as_posix(),
              'before_sha256': digest(before_path), 'after_sha256': digest(after_path),
              'exact_updates': updates}
    atomic_json(REPAIR / 'REPAIR.json', {'created_utc': utc_now(),
        'status': 'PROSPECTIVE_RECOVERY_GRAPH_PREPARED', 'changed_queue': record,
        'replacement_jobs': [probe['id'], resolver['id']],
        'policy_sha256': digest(REPAIR / 'POLICY_LOCK.json'),
        'original_failed_state_preserved': 'FA008_EXECUTION_GRAPH',
        'new_fits': 0, 'new_predictions': 0, 'new_draws': 0,
        'successful_recovery_not_yet_claimed': True}, immutable=True)
    revised = {**previous, 'created_utc': utc_now(),
        'status': 'REGISTERED_SEPARATE_AUDIT_AFTER_REVIEWED_GRAPH_RECOVERY',
        'previous_plan_sha256': digest(previous_path),
        'graph_recovery_record_sha256': digest(REPAIR / 'REPAIR.json'),
        'graph_recovery_policy_sha256': digest(REPAIR / 'POLICY_LOCK.json'),
        'source_sha256': {**previous['source_sha256'], **source_hashes},
        'source_script_sha256': digest(Path(__file__))}
    revised['jobs'] = []
    for job in previous['jobs']:
        revised['jobs'].extend([probe['id'], resolver['id']] if job == 'FA008_EXECUTION_GRAPH' else [job])
    atomic_json(BASE / 'final_audit/PLAN_AFTER_GRAPH_REPAIR.json', revised, immutable=True)
    atomic_json(BASE / 'queue' / (resolver['id'] + '.json'), resolver, immutable=True)
    atomic_json(path, updated)
    atomic_json(BASE / 'queue' / (probe['id'] + '.json'), probe, immutable=True)
    assert digest(path) == record['after_sha256']
    progress('Reviewed graph policy repair registered', 'Original failure preserved; exact fixture scope contracts passed; only unstarted portable dependency reconnected', phase='FINAL_AUDIT')
    print('NEXT5 GRAPH POLICY SCOPE RECOVERY REGISTERED', len(revised['jobs']), 'required audit jobs', flush=True)


if __name__ == '__main__':
    main()
