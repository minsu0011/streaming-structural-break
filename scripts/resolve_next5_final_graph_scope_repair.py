"""Resolve from a completed graph replay, then verify the final graph again."""
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, read_json, digest, atomic_json, utc_now, progress
from src.next5.final_graph_repair_evidence import REPAIR, verify_sources, verify_reviewed_graph_failure


def main():
    verify_sources()
    failed = verify_reviewed_graph_failure()
    job_path = BASE / 'jobs/FA008R_GRAPH_CHECK/STATE.json'
    job = read_json(job_path)
    assert job['status'] == 'COMPLETE' and job['exit_code'] == 0
    probe_path = BASE / 'final_audit/graph_scope_probe_v1/execution_graph/RESULTS.json'
    probe = read_json(probe_path)
    assert probe['status'] == 'PASS' and probe['acyclic_graph']
    assert probe['actual_successful_dependencies_precede_starts']
    assert probe['completed_worker_logs_hash_verified']
    assert probe['pending_preserved_failure_resolutions'] == [{
        'original_job': 'FA008_EXECUTION_GRAPH', 'replacement_job': 'FA008R_GRAPH_CHECK',
        'resolution_claimed': False, 'reviewed_policy_scope_repair': True}]
    evidence_paths = [probe_path, job_path, ROOT / job['attempt_directory'] / 'RUN.log',
                      REPAIR / 'POLICY_LOCK.json', REPAIR / 'REVIEW_BEFORE_CHANGES.json',
                      REPAIR / 'CONTRACT_CHECKS.json']
    assert digest(ROOT / job['attempt_directory'] / 'RUN.log') == job['log_sha256']
    state_path = BASE / 'jobs/FA008_EXECUTION_GRAPH/STATE.json'
    atomic_json(BASE / 'failure_resolutions/FA008_EXECUTION_GRAPH.json', {
        'created_utc': utc_now(), 'status': 'RESOLVED_WITH_PRESERVED_FAILURE',
        'job': 'FA008_EXECUTION_GRAPH', 'classification': 'BUG',
        'quarantined_state_sha256': digest(state_path),
        'reason': 'Exact completed-suite fixture scope exclusion; completed replacement verified all production policy pins and job chronology',
        'replacement_jobs': ['FA008R_GRAPH_CHECK'],
        'evidence_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in evidence_paths},
        'original_failed_execution_preserved': True, 'new_fits': 0}, immutable=True)
    previous = sys.argv
    try:
        script = ROOT / 'scripts/audit_next5_execution_graph_after_scope_repair.py'
        sys.argv = [str(script), '--tag', 'final_graph_repair_v1', '--final-audit']
        runpy.run_path(str(script), run_name='__main__')
    finally:
        sys.argv = previous
    final_path = BASE / 'final_audit/final_graph_repair_v1/execution_graph/RESULTS.json'
    final = read_json(final_path)
    assert final['status'] == 'PASS' and final['all_preserved_failures_resolved']
    assert not final['pending_preserved_failure_resolutions']
    atomic_json(REPAIR / 'RESOLVED.json', {'created_utc': utc_now(), 'status': 'PASS',
        'completed_probe_job': 'FA008R_GRAPH_CHECK', 'probe_result_sha256': digest(probe_path),
        'final_graph_result_sha256': digest(final_path),
        'failure_resolution_sha256': digest(BASE / 'failure_resolutions/FA008_EXECUTION_GRAPH.json'),
        'failed_state_sha256': digest(state_path), 'failed_log_sha256': failed['log_sha256'],
        'all_production_policy_and_graph_checks_passed': True,
        'all_13_preserved_quarantines_resolved': True,
        'new_fits': 0, 'new_predictions': 0, 'new_draws': 0,
        'source_sha256': digest(Path(__file__))}, immutable=True)
    progress('Final graph scope recovery completed', 'Completed replay, preserved failure resolution and final graph checks passed', phase='FINAL_AUDIT')
    print('NEXT5 FINAL GRAPH SCOPE RECOVERY PASS', flush=True)


if __name__ == '__main__':
    main()
