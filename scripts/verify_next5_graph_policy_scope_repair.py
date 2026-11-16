"""Focused contracts for the final graph policy inventory repair; no test-suite rerun."""
from pathlib import Path
import ast
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.final_graph_repair_evidence import (
    REPAIR, FIXTURE_ROOTS, is_completed_suite_fixture, production_policy_paths,
    verify_fixture_inventory, verify_reviewed_graph_failure)


def main():
    before = verify_fixture_inventory()
    verify_reviewed_graph_failure()
    original = {p for p in BASE.rglob('*LOCK*.json')
                if 'jobs' not in p.parts and 'report_tables' not in p.parts}
    production = set(production_policy_paths())
    excluded = {p for p in original if is_completed_suite_fixture(p)}
    assert original - production == excluded and len(excluded) == 44
    assert production | excluded == original
    parsed = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(production)}
    for p in production:
        read_json(p)  # Invalid JSON outside the exact fixture roots still raises.
    for root in FIXTURE_ROOTS:
        assert is_completed_suite_fixture(root / 'new_synthetic_test/POLICY_LOCK.json')
    assert not is_completed_suite_fixture(ROOT / 'configs/next5/pytest_tmp/POLICY_LOCK.json')
    assert not is_completed_suite_fixture(BASE / 'locks/POLICY_LOCK.json')
    assert not is_completed_suite_fixture(BASE / 'final_audit/another_suite/full_tests/pytest_tmp/POLICY_LOCK.json')
    invalid = read_json(REPAIR / 'REVIEW_BEFORE_CHANGES.json')['invalid_fixture_locks']
    for record in invalid:
        try:
            read_json(ROOT / record['path'])
        except json.JSONDecodeError:
            pass
        else:
            raise AssertionError('Intentionally invalid fixture unexpectedly changed')
    source_names = ['src/next5/final_graph_repair_evidence.py',
                    'scripts/verify_next5_graph_policy_scope_repair.py',
                    'scripts/audit_next5_execution_graph_after_scope_repair.py']
    for name in source_names:
        ast.parse((ROOT / name).read_text(encoding='utf-8'))
    assert before == verify_fixture_inventory()
    atomic_json(REPAIR / 'CONTRACT_CHECKS.json', {'created_utc': utc_now(), 'status': 'PASS',
        'checks': {'exact_44_fixture_policies_excluded': True,
                   'all_other_original_policy_paths_preserved_and_parsed': True,
                   'similarly_named_external_paths_not_excluded': True,
                   'both_invalid_original_fixtures_remain_invalid_and_unchanged': True,
                   'failed_job_state_and_log_preserved': True},
        'production_policy_files': parsed, 'fixture_scope': before,
        'source_sha256': {n: digest(ROOT / n) for n in source_names},
        'new_fits': 0, 'new_predictions': 0, 'new_draws': 0,
        'complete_test_suite_repeated': False}, immutable=True)
    progress('Final graph policy-scope contracts passed', 'Exact fixture exclusions, unchanged production policies and preserved failure verified', phase='FINAL_AUDIT')
    print('NEXT5 GRAPH POLICY SCOPE CONTRACTS PASS', len(production), 'production policy files', flush=True)


if __name__ == '__main__':
    main()
