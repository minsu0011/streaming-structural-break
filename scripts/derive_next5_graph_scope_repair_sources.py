"""Produce exact, reviewable audit/publisher variants without editing frozen sources."""
from pathlib import Path
import ast
import difflib
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, digest, utc_now


def main():
    out = BASE / 'maintenance/FINAL_GRAPH_POLICY_SCOPE_REPAIR'
    records = []
    def derive(original, target, changes):
        before = (ROOT / original).read_text(encoding='utf-8')
        after = before
        for old, new in changes:
            assert after.count(old) == 1, old
            after = after.replace(old, new)
        ast.parse(after)
        with (ROOT / target).open('x', encoding='utf-8', newline='') as f:
            f.write(after)
        diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                          fromfile=original, tofile=target))
        diff_path = out / (Path(target).stem + '.diff')
        diff_path.write_text(diff, encoding='utf-8')
        records.append({'original': original, 'original_sha256': digest(ROOT / original),
                        'derived': target, 'derived_sha256': digest(ROOT / target),
                        'diff_sha256': digest(diff_path)})
    derive('scripts/audit_next5_execution_graph_after_test_repair.py',
           'scripts/audit_next5_execution_graph_after_scope_repair.py', [
        ("    args = parser.parse_args()", "    parser.add_argument('--reviewed-recovery-probe', action='store_true')\n    args = parser.parse_args()"),
        ("    verify_execution_changes(jobs, paths, allowed_late_dependencies)",
         "    verify_execution_changes(jobs, paths, allowed_late_dependencies)\n    from src.next5.final_graph_repair_evidence import verify_execution_changes as verify_scope_repair\n    from src.next5.final_graph_repair_evidence import production_policy_paths, verify_fixture_inventory, verify_reviewed_graph_failure\n    verify_scope_repair(jobs, paths, allowed_late_dependencies)"),
        ("            else:\n                assert not args.final_audit and api_repair and identifier in api_repair['replacement_jobs']",
         "            elif args.reviewed_recovery_probe and identifier == 'FA008_EXECUTION_GRAPH':\n                verify_reviewed_graph_failure()\n                pending_failure_resolutions.append({'original_job': identifier, 'replacement_job': 'FA008R_GRAPH_CHECK',\n                    'resolution_claimed': False, 'reviewed_policy_scope_repair': True})\n            else:\n                assert not args.final_audit and api_repair and identifier in api_repair['replacement_jobs']"),
        ("    policy_paths += [p for p in BASE.rglob('*LOCK*.json') if 'jobs' not in p.parts and 'report_tables' not in p.parts]",
         "    policy_paths += production_policy_paths()"),
        ("        'running_scheduler_source_unchanged': True, 'snapshot_inputs_sha256': inputs,",
         "        'reviewed_fixture_policy_scope': verify_fixture_inventory(),\n        'running_scheduler_source_unchanged': True, 'snapshot_inputs_sha256': inputs,")])
    derive('scripts/publish_next5_final_reports_after_test_repair.py',
           'scripts/publish_next5_final_reports_after_graph_repair.py', [
        ('from src.next5.korean_reports_after_test_repair import documents',
         'from src.next5.korean_reports_after_graph_repair import documents'),
        ('    from src.next5.final_test_repair_evidence import verify_sources',
         '    from src.next5.final_graph_repair_evidence import verify_sources'),
        ("    plan = read(BASE / 'final_audit/PLAN_AFTER_TEST_REPAIR.json')",
         "    read(BASE / 'final_audit/PLAN_AFTER_TEST_REPAIR.json')\n    read(BASE / 'maintenance/FINAL_GRAPH_POLICY_SCOPE_REPAIR/POLICY_LOCK.json')\n    read(BASE / 'maintenance/FINAL_GRAPH_POLICY_SCOPE_REPAIR/REPAIR.json')\n    read(BASE / 'maintenance/FINAL_GRAPH_POLICY_SCOPE_REPAIR/RESOLVED.json', 'PASS')\n    plan = read(BASE / 'final_audit/PLAN_AFTER_GRAPH_REPAIR.json')"),
        ("    graph = read(audit_root / 'execution_graph/RESULTS.json', 'PASS')",
         "    graph = read(BASE / 'final_audit/final_graph_repair_v1/execution_graph/RESULTS.json', 'PASS')"),
        ('    from src.next5.final_test_repair_evidence import completed_failure_rows',
         '    from src.next5.final_graph_repair_evidence import completed_failure_rows')])
    atomic_json(out / 'DERIVATION.json', {'created_utc': utc_now(),
        'status': 'EXACT_VERSIONED_DERIVATION', 'records': records,
        'all_previous_sources_unchanged': True, 'new_fits': 0,
        'source_sha256': digest(Path(__file__))}, immutable=True)
    print('NEXT5 GRAPH SCOPE VARIANTS DERIVED', len(records), flush=True)


if __name__ == '__main__':
    main()
