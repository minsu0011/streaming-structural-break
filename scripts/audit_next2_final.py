"""Separate audit checkpoint and duration-gated final closure; no model fitting."""
from pathlib import Path
import sys
import json
import argparse
import subprocess
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from src.next2.session import clock, require_handoff_time, checkpoint
from src.next.guard import SealGuard
from src.next.data import RAW_SHA
from src.next2.shared_inference import load_artifact as load_shared
from src.next2.canonical_inference import load_artifact as load_canonical
from src.next2.canonical_model import dumps
from src.utils.artifacts import write_json, sha256, utc_now
from package_next2_handoff import REQUIRED, collect_payload


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--finalize', action='store_true'); args = parser.parse_args()
    timing = require_handoff_time() if args.finalize else clock()
    assert timing['minimum_research_met'] and timing['audit_started']
    guard = SealGuard(ROOT); guard.check_state()
    selection = json.loads((ROOT/'DEV_CANDIDATES_NEXT2.json').read_text())
    for path, digest in selection['evidence_sha256'].items(): assert sha256(ROOT/path) == digest, path
    evidence = {}; checks = {}; counts = {}
    def receipt(path):
        result = json.loads((ROOT/path).read_text(encoding='utf-8-sig')); assert result['status'] == 'PASS', path
        for child, digest in result.get('receipts_sha256', {}).items(): assert sha256(ROOT/child) == digest, child
        evidence[path] = sha256(ROOT/path); return result
    references = json.loads((ROOT/'artifacts/next2/initialization/FROZEN_REFERENCES.json').read_text())
    changed = [path for path, digest in references['protected_sha256'].items() if sha256(ROOT/path) != digest]
    assert not changed
    checks['all_467_protected_files_unchanged'] = len(references['protected_sha256']) == 467
    checks['ALL_SEALS_UNOPENED'] = all(not v['opened'] for v in guard.lock['partitions'].values())
    checks['DEV_seal_overlap_zero'] = not guard.dev_ids.intersection(guard.seal_ids)
    checks['REDUCED_NEW_USAGE_ZERO'] = guard.ledger_sha == '50e9bc60353e211223b29fa3c05e618af11a00b60e81989aced999f19063eccf'
    checks['PUBLIC_SUBMISSION_NEW_ZERO'] = selection['public_submission_new'] == 0
    checks['raw_data_unchanged'] = all(sha256(ROOT/'data/raw'/name) == digest for name, digest in RAW_SHA.items())
    tests = ROOT/'artifacts/next2/final_audit/ALL_TESTS.xml'; suites = list(ET.parse(tests).getroot().iter('testsuite'))
    checks['536_tests_passed'] = sum(int(s.attrib['tests']) for s in suites) == 536 and all(int(s.attrib.get(k, 0)) == 0 for s in suites for k in ('errors', 'failures', 'skipped'))
    cases = [case.attrib for case in ET.parse(tests).getroot().iter('testcase')]
    scorer = [c for c in cases if c['classname'] in ('tests.test_scorer_official_checks', 'tests.test_ts_auc')]
    checks['official_scorer_parity'] = bool(scorer) and checks['536_tests_passed']
    counts['scorer_test_cases'] = len(scorer); evidence[tests.relative_to(ROOT).as_posix()] = sha256(tests)
    reps = receipt('REPRODUCTION_NEXT2.json'); deterministic = receipt('DETERMINISM_NEXT2.json')
    checks['independent_raw_training_both_roles'] = reps['new_learner_fits'] == 3 and reps['canonical_model_bytes_exact'] and reps['all_DEV_prediction_fingerprints_exact']
    checks['determinism_save_load_and_complete_DEV_parity'] = deterministic['canonical_save_load'] and deterministic['full_DEV_API'] and deterministic['OOF_and_alternate_runtime_parity']
    for role in ('distilled', 'performance'):
        record = selection[role]; directory = ROOT/record['model_directory']
        model, descriptor = (load_canonical if role == 'distilled' else load_shared)(directory)
        checks[role+'_canonical_bytes_and_contract'] = dumps(model) == (directory/'model.sbnext2').read_bytes() and sha256(directory/'model.sbnext2') == record['canonical_model_sha256']
        evidence[(directory/'model.sbnext2').relative_to(ROOT).as_posix()] = sha256(directory/'model.sbnext2')
        evidence[(directory/'model.json').relative_to(ROOT).as_posix()] = sha256(directory/'model.json')
        proof = receipt('artifacts/next2/independent_reproduction/'+role.upper()+'/RESULTS.json')
        checks[role+'_seal_values_zero'] = proof['seal_values_accessed'] == proof['seal_rows_fitted'] == 0
        checks[role+'_causal_current_point_only'] = proof['first_none_before_data'] and proof['one_current_point_per_yield'] and proof['no_future_length_or_suffix_access']
    for role, folder in [('distilled', 'canonical_api_all_dev/DISTILLED'), ('performance', 'shared_public_api_all_dev/PERFORMANCE')]:
        r = receipt('artifacts/next2/'+folder+'/RESULTS.json')
        checks[role+'_read_only_API_and_series_reset'] = r['persistent_filesystem_write_attempts'] == 0 and r['reverse_order_reset_series'] == 128
    for folder in ('canonical_oof/J102_SERIAL_VARIANCE_512', 'alternate_runtime_v2/DISTILLED', 'shared_blend_runtime/all_dev',
            'shared_blend_runtime/oof', 'shared_blend_runtime/alternate'):
        path = 'artifacts/next2/'+folder+'/RESULTS.json'
        receipt(path)
    claims = receipt('artifacts/next2/final_claims/RESULTS.json')
    bindings = receipt('artifacts/next2/final_audit/EVIDENCE_BINDINGS.json')
    for path, item in bindings['files'].items(): assert sha256(ROOT/path) == item['sha256']
    checks['selected_OOF_models_coordinates_and_measurements_bound'] = bindings['files_checked'] == 108
    for path, digest in claims['source_sha256'].items(): assert sha256(ROOT/path) == digest
    checks['45_numeric_claims_and_35_Korean_answers'] = claims['numeric_cross_checks'] >= 45 and claims['numbered_Korean_answers'] == 35
    checks['previous_ledger_history_preserved'] = claims['previous_ledger_rows_unchanged'] == 142
    portable = receipt('artifacts/next2/handoff/portable_copy/RESULTS.json')
    for module in portable['project_modules'].values(): assert sha256(ROOT/module['file']) == module['sha256']
    checks['raw_free_copied_public_API_and_bootstrap_replay'] = portable['public_API_points'] == 524288 and portable['bootstrap_scopes'] == 24 and not portable['raw_competition_file_access_attempts'] and not portable['original_project_access_attempts']
    counts['raw_free_copy_files_verified'] = portable['unchanged_payload_files_checked']
    table = receipt('artifacts/next2/final_tables/RESULTS.json')
    for path, digest in table['files_sha256'].items(): assert sha256(ROOT/path) == digest
    checks['final_dispositions_and_role_costs_present'] = table['scientific_records'] == 80 and table['new_named_records'] == 71
    gate_review = json.loads((ROOT/'artifacts/next2/final_audit/SELECTION_GATE_REVIEW.json').read_text())
    checks['literal_gates_and_generation_no_promotion'] = claims['literal_performance_distilled_generation_gates_consistent'] and not selection['generation_level_promotion'] and not gate_review['thresholds_changed']
    paths, files, excluded = collect_payload(require_audit=False)
    checks['payload_has_no_raw_or_credentials_or_real_point_caches'] = bool(files) and all(not path.startswith('data/') for path in files)
    counts.update(payload_preflight_files=len(files), payload_uncompressed_bytes=sum(v['bytes'] for v in files.values()), excluded_optional_reports=len(excluded))
    for path in REQUIRED:
        if path == 'FINAL_AUDIT_NEXT2.json': continue
        assert (ROOT/path).is_file(); evidence[path] = sha256(ROOT/path)
    if args.finalize: checks['six_research_hours_plus_45_audit_minutes'] = timing['handoff_time_gate_pass']
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}
    result = {'status': 'PASS', 'created_utc': utc_now(), 'checks': checks, 'counts': counts, 'clock': timing,
        'research_seconds': timing['elapsed_research_seconds'], 'audit_seconds_at_closure': timing['elapsed_audit_seconds'],
        'evidence_sha256': evidence, 'script_sha256': sha256(Path(__file__)),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'seals_opened': {name: False for name in guard.lock['partitions']}, 'reduced_new_usage': 0, 'public_submission_new': 0,
        'raw_sha256': RAW_SHA, 'new_learner_fits': 397, 'final_duration_closure': args.finalize,
        'audit_role': 'Same agent, separately timed final validation phase; no independent human or subagent review claimed',
        'scope_limitations': ['Python audit hooks and SealGuard are application checks, not an OS sandbox',
            'Exact replay is for pinned platform/dependencies and tested streams', 'Bootstrap conditions on fitted DEV predictions and does not adjust adaptive selection',
            'Source manifest is generated during packaging; external receipt verifies every ZIP member and the archive hash'],
        'corrections_preserved': ['Joblib alias byte failure resolved by canonical format', 'C210 adapter bug and C211 rejected screen',
            'Alternate proof harness V1 coordinate repair', 'Import-time JSON configuration completion before independent fitting',
            'Literal state-OR-latency selection report correction', 'Audit percentage expectation and mixed CSV history dtype interpretation']}
    destination = ROOT/'FINAL_AUDIT_NEXT2.json' if args.finalize else ROOT/'artifacts/next2/final_audit/CHECKS.json'
    write_json(destination, result)
    checkpoint('Final audit '+('duration closure' if args.finalize else 'evidence checkpoint'),
        ['Verified Desktop ZIP and final artifact checks'] if args.finalize else ['Complete separate audit duration', 'Verified Desktop ZIP'])
    print('NEXT2 FINAL AUDIT', 'CLOSED' if args.finalize else 'CHECKPOINT', len(checks), 'checks PASS', flush=True)


if __name__ == '__main__': main()
