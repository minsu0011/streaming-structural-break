"""Final evidence/identity audit after research and the additional audit window.

The Desktop packager consumes this receipt and checks every pinned input again.
"""
from pathlib import Path
import json
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.guard import SealGuard
from src.next.data import prepare_dev, RAW_SHA
from src.next.research_clock import ResearchClock
from src.next.active_research_time import verify_final_duration
from src.next.references import verify_reference_sources
from src.next.decision import evaluate_candidate
from src.utils.artifacts import write_json, sha256, utc_now
from scripts.freeze_next_roles import select_roles
from scripts.package_next_handoff import REQUIRED, ordinary_files
from scripts.schedule_next_research import completed


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def main():
    clock = ResearchClock(ROOT).assert_handoff_ready()
    substantive_duration = verify_final_duration(ROOT)
    out = ROOT/'artifacts/next/final_audit'
    require(not (out/'RESULTS.json').exists(), 'Preserve the final audit receipt; no silent overwriting')
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    evidence, checks = {}, {}
    def pin(relative, digest=None):
        path = ROOT/relative
        actual = sha256(path)
        require(digest is None or actual == digest, 'Final audit SHA256 differs: '+str(relative))
        evidence[path.relative_to(ROOT).as_posix()] = actual
        return actual
    def read(relative, status=None):
        pin(relative)
        result = json.loads((ROOT/relative).read_text(encoding='utf-8'))
        require(status is None or result.get('status') == status, 'Incomplete evidence: '+str(relative))
        return result
    guard = SealGuard(ROOT)
    guard.check_state()
    session = read('artifacts/next/SESSION_NEXT.json')
    for relative, digest in session['old_reference_hashes'].items():
        pin(relative, digest)
    verify_reference_sources(ROOT)
    prior = read('artifacts/next/prior_preservation/RESULTS.json', 'PASS')
    pin('SOURCE_MANIFEST.json', prior['manifest_sha256'])
    for relative, digest in prior['preserved_files_sha256'].items():
        pin(relative, digest)
    require(prior['checked_prior_final_models'] == 3 and prior['checked_prior_cv_models'] == 15,
        'The prior published champion artifact coverage changed')
    require(sha256(Path(prior['old_archive'])) == prior['old_archive_sha256'], 'Prior Desktop archive changed')
    checks['old_champion_code_models_split_and_reduced_ledger_preserved'] = True
    lock = read('SEAL_PARTITION_500x4_LOCK.json')
    require(len(lock['partitions']) == 4 and all(not row['opened'] for row in lock['partitions'].values()), 'A seal partition was opened')
    require(len(guard.dev_ids) == 8000 and len(guard.seal_ids) == 2000 and not guard.dev_ids.intersection(guard.seal_ids), 'DEV/seal partition changed')
    for letter in 'ABCD':
        pin('SEAL_'+letter+'_IDS.txt')
    checks['all_four_seals_remain_closed_and_membership_exact'] = True
    for name, digest in RAW_SHA.items():
        pin(Path('data/raw')/name, digest)
    data = prepare_dev(ROOT)
    metadata = data.rows()
    require(len(metadata['target']) == 4014405 and set(metadata['dataset_id']) == set(guard.dev_ids), 'DEV observation/target coverage changed')
    guard.record_rows(metadata['dataset_id'], purpose='final source contamination audit')
    checks['guarded_dev_array_ids_and_rows_contamination_zero'] = True
    selection = read('DEV_CANDIDATES_NEXT.json', 'FROZEN_DEV_SELECTION')
    frozen = read('artifacts/next/final_selection/RESULTS.json', 'FROZEN_DEV_SELECTION')
    pin('artifacts/next/scheduler/SUBSTANTIVE_RESEARCH_DURATION.json', frozen['substantive_research_report_sha256'])
    pin('src/next/active_research_time.py', substantive_duration['active_clock_source_sha256'])
    pin('scripts/begin_next_final_audit.py', substantive_duration['start_driver_sha256'])
    for relative, digest in substantive_duration['evidence_files_sha256'].items():
        pin(relative, digest)
    require(substantive_duration['substantive_research_seconds'] >= 36000,
        'Interrupted wall time cannot satisfy the research minimum')
    checks['ten_substantive_research_hours_excluding_audited_reboot_gap'] = True
    pin('scripts/freeze_next_roles.py', frozen['selector_sha256'])
    pin('configs/next_final_role_policy.json', frozen['role_policy_sha256'])
    require(selection['roles'] == frozen['roles'] and 1 <= len(selection['candidates']) <= 3, 'Final roles changed or exceed three')
    require(selection['frozen_selection_report_sha256'] == sha256(ROOT/'artifacts/next/final_selection/RESULTS.json'), 'Selection bridge is invalid')
    policy = read('ROBUSTNESS_SPLIT_LOCK.json')
    snapshot_dir = Path(frozen['evidence_snapshot'])
    snapshot = read(snapshot_dir/'RESULTS.json', 'PASS')
    require(sha256(ROOT/snapshot_dir/'RESULTS.json') == frozen['evidence_snapshot_sha256'],
        'The role freeze refers to different snapshot bytes')
    require(snapshot['payload_byte_checks_requested'], 'Final source snapshot did not verify payload bytes')
    for relative, digest in snapshot['source_reports_sha256'].items():
        pin(relative, digest)
    for relative, digest in snapshot['files_sha256'].items():
        pin(snapshot_dir/relative, digest)
    snapshot_rows = {r['candidate']['experiment_id']: r for r in read(snapshot_dir/'CANDIDATE_EVIDENCE.json')}
    role_rows = []
    for eid, record in snapshot_rows.items():
        original = read(Path('artifacts/next/experiments')/eid/'RESULTS.json')
        full = original.get('full', {})
        if original['status'] == 'BUG' or full.get('status') != 'COMPLETE':
            continue
        decision = evaluate_candidate(policy, record['evidence'])
        require(decision == record['decision'], 'A snapshot promotion decision changed')
        latency = record['supplemental'].get('latency', {})
        role_rows.append({'id': eid, 'mean': full['mean'], 'median': full['median'], 'worst': full['worst_fold'],
            'std': float(np.std(full['fold_scores'])), 'decision': decision,
            'marginal_complexity_gate_pass': record['marginal_complexity_gate_pass'],
            'point_us': latency.get('point_distribution_us', {}).get('median', float('inf')),
            'amortized_us': latency.get('real_history_initialization', {}).get('amortized_us_per_point', float('inf'))})
    require(select_roles(role_rows) == frozen['roles'], 'The complete candidate evidence does not reproduce frozen roles')
    checks['frozen_role_order_reconstructed_from_all_candidates'] = True
    for relative, digest in frozen['specification_files_sha256'].items():
        pin(relative, digest)
    for row, original in zip(selection['candidates'], frozen['candidates']):
        require(all(row[key] == value for key, value in original.items()), 'A selected candidate changed after freeze')
        record = snapshot_rows[row['experiment_id']]
        decision = evaluate_candidate(policy, record['evidence'])
        require(decision == record['decision'] and decision['promoted'] and record['marginal_complexity_gate_pass'], 'Selected promotion is not reconstructible')
        training = read(Path('artifacts/next/final_training')/row['role']/'RESULTS.json', 'PASS')
        require(all(value is True for value in training['checks'].values()), 'A final raw-iterator training check failed')
        for relative, digest in training['evidence_files_sha256'].items():
            pin(relative, digest)
        directory = Path(row['delivery_model_directory'])
        pin(directory/'model.joblib', row['model_sha256'])
        pin(directory/'model.json', row['delivery_descriptor_sha256'])
        descriptor = read(directory/'model.json')
        require(descriptor['candidate'] == row['candidate'] and descriptor['seal_rows_fitted'] == 0 and descriptor['seal_series_skipped_before_value_or_label_access'] == 2000, 'Delivered training boundary differs')
        engineering = read(Path('artifacts/next/final_engineering')/row['experiment_id']/'RESULTS.json', 'PASS')
        require(len(engineering['checks']) == 9 and all(value is True for value in engineering['checks'].values()), 'Incomplete selected engineering')
        for relative, digest in engineering['evidence_files_sha256'].items():
            pin(relative, digest)
    checks['selected_decisions_roles_artifacts_and_nine_engineering_checks'] = True
    checks['selected_raw_iterator_retraining_matches_independent_model_bytes'] = True
    checks['selected_full_api_future_suffix_readonly_determinism_and_latency'] = True
    edges = read('artifacts/next/history_edge_contract/RESULTS.json', 'PASS')
    pin('configs/next_history_edge_policy.json', edges['policy_sha256'])
    pin('scripts/audit_next_history_edges.py', edges['driver_sha256'])
    require(edges['streams'] == 504 and edges['scalar_predictions_per_implementation'] == 516096,
        'The fixed history-edge coverage is incomplete')
    require(all(r['bitwise_scalar_bundle_parity'] and r['finite_bounded_predictions'] and r['fixed_state_array_bytes'] for r in edges['cases']),
        'A historical-state edge contract failed')
    for row in selection['candidates']:
        require(edges['sources'][row['experiment_id']]['model_sha256'] == row['model_sha256'],
            'Selected edge contract used different weights')
    checks['selected_historical_extremes_and_fixed_state_contract'] = True
    scores = read('artifacts/next/final_scoring/RESULTS.json', 'PASS')
    require(scores['selection_sha256'] == sha256(ROOT/'artifacts/next/final_selection/RESULTS.json'),
        'The selected scorer audit used a different frozen role selection')
    require(scores['selected_original_folds'] == len(selection['candidates'])*5 and scores['official_code_executed'] and scores['maximum_absolute_error'] <= 1e-12, 'Final official scorer parity incomplete')
    for relative, digest in scores['evidence_files_sha256'].items():
        pin(relative, digest)
    checks['selected_original_oof_official_scorer_and_coordinate_parity'] = True
    portable = read('artifacts/next/portable_delivery/RESULTS.json', 'PASS')
    require(all(value is True for value in portable['checks'].values()) and
        portable['points'] == len(selection['candidates'])*17771,
        'Fresh copied-payload inference or pinned environment preflight incomplete')
    require({row['role']: row['candidate'] for row in portable['selected_roles']} ==
        {row['role']: row['experiment_id'] for row in selection['candidates']},
        'Portable preflight selected different roles')
    for relative, digest in portable['evidence_files_sha256'].items():
        pin(relative, digest)
    copied = read('artifacts/next/portable_delivery/COPIED_PAYLOAD_MANIFEST.json', 'PASS')
    for relative, item in copied['files'].items():
        pin(relative, item['sha256'])
    for item in portable['selected_roles']:
        receipt = read(item['report'], 'PASS')
        require(sha256(ROOT/item['report']) == item['sha256'] and
            receipt['origin_project_read_attempts'] == [] and receipt['project_modules_all_from_payload'],
            'Portable inference read from the origin or changed')
        for module in receipt['project_modules'].values():
            pin(module['relative'], module['sha256'])
    checks['fresh_pinned_environment_and_copied_delivery_portability'] = True
    tests = read('artifacts/next/final_tests/RESULTS.json', 'PASS')
    require(tests['test_cases'] >= 350 and tests['failures'] == tests['errors'] == tests['skipped'] == 0 and
        tests['source_files_unchanged_during_suite'] and tests['fresh_pinned_environment'], 'Incomplete final suite')
    for relative, digest in tests['files_sha256'].items():
        pin(Path('artifacts/next/final_tests')/relative, digest)
    for relative, digest in tests['source_files_sha256'].items():
        pin(relative, digest)
    checks['complete_current_source_test_suite'] = True
    schedule = read('configs/next_research_schedule.json')
    missing = [job['id'] for job in schedule['jobs'] if not completed(job)]
    require(not missing, 'Required research backlog remains: '+', '.join(missing))
    checks['all_registered_research_jobs_complete'] = True
    secondary = read('artifacts/next/research_synthesis/RESULTS.json', 'PASS')
    pin('scripts/summarize_next_secondary_evidence.py', secondary['script_sha256'])
    for relative, digest in secondary['source_files_sha256'].items():
        pin(relative, digest)
    for name, digest in secondary['files_sha256'].items():
        pin(Path('artifacts/next/research_synthesis')/name, digest)
    require(secondary['new_fits'] == secondary['new_predictions'] == secondary['new_bootstrap_draws'] == secondary['new_selection_rules'] == 0,
        'Secondary evidence synthesis changed the research procedure')
    checks['long_horizon_direction_and_fixed_contribution_evidence_hashed'] = True
    reports = read('artifacts/next/final_reports/RESULTS.json', 'PASS')
    exported_tables = read('artifacts/next/final_report_audit/RESULTS.json', 'PASS')
    require(exported_tables['candidate_rows'] == 142 and all(value is True for value in exported_tables['checks'].values()),
        'Final exported table reconciliation incomplete')
    pin('scripts/audit_next_report_tables.py', exported_tables['driver_sha256'])
    for relative, digest in exported_tables['evidence_files_sha256'].items():
        pin(relative, digest)
    checks['exported_model_cv_bootstrap_and_role_tables_match_original_evidence'] = True
    for relative, digest in reports['output_files_sha256'].items():
        pin(relative, digest)
    for name in REQUIRED:
        require((ROOT/name).is_file() and (ROOT/name).stat().st_size > 0, 'Missing final output: '+name)
        pin(name)
    read_first = (ROOT/'00_GPT_READ_FIRST_NEXT.md').read_text(encoding='utf-8')
    require(all('\n'+str(j)+'. ' in read_first for j in range(1, 27)), 'The 26 read-first requirements are incomplete')
    require('\ufffd' not in read_first, 'Korean handoff contains replacement characters')
    checks['required_final_reports_present_hashed_and_korean_read_first_complete'] = True
    checks['reduced_new_usage_zero_by_unchanged_exhausted_ledger'] = sha256(ROOT/'REDUCED_TEST_USAGE_LEDGER.json') == session['old_reference_hashes']['REDUCED_TEST_USAGE_LEDGER.json']
    require(selection['reduced_new_uses'] == 0 and selection['public_submissions'] == 0 and all(v is False for v in selection['seal_opened'].values()), 'Selection boundary differs')
    checks['recorded_seal_reduced_and_public_actions_zero'] = True
    # Inventory binary intermediate results without placing them in the ZIP.
    # This includes retained failures; it does not label every artifact PASS.
    inventory = {}
    for path in ordinary_files(ROOT/'artifacts/next'):
        if 'handoff' in path.relative_to(ROOT).parts or path.is_relative_to(out):
            continue
        inventory[path.relative_to(ROOT).as_posix()] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    inventory_path = out/'RESULT_ARTIFACT_INVENTORY.json'
    write_json(inventory_path, {'status': 'INTEGRITY_INVENTORY', 'created_utc': utc_now(), 'files': inventory,
        'files_count': len(inventory), 'bytes': sum(r['bytes'] for r in inventory.values()),
        'scope': 'All ordinary NEXT artifact files except reproduction workspace trees, environments, this final-audit directory, and archive receipts. Includes failures and unselected binary results; not a claim that each scientific hypothesis passed.'})
    pin(inventory_path.relative_to(ROOT))
    checks['intermediate_result_artifact_sha256_inventory_complete'] = True
    guard.check_state()
    final_clock = ResearchClock(ROOT).assert_handoff_ready()
    checks['ten_research_hours_plus_at_least_45_additional_audit_minutes'] = True
    require(all(value is True for value in checks.values()), 'A final audit gate failed')
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'clock': final_clock,
        'research_duration_seconds_before_additional_audit': substantive_duration['substantive_research_seconds'],
        'excluded_reboot_and_interruption_seconds': substantive_duration['excluded_seconds'],
        'wall_elapsed_before_additional_audit_seconds': substantive_duration['wall_seconds'],
        'additional_audit_elapsed_seconds': final_clock['final_audit_elapsed_seconds'],
        'checks': checks, 'evidence_files_sha256': evidence, 'auditor_sha256': sha256(Path(__file__)),
        'elapsed_this_audit_driver_seconds': time.perf_counter()-start, 'tests_passed': tests['test_cases'],
        'selected_candidates': [r['experiment_id'] for r in selection['candidates']],
        'seal_opened': {letter: False for letter in 'ABCD'}, 'reduced_new_uses': 0, 'public_submissions': 0,
        'limitations': 'Application-level guarded data use and pinned local evidence; not OS sandbox proof, external account activity attestation, cloud execution, or independent seal generalization.'})
    print('FINAL NEXT AUDIT PASS', len(checks), flush=True)


if __name__ == '__main__':
    main()
