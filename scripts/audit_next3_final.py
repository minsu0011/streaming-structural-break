"""Evidence-complete core checkpoint and separately duration-gated final closure."""
from pathlib import Path
import argparse
import csv
import datetime
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.final_core_contract import literal_policy, protected_files, registered_hypothesis, require_fresh
from src.next3.final_source_contract import assert_source_snapshot
from src.next3.final_validation import all_true, final_clock, prepared_table_proof, successful_receipt, tests_proof
from src.next.guard import SealGuard
from scripts.package_next3_handoff import REQUIRED, snapshot_payload
from scripts.verify_next3_old_desktop_preservation import verify_existing_desktop_deliveries


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream: return list(csv.DictReader(stream))


def current_clock(*, complete=False):
    return final_clock(read_json(BASE/'SESSION.json'), datetime.datetime.now(datetime.timezone.utc), require_complete_duration=complete)


def evaluate_checkpoint():
    timing = current_clock(); out = BASE/'final_audit/core_checkpoint'
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'START.json', {'created_utc': utc_now(), 'clock': timing,
        'script_sha256': digest(Path(__file__)), 'scope': 'Final core evidence checkpoint; duration closure and actual Desktop archive remain separate.'}, immutable=True)
    guard = SealGuard(ROOT); guard.check_state(); checks = {}; counts = {}; evidence = {}
    def bind(path):
        path = Path(path)
        if not path.resolve().is_relative_to(ROOT): raise ValueError('Core evidence must be inside the original project')
        evidence[path.relative_to(ROOT).as_posix()] = digest(path); return path
    def receipt(path, *, statuses=('PASS',), fresh=False):
        value = successful_receipt(bind(path), statuses=statuses)
        if fresh: require_fresh(value, timing['audit_started_utc'])
        return value
    def file_bindings(mapping, directory=ROOT):
        for name, sha in mapping.items():
            if digest(bind(Path(directory)/name)) != sha: raise RuntimeError('Core evidence checksum differs: '+name)
    frozen = read_json(bind(BASE/'preflight/FROZEN_REFERENCES.json'))
    old_ledger = bind(BASE/'preflight/EXPERIMENT_LEDGER_BEFORE_NEXT3.csv').read_bytes()
    protected = protected_files(ROOT, frozen['protected_sha256'], old_ledger)
    checks['all_10110_protected_old_files_exact_with_ledger_append_only'] = protected['protected_files_checked'] == 10110
    counts['protected_old_files'] = protected['protected_files_checked']
    old_desktop = verify_existing_desktop_deliveries(ROOT)
    bind(ROOT/'scripts/verify_next3_old_desktop_preservation.py')
    bind(ROOT/'artifacts/next2/handoff/DESKTOP_ZIP_VERIFICATION_NEXT2.json')
    checks['earlier_Desktop_ZIP_and_exact_20_old_result_files_unchanged'] = (
        old_desktop['file_count'] == 20 and old_desktop['all_old_Desktop_delivery_bytes_unchanged'] is True)
    counts['old_Desktop_result_files_preserved'] = old_desktop['file_count']
    original_j = frozen['roles']['FROZEN_J102_SCIENTIFIC']
    checks['original_scientific_J102_joblib_unchanged'] = digest(bind(ROOT/original_j['model'])) == original_j['sha256']
    checks['all_four_seals_unopened'] = len(guard.lock['partitions']) == 4 and all(item['opened'] is False for item in guard.lock['partitions'].values())
    checks['DEV_8000_seal_2000_overlap_zero'] = len(guard.dev_ids) == 8000 and len(guard.seal_ids) == 2000 and not (guard.dev_ids & guard.seal_ids)
    checks['seal_partition_lock_unchanged'] = guard.lock_sha == frozen['seal_lock_sha256']
    checks['reduced_ledger_unchanged_no_new_usage'] = guard.ledger_sha == frozen['reduced_ledger_sha256']
    for name, sha in frozen['raw_sha256'].items():
        checks['original_raw_bytes_'+name] = digest(bind(ROOT/'data/raw'/name)) == sha
    bind(ROOT/'artifacts/validation/SPLIT_LOCK_REAL.json')
    checks['original_group_split_assignments_unchanged'] = digest(bind(ROOT/'artifacts/validation/GROUP_ASSIGNMENTS.parquet')) == frozen['split_sha256'] == guard.split.digest
    source_lock_path = BASE/'final_audit/FINAL_EXECUTION_SOURCE_LOCK.json'
    source_lock = receipt(source_lock_path, statuses=('FROZEN_FINAL_EXECUTION_SOURCE',), fresh=True)
    sources = assert_source_snapshot(ROOT, source_lock)
    checks['final_code_config_dependency_set_unchanged'] = sources['file_count'] > 800
    counts['final_bound_code_config_files'] = sources['file_count']
    policy_path = ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json'; policy = read_json(bind(policy_path))
    policy_check = literal_policy(policy); counts.update(policy_check)
    checks['all_literal_scientific_and_engineering_gates_unchanged'] = policy_check['thresholds_changed'] is False
    checks['policy_bound_to_original_frozen_selection'] = policy['reference_sha256'] == digest(BASE/'preflight/FROZEN_REFERENCES.json') and policy['selection_sha256'] == digest(bind(ROOT/'DEV_CANDIDATES_NEXT2.json'))
    model_rows = csv_rows(bind(ROOT/'MODEL_RESULTS_NEXT3.csv'))
    recipes = [row for row in model_rows if row['record_type'] == 'NEXT3_NEW_SCIENTIFIC_RECIPE']
    for row in recipes:
        path = BASE/'hypotheses'/(row['experiment_id']+'.json'); hypothesis = read_json(bind(path))
        registered_hypothesis(hypothesis)
        if hypothesis['policy_sha256'] != digest(policy_path) or row['hypothesis_sha256'] != digest(path):
            raise RuntimeError('Scientific recipe was not bound to the frozen policy')
    checks['13_registered_recipes_12_full_one_screen_failure_no_promotions'] = len(recipes) == 13 and sum(bool(row['full_mean']) for row in recipes) == 12 and all(row['promoted'] == 'False' for row in recipes)
    science_lock = receipt(ROOT/'configs/next3/SCIENTIFIC_ROLE_FREEZE.json', statuses=('PRE_AUDIT_SCIENTIFIC_ROLES_FROZEN',))
    checks['scientific_roles_frozen_before_final_audit'] = science_lock['scientific_role_count'] == 2 and not science_lock['new_scientific_search_authorized'] and not science_lock['generation_gate_passed']
    lineage_dir = BASE/'maintenance/lineage_final_audit'
    lineage = receipt(lineage_dir/'RESULTS.json', fresh=True)
    lineage_start = read_json(bind(lineage_dir/'START.json'))
    checks['fresh_final_nested_and_OOF_lineage_reconstructed'] = lineage['nested_base_fits_reconstructed'] == 56 and lineage['candidate_fold_receipts_checked'] == 86 and lineage['new_fits'] == 0 and lineage['seal_rows'] == 0
    checks['lineage_uses_current_audited_script'] = lineage_start['script_sha256'] == digest(bind(ROOT/'scripts/audit_next3_lineage.py'))
    counts.update(nested_base_fits_reconstructed=lineage['nested_base_fits_reconstructed'], candidate_fold_receipts_checked=lineage['candidate_fold_receipts_checked'])
    primitive_fits = sum(read_json(path).get('new_fits', 0)+read_json(path).get('new_gate_fits', 0)+read_json(path).get('new_residual_fits', 0)
        for path in (BASE/'experiments').glob('*/*/fold_?.json'))
    nested_fits = len(list((BASE/'nested').glob('*/outer_*/inner_*/attempt_*/COMPLETE.json')))
    refit_fits = sum(read_json(path).get('new_fits', 0)+read_json(path).get('new_gate_fits', 0)+read_json(path).get('new_base_fits', 0)
        for path in (BASE/'refits').glob('*/*/RESULTS.json') if read_json(path)['status'] == 'PASS')
    checks['129_scientific_fits_from_primitive_receipts'] = primitive_fits+nested_fits+refit_fits == 129
    counts.update(scientific_fits=primitive_fits+nested_fits+refit_fits)
    chronology_dir = BASE/'maintenance/registration_chronology/v1'
    chronology = receipt(chronology_dir/'RESULTS.json', statuses=('PASS_RECORDED_REGISTRATION_CHRONOLOGY',))
    file_bindings(chronology['evidence_sha256']); file_bindings(chronology['source_sha256'])
    file_bindings(chronology['files_sha256'], chronology_dir)
    checks['172_recorded_UTC_chains_cover_129_fits_without_inferred_times'] = (
        chronology['event_chains_checked'] == 172 and chronology['science_fits_accounted'] == 129
        and chronology['chronology_reversals'] == chronology['new_fits_in_this_audit'] == 0
        and chronology['missing_per_fold_completion_times'] == 86
        and chronology['missing_nested_policy_times'] == 2
        and chronology['observations_labels_tau_features_predictions_read'] is False)
    test_path = BASE/'final_audit/FULL_TESTS.xml'
    tests = tests_proof(bind(test_path), minimum_count=894,
        required_classnames={'tests.test_ts_auc', 'tests.test_scorer_official_checks',
            'tests.test_next3_final_validation', 'tests.test_next3_final_runtime_probe',
            'tests.test_next3_final_source_contract', 'tests.test_next3_final_core_contract',
            'tests.test_next3_claim_checks', 'tests.test_next3_portable_report_contract', 'tests.test_next3_offline_environment',
            'tests.test_next3_path_budget', 'tests.test_next3_payload_filters'},
        not_before_timestamp=datetime.datetime.fromisoformat(timing['audit_started_utc']).timestamp())
    full_tests = receipt(BASE/'final_audit/FULL_TESTS_RESULT.json', fresh=True)
    checks['fresh_complete_tests_without_failure_error_or_skip'] = full_tests['tests']['report_sha256'] == tests['report_sha256'] and full_tests['tests']['tests'] == tests['tests'] == full_tests['collected_tests'] and full_tests['tests']['seconds'] == tests['seconds']
    checks['test_collection_execution_and_source_lock_identical'] = full_tests['collected_and_executed_counts_equal'] is True and full_tests['source_set_unchanged_before_and_after'] is True and full_tests['source_lock_sha256'] == digest(source_lock_path)
    counts['passed_final_tests'] = tests['tests']; counts['final_test_seconds'] = tests['seconds']
    preparation = receipt(BASE/'independent_reproduction/PREPARATION.json', statuses=('PREPARED',), fresh=True)
    raw_reproduction = receipt(BASE/'independent_reproduction/RESULTS.json', fresh=True)
    checks['fresh_pinned_raw_training_ordinary_copies_no_prior_feature_cache'] = preparation['ordinary_file_copies'] is True and preparation['offline_install_from_verified_ordinary_wheel_copies'] is True and raw_reproduction['fresh_environment'] is True and raw_reproduction['raw_copies_only_no_prior_feature_cache'] is True
    checks['raw_reproduction_three_separate_fits_no_new_search'] = raw_reproduction['raw_reproduction_fits'] == 3 and raw_reproduction['new_candidate_search_fits'] == 0
    wheel_copy_path = BASE/'independent_reproduction/OFFLINE_WHEEL_COPIES.json'
    checks['verified_offline_wheel_copy_receipt_bound'] = digest(bind(wheel_copy_path)) == preparation['offline_wheel_copy_receipt_sha256']
    selection = receipt(ROOT/'DEV_CANDIDATES_NEXT3.json', statuses=('FROZEN_NEXT3_DEV_SELECTION',), fresh=True)
    file_bindings(selection['evidence_sha256'])
    checks['exactly_two_original_scientific_roles_no_new_promotion'] = selection['scientific_role_count'] == 2 and selection['new_performance_candidate'] is None and selection['new_robustness_candidate'] is None and not selection['generation_level_promotion']
    checks['selected_quality_delta_zero_with_strict_identity_probability'] = selection['selected_quality_delta_vs_starting_performance'] == 0.0 and selection['selected_delta_ci'] == [0., 0.] and selection['selected_strict_probability_delta_positive'] == 0.0 and selection['selected_contrast_is_analytic_identity'] is True
    checks['new_public_cloud_and_reduced_usage_zero'] = selection['reduced_new_usage'] == selection['public_submission_new'] == selection['cloud_submission_new'] == 0
    entry_points = receipt(ROOT/'DEPLOYMENT_ENTRY_POINTS_NEXT3.json', statuses=('FROZEN_NEXT3_PUBLIC_ENTRY_POINTS',), fresh=True)
    checks['final_public_entry_contract_two_roles_and_one_stream'] = set(entry_points['entries']) == {'DISTILLED', 'PERFORMANCE'} and entry_points['INFER_PARALLELISM'] == 1
    for role in ('DISTILLED', 'PERFORMANCE'):
        record = frozen['roles']['FROZEN_'+role]; result = receipt(BASE/'independent_reproduction'/role/'RESULTS.json', fresh=True)
        role_start = read_json(bind(BASE/'independent_reproduction'/role/'START.json'))
        require_fresh(role_start, timing['audit_started_utc'])
        checks[role+'_fresh_child_explicit_isolated_python_and_SDK_bootstrap_disabled'] = (
            role_start['command'][1:3] == ['-I', '-u']
            and role_start['isolated_python_mode'] is True
            and role_start['SDK_auto_monkey_patch_disabled_for_child'] is True)
        chosen = selection[role.lower()]; entry = entry_points['entries'][role]
        model = ROOT/chosen['model_directory']; model_sha = digest(bind(model/'model.sbnext2'))
        bind(model/'model.json')
        checks[role+'_canonical_model_bytes_exact_after_raw_training'] = result['canonical_model_bytes_exact'] is True and model_sha == record['canonical_model_sha256'] == result['canonical_model_sha256'] == entry['canonical_model_sha256']
        checks[role+'_all_DEV_public_output_fingerprint_exact'] = result['series'] == 8000 and result['points'] == 4014405 and result['all_dev_prediction_sha256'] == record['prediction_fingerprint_sha256']
        checks[role+'_raw_training_causal_seal_admission_and_63_pins'] = result['pin_count'] == 63 and result['opaque_seal_series_skipped'] == 2000 and result['seal_values_accessed'] == result['seal_rows_fitted'] == 0 and result['new_fits'] == (1 if role == 'DISTILLED' else 2)
        checks[role+'_origin_expected_early_network_outside_write_denied'] = not any(result[key] for key in ('original_project_access_attempts', 'expected_reads_before_training', 'outside_workspace_write_attempts', 'network_attempts'))
        checks[role+'_first_None_current_point_only_no_future_length_or_suffix'] = result['first_none_before_data'] is True and result['one_current_point_per_yield'] is True and result['no_future_length_or_suffix_access'] is True
        checks[role+'_published_entry_is_the_actual_reproduced_entry'] = entry['entry_module'] == result['entry_module'] == chosen['entry'] and entry['model_directory'] == chosen['model_directory']
    engineering = receipt(BASE/'engineering/lean_shared/ENGINEERING_GATE_DECISION.json', statuses=('PROVISIONAL_ENGINEERING_ELIGIBLE',))
    lock_path = BASE/'engineering/lean_shared/IMPLEMENTATION_LOCK_CORRECTION_001.json'; lock = read_json(bind(lock_path))
    harness_path = BASE/'engineering/lean_shared/VALIDATION_HARNESS_CORRECTION_002.json'; harness = read_json(bind(harness_path))
    checks['engineering_core_and_validation_correction_hashes_frozen'] = digest(lock_path) == engineering['core_implementation_lock_sha256'] and digest(harness_path) == engineering['validation_harness_correction_sha256']
    file_bindings(lock['source_sha256']); file_bindings(harness['source_sha256'])
    proof_paths = {'all_dev': 'engineering/lean_shared/all_dev/RESULTS.json',
        'oof': 'engineering/lean_shared/heldout_metadata_corrected_v1/oof/RESULTS.json',
        'alternate': 'engineering/lean_shared/heldout_metadata_corrected_v1/alternate/RESULTS.json',
        'cold': 'engineering/lean_shared/cold/RESULTS.json', 'direct': 'engineering/lean_shared/direct/RESULTS.json',
        'public_api': 'lean_public_api_all_dev/RESULTS.json'}
    parity = {}
    for name, relative in proof_paths.items():
        path = BASE/relative; parity[name] = receipt(path)
        if digest(path) != engineering['proofs_sha256'][name]: raise RuntimeError('Engineering gate proof changed: '+name)
    checks['E003_full_heldout_OOF_parity'] = parity['oof']['heldout_oof_checked'] is True
    checks['E003_all_20_ALT_H_models_16057620_points_exact'] = parity['alternate']['model_fits_checked_per_variant'] == 20 and parity['alternate']['points_per_variant'] == 16057620 and parity['alternate']['prediction_max_abs_diff'] == 0.
    checks['E003_all_DEV_public_API_readonly_and_reverse_128_reset'] = parity['public_api']['series'] == 8000 and parity['public_api']['points'] == 4014405 and parity['public_api']['persistent_filesystem_write_attempts'] == 0 and parity['public_api']['reverse_order_reset_series'] == 128
    checks['E003_cold_memory_route_pass_startup_route_not_claimed'] = parity['cold']['cold_cost_gate_pass'] is True and parity['cold']['memory_route_pass'] is True and parity['cold']['median_RSS_ratio'] <= .8 and all(row['RSS_ratio'] < 1 for row in parity['cold']['pairs']) and parity['cold']['startup_route_pass'] is False
    checks['E003_three_paired_10M_cost_trials_and_constant_state'] = parity['direct']['direct_cost_gate_pass'] is True and parity['direct']['total_direct_points'] == 60000000 and parity['direct']['median_direct_10M_ratio'] <= 1.05 and parity['direct']['constant_state'] is True
    long_dir = BASE/'maintenance/lean_long_horizon_parity_v1'
    long_parity = receipt(long_dir/'RESULTS.json')
    long_protocol_path = BASE/'maintenance/LEAN_LONG_HORIZON_PARITY_PROTOCOL.json'
    long_protocol = read_json(bind(long_protocol_path))
    file_bindings(long_protocol['source_sha256']); file_bindings(long_parity['files_sha256'], long_dir)
    file_bindings(long_parity['existing_synthetic_score_files_sha256'])
    checks['E003_34603008_original_long_horizon_public_outputs_exact'] = (
        long_parity['public_API_prediction_points'] == 34603008 and long_parity['synthetic_streams'] == 3072
        and long_parity['all_original_float32_predictions_exact'] is True
        and long_parity['constant_retained_numeric_bytes'] == 16925
        and long_parity['input_arrays_unchanged'] is True
        and long_parity['first_None_before_dataset_access'] is True
        and long_parity['one_current_point_per_yield'] is True
        and long_parity['new_fits'] == long_parity['new_quality_scores'] == long_parity['new_runtime_families'] == 0
        and long_parity['protocol_sha256'] == digest(long_protocol_path))
    long_copy_dir = BASE/'maintenance/long_horizon_copy_replay_v1'
    long_copy = receipt(long_copy_dir/'RESULTS.json')
    file_bindings(long_copy['source_sha256']); file_bindings(long_copy['files_sha256'], long_copy_dir)
    long_child = receipt(long_copy_dir/'CHILD_RESULT.json')
    checks['optional_long_replay_actually_runs_without_raw_or_saved_prediction_arrays'] = (
        long_copy['public_API_prediction_points'] == long_child['public_API_prediction_points'] == 34603008
        and long_copy['synthetic_streams'] == long_child['synthetic_streams'] == 3072
        and long_copy['copied_files'] == 249 and long_copy['project_modules_loaded_from_copy'] == 32
        and long_copy['source_and_inputs_unchanged'] is True
        and long_copy['raw_observations_and_saved_prediction_arrays_absent'] is True
        and long_copy['fresh_environment_claimed'] is False
        and long_copy['actual_final_archive_replay_claimed'] is False
        and long_child['all_input_and_original_float32_output_hashes_exact'] is True
        and long_child['saved_prediction_arrays_needed'] is False
        and long_child['raw_competition_inputs_needed'] is False
        and long_child['retained_numeric_bytes'] == 16925
        and long_child['persistent_write_attempts'] == long_child['competition_or_saved_prediction_access_attempts'] == long_child['network_attempts'] == []
        and long_copy['new_fits'] == long_child['new_fits'] == long_child['new_quality_scores'] == 0)
    bind(ROOT/'docs/NEXT3_OPTIONAL_LONG_HORIZON_REPLAY.md')
    horizon_dir = BASE/'maintenance/horizon_calibration_logic_v1'
    horizon_logic = receipt(horizon_dir/'RESULTS.json', statuses=('PASS_EXPLANATORY_RANK_ALGEBRA',))
    file_bindings(horizon_logic['source_sha256'])
    horizon_report = receipt(horizon_dir/'REPORT.json')
    checks['fixed_horizon_calibration_scope_explained_without_new_alarm_policy'] = (
        horizon_logic['new_fits'] == horizon_logic['new_Monte_Carlo_draws'] == 0
        and horizon_logic['thresholds_recalibrated'] is False
        and horizon_logic['hypothetical_16_blocks']['premises_verified_for_current_carried_state_process'] is False
        and horizon_report['algebra_result_sha256'] == digest(horizon_dir/'RESULTS.json')
        and horizon_report['document_sha256'] == digest(bind(ROOT/horizon_report['document'])))
    tree_dir = BASE/'maintenance/tree_structural_final_audit'
    tree = receipt(tree_dir/'RESULTS.json', statuses=('PASS_TREE_STRUCTURE_AND_ARTIFICIAL_BOUNDARIES',), fresh=True)
    file_bindings(tree['source_sha256']); file_bindings(tree['files_sha256'], tree_dir)
    tree_lock_path = BASE/'maintenance/TREE_STRUCTURAL_AUDIT_IMPLEMENTATION_LOCK_CORRECTION_001.json'
    tree_lock = read_json(bind(tree_lock_path)); file_bindings(tree_lock['source_sha256'])
    tree_protocol_path = bind(BASE/'maintenance/TREE_STRUCTURAL_AUDIT_PROTOCOL.json')
    checks['fresh_raw_model_5800_nodes_and_38732_artificial_margins_exact'] = (
        tree['final_audit'] is True and tree['forest_components'] == 2
        and tree['model_directory'] == 'artifacts/next3/independent_reproduction/PERFORMANCE/model'
        and tree['canonical_model_sha256'] == frozen['roles']['FROZEN_PERFORMANCE']['canonical_model_sha256']
        and tree['unique_model_nodes_checked'] == 5800
        and tree['artificial_full_forest_margin_comparisons'] == 38732
        and tree['all_live_fields_column_maps_and_evaluated_margin_bits_exact'] is True
        and tree['primary_old_and_new_output_selectors_match_scientific_names'] is True
        and tree['unreachable_cases_retained_not_counted_as_executed'] is True
        and tree['new_fits'] == tree['new_quality_scores'] == tree['new_runtime_families'] == 0
        and tree['competition_data_or_saved_array_access_attempts'] == tree['network_attempts'] == []
        and tree['implementation_lock_sha256'] == digest(tree_lock_path)
        and tree['protocol_sha256'] == tree_lock['protocol_sha256'] == digest(tree_protocol_path))
    bind(ROOT/'docs/NEXT3_TREE_STRUCTURAL_AUDIT.md')
    runtime = receipt(BASE/'final_audit/runtime/RESULTS.json', fresh=True)
    file_bindings(runtime['files_sha256'], BASE/'final_audit/runtime')
    runtime_registration = read_json(bind(BASE/'final_audit/FINAL_RUNTIME_PROTOCOL.json'))
    file_bindings(runtime_registration['source_sha256'])
    checks['final_actual_public_API_30M_fingerprints_exact'] = runtime['total_actual_public_API_points'] == 30000000 and runtime['public_API_output_fingerprints_exact'] is True
    checks['final_runtime_constant_state_current_point_and_isolation'] = runtime['constant_numeric_state'] is True and runtime['one_current_point_per_yield'] is True and runtime['isolated_from_other_project_python_workers'] is True
    checks['final_same_protocol_performance_cost_below_1point5_reference'] = runtime['selected_performance_vs_reference_public_API_10M_ratio'] <= 1.5
    state = receipt(ROOT/'STATE_MEMORY_NEXT3.json')
    require_fresh({'created_utc': state['publication_utc']}, timing['audit_started_utc'])
    checks['selected_private_numeric_memory_below_two_reference'] = state['PERFORMANCE']['private_per_series_numeric_bytes'] <= 2*16441 and state['PERFORMANCE']['10M_retained_numeric_growth_bytes'] == 0
    prepared = prepared_table_proof(ROOT, BASE/'final_tables/prepared_v2')
    publication = receipt(BASE/'final_audit/publication/RESULTS.json', fresh=True)
    file_bindings(publication['files_sha256'])
    checks['closed_scientific_tables_preserve_all_original_ledger_bytes'] = prepared['scientific_table_rows'] == 17 and publication['original_ledger_byte_prefix_preserved'] is True and publication['original_ledger_rows'] == 213 and publication['new_ledger_rows'] == 29
    reports = receipt(BASE/'final_reports/RESULTS.json', fresh=True); file_bindings(reports['files_sha256']); file_bindings(reports['source_sha256'])
    checks['44_Korean_answers_and_all_supporting_reports_published'] = reports['required_numbered_answers'] == 44 and reports['language'] == 'Korean' and reports['preview'] is False
    claims = receipt(BASE/'final_claims/RESULTS.json', fresh=True); file_bindings(claims['source_sha256'])
    checks['at_least_245_independent_numeric_joins_and_44_answer_checks'] = claims['numeric_cross_checks'] >= 245 and claims['numbered_Korean_answers'] == 44
    checks['claims_keep_inherited_validation_and_undefined_null_metrics_separate'] = claims['inherited_ALT_H_not_new_generalization'] is True and claims['null_and_outlier_only_TS_AUC_undefined'] is True and claims['selection_adjustment_not_claimed'] is True
    counts['numeric_claim_cross_checks'] = claims['numeric_cross_checks']
    receipt(ROOT/'REPRODUCTION_NEXT3.json', fresh=True); receipt(ROOT/'DETERMINISM_NEXT3.json', fresh=True)
    for relative, count in [('pair_matrix_audit/run_v1', 130065), ('outlier_state_transfer/run_v1', None), ('legacy_state_transfer/run_v1', None)]:
        value = receipt(BASE/relative/'RESULTS.json')
        if count is not None: checks['D007_130065_scores_reconstructed_and_existing_arrays_exact'] = value['total_bootstrap_scores_including_observed'] == count and value['max_existing_fold_abs_difference'] == value['max_contrast_abs_difference'] == 0.
        checks[relative.replace('/', '_')+'_zero_fit_diagnostic'] = value['new_fits'] == 0
    finite = receipt(BASE/'bottlenecks/FINITE_MENU_CLOSURE_REVIEW.json', statuses=('FINITE_SCIENTIFIC_MENU_CLOSED_MAINTENANCE_CONTINUES',))
    checks['finite_menu_closed_without_falsely_triggering_saturation'] = finite['DEV_RESEARCH_SATURATION'] == 'NOT_FORMALLY_TRIGGERED' and finite['formal_section_60_all_ten_failed_screen_condition'] is False
    final_board = read_json(bind(ROOT/'BOTTLENECK_BOARD_FINAL.json'))
    allowed_statuses = {'RESOLVED', 'PARTIALLY_RESOLVED', 'UNCHANGED', 'WORSE', 'DEFERRED', 'NOT_RELEVANT'}
    checks['all_eight_final_bottlenecks_have_honest_terminal_assessments'] = len(final_board['bottlenecks']) == 8 and all(item['status'] in allowed_statuses for item in final_board['bottlenecks'])
    golden = receipt(ROOT/'PORTABLE_RUNTIME_GOLDEN_NEXT3.json', statuses=('FROZEN_SYNTHETIC_PUBLIC_API_GOLDEN',))
    checks['raw_free_portable_probe_source_and_98372_outputs_frozen'] = digest(bind(ROOT/'src/next3/portable_probes.py')) == golden['probe_source_sha256'] and sum(item['points'] for item in golden['roles'].values())*2 == 98372
    navigation_path = BASE/'maintenance/evidence_navigation_v1/RESULTS.json'
    navigation = receipt(navigation_path, statuses=('PASS_EXISTING_EVIDENCE_NAVIGATION',))
    checks['25_existing_evidence_records_linked_without_new_quality_claims'] = (
        navigation['evidence_records'] == len(navigation['records']) == 25
        and navigation['new_fits'] == navigation['new_quality_scores'] == 0
        and navigation['E003_six_proofs_matched_to_original_gate_decision'] is True
        and navigation['document_sha256'] == digest(bind(ROOT/navigation['document']))
        and navigation['source_sha256'] == digest(bind(ROOT/'scripts/build_next3_evidence_navigation.py')))
    file_bindings({item['path']: item['sha256'] for item in navigation['records']})
    statistics_dir = BASE/'maintenance/statistics_copy_replay_v1'
    statistics = receipt(statistics_dir/'RESULTS.json', statuses=('PASS_EXISTING_STATISTICS_FROM_MINIMAL_COPY',))
    file_bindings(statistics['source_sha256']); file_bindings(statistics['files_sha256'], statistics_dir)
    statistics_child = receipt(statistics_dir/'CHILD_RESULT.json', statuses=('PASS_EXISTING_CONDITIONAL_STATISTICS_REPLAY',))
    statistics_lock_path = bind(BASE/'maintenance/CONDITIONAL_STATISTICS_REPLAY_IMPLEMENTATION_LOCK.json')
    statistics_lock = read_json(statistics_lock_path); file_bindings(statistics_lock['source_sha256'])
    statistics_protocol_path = bind(BASE/'maintenance/CONDITIONAL_STATISTICS_REPLAY_PROTOCOL.json')
    checks['existing_conditional_statistics_replayed_from_15_raw_free_model_free_files'] = (
        statistics['copied_files'] == 15 and statistics['project_modules_loaded_from_copy'] == 7
        and statistics['numeric_cross_checks'] == statistics_child['numeric_cross_checks'] == 169
        and statistics['candidate_count'] == statistics_child['candidate_count'] == 12
        and statistics['existing_draw_rows'] == statistics_child['existing_draw_rows'] == 2000
        and statistics['raw_model_and_saved_point_array_files_absent'] is True
        and statistics['all_copied_source_and_inputs_unchanged'] is True
        and statistics['new_fits'] == statistics['new_Monte_Carlo_draws'] == statistics['new_point_prediction_or_quality_evaluations'] == 0
        and statistics_child['selection_adjusted'] is statistics_child['independent_validation_of_generalization'] is False
        and statistics_child['persistent_write_attempts'] == statistics_child['raw_model_or_point_array_access_attempts'] == statistics_child['network_attempts'] == []
        and statistics['implementation_lock_sha256'] == digest(statistics_lock_path)
        and statistics['protocol_sha256'] == statistics_lock['protocol_sha256'] == digest(statistics_protocol_path))
    bind(ROOT/'docs/NEXT3_CONDITIONAL_STATISTICS_REPLAY.md')
    support_dir = BASE/'maintenance/diagnostic_cohort_support_v1'
    support = receipt(support_dir/'RESULTS.json', statuses=('PASS_HYPOTHETICAL_INDEPENDENT_COHORT_SUPPORT_ARITHMETIC',))
    file_bindings(support['source_sha256']); file_bindings(support['files_sha256'], support_dir)
    support_precision = receipt(support_dir/'PRECISION_REVIEW.json')
    checks['hypothetical_diagnostic_support_arithmetic_keeps_power_and_seals_out_of_scope'] = (
        support['hypothetical_grid_rows'] == 48 and support['expected_scale_rows'] == 12
        and support['maximum_absolute_log_probability_difference'] <= 1e-8
        and support['diagnostic_minimum'] == 100 and support['existing_DEV_series'] == 8000
        and support['new_fits'] == support['new_Monte_Carlo_draws'] == support['new_quality_scores'] == 0
        and support['sealed_partition_composition_analyzed'] is support['TS_AUC_power_analysis'] is False
        and support['assumptions_verified_for_any_future_cohort'] is support['new_promotion_or_seal_rule'] is False
        and support['protocol_sha256'] == digest(bind(BASE/'maintenance/DIAGNOSTIC_COHORT_SUPPORT_PROTOCOL.json'))
        and support_precision['result_sha256'] == digest(support_dir/'RESULTS.json')
        and support_precision['original_report_sha256'] == digest(bind(support_dir/'REPORT.json'))
        and support_precision['original_document_sha256'] == digest(bind(ROOT/support_precision['original_document']))
        and support_precision['reviewed_document_sha256'] == digest(bind(ROOT/support_precision['reviewed_document'])))
    requirement_dir = BASE/'maintenance/requirements_traceability_pre_audit_v1'
    requirement = receipt(requirement_dir/'RESULTS.json', statuses=('PASS_SPECIFICATION_TRACEABILITY_MAP_NOT_FINAL_AUDIT',))
    file_bindings({item['snapshot']:item['sha256'] for item in requirement['input_snapshots'].values()})
    checks['all_81_user_sections_have_historical_evidence_or_explicit_pending_scope'] = (
        requirement['user_sections'] == 81 and {row['section'] for row in requirement['rows']} == set(range(81))
        and len(requirement['input_snapshots']) == 37
        and requirement['new_fits'] == requirement['new_quality_scores'] == requirement['new_Monte_Carlo_draws'] == 0
        and requirement['all_requirements_claimed_complete'] is requirement['actual_final_audit_claimed_complete'] is False
        and requirement['learning_curve_fits_performed'] is requirement['label_permutation_or_time_shift_learner_fits_performed'] is False
        and requirement['source_sha256'] == digest(bind(ROOT/'scripts/build_next3_requirements_traceability.py'))
        and requirement['table_sha256'] == digest(bind(requirement_dir/'TRACEABILITY.csv'))
        and requirement['document_sha256'] == digest(bind(ROOT/requirement['document'])))
    phase_lock = receipt(BASE/'maintenance/PHASE_ACCOUNTING_IMPLEMENTATION_LOCK.json',
        statuses=('REGISTERED_PROSPECTIVE_ACCOUNTING_FIX',))
    file_bindings(phase_lock['source_sha256'])
    replacement = receipt(BASE/'runtime/maintenance_updates/SUPERVISOR_V5_REPLACED.json')
    bind(BASE/'runtime/maintenance_updates/SUPERVISOR_V5_REPLACEMENT_PLAN.json')
    bind(ROOT/'docs/NEXT3_PHASE_ACCOUNTING_REVIEW.md')
    phase_state = read_json(bind(BASE/'jobs/M689_PHASE_ACCOUNTING_REGRESSIONS/STATE.json'))
    phase_tests = tests_proof(bind(BASE/'maintenance/PHASE_ACCOUNTING_REGRESSIONS.xml'),
        minimum_count=71, required_classnames={'tests.test_next3_phase_accounting', 'tests.test_next3_supervisor_lifecycle'},
        not_before_timestamp=datetime.datetime.fromisoformat(phase_state['started_utc']).timestamp())
    checks['phase_accounting_fixed_and_regression_tested_before_actual_audit_without_backcredit'] = (
        phase_state['status'] == 'COMPLETE' and phase_state['exit_code'] == 0
        and phase_tests['tests'] == 71 and phase_tests == replacement['targeted_tests']
        and len(phase_lock['source_sha256']) == 4 and phase_lock['new_fits'] == 0
        and replacement['new_source_sha256'] == phase_lock['source_sha256']['src/next3/maintenance.py']
        and replacement['replacement']['source_sha256'] == replacement['new_source_sha256']
        and replacement['retroactive_credit'] is False and replacement['restart_gap_kept_unobserved'] is True
        and replacement['active_children'] == 0
        and replacement['accounting_before']['audit_productive_seconds'] == 0
        and replacement['accounting_after_start']['audit_productive_seconds'] == 0
        and datetime.datetime.fromisoformat(replacement['finished_utc']) < datetime.datetime.fromisoformat(timing['audit_started_utc']))
    for name in REQUIRED:
        if name != 'FINAL_AUDIT_NEXT3.json': bind(ROOT/name)
    # Snapshot into the excluded handoff tree, so a later package cannot recurse into this copy.
    payload = BASE/'handoff/cp/payload'
    files, excluded, missing = snapshot_payload(payload, require_audit=False)
    checks['actual_payload_snapshot_has_all_required_reports_except_pending_core_audit'] = set(missing) == {'FINAL_AUDIT_NEXT3.json'}
    checks['bounded_raw_free_ordinary_payload_content_verified'] = bool(files) and all(not name.startswith('data/') for name in files)
    checks['final_payload_contains_every_frozen_execution_source_and_config_byte'] = all(
        name in files and files[name] == record for name,record in sources['files'].items())
    counts.update(payload_preflight_files=len(files), payload_preflight_bytes=sum(item['bytes'] for item in files.values()),
        excluded_optional_reports=len(excluded), final_raw_reproduction_fits=3, total_completed_fits=132)
    atomic_json(out/'PAYLOAD_PREFLIGHT.json', {'status': 'PASS', 'created_utc': utc_now(),
        'snapshot_directory': str(payload), 'files': files, 'excluded': excluded, 'missing_final_core_audit_only': missing,
        'actual_Desktop_archive_created': False}, immutable=True)
    bind(out/'PAYLOAD_PREFLIGHT.json')
    guard.check_state(); all_true(checks)
    result = {'status': 'PASS', 'created_utc': utc_now(), 'clock': timing, 'checks': checks, 'counts': counts,
        'evidence_sha256': evidence, 'protected_originals': protected, 'final_duration_closure': False,
        'source_sha256': digest(Path(__file__)), 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'conclusion': 'NEXT3_ENGINEERING_IMPROVED', 'scientific_quality_delta': 0.0,
        'scientific_promotions': 0, 'actual_archive_replay': 'PENDING_EXTERNAL_AFTER_FINAL_ARCHIVE_CREATION',
        'audit_role': 'Same agent in a separately timed validation phase; no independent human or second-agent review claimed',
        'scope_limitations': ['Conditional fitted-OOF intervals do not adjust adaptive DEV selection',
            'Python admission/audit hooks are not an OS sandbox', 'Pinned Windows and tested-stream exactness is not cross-platform or hidden-test validation',
            'Source manifest is generated during packaging; actual archive SHA/CRC and extracted inference replay are external post-creation proofs']}
    atomic_json(out/'RESULTS.json', result, immutable=True)
    progress('NEXT3 final core evidence checkpoint passed', str(len(checks))+' checks; separate 50-minute closure and actual Desktop ZIP replay remain required', phase='FINAL_AUDIT')
    print('NEXT3 FINAL CORE CHECKPOINT PASS', len(checks), flush=True)


def finalize():
    timing = current_clock(complete=True)
    checkpoint_path = BASE/'final_audit/core_checkpoint/RESULTS.json'
    checkpoint = successful_receipt(checkpoint_path); require_fresh(checkpoint, timing['audit_started_utc'])
    all_true(checkpoint['checks'])
    if checkpoint['source_sha256'] != digest(Path(__file__)): raise RuntimeError('Core audit source changed after the evidence checkpoint')
    for name, sha in checkpoint['evidence_sha256'].items():
        if digest(ROOT/name) != sha: raise RuntimeError('Core evidence changed before duration closure: '+name)
    source_lock = read_json(BASE/'final_audit/FINAL_EXECUTION_SOURCE_LOCK.json'); assert_source_snapshot(ROOT, source_lock)
    frozen = read_json(BASE/'preflight/FROZEN_REFERENCES.json')
    protected_files(ROOT, frozen['protected_sha256'], (BASE/'preflight/EXPERIMENT_LEDGER_BEFORE_NEXT3.csv').read_bytes())
    old_desktop = verify_existing_desktop_deliveries(ROOT)
    guard = SealGuard(ROOT); guard.check_state()
    if any(item['opened'] for item in guard.lock['partitions'].values()): raise RuntimeError('A seal was opened before closure')
    checks = {**checkpoint['checks'], 'research_11h10_and_separate_audit_50m_wall_minima': True,
        'all_checkpoint_evidence_and_final_sources_unchanged_at_closure': True}
    all_true(checks)
    result = {**checkpoint, 'created_utc': utc_now(), 'clock': timing, 'checks': checks,
        'final_duration_closure': True, 'research_wall_seconds': timing['research_wall_seconds'],
        'audit_wall_seconds_at_closure': timing['audit_wall_seconds'],
        'observed_accounting_at_closure': read_json(BASE/'runtime/ACCOUNTING.json'),
        'earlier_Desktop_deliveries_verified_at_closure': old_desktop,
        'evidence_sha256': {**checkpoint['evidence_sha256'], checkpoint_path.relative_to(ROOT).as_posix(): digest(checkpoint_path)},
        'seals_opened': {name: False for name in guard.lock['partitions']},
        'reduced_new_usage': 0, 'public_submission_new': 0, 'cloud_submission_new': 0,
        'SESSION_COMPLETE_at_core_audit_time': False,
        'remaining_required_distribution_work': 'Create and byte-verify actual Desktop ZIP; independently extract and replay it; execute the documented reviewer command; create exactly 20 core Desktop result files; publish the external session completion receipt.'}
    atomic_json(ROOT/'FINAL_AUDIT_NEXT3.json', result, immutable=True)
    progress('NEXT3 final core audit duration closure passed', 'Source/evidence unchanged; actual archive and 20-file Desktop copy remain', phase='FINAL_AUDIT')
    print('NEXT3 FINAL CORE AUDIT PASS; ACTUAL DESKTOP ARCHIVE STILL REQUIRED', flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--finalize', action='store_true'); args = parser.parse_args()
    finalize() if args.finalize else evaluate_checkpoint()


if __name__ == '__main__': main()
