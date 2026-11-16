"""Publish two-role evidence only after final raw, runtime and full-test proofs."""
from pathlib import Path
import copy
import datetime
import os
import sys
import uuid
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.final_validation import prepared_table_proof, successful_receipt, tests_proof
from src.next3.final_source_contract import assert_source_snapshot
from src.next.guard import SealGuard
from scripts.prepare_next3_independent_reproduction import time_gate, selected_performance


def replace_bytes(path, content):
    path = Path(path); temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    timing = time_gate(); guard = SealGuard(ROOT); guard.check_state()
    source_lock_path = BASE/'final_audit/FINAL_EXECUTION_SOURCE_LOCK.json'
    source_lock = successful_receipt(source_lock_path, statuses=('FROZEN_FINAL_EXECUTION_SOURCE',))
    assert_source_snapshot(ROOT, source_lock)
    final_tests_result_path = BASE/'final_audit/FULL_TESTS_RESULT.json'
    final_tests_result = successful_receipt(final_tests_result_path)
    assert final_tests_result['source_lock_sha256'] == digest(source_lock_path)
    out = BASE/'final_audit/publication'; out.mkdir(parents=True, exist_ok=False)
    prepared = BASE/'final_tables/prepared_v2'; table_proof = prepared_table_proof(ROOT, prepared)
    frozen_path = BASE/'preflight/FROZEN_REFERENCES.json'; frozen = read_json(frozen_path)
    science_lock_path = ROOT/'configs/next3/SCIENTIFIC_ROLE_FREEZE.json'; science_lock = read_json(science_lock_path)
    assert science_lock['status'] == 'PRE_AUDIT_SCIENTIFIC_ROLES_FROZEN' and science_lock['scientific_role_count'] == 2
    assert not science_lock['generation_gate_passed'] and not science_lock['new_scientific_search_authorized']
    reproduction_path = BASE/'independent_reproduction/RESULTS.json'; reproduction = successful_receipt(reproduction_path)
    assert reproduction['fresh_environment'] and reproduction['raw_copies_only_no_prior_feature_cache']
    assert reproduction['raw_reproduction_fits'] == 3 and reproduction['new_candidate_search_fits'] == 0
    runtime_path = BASE/'final_audit/runtime/RESULTS.json'; runtime = successful_receipt(runtime_path)
    assert runtime['selected_roles_verified'] == ['DISTILLED', 'PERFORMANCE']
    assert runtime['public_API_output_fingerprints_exact'] and runtime['one_current_point_per_yield']
    assert runtime['constant_numeric_state'] and runtime['isolated_from_other_project_python_workers']
    assert runtime['selected_performance_vs_reference_public_API_10M_ratio'] <= 1.5
    test_path = BASE/'final_audit/FULL_TESTS.xml'
    tests = tests_proof(test_path, minimum_count=798,
        required_classnames={'tests.test_ts_auc', 'tests.test_scorer_official_checks',
            'tests.test_next3_final_reproduction_gate', 'tests.test_next3_offline_environment',
            'tests.test_next3_result_tables', 'tests.test_next3_final_validation', 'tests.test_next3_final_runtime_probe'},
        not_before_timestamp=datetime.datetime.fromisoformat(timing['audit_started_utc']).timestamp())
    assert final_tests_result['tests']['report_sha256'] == tests['report_sha256']
    assert final_tests_result['collected_tests'] == tests['tests']
    assert final_tests_result['source_set_unchanged_before_and_after'] is True
    expected_performance_entry, _ = selected_performance()
    assert expected_performance_entry == 'next3_lean_performance_main', 'This publication requires the actually verified E003 selection'
    evidence = {path.relative_to(ROOT).as_posix(): digest(path) for path in
        (frozen_path, science_lock_path, reproduction_path, runtime_path, test_path,
         prepared/'RESULTS.json', prepared/'INPUT_RETENTION.json', source_lock_path, final_tests_result_path)}
    engineering_path = BASE/'engineering/lean_shared/ENGINEERING_GATE_DECISION.json'
    engineering = successful_receipt(engineering_path, statuses=('PROVISIONAL_ENGINEERING_ELIGIBLE',))
    lock_path = BASE/'engineering/lean_shared/IMPLEMENTATION_LOCK_CORRECTION_001.json'
    assert digest(lock_path) == engineering['core_implementation_lock_sha256']
    lock = read_json(lock_path)
    for relative, sha in lock['source_sha256'].items(): assert digest(ROOT/relative) == sha
    evidence[lock_path.relative_to(ROOT).as_posix()] = digest(lock_path)
    evidence[engineering_path.relative_to(ROOT).as_posix()] = digest(engineering_path)
    proof_paths = {'all_dev': 'engineering/lean_shared/all_dev/RESULTS.json',
        'oof': 'engineering/lean_shared/heldout_metadata_corrected_v1/oof/RESULTS.json',
        'alternate': 'engineering/lean_shared/heldout_metadata_corrected_v1/alternate/RESULTS.json',
        'cold': 'engineering/lean_shared/cold/RESULTS.json', 'direct': 'engineering/lean_shared/direct/RESULTS.json',
        'public_api': 'lean_public_api_all_dev/RESULTS.json'}
    parity = {}
    for name, relative in proof_paths.items():
        path = BASE/relative; parity[name] = successful_receipt(path)
        assert digest(path) == engineering['proofs_sha256'][name]
        evidence[path.relative_to(ROOT).as_posix()] = digest(path)
    assert parity['oof']['heldout_oof_checked'] is True
    assert parity['alternate']['model_fits_checked_per_variant'] == 20
    assert parity['alternate']['points_per_variant'] == 16057620 and parity['alternate']['prediction_max_abs_diff'] == 0
    for path in (BASE/'lean_public_api_all_dev/RESULTS.json', ROOT/'artifacts/next2/canonical_api_all_dev/DISTILLED/RESULTS.json'):
        api = successful_receipt(path)
        assert api['series'] == 8000 and api['points'] == 4014405 and api['reverse_order_reset_series'] == 128
        assert api['persistent_filesystem_write_attempts'] == api['seal_rows'] == 0
        evidence[path.relative_to(ROOT).as_posix()] = digest(path)
    entries = {}; roles = {}; raw_proofs = {}
    for role in ('DISTILLED', 'PERFORMANCE'):
        reference = frozen['roles']['FROZEN_'+role]
        path = BASE/'independent_reproduction'/role/'RESULTS.json'; proof = successful_receipt(path)
        assert proof['role'] == role and proof['series'] == 8000 and proof['points'] == 4014405
        assert proof['new_fits'] == (1 if role == 'DISTILLED' else 2) and proof['pin_count'] == 63
        assert proof['canonical_model_bytes_exact'] and proof['canonical_model_sha256'] == reference['canonical_model_sha256']
        assert proof['all_dev_prediction_sha256'] == reference['prediction_fingerprint_sha256']
        assert proof['seal_rows_fitted'] == proof['seal_values_accessed'] == 0
        assert proof['opaque_seal_series_skipped'] == 2000
        assert not any(proof[key] for key in ('original_project_access_attempts', 'expected_reads_before_training',
            'outside_workspace_write_attempts', 'network_attempts'))
        assert proof['first_none_before_data'] and proof['one_current_point_per_yield'] and proof['no_future_length_or_suffix_access']
        entry = 'next2_canonical_distilled_main' if role == 'DISTILLED' else expected_performance_entry
        assert proof['entry_module'] == entry
        model = path.parent/'model'
        assert digest(model/'model.sbnext2') == reference['canonical_model_sha256']
        for source in (path, model/'model.sbnext2', model/'model.json'):
            evidence[source.relative_to(ROOT).as_posix()] = digest(source)
        entries[role] = {'entry_module': entry, 'model_directory': model.relative_to(ROOT).as_posix(),
            'canonical_model_sha256': reference['canonical_model_sha256'], 'scientific_experiment_id': reference['experiment_id']}
        roles[role.lower()] = {**copy.deepcopy(reference), 'entry': entry,
            'model_directory': model.relative_to(ROOT).as_posix(), 'raw_reproduction_result': path.relative_to(ROOT).as_posix(),
            'raw_reproduction_result_sha256': digest(path), 'canonical_model_sha256': digest(model/'model.sbnext2'),
            'model_bytes': (model/'model.sbnext2').stat().st_size,
            'descriptor_sha256': digest(model/'model.json'),
            'runtime_variant': 'DISTILLED_tree_used_v1' if role == 'DISTILLED' else 'LEAN_PACKED_SHARED',
            'quality_change_since_session_start_for_same_role': 0.0,
            'scientific_promotion': False, 'final_delivery_acceptance_authority': 'FINAL_AUDIT_NEXT3.json plus the external verified actual archive receipt'}
        raw_proofs[role] = proof
    original = (BASE/'preflight/EXPERIMENT_LEDGER_BEFORE_NEXT3.csv').read_bytes()
    if (ROOT/'EXPERIMENT_LEDGER.csv').read_bytes() != original:
        raise RuntimeError('Preserve an earlier root ledger publication; explicit review required')
    for name, details in table_proof['files'].items():
        content = (prepared/name).read_bytes()
        if name.endswith('.json'):
            record = read_json(prepared/name)
            record.update(status='PASS', final_raw_reproduction_sha256=digest(reproduction_path),
                final_public_API_runtime_verification_sha256=digest(runtime_path),
                publication_utc=utc_now(), final_audit_and_archive_receipts_separate=True)
            atomic_json(ROOT/name, record)
        else:
            if name == 'EXPERIMENT_LEDGER.csv': assert content.startswith(original)
            replace_bytes(ROOT/name, content)
    selection = {'status': 'FROZEN_NEXT3_DEV_SELECTION', 'created_utc': utc_now(), **roles,
        'scientific_role_count': 2, 'starting_performance': copy.deepcopy(frozen['roles']['FROZEN_PERFORMANCE']),
        'selected_quality_delta_vs_starting_performance': 0.0, 'selected_delta_ci': [0.0, 0.0],
        'selected_strict_probability_delta_positive': 0.0, 'selected_contrast_is_analytic_identity': True,
        'new_performance_candidate': None, 'new_robustness_candidate': None,
        'generation_level_promotion': False, 'engineering_variant_selected': 'LEAN_PACKED_SHARED',
        'engineering_same_scientific_role': True, 'canonical_predictions_changed': False,
        'scientific_search_fits': 129, 'separate_final_raw_reproduction_fits': 3, 'total_completed_fits': 132,
        'new_scientific_recipes': 13, 'inherited_roles_recounted_as_new_candidates': 0,
        'final_raw_reproduction_passed': True, 'final_runtime_verification_passed': True,
        'final_full_tests': tests, 'evidence_sha256': evidence,
        'all_seals_unopened': True, 'reduced_new_usage': 0, 'public_submission_new': 0, 'cloud_submission_new': 0,
        'runtime_configuration_history': 'The original PROVISIONAL engineering config remains byte-for-byte preserved for reproduction. This selection freezes the actual deliverable paths; final core-audit and external actual-archive receipts complete acceptance.',
        'remaining_distribution_proof': 'Actual final Desktop ZIP replay; verified externally after the archive is built.',
        'scope': 'Frozen scientific roles and reproduced runtime delivery; source core audit and external archive verification are separately reported.'}
    atomic_json(ROOT/'DEV_CANDIDATES_NEXT3.json', selection)
    atomic_json(ROOT/'DEPLOYMENT_ENTRY_POINTS_NEXT3.json', {'status': 'FROZEN_NEXT3_PUBLIC_ENTRY_POINTS',
        'created_utc': utc_now(), 'entries': entries, 'INFER_PARALLELISM': 1,
        'scientific_role_count': 2, 'new_fits_for_selection': 0, 'selection_sha256': digest(ROOT/'DEV_CANDIDATES_NEXT3.json')})
    reproduced = {'status': 'PASS', 'created_utc': utc_now(), 'roles': reproduction['roles'],
        'new_learner_fits': 3, 'prior_scientific_fits': 129, 'fresh_environment': True,
        'canonical_model_bytes_exact': True, 'all_DEV_prediction_fingerprints_exact': True,
        'prior_feature_caches_used': False, 'expected_weights_inaccessible_until_training_complete': True,
        'original_project_access_attempts': [], 'seal_values_accessed': 0,
        'raw_source': 'Ordinary copied original bytes, with DEV-only admission before value access.',
        'original_result_sha256': digest(reproduction_path),
        'evidence_sha256': {path: sha for path, sha in evidence.items() if '/independent_reproduction/' in path},
        'boundary': 'Python audit hooks and complete input/source checks; not an OS sandbox.'}
    atomic_json(ROOT/'REPRODUCTION_NEXT3.json', reproduced)
    deterministic = {'status': 'PASS', 'created_utc': utc_now(), 'selected_roles': list(entries),
        'canonical_save_load': True, 'fresh_raw_training': True, 'full_DEV_API': True,
        'OOF_and_alternate_runtime_parity': True, 'reversed_series_reset': True,
        'one_shot_causal_online_stream': True, 'persistent_inference_writes': 0,
        'actual_final_10M_API_per_entry': True, 'actual_archive_replay': 'EXTERNAL_RECEIPT_REQUIRED',
        'evidence_sha256': evidence, 'scope': 'Pinned Windows environment and tested streams; no claim of cross-platform or hidden-test validation.'}
    atomic_json(ROOT/'DETERMINISM_NEXT3.json', deterministic)
    atomic_json(ROOT/'RAW_DATA_REQUIRED_NEXT3.json', {'status': 'EXTERNAL_RAW_INPUTS_REQUIRED', 'created_utc': utc_now(),
        'included_in_GPT_archive': False, 'raw_sha256': frozen['raw_sha256'], 'DEV_series': 8000,
        'seal_series': 2000, 'seal_values_must_remain_unread': True,
        'required_files': ['data/raw/X_train.parquet', 'data/raw/y_train_index.parquet'],
        'training_code': {role: entry['entry_module']+'.train' for role, entry in entries.items()},
        'purpose': 'Only needed for a new training reproduction. Archived inference replay uses fixed synthetic probes and needs no competition raw input.',
        'credentials_included': False})
    published = [*table_proof['files'], 'DEV_CANDIDATES_NEXT3.json', 'DEPLOYMENT_ENTRY_POINTS_NEXT3.json',
                 'REPRODUCTION_NEXT3.json', 'DETERMINISM_NEXT3.json', 'RAW_DATA_REQUIRED_NEXT3.json']
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'time_gate': timing,
        'scientific_role_count': 2, 'scientific_promotions': 0, 'scientific_fits': 129, 'raw_reproduction_fits': 3,
        'original_ledger_byte_prefix_preserved': True, 'original_ledger_rows': 213, 'new_ledger_rows': 29,
        'files_sha256': {name: digest(ROOT/name) for name in published},
        'source_sha256': digest(Path(__file__)), 'final_core_audit_completed': False, 'actual_archive_replay_completed': False}, immutable=True)
    assert_source_snapshot(ROOT, source_lock)
    progress('NEXT3 final evidence published', 'Both raw-reproduced roles and final runtime/tests verified; final core audit and actual archive replay remain separate', phase='FINAL_AUDIT')
    print('NEXT3 FINAL EVIDENCE PUBLISHED TWO ROLES; CORE AUDIT AND ACTUAL ARCHIVE REMAIN REQUIRED', flush=True)


if __name__ == '__main__': main()
