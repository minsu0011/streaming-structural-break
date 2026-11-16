"""Aggregate independent full-population, provenance and execution receipts.

This module never changes a fitted procedure or treats an inference replay as a
new performance experiment. Final archive reproduction remains a separate gate.
"""
from collections import Counter
import math
from src.next8.common import ROOT, BASE, read_json, digest

FLOAT32_EPS = 2.0 ** -23


def verified(directory, name):
    directory = BASE / directory
    done = read_json(directory / 'DONE.json')
    assert done['status'] in ('COMPLETE', 'PASS')
    assert digest(directory / name) == done['files_sha256'][name], name
    return read_json(directory / name)


def finite_error(value, points, limit, *, strict=False):
    if points == 0:
        assert value is None
        return 0.
    assert value is not None and math.isfinite(value) and value >= 0
    assert value < limit if strict else value <= limit
    return value


def aggregate_series(records, expected_folds, expected_points):
    """Reject omitted, repeated, forbidden and incorrectly folded replay cases."""
    ids = [int(r['series_id']) for r in records]
    assert len(ids) == len(set(ids)), 'A repeated series cannot replace an omitted series'
    assert set(ids) == set(expected_folds), 'Full replay must cover exactly original RP membership'
    assert sum(r['online_points'] for r in records) == expected_points
    by_fold = Counter()
    comparisons = {}
    for mode in ('NORMAL', 'EARLY'):
        comparisons[mode] = {
            'saved_points_per_head': 0, 'baseline_points': 0,
            'maximum_baseline_float64_error': 0., 'maximum_M6_composition_error': 0.,
            'M6_eligible_series': 0,
            'heads': {m: {'maximum_saved_float32_error': 0., 'bitwise_equal_series': 0,
                           'compared_series': 0, 'no_saved_points_series': 0}
                      for m in ('M1', 'M2', 'M3', 'M4', 'M5')}}
    for r in records:
        sid = int(r['series_id'])
        assert r['status'] == 'PASS' and r['outer_fold'] == expected_folds[sid]
        by_fold[r['outer_fold']] += 1
        assert r['online_points'] > 0 and r['normalized_features'] == 153
        assert r['fresh_scalar_kernel_feature_cache_bitwise_equal']
        assert r['head_features_and_graph_states_built_from_fresh_H_and_scalar_online_kernel']
        assert r['saved_feature_cache_used_only_as_comparison_target']
        assert not r['break_labels_or_tau_as_inference_input'] and r['new_model_fits'] == 0
        assert {'center', 'spread', 'fit_length', 'epsilon', 'seed_values', 'valid', 'reference_sha256'} <= set(r['all_H_metadata_equal'])
        assert set(r['full_H_original_reference_feature_parity']) == {'J102', 'C205'}
        assert all(a['bitwise_equal'] for a in r['full_H_original_reference_feature_parity'].values())
        assert set(r['predictions']) == {'NORMAL', 'EARLY'}
        for mode, result in r['predictions'].items():
            target = comparisons[mode]
            count = r['online_points'] if mode == 'NORMAL' else max(0, r['online_points'] - 400)
            assert result['baseline_points'] == count
            target['baseline_points'] += count
            target['saved_points_per_head'] += count
            target['maximum_baseline_float64_error'] = max(target['maximum_baseline_float64_error'],
                finite_error(result['baseline_float64_max_error'], count, 1e-12, strict=True))
            assert result['M6_fixed_composition_source_verified']
            target['maximum_M6_composition_error'] = max(target['maximum_M6_composition_error'],
                finite_error(result['M6_replay_composition_max_abs'], count, .05 * FLOAT32_EPS + 1e-12))
            target['M6_eligible_series'] += int(result['M6_eligible'])
            assert set(result['head_results']) == set(target['heads'])
            for method, head in result['head_results'].items():
                assert head['saved_points'] == count
                row = target['heads'][method]
                row['maximum_saved_float32_error'] = max(row['maximum_saved_float32_error'],
                    finite_error(head['maximum_float32_error'], count, FLOAT32_EPS))
                if count:
                    assert isinstance(head['bitwise_equal'], bool)
                    row['compared_series'] += 1
                    row['bitwise_equal_series'] += int(head['bitwise_equal'])
                else:
                    assert head['bitwise_equal'] is None
                    row['no_saved_points_series'] += 1
    return {'status': 'PASS', 'unique_RP_series': len(ids), 'online_points': expected_points,
            'series_per_outer_fold': dict(sorted(by_fold.items())),
            'all153_fresh_scalar_features_and_H_metadata_exact': True,
            'J102_C205_original_full_H_features_bitwise_equal': True,
            'normal_and_early_replay': comparisons,
            'head_saved_float32_tolerance': FLOAT32_EPS,
            'baseline_saved_float64_strict_tolerance': 1e-12,
            'M6_composition_tolerance': .05 * FLOAT32_EPS + 1e-12,
            'bitwise_feature_parity_is_distinct_from_prediction_tolerance': True,
            'all_population_head_designs_use_fresh_causal_features_and_states': True,
            'all_population_head_prediction_uses_batched_frozen_estimator': True,
            'scalar_model_API_chunk_snapshot_and_suffix_checks_are_separate_fixed_sample_and_synthetic_audits': True,
            'no_model_refits_no_label_or_tau_inference_inputs': True}


def summarize_full_replay():
    from src.next5.guard import ResearchGuard
    guard = ResearchGuard(ROOT)
    expected_folds = {int(s): guard.split.assignment(int(s))[1] for s in guard.research_ids}
    guard.check_state()
    assert len(expected_folds) == 6400
    directory = BASE / 'full_replay_audit'
    records, sources, shard_ids = [], {}, []
    for part in (0, 1):
        path = directory / f'PART_{part}_DONE.json'
        done = read_json(path)
        assert done['status'] == 'COMPLETE' and done['part'] == part and done['parts'] == 2
        assert len(done['shards']) == 50 and done['series'] == 3200
        sources[path.name] = digest(path)
        part_series = part_points = 0
        for shard in done['shards']:
            index = shard['shard']
            assert index % 2 == part and shard['path'] == f'SHARD_{index:03d}.json'
            receipt_path = directory / shard['path']
            assert digest(receipt_path) == shard['sha256']
            receipt = read_json(receipt_path)
            assert receipt['status'] == 'PASS' and receipt['part'] == part and receipt['shard'] == index
            assert receipt['implementation_lock_sha256'] == digest(BASE / 'FULL_REPLAY_V2_IMPLEMENTATION_LOCK_NEXT8.json')
            assert receipt['admission_sha256'] == digest(BASE / 'runtime/FINAL_AUDIT_ADMISSION.json')
            assert receipt['start'] == index * 64 and receipt['stop'] == (index + 1) * 64
            assert receipt['series'] == len(receipt['results']) == shard['series'] == 64
            assert receipt['points'] == sum(r['online_points'] for r in receipt['results']) == shard['points']
            records.extend(receipt['results'])
            part_series += receipt['series']; part_points += receipt['points']
            sources[shard['path']] = digest(receipt_path); shard_ids.append(index)
        assert (part_series, part_points) == (done['series'], done['points'])
    assert sorted(shard_ids) == list(range(100))
    result = aggregate_series(records, expected_folds, read_json(BASE / 'PREFLIGHT_NEXT8.json')['RP_online_rows'])
    return {**result, 'shards': 100, 'source_receipts_sha256': sources,
            'fresh_feature_sha256_by_series': {str(r['series_id']): r['fresh_features_sha256'] for r in records},
            'no_shadow_seal_reduced_public_cloud_values': True}


def summarize_execution():
    latency = verified('execution_summary', 'LATENCY_NEXT8.json')
    records = []
    for name, sha in latency['scope_receipts_sha256'].items():
        path = BASE / 'runtime_validation' / name / 'DONE.json'
        assert digest(path) == sha
        r = read_json(path)
        assert r['status'] == 'COMPLETE' and r['parity_status'] == 'PASS'
        for case in r['records']:
            assert case['status'] == 'PASS' and not case['labels_or_tau_used']
            assert case['maximum_coordinate_error'] == case['maximum_normalized_feature_error'] == 0.
            assert max(case['native_float64_error'].values()) < 1e-12
            assert all(e is None or e <= FLOAT32_EPS for e in case['saved_float32_error'].values())
            assert case['exact_saved_prefix_chunk_future_suffix'] and case['standalone_vs_shared_bitwise']
            assert case['baseline_float64_error'] is None or case['baseline_float64_error'] < 1e-12
            records.append({'scope': name, **case})
    return {'status': 'PASS', 'scope_count': len(latency['scope_receipts_sha256']),
            'procedure_series_cases': len(records), 'unique_RP_series': len({r['dataset_id'] for r in records}),
            'all_recorded_scalar_paths_share_exact_coordinates_and_normalized_features': True,
            'float64_native_probability_strict_max_error': max(max(r['native_float64_error'].values()) for r in records),
            'scalar_shared_standalone_chunk_snapshot_changed_future_prefix_bitwise': True,
            'same_series_normal_and_early_not_independent_replications': True,
            'scope_receipts_sha256': latency['scope_receipts_sha256'],
            'portable_pickle_and_public_entry_checks_are_separate': True}


def summarize_computation(full, execution):
    protected = verified('final_integrity', 'PROTECTED_INTEGRITY.json')
    data = verified('final_integrity', 'DATA_BOUNDARY_INTEGRITY.json')
    frozen = verified('final_integrity', 'FROZEN_PREDICTION_INTEGRITY.json')
    lineage_path = BASE / 'TRAINING_LINEAGE_AUDIT_NEXT8.json'
    lineage = read_json(lineage_path)
    assert lineage['status'] == 'PASS'
    tests = read_json(BASE / 'final_tests/DONE.json')
    assert tests['status'] == 'COMPLETE' and tests['counts']['failures'] == tests['counts']['errors'] == 0
    for name, sha in tests['files_sha256'].items():
        assert digest(BASE / 'final_tests' / name) == sha
    access_path = BASE / 'audit/FINAL_COMPLETE_TEST_ACCESS_NEXT8.json'
    assert digest(access_path) == tests['guard_receipt_sha256']
    access = read_json(access_path)
    assert access['status'] == 'PASS' and not access['blocked_reads']
    for name in ('PREFLIGHT_NEXT8.json', 'CAUSAL_CACHE_AUDIT_NEXT8.json', 'BASELINE_PROVENANCE_NEXT8.json',
                 'MODEL_ALIAS_CONTRACT_GATE_NEXT8.json'):
        assert read_json(BASE / name)['status'] == 'PASS', name
    proof = {name: digest(BASE / name) for name in (
        'PREFLIGHT_NEXT8.json', 'CAUSAL_CACHE_AUDIT_NEXT8.json', 'BASELINE_PROVENANCE_NEXT8.json',
        'MODEL_ALIAS_CONTRACT_GATE_NEXT8.json', 'TRAINING_LINEAGE_AUDIT_NEXT8.json',
        'runtime/FINAL_AUDIT_ADMISSION.json', 'runtime/FINAL_SOURCE_CLOSURE_NEXT8.json',
        'PREDICTION_FREEZE_NEXT8.json', 'final_integrity/DONE.json', 'final_tests/DONE.json',
        'execution_summary/DONE.json')}
    deterministic = {'status': 'PASS', 'full_RP_replay': full, 'fixed_RP_scalar_execution': execution,
        'source_and_environment_frozen_integrity': frozen,
        'portable_new_process_and_public_entry_bitwise_proof_pending_separate_archive_receipt': True,
        'no_universal_cross_hardware_or_library_version_bitwise_claim': True}
    leakage = {'status': 'PASS', 'original_data_boundary': data, 'protected_authorities': protected,
        'training_lineage_audit_sha256': digest(lineage_path),
        'full_population_causal_feature_and_state_replay': full['status'],
        'scalar_future_suffix_prefix_and_state_restore_checks': execution['status'],
        'complete_test_counts': tests['counts'], 'complete_guarded_tests_passed': tests['passed'],
        'source_receipts_sha256': proof, 'outer_selection_scoring_release_is_separate_and_hash_frozen': True,
        'historical_adaptive_RP_not_erased_by_current_nested_validation': True,
        'application_guards_and_source_provenance_not_an_OS_information_flow_proof': True,
        'new_Shadow_Seal_reduced_public_cloud_value_usage': 0,
        'canonical_model_role_or_raw_authority_changes': 0}
    audit = {'status': 'PASS', 'scope': 'Computation, lineage, complete tests and protected-byte integrity only',
        'source_receipts_sha256': proof, 'unique_RP_series': full['unique_RP_series'],
        'online_points': full['online_points'], 'tests': tests['counts'],
        'all_required_computational_audits_passed': True,
        'model_promotion_assigned': False, 'archive_reproduction_and_final_time_contract_still_separate': True,
        'no_finished_delivery_claim_from_this_receipt': True}
    return deterministic, leakage, audit
