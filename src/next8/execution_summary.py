"""Count actual recorded fits and summarize measured execution, not model gain."""
from collections import Counter
import numpy as np
from src.next8.common import BASE, read_json, digest
from src.next8.protocol import H_BUCKETS
from src.next8.runtime_audit import latency_summary


def fit_inventory():
    counts = Counter()
    evidence = []
    joint = []

    def add(path, category, fits, reused=0, detail=None):
        assert fits >= 0 and reused >= 0
        counts[category] += fits
        evidence.append({'path': path.relative_to(BASE).as_posix(), 'sha256': digest(path),
                         'category': category, 'new_estimator_training_calls': fits,
                         'reused_or_alias_objects': reused, 'detail': detail})

    # These feature-level receipts also cover any conditional extra early meta
    # and auxiliary leaf procedures. Summary copies are not input receipts.
    for path in sorted(BASE.rglob('F*.json')):
        if not path.parent.name.startswith('split_'):
            continue
        record = read_json(path)
        if 'scores' in record and 'atlas' in record and 'conditional' in record:
            count = 3 + len(record['conditional'].get('models', {}))
            add(path, 'INNER_FEATURE_SCREEN', count, detail='Three O0/O1/context fits plus actually supported conditional model paths')
        elif 'H_atlas' in record and 'reconstructed_O1' in record:
            count = int(record['reconstructed_O1'] is not None) + sum(len(c.get('models', {})) for c in record['conditional'].values())
            add(path, 'INNER_H_CONTEXT', count, detail='Only newly reconstructed O1 or actually fit H-conditional models; reused OWN evidence excluded')
        elif 'mechanical_feature_columns' in record and 'candidates' in record:
            count = 1 + sum(c['edge_type'] == 'CROSS_PREDICTIVE_CANDIDATE' for c in record['candidates'])
            add(path, 'INNER_PARENT_SCREEN', count, detail='One strong mechanical ridge plus each actually tested cross-parent ridge')
    roots = ('forecasts', 'transitions', 'state_forecasts', 'graph_states', 'meta_state_inputs',
             'early_state_inputs', 'models', 'orientation_refits', 'baseline_conditioned')
    for root in roots:
        for path in sorted((BASE / root).rglob('DONE.json')):
            r = read_json(path)
            if r.get('status') != 'COMPLETE':
                continue
            if 'method_aliases' in r and 'D0_policy' in r:
                fits = sum(sum(v.get('status') != 'NO_SUPPORTED_H_STATE' for v in item['targets'].values()) for item in r['fitting'])
                fits += int('T2_state.joblib' in r['files_sha256'])
                reused = sum(name != actual for name, actual in r['method_aliases'].items())
                add(path, 'OUTER_MOVEMENT_COMPARATORS', fits, reused, 'Fit count is individual estimators; reuse count is logical D-bundle aliases. A saved bundle contains separate delta/movement/reversal/known-state estimators; D0 is a direct analytic baseline')
            elif 'T4_alias_T3_identical_parents' in r:
                add(path, 'OUTER_TRANSITION_TABLES', 1 + int(not r['T4_alias_T3_identical_parents']),
                    1 + int(r['T4_alias_T3_identical_parents']), 'T1 is marginal use of fitted T3; identical T4 does not refit')
            elif isinstance(r.get('models'), list) and ('state_estimator' in r or 'pre_current_F_only' in r):
                complete = [v for v in r['models'] if v['status'] == 'COMPLETE']
                add(path, 'CURRENT_STATE_T3' if 'state_estimator' in r else 'CURRENT_STATE_RIDGE',
                    sum(v['new_fit'] for v in complete), sum(not v['new_fit'] for v in complete))
            elif 'model_aliases' in r and 'fitting' in r and 'methods' in r:
                fitted = [v for v in r['fitting'] if v['alias'] is None]
                add(path, 'FINAL_M_HEADS', len(fitted), len(r['fitting']) - len(fitted))
                joint.append({'scope': r['scope']['scope'], 'requested_methods': r['methods'],
                              'model_aliases': r['model_aliases'], 'new_fitted_heads': len(fitted),
                              'active_columns_by_fitted_head': {v['method']: v['active_columns'] for v in fitted},
                              'stable_features': r['stable_features'], 'variable_features': r['variable_features'],
                              'training_rows': r['training_rows'], 'prediction_rows': r['prediction_rows'],
                              'scope_sha256': digest(path)})
            elif 'models_training' in r or 'model_training' in r:
                # Count only direct files: parent orientation receipts also
                # contain nested conditional paths, which have their own DONE.
                count = sum(name.endswith('.joblib') and '/' not in name for name in r['files_sha256'])
                add(path, 'BASELINE_CONDITIONAL_INFORMATION' if root == 'baseline_conditioned' else 'OUTER_ORIENTATION_REFITS', count)
    for root in ('early_baseline', 'early_integration'):
        for path in sorted((BASE / root).rglob('model.joblib')):
            if root == 'early_integration' and 'baseline' not in path.parts:
                continue
            add(path, 'EARLY_REFERENCE_LIGHTGBM', 1)
    control = read_json(BASE / 'negative_controls/DONE.json')
    add(BASE / 'negative_controls/DONE.json', 'SYNTHETIC_METHODOLOGICAL_CONTROL',
        sum(r.get('fitted_control_models', 0) for r in control['records']))
    integrations = []
    for root in ('integration', 'early_integration'):
        for path in sorted((BASE / root).glob('*/DONE.json')):
            r = read_json(path)
            integrations.append({'scope': r['scope']['scope'], 'eligible': r['eligible'],
                                 'M6_status': r['M6_status'], 'new_blend_weight_fits': 0, 'receipt_sha256': digest(path)})
    return {'status': 'COMPLETE', 'fit_call_counts_by_stage': dict(counts),
            'total_recorded_new_estimator_training_calls': sum(counts.values()), 'fit_count_evidence': evidence,
            'declared_scientific_classifier_families': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'],
            'final_head_procedures': joint, 'M6_qualification_procedures': integrations,
            'unique_final_head_training_calls': counts['FINAL_M_HEADS'],
            'aliases_are_not_additional_fitted_scientific_candidates': True,
            'a_training_call_can_be_analytic_ridge_pseudocount_or_constant_class_fit': True,
            'reference_normal_models_are_reused_not_new_fits': True,
            'feature_H_normalization_and_population_quantile_fits_not_counted_as_estimators': True,
            'D0_direct_persistence_majority_calculations_excluded_from_estimator_count': True,
            'tests_and_debug_fixture_fits_excluded_from_scientific_run_count': True,
            'failed_or_uncommitted_partial_fit_attempts_not_in_successful_receipt_total': True,
            'large_training_call_count_is_not_a_count_of_independent_scientific_trials': True}


def summarize_latency():
    cases = []
    sources = {}
    for early in (False, True):
        for fold in range(5):
            name = f'outer_{fold}' + ('_EARLY_T199' if early else '')
            parent = BASE / 'runtime_validation' / name
            done = read_json(parent / 'DONE.json')
            assert done['status'] == 'COMPLETE' and done['parity_status'] == 'PASS'
            sources[name] = digest(parent / 'DONE.json')
            for group in done['cases']:
                for sid in group['selected_ids']:
                    p = parent / f'ID{sid}'
                    assert digest(p / 'DONE.json') == done['files_sha256'][f'ID{sid}/DONE.json']
                    r = read_json(p / 'DONE.json')
                    assert r['status'] == 'PASS' and r['dataset_id'] == sid
                    assert digest(p / 'TIMING.npz') == r['timing_sha256']
                    with np.load(p / 'TIMING.npz', allow_pickle=False) as values:
                        times = {key: values[key].copy() for key in values.files}
                    assert all(len(v) == r['online_points'] for v in times.values())
                    cases.append({'scope': name, 'early_only': early, 'fold': fold,
                                  'H_bin': group['H_bin'], 'record': r, 'times': times})
    rows = []
    for early in (False, True):
        for bucket in (None, *range(len(H_BUCKETS))):
            available = [c for c in cases if c['early_only'] == early and (bucket is None or c['H_bin'] == bucket)]
            for method in ('M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6'):
                matched = [c for c in available if method in c['times']]
                base = {'early_only': early, 'H_bin': 'ALL' if bucket is None else bucket,
                        'method': method, 'measured_series': len(matched), 'available_series': len(available),
                        'initialization_in_point_latency': False}
                if not matched:
                    rows.append({**base, 'status': 'NOT_MEASURED', 'runtime_ratio_gate': None,
                                 'reason': 'No separately measured eligible path; missing timing is not replaced by a BL80 estimate'})
                    continue
                result = latency_summary(np.concatenate([c['times'][method] for c in matched]))
                baseline = latency_summary(np.concatenate([c['times']['M0'] for c in matched]))
                median = result['median_microseconds'] / baseline['median_microseconds']
                p99 = result['p99_microseconds'] / baseline['p99_microseconds']
                initialization = [c['record']['initialization_nanoseconds'][method] / 1e6 for c in matched]
                memory = [c['record']['instance_array_bytes'].get(method) for c in matched]
                measured_memory = [v for v in memory if v is not None]
                rows.append({**base, 'status': 'MEASURED', **result, 'median_ratio_matched_BL80': median,
                             'p99_ratio_matched_BL80': p99, 'runtime_ratio_gate': median <= 3 and p99 <= 5,
                             'initialization_median_ms': float(np.median(initialization)),
                             'initialization_max_ms': max(initialization),
                             'tracked_instance_array_bytes_max': max(measured_memory) if measured_memory else None,
                             'array_memory_not_total_process_RSS': True,
                             'tracked_memory_series': len(measured_memory),
                             'M6_measured_eligible_subset_only': method == 'M6',
                             'unmeasured_M6_ineligible_wrapper_overhead_not_inferred': method == 'M6'})
    unique = sorted({c['record']['dataset_id'] for c in cases})
    return {'status': 'COMPLETE', 'rows': rows, 'scope_receipts_sha256': sources,
            'sample_protocol': 'First2 validation IDs in each of5 fixed H-length bins/fold; same IDs normal and early',
            'unique_RP_series': len(unique), 'procedure_series_cases': len(cases),
            'measured_online_points_across_procedures': sum(c['record']['online_points'] for c in cases),
            'same_series_normal_early_not_independent_replications': True,
            'measurement': 'Standalone warmed one-thread scalar APIs with actual concurrent research background; no isolated CPU claim',
            'latency_gate': {'median_ratio_BL80_max': 3, 'p99_ratio_BL80_max': 5},
            'pooling': 'Per-point pooled distribution; each candidate uses identical measured baseline case set',
            'initialization_and_memory_reported_separately': True,
            'model_settings_not_changed_by_timing_results': True,
            'full_population_latency_or_peak_process_memory_not_claimed': True}
