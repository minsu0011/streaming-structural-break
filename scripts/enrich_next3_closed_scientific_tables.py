"""Add inherited controls, runtime evidence and zero-fit diagnostic ledger rows."""
from pathlib import Path
import csv
import io
import statistics
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.result_tables import append_history, csv_bytes
from scripts.prepare_next3_closed_scientific_tables import Evidence


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream: return list(csv.DictReader(stream))


def main():
    source = BASE/'final_tables/prepared_v1'; proof = read_json(source/'RESULTS.json')
    assert proof['status'] == 'PREPARED_CLOSED_SCIENTIFIC_TABLES_NOT_FINAL_DELIVERY'
    for name, value in proof['files'].items(): assert digest(source/name) == value['sha256']
    evidence = Evidence(); evidence.files.update(proof['source_evidence_sha256'])
    for name, sha in evidence.files.items(): assert digest(ROOT/name) == sha
    out = BASE/'final_tables/prepared_v2'; out.mkdir(parents=True, exist_ok=False)
    tables = {name: rows(source/name) for name in proof['files'] if name.endswith('.csv') and name != 'EXPERIMENT_LEDGER.csv'}
    roles = {'J102_SERIAL_VARIANCE_512', 'BL2_C205_DEPENDENCE_80'}
    aliases = {'BL80': 'BL2_C205_DEPENDENCE_80'}
    for destination, old_name in (('COUNTEREXAMPLE_RESULTS_NEXT3.csv', 'COUNTEREXAMPLE_RESULTS.csv'),
                                  ('LONG_NULL_RESULTS_NEXT3.csv', 'LONG_NULL_RESULTS_NEXT2.csv')):
        for row in tables[destination]:
            row['scientific_candidate_id'] = aliases.get(row['candidate'], row['candidate'])
            row['source_candidate_label'] = row['candidate']
            row['evidence_origin'] = 'NEXT3_EXISTING_FIXED_SYNTHETIC_DIAGNOSTIC'
            if destination.startswith('LONG_NULL'): row['ts_auc'] = ''
        inherited = [row for row in evidence.csv(old_name) if row['candidate'] in roles
                     and row.get('artifact_kind') == 'frozen all-DEV model']
        assert inherited and {row['candidate'] for row in inherited} == roles
        for row in inherited:
            row.update(scientific_candidate_id=row['candidate'], source_candidate_label=row['candidate'],
                record_type='INHERITED_NEXT2_FROZEN_SYNTHETIC_DIAGNOSTIC',
                evidence_origin='NEXT2_FROZEN_NO_NEXT3_REFIT', source_table=old_name,
                source_table_sha256=evidence.files[old_name], new_fits_in_NEXT3=0,
                scientific_promotion_authority=False,
                scope_note='Original fixed scenarios, fitted models and alpha-specific thresholds. Distinct receipt rows are not pooled as independent samples.')
            if destination.startswith('LONG_NULL'):
                row['ts_auc'] = ''; row['background'] = row['regime']
            elif row['scenario'] in ('N0', 'N1', 'N2', 'N3', 'O1'):
                assert row['ts_auc'] == '', 'Null/outlier-only TS-AUC cannot be a number'
        tables[destination].extend(inherited)
    diagnostic_groups = [
        ('F001_PAIR_CONTRIBUTIONS', ['pair_contributions/RESULTS.json']),
        ('F002_SCALE_STAGE_TRACE', ['scale_stages/RESULTS.json']),
        ('D001_DIRECTIONAL_SCALE_FORENSIC', ['directional_scale/RESULTS.json']),
        ('D002_JOINT_CONDITIONAL_UNCERTAINTY', ['selection_uncertainty/RESULTS.json']),
        ('D003_FEATURE_CACHE_STREAMING', [p.relative_to(BASE).as_posix()
            for p in sorted((BASE/'maintenance/feature_streaming').glob('*/RESULTS.json'))]),
        ('D004_ROUTER_METRIC_ALIGNMENT', ['router_metric_alignment/RESULTS.json']),
        ('D005_OUTLIER_STATE_TRANSFER', ['outlier_state_transfer/run_v1/RESULTS.json']),
        ('D006_LEGACY_STATE_REFINEMENT', ['legacy_state_transfer/run_v1/RESULTS.json']),
        ('D007_PAIR_MATRIX_BOOTSTRAP_AUDIT', ['pair_matrix_audit/run_v1/RESULTS.json']),
    ]
    ledger_rows = []
    for name, paths in diagnostic_groups:
        hypothesis_relative = 'artifacts/next3/hypotheses/'+name+'.json'; hypothesis = evidence.json(hypothesis_relative)
        assert paths
        for path in paths:
            report = evidence.json('artifacts/next3/'+path); assert report['status'] == 'PASS'
        ledger_rows.append({'experiment_id': 'NEXT3_'+name, 'generation': 'NEXT3', 'phase': 'EVALUATION_ONLY_DIAGNOSTIC',
            'hypothesis': hypothesis['HYPOTHESIS'], 'change': hypothesis['CHANGE'],
            'expected_effect': hypothesis['EXPECTED_EFFECT'], 'risks': hypothesis['RISKS'],
            'status': 'PASS_DIAGNOSTIC_NOT_SCIENTIFIC_PROMOTION', 'final_disposition': 'DIAGNOSTIC_ONLY',
            'reason': 'Completed fixed diagnostic. Does not establish a deployable remedy or independent generalization.',
            'new_fits': 0, 'seal_rows': 0, 'reduced_new_uses': 0, 'selected_role': 'NONE',
            'ledger_section': 'NEXT3_ZERO_FIT_DIAGNOSTICS', 'evidence_origin': ';'.join('artifacts/next3/'+p for p in paths),
            'experiment_report_sha256': ';'.join(evidence.files['artifacts/next3/'+p] for p in paths)})
    engineering = [
        ('E001_PRECOMPUTED_SHARED', 'engineering/precomputed_shared/timing/RESULTS.json', 'REJECTED_FIXED_TIMING_GATE'),
        ('E002_PACKED_SHARED', 'engineering/packed_shared/timing/RESULTS.json', 'ENGINEERING_ELIGIBLE_PREDECESSOR'),
        ('E003_LEAN_SHARED', 'engineering/lean_shared/ENGINEERING_GATE_DECISION.json', 'PROVISIONAL_ENGINEERING_ELIGIBLE'),
    ]
    for name, path, disposition in engineering:
        hypothesis = evidence.json('artifacts/next3/hypotheses/'+name+'.json')
        record = evidence.json('artifacts/next3/'+path)
        assert record['status'] in ('PASS', 'PROVISIONAL_ENGINEERING_ELIGIBLE')
        if name.startswith('E001'): assert record['timing_gate_pass'] is False
        ledger_rows.append({'experiment_id': 'NEXT3_'+name, 'generation': 'NEXT3', 'phase': 'EXACT_RUNTIME_ENGINEERING',
            'hypothesis': hypothesis['HYPOTHESIS'], 'change': hypothesis['CHANGE'],
            'expected_effect': hypothesis['EXPECTED_EFFECT'], 'risks': hypothesis['RISKS'],
            'status': disposition, 'final_disposition': disposition,
            'reason': 'Exact representations of frozen science; final raw reproduction, core audit and archived replay are separate acceptance requirements.',
            'new_fits': 0, 'seal_rows': 0, 'reduced_new_uses': 0, 'selected_role': 'SAME_FROZEN_PERFORMANCE_SCIENCE',
            'ledger_section': 'NEXT3_ZERO_FIT_ENGINEERING', 'evidence_origin': 'artifacts/next3/'+path,
            'experiment_report_sha256': evidence.files['artifacts/next3/'+path]})
    for name, disposition in (('I1_SIGN_PRODUCTS', 'EXACT_PRIOR_FAILURE_REUSED_ZERO_FITS'),
                              ('S4_CONDITIONAL_RATIO_DOWN', 'DEFERRED_WITHOUT_FIT')):
        ledger_rows.append({'experiment_id': name, 'generation': 'NEXT3', 'phase': 'UNEXECUTED_MENU_ITEM',
            'status': disposition, 'final_disposition': disposition, 'new_fits': 0, 'seal_rows': 0, 'reduced_new_uses': 0,
            'selected_role': 'NONE', 'reason': 'Existing reuse/closure receipts; no new hypothesis result or fit invented.',
            'ledger_section': 'NEXT3_ZERO_FIT_REUSE_OR_DEFERRED',
            'evidence_origin': 'artifacts/next3/ledger/failure_index_v2_policy_schema/RESULTS.json'})
    assert len(ledger_rows) == 14 and all(row['new_fits'] == 0 for row in ledger_rows)
    ledger = append_history((source/'EXPERIMENT_LEDGER.csv').read_bytes(), ledger_rows)
    before = (BASE/'preflight/EXPERIMENT_LEDGER_BEFORE_NEXT3.csv').read_bytes(); assert ledger.startswith(before)
    ledger_parsed = list(csv.DictReader(io.StringIO(ledger.decode('utf-8-sig'))))
    assert len(ledger_parsed) == 242 and sum(int(row['new_fits']) for row in ledger_parsed[213:]) == 129
    cold_relative = 'artifacts/next3/engineering/lean_shared/cold/RESULTS.json'
    direct_relative = 'artifacts/next3/engineering/lean_shared/direct/RESULTS.json'
    cold = evidence.json(cold_relative); direct = evidence.json(direct_relative)
    assert cold['status'] == direct['status'] == 'PASS'
    assert cold['cold_cost_gate_pass'] and direct['direct_cost_gate_pass']
    assert len(cold['pairs']) == len(direct['pairs']) == 3
    assert cold['median_RSS_ratio'] == statistics.median(p['lean_RSS_bytes'] for p in cold['pairs'])/statistics.median(p['packed_RSS_bytes'] for p in cold['pairs'])
    assert cold['median_first_none_ratio'] == statistics.median(p['lean_first_none_seconds'] for p in cold['pairs'])/statistics.median(p['packed_first_none_seconds'] for p in cold['pairs'])
    assert direct['median_direct_10M_ratio'] == statistics.median(p['direct_10M_ratio'] for p in direct['pairs'])
    trials = []
    for number in range(3):
        relative = 'artifacts/next3/engineering/lean_shared/direct/trial_'+str(number)+'_LEAN_PACKED_SHARED/RESULTS.json'
        record = evidence.json(relative); assert record['status'] == 'PASS'
        assert direct['files_sha256']['trial_'+str(number)+'_LEAN_PACKED_SHARED/RESULTS.json'] == evidence.files[relative]
        assert record['continuous_stream']['points'] == 10_000_000
        assert record['continuous_stream']['state_array_bytes_before'] == record['continuous_stream']['state_array_bytes_after']
        assert record['continuous_stream']['measured_prediction_seconds'] == direct['pairs'][number]['lean_seconds']
        trials.append(record)
    latency_old = evidence.json('LATENCY_NEXT2.json'); distilled = latency_old['variants']['J_USED']
    assert distilled['status'] == 'PASS'
    memory_old = evidence.json('STATE_MEMORY_NEXT2.json')
    distilled_observations = [row for row in memory_old['observations'] if row['variant'] == 'J_USED']
    assert len(distilled_observations) >= 2
    memory_fields = ('private_per_series_numeric_bytes', 'shared_selector_bytes_reachable_from_state',
        'retained_state_numeric_bytes_including_shared_selectors', 'model_and_prepared_shared_numeric_bytes',
        'total_one_stream_plus_one_model_numeric_bytes', 'private_allocation_count', 'state_allocation_count')
    distilled_memory = {key: distilled_observations[0][key] for key in memory_fields}
    assert all(all(row[key] == distilled_memory[key] for key in memory_fields) for row in distilled_observations)
    performance_memory = trials[0]['numeric_memory']
    assert all(record['numeric_memory'] == performance_memory for record in trials)
    aggregate = {name: {key: statistics.median(record[name][key] for record in trials)
        for key in ('median_us', 'p95_us', 'p99_us', 'mean_us')} for name in ('point_distribution', 'historical_initialization')}
    performance_cost = {'runtime_variant': 'LEAN_PACKED_SHARED', 'measurement_origin': 'NEXT3_E003_FIXED_THREE_TRIALS',
        'three_actual_10M_seconds': [r['continuous_stream']['measured_prediction_seconds'] for r in trials],
        'median_actual_10M_seconds': statistics.median(r['continuous_stream']['measured_prediction_seconds'] for r in trials),
        'three_actual_1M_seconds': [r['continuous_stream']['measured_first_million_seconds'] for r in trials],
        'trial_distributions': [{key: r[key] for key in ('trial', 'point_distribution', 'historical_initialization')} for r in trials],
        'median_of_trial_summary_statistics': aggregate,
        'summary_rule': 'Median across three per-trial summaries; percentile-of-pooled-samples is not claimed.',
        'estimated_10M_seconds_from_point_median': aggregate['point_distribution']['median_us']*10,
        'estimated_10M_seconds_with_10000_H_from_sample_means': aggregate['point_distribution']['mean_us']*10+
            aggregate['historical_initialization']['mean_us']*.01,
        'median_direct_time_ratio_vs_E002': direct['median_direct_10M_ratio'],
        'direct_throughput_improvement_claimed': False,
        'cold_process': {'trials': cold['pairs'], 'ratio_of_median_postwarm_RSS_vs_E002': cold['median_RSS_ratio'],
            'ratio_of_median_first_none_seconds_vs_E002': cold['median_first_none_ratio'],
            'relative_postwarm_RSS_reduction': 1-cold['median_RSS_ratio'],
            'median_postwarm_RSS_bytes': statistics.median(p['lean_RSS_bytes'] for p in cold['pairs']),
            'median_first_none_seconds': statistics.median(p['lean_first_none_seconds'] for p in cold['pairs']),
            'comparison_reference': 'E002 PACKED_SHARED; not the original NEXT2 shared runtime',
            'distinct_from_validation_process_RSS': True}}
    latency = {'status': 'PREPARED_MEASURED_RUNTIME_EVIDENCE_NOT_FINAL_ACCEPTANCE', 'created_utc': utc_now(),
        'DISTILLED': {'runtime_variant': 'DISTILLED_tree_used_v1', 'measurement_origin': 'INHERITED_NEXT2_FIXED_RUNTIME_MEASUREMENT',
            **{key: distilled[key] for key in ('point_distribution', 'historical_initialization', 'continuous_stream',
                'h_plus_online_amortized_us_per_point', 'h_plus_online_extrapolated_10M_seconds')}},
        'PERFORMANCE': performance_cost, 'new_measurements_performed_by_this_report': 0,
        'limitations': ['Host, process, JIT, CPU and pinned dependency conditions matter.',
            'Direct 10M traverses one continuous stream; 10000 H initializations are a separate extrapolation.',
            'Cold process RSS includes Python/JIT/import allocations; retained numeric state is a separate quantity.',
            'No cloud SDK IPC or Linux timing measured.'],
        'evidence_sha256': {key: value for key, value in evidence.files.items() if key in
            (cold_relative, direct_relative, 'LATENCY_NEXT2.json') or '/direct/trial_' in key}}
    memory = {'status': 'PREPARED_MEASURED_STATE_EVIDENCE_NOT_FINAL_ACCEPTANCE', 'created_utc': utc_now(),
        'DISTILLED': {'runtime_variant': 'DISTILLED_tree_used_v1', 'measurement_origin': 'INHERITED_NEXT2',
            **distilled_memory, '10M_retained_numeric_growth_bytes': 0, 'canonical_model_file_bytes': 35180},
        'PERFORMANCE': {'runtime_variant': 'LEAN_PACKED_SHARED', 'measurement_origin': 'NEXT3_E003_THREE_FIXED_TRIALS',
            **performance_memory, '10M_retained_numeric_growth_bytes': 0, 'canonical_model_file_bytes': 68999},
        'boundaries': memory_old['boundaries'], 'new_measurements_performed_by_this_report': 0,
        'cold_process_RSS_report': 'LATENCY_NEXT3.json / PERFORMANCE / cold_process',
        'evidence_sha256': {key: value for key, value in evidence.files.items()
                           if key == 'STATE_MEMORY_NEXT2.json' or '/direct/trial_' in key}}
    files = {}
    for name, table in tables.items():
        with (out/name).open('xb') as stream: stream.write(csv_bytes(table))
        files[name] = {'sha256': digest(out/name), 'rows': len(table)}
    with (out/'EXPERIMENT_LEDGER.csv').open('xb') as stream: stream.write(ledger)
    files['EXPERIMENT_LEDGER.csv'] = {'sha256': digest(out/'EXPERIMENT_LEDGER.csv'), 'rows': len(ledger_parsed)}
    for name, value in (('LATENCY_NEXT3.json', latency), ('STATE_MEMORY_NEXT3.json', memory)):
        atomic_json(out/name, value, immutable=True); files[name] = {'sha256': digest(out/name)}
    result = {**proof, 'created_utc': utc_now(), 'files': files, 'source_evidence_sha256': evidence.files,
        'source_sha256': {**proof['source_sha256'], 'scripts/enrich_next3_closed_scientific_tables.py': digest(Path(__file__))},
        'previous_preparation_receipt_sha256': digest(source/'RESULTS.json'),
        'appended_total_ledger_rows': 29, 'zero_fit_engineering_rows': 3, 'zero_fit_diagnostic_rows': 9,
        'zero_fit_reuse_or_deferred_rows': 2, 'new_runtime_measurements_performed': 0,
        'metadata_refinement': 'Explicit candidate aliases, inherited two-role synthetic controls, ratio-of-medians cold cost, median-of-paired-ratios direct cost, and all zero-fit diagnostic/engineering ledger records.'}
    atomic_json(out/'RESULTS.json', result, immutable=True)
    progress('Closed scientific tables enriched', '29 new ledger records preserve 213 old rows and 129 actual fits; runtime and inherited synthetic controls bound to existing evidence')
    print('NEXT3 ENRICHED TABLES PREPARED', len(files), 'files;', len(ledger_parsed), 'ledger rows; no new fit or final acceptance', flush=True)


if __name__ == '__main__': main()
