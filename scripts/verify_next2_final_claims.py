"""Cross-check narrative numbers, literal gates, preserved histories and final-role evidence."""
from pathlib import Path
import sys
import json
import re
import ast
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now
from src.next2.session import clock


def main():
    assert clock()['audit_started']
    out = ROOT/'artifacts/next2/final_claims'; out.mkdir(parents=True, exist_ok=True)
    selection = json.loads((ROOT/'DEV_CANDIDATES_NEXT2.json').read_text())
    performance = selection['performance']; distilled = selection['distilled']
    claims = []; sources = {}
    def claim(name, value, expected, source, location, tolerance=1e-14):
        assert np.isclose(float(value), float(expected), atol=tolerance, rtol=0), (name, value, expected)
        claims.append({'claim': name, 'value': value, 'cross_checked_value': expected, 'source': source, 'location': location, 'tolerance': tolerance})
        sources[source] = sha256(ROOT/source)
    full_path = 'artifacts/next2/experiments/BL2_C205_DEPENDENCE_80/RESULTS.json'
    full = json.loads((ROOT/full_path).read_text())['full']
    cv = pd.read_csv(ROOT/'CV_RESULTS_NEXT2.csv', float_precision='round_trip')
    chosen = cv[(cv.experiment_id == performance['experiment_id']) & (cv.scope == 'full')].sort_values('fold')
    np.testing.assert_array_equal(chosen.ts_auc.to_numpy(), full['fold_scores'])
    for key, expected in [('mean', np.mean(full['fold_scores'])), ('median', np.median(full['fold_scores'])), ('worst_fold', np.min(full['fold_scores']))]:
        claim('best_'+key, performance['full'][key], expected, full_path, 'full.'+key)
    primary = selection['previous_primary']['full']
    claim('primary_mean', primary['mean'], .62163175031248, 'artifacts/next2/initialization/FROZEN_REFERENCES.json', 'references.FROZEN_PRIMARY.full.mean')
    claim('mean_delta', performance['full']['mean']-primary['mean'], .0016366625048295225, 'PAIRED_BOOTSTRAP_NEXT2.csv', 'BL2_C205_DEPENDENCE_80/J102/all_time.estimate_mean_delta')
    claim('distilled_quality_delta', distilled['full']['mean']-primary['mean'], 0., 'DEV_CANDIDATES_NEXT2.json', 'distilled.full.mean')
    pair = performance['paired_all_time']
    for key, expected in [('ci_2_5', .0006743151264927167), ('ci_97_5', .0025479387096717654), ('probability_delta_positive', .999)]:
        claim('best_bootstrap_'+key, pair[key], expected, 'PAIRED_BOOTSTRAP_NEXT2.csv', 'best/all_time.'+key)
    runtime = pd.read_csv(ROOT/'RUNTIME_PRUNING_RESULTS.csv', float_precision='round_trip').set_index('runtime_variant')
    assert len(runtime) == 12
    for variant, values in [('J_USED', {'point_median_us': 5.1, 'historical_init_median_us': 4639.25,
            'private_per_series_numeric_bytes': 11729, 'direct_10M_seconds': 50.7685899}),
        ('BL80_SHARED_V1', {'point_median_us': 9.3, 'historical_init_median_us': 5140.05,
            'private_per_series_numeric_bytes': 16441, 'direct_10M_seconds': 93.1921636})]:
        receipt = json.loads((ROOT/'artifacts/next2/latency'/variant/'RESULTS.json').read_text())
        for key, expected in values.items(): claim(variant+'_'+key, runtime.loc[variant, key], expected, 'RUNTIME_PRUNING_RESULTS.csv', variant+'.'+key, 1e-9)
        claim(variant+'_direct10M_receipt', runtime.loc[variant, 'direct_10M_seconds'], receipt['continuous_stream']['measured_prediction_seconds'],
            'artifacts/next2/latency/'+variant+'/RESULTS.json', 'continuous_stream.measured_prediction_seconds')
        assert receipt['continuous_stream']['state_array_bytes_before'] == receipt['continuous_stream']['state_array_bytes_after']
    claim('distilled_input_reduction_percent', 100*(1-48/77), 37.66233766233766, 'DEV_CANDIDATES_NEXT2.json', 'distilled.materialized_tree_inputs')
    claim('distilled_private_reduction_percent', 100*(1-11729/29173), 100*runtime.loc['J_USED', 'private_state_reduction_fraction'], 'RUNTIME_PRUNING_RESULTS.csv', 'J_ORIGINAL/J_USED.private_per_series_numeric_bytes', 1e-12)
    counter = pd.read_csv(ROOT/'COUNTEREXAMPLE_RESULTS.csv', float_precision='round_trip')
    all_dev = counter[(counter.artifact_kind == 'frozen all-DEV model') & (counter.alpha == .05)]
    for eid, expected_values in [('J102_SERIAL_VARIANCE_512', {'V3': .473043, 'V4': .588660}),
            ('BL2_C205_DEPENDENCE_80', {'V3': .462974, 'V4': .571729})]:
        for scenario, expected in expected_values.items():
            row = all_dev[(all_dev.candidate == eid) & (all_dev.scenario == scenario)]; assert len(row) == 1
            claim(eid+'_'+scenario, row.iloc[0].ts_auc, expected, 'COUNTEREXAMPLE_RESULTS.csv', eid+'/'+scenario+'/all_DEV/alpha.05', 1e-6)
    for eid, expected in [('J102_SERIAL_VARIANCE_512', .693359), ('BL2_C205_DEPENDENCE_80', .744141)]:
        row = all_dev[(all_dev.candidate == eid) & (all_dev.scenario == 'O1')]; assert len(row) == 1
        assert np.isnan(row.iloc[0].ts_auc)
        claim(eid+'_outlier_excess', row.iloc[0].outlier_alarm_excess, expected, 'COUNTEREXAMPLE_RESULTS.csv', eid+'/O1/all_DEV/alpha.05', 1e-6)
    assert counter[counter.scenario.str.startswith('N') | counter.scenario.eq('O1')].ts_auc.isna().all()
    nulls = pd.read_csv(ROOT/'LONG_NULL_RESULTS_NEXT2.csv', float_precision='round_trip')
    for eid, horizon, expected in [('J102_SERIAL_VARIANCE_512', 1024, .037109375),
        ('J102_SERIAL_VARIANCE_512', 4096, .12890625), ('J102_SERIAL_VARIANCE_512', 16384, .322265625),
        ('BL2_C205_DEPENDENCE_80', 16384, .330078125)]:
        row = nulls[(nulls.candidate == eid) & (nulls.regime == 'N0') & (nulls.horizon == horizon) &
            (nulls.alpha == .05) & (nulls.artifact_kind == 'frozen all-DEV model')]; assert len(row) == 1
        claim(eid+'_null_'+str(horizon), row.iloc[0].eventual_alarm_rate, expected, 'LONG_NULL_RESULTS_NEXT2.csv', eid+'/N0/'+str(horizon)+'/all_DEV/alpha.05')
    late_path = 'artifacts/next2/late_alarm_diagnostic/N0_ONSET_49152/ANALYSIS.csv'
    late = pd.read_csv(ROOT/late_path, float_precision='round_trip')
    for eid, expected in [('J102_SERIAL_VARIANCE_512', .40625), ('BL2_C205_DEPENDENCE_80', .3203125)]:
        for beta in (0., .5, 1.):
            row = late[(late.candidate == eid) & (late.beta == beta) & (late.alpha == .05)]; assert len(row) == 1
            claim(eid+'_late_clean_beta'+str(beta), row.iloc[0].postbreak_first_detection_rate, expected if beta == 0 else 0., late_path, eid+'/beta'+str(beta)+'/alpha.05')
    correlation = pd.read_csv(ROOT/'COMPLEMENT_CORRELATION.csv', float_precision='round_trip')
    corr = correlation[(correlation.candidate == 'C205_DEPENDENCE') & (correlation.scope == 'full')]
    assert len(corr) == 5
    claim('C205_admission_rank_correlation', corr.within_time_rank_correlation.mean(), .42193977197036003, 'COMPLEMENT_CORRELATION.csv', 'C205/full/mean_of_5_folds')
    for alternate, expected in zip(performance['alternate_splits'], [.622572491375891, .6211655235665667, .6234311259085243]):
        claim('best_'+alternate['split'], alternate['mean'], expected, 'DEV_CANDIDATES_NEXT2.json', 'performance.alternate_splits.'+alternate['split'])
    claim('best_H_holdout_worst', performance['heldout_H']['worst'], .6006542695774002, 'DEV_CANDIDATES_NEXT2.json', 'performance.heldout_H.worst')
    mechanism = pd.read_csv(ROOT/'MECHANISM_RESULTS_NEXT2.csv', float_precision='round_trip')
    for signature, expected in [('DEPENDENCE_DOMINANT', .004261), ('SCALE_DOMINANT', -.004355)]:
        row = mechanism[(mechanism.candidate == performance['experiment_id']) & (mechanism.reference == 'J102_SERIAL_VARIANCE_512') & (mechanism.scope == 'mechanism_'+signature)]; assert len(row) == 1 and row.iloc[0].supported == True
        claim('best_mechanism_'+signature, row.iloc[0].estimate_mean_delta, expected, 'MECHANISM_RESULTS_NEXT2.csv', signature, 1e-6)
    models = pd.read_csv(ROOT/'MODEL_RESULTS_NEXT2.csv', float_precision='round_trip')
    assert len(models) == models.experiment_id.nunique() == 80
    assert int(models.evidence_origin.eq('NEXT2_RESEARCH').sum()) == 71
    assert models.full_mean.max() == full['mean'] and models.final_disposition.notna().all()
    assert performance['gates']['performance']['status'] == 'PASS' and distilled['gates']['distilled']['status'] == 'PASS'
    assert performance['gates']['distilled']['status'] == 'PASS'
    assert not selection['generation_level_promotion'] and full['mean']-primary['mean'] < .0025
    assert all(v is True for k, v in performance['gates']['checks'].items() if k.startswith('performance_'))
    history = pd.read_csv(ROOT/'artifacts/next2/initialization/EXPERIMENT_LEDGER_NEXT_PRESERVED.csv', float_precision='round_trip')
    ledger = pd.read_csv(ROOT/'EXPERIMENT_LEDGER.csv', float_precision='round_trip')
    old = ledger[ledger.ledger_section == 'PREVIOUS_SESSION_PRESERVED'].reset_index(drop=True)
    comparable = old[history.columns].copy()
    normalized_numeric_columns = []
    for column in history:
        # New textual generation labels make the combined CSV column textual;
        # apply the preserved history's own schema before exact value comparison.
        if pd.api.types.is_numeric_dtype(history[column].dtype) and not pd.api.types.is_bool_dtype(history[column].dtype):
            comparable[column] = pd.to_numeric(comparable[column], errors='raise')
            normalized_numeric_columns.append(column)
        elif pd.api.types.is_bool_dtype(history[column].dtype) and not pd.api.types.is_bool_dtype(comparable[column].dtype):
            assert comparable[column].isin(['True', 'False', True, False]).all()
            comparable[column] = comparable[column].map({'True': True, 'False': False, True: True, False: False})
    pd.testing.assert_frame_equal(comparable, history, check_dtype=False, check_exact=True)
    inventory_path = 'artifacts/next2/final_tables/FINAL_TRAINING_INVENTORY.json'
    inventory = json.loads((ROOT/inventory_path).read_text()); claim('learner_fits', inventory['total_NEXT2_learner_fits'], 397, inventory_path, 'total_NEXT2_learner_fits', 0.)
    text = (ROOT/'00_GPT_READ_FIRST_NEXT2.md').read_text(encoding='utf-8')
    assert [int(x) for x in re.findall(r'(?m)^(\d+)\. \*\*', text)] == list(range(1, 36))
    assert repr(full['mean']) in text and repr(full['mean']-primary['mean']) in text
    for path in [ROOT/p for p in ('00_GPT_READ_FIRST_NEXT2.md', '01_GPT_HANDOFF_NEXT2.md', 'CURRENT_GENERATION_COMPONENT_GRAPH.md', 'LIMITATIONS_NEXT2.md', 'NEXT_SESSION_PLAN_NEXT2.md')]:
        assert '\ufffd' not in path.read_text(encoding='utf-8'); sources[path.relative_to(ROOT).as_posix()] = sha256(path)
    suites = list(ET.parse(ROOT/'artifacts/next2/final_audit/ALL_TESTS.xml').getroot().iter('testsuite'))
    assert sum(int(s.attrib['tests']) for s in suites) == 536
    assert all(int(s.attrib.get(k, 0)) == 0 for s in suites for k in ('errors', 'failures', 'skipped'))
    files = list((ROOT/'src/next2').glob('*.py'))+list((ROOT/'scripts').glob('*next2*.py'))+list((ROOT/'tests').glob('test_next2*.py'))+list(ROOT.glob('next2_*.py'))
    for path in files: ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    portable = json.loads((ROOT/'artifacts/next2/handoff/portable_copy/RESULTS.json').read_text()); assert portable['status'] == 'PASS'
    for module in portable['project_modules'].values(): assert sha256(ROOT/module['file']) == module['sha256']
    pd.DataFrame(claims).to_csv(out/'CLAIM_PROVENANCE.csv', index=False)
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'script_sha256': sha256(Path(__file__)),
        'numeric_cross_checks': len(claims), 'claim_table_sha256': sha256(out/'CLAIM_PROVENANCE.csv'), 'source_sha256': sources,
        'numbered_Korean_answers': 35, 'source_files_AST_checked': len(files), 'full_tests': 536,
        'previous_ledger_rows_unchanged': len(history), 'history_numeric_columns_parsed_using_original_schema': normalized_numeric_columns,
        'portable_executed_project_modules_still_identical': len(portable['project_modules']),
        'literal_performance_distilled_generation_gates_consistent': True})
    print('NEXT2 FINAL CLAIM CROSS-CHECKS PASS', len(claims), flush=True)


if __name__ == '__main__': main()
