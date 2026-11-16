"""Consolidate only fully completed diagnostic cohorts into final result tables."""
from pathlib import Path
import argparse
import csv
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, note, require_guard
from src.next6_extension.synthetic import CASES


def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def completed(directory, inputs):
    receipt = read_json(directory / 'DONE.json')
    assert receipt['status'] == 'COMPLETE'
    for name, sha in receipt.get('files_sha256', receipt.get('source_files_sha256', {})).items():
        assert digest(directory / name) == sha
    inputs[(directory / 'DONE.json').relative_to(BASE).as_posix()] = digest(directory / 'DONE.json')
    return receipt


def emit(kind, tables, inputs):
    for name, records in tables.items():
        assert records, 'An advertised completed table cannot be empty: ' + name
        csv_write(name, records)
    save('COMPLETE_DIAGNOSTICS_' + kind.upper() + '_NEXT7.json', {
        'status': 'COMPLETE', 'created_utc': utc_now(), 'kind': kind,
        'source_sha256': digest(__file__), 'inputs_sha256': inputs,
        'outputs_sha256': {name: digest(BASE / name) for name in tables},
        'row_counts': {name: len(records) for name, records in tables.items()},
        'new_fits': 0, 'new_draws': 0, 'partial_cohorts_included': False})


def gradients():
    inputs, summaries, coordinates = {}, [], []
    for fold in range(5):
        directory = BASE / f'g3_full_gradient/fold_{fold}'
        receipt = completed(directory, inputs)
        assert receipt['original_sampled_optimizer_objective_and_gradient_parity']
        assert receipt['fold'] == fold and receipt['new_fits'] == 0
        assert digest(ROOT / 'scripts/next7/full_g3_gradient.py') == receipt['source_sha256']
        for point in receipt['per_time_receipts']:
            assert digest(directory / f"time_{point['time']:04d}.npz") == point['file_sha256']
        objective_path = directory / 'OBJECTIVE_SUMMARY.csv'
        gradient_path = directory / 'GRADIENT_COORDINATES.csv'
        objective = rows(objective_path)
        assert {row['beta'] for row in objective} == {'ZERO', 'G2_PARENT', 'G3_FITTED'}
        assert len(objective) == 3 and all(int(row['fold']) == fold for row in objective)
        inputs[objective_path.relative_to(BASE).as_posix()] = digest(objective_path)
        inputs[gradient_path.relative_to(BASE).as_posix()] = digest(gradient_path)
        summaries.extend(objective)
        coordinates.extend(rows(gradient_path))
    emit('gradients', {'G3_ALL_PAIRS_OBJECTIVE_DIAGNOSTICS_NEXT7.csv': summaries,
                      'G3_ALL_PAIRS_GRADIENT_DIAGNOSTICS_NEXT7.csv': coordinates}, inputs)


def conditional():
    inputs = {}
    tables = {name: [] for name in ('CONDITIONAL_NULL_RESULTS_NEXT7.csv', 'PER_H_CONDITIONAL_NULL_RISK_NEXT7.csv',
        'EMPIRICAL_SUPPORT_PARTITION_NEXT7.csv', 'CROSSED_H_LENGTH_CONTRASTS_NEXT7.csv',
        'CROSSED_H_LENGTH_DISCORDANCE_NEXT7.csv', 'H_NORMALIZER_RISK_ASSOCIATIONS_NEXT7.csv')}
    for case in ('N0', 'N1', 'N2', 'N3'):
        for length in (1000, 2048, 5000):
            stats = BASE / f'conditional_statistics/{case}/H_{length}'
            support = BASE / f'support_partition/{case}/H_{length}'
            normalizers = BASE / f'normalizer_transport/correlations/{case}/H_{length}'
            for directory in (stats, support, normalizers):
                completed(directory, inputs)
            tables['CONDITIONAL_NULL_RESULTS_NEXT7.csv'].extend(rows(stats / 'SUMMARY.csv'))
            tables['PER_H_CONDITIONAL_NULL_RISK_NEXT7.csv'].extend(rows(stats / 'PER_H_CONDITIONAL_RISK.csv'))
            tables['EMPIRICAL_SUPPORT_PARTITION_NEXT7.csv'].extend(rows(support / 'SUMMARY.csv'))
            tables['H_NORMALIZER_RISK_ASSOCIATIONS_NEXT7.csv'].extend(rows(normalizers / 'CORRELATIONS.csv'))
        paired = BASE / f'crossed_H_length_comparison/{case}'
        completed(paired, inputs)
        tables['CROSSED_H_LENGTH_CONTRASTS_NEXT7.csv'].extend(rows(paired / 'PAIRED_H_LENGTH_CONTRASTS.csv'))
        tables['CROSSED_H_LENGTH_DISCORDANCE_NEXT7.csv'].extend(rows(paired / 'QUALIFIED_DECISION_DISCORDANCE.csv'))
    assert len(tables['CONDITIONAL_NULL_RESULTS_NEXT7.csv']) == 12 * 3 * 5
    assert len(tables['EMPIRICAL_SUPPORT_PARTITION_NEXT7.csv']) == 12 * 3 * 3
    assert len(tables['H_NORMALIZER_RISK_ASSOCIATIONS_NEXT7.csv']) == 12 * 3 * 23
    emit('conditional', tables, inputs)


def classifiers():
    inputs, individual, paired, amplitude = {}, [], [], []
    for case in CASES:
        directory = BASE / 'classifier_statistics' / case
        receipt = completed(directory, inputs)
        assert receipt['fixed_folds'] == 5 and receipt['draws'] == 2000
        assert receipt['own_cut_amplitude_alarm_metrics_exact_M0']
        assert not receipt['fold_resampling'] and receipt['undefined_support_not_imputed']
        case_rows = rows(directory / 'SUMMARY.csv')
        assert all(row['case'] == case and row['fixed_folds'] == '5' for row in case_rows)
        individual.extend(case_rows)
        paired.extend(rows(directory / 'PAIRED_CONTRASTS.csv'))
        amplitude.extend(rows(directory / 'AMPLITUDE_CONTROL.csv'))
    assert set(row['case'] for row in individual) == set(CASES)
    tables = {'CLASSIFIER_STRESS_RESULTS_NEXT7.csv': individual,
              'CLASSIFIER_STRESS_PAIRED_NEXT7.csv': paired,
              'CLASSIFIER_AMPLITUDE_CONTROL_NEXT7.csv': amplitude,
              'CLASSIFIER_LONG_NULL_RESULTS_NEXT7.csv': [row for row in individual if row['case'] in ('N0', 'N1', 'N2', 'N3')]}
    state_files = {'SCALE_EXPANSION_CONTRACTION_RESULTS.csv': 'SCALE_UP_DOWN_RESULTS_NEXT7.csv',
                   'SHOCK_RESULTS_NEXT7.csv': 'SHOCK_EVENT_CONTAMINATION_NEXT7.csv',
                   'FALSE_LATCH_RESULTS_NEXT7.csv': 'LONG_NULL_RESULTS_NEXT7.csv'}
    selectors = {'SCALE_EXPANSION_CONTRACTION_RESULTS.csv': {'V1', 'V2', 'V3', 'V4'},
                 'SHOCK_RESULTS_NEXT7.csv': {'O1', 'OB16', 'REPEATED_SHOCK'},
                 'FALSE_LATCH_RESULTS_NEXT7.csv': {'N0', 'N1', 'N2', 'N3'}}
    for output, state_source in state_files.items():
        inputs[state_source] = digest(BASE / state_source)
        tables[output] = [{'record_type': 'STATE_EVENT_MEMORY', **row,
            'source_table': state_source, 'classifier_alarm_interpretation': False} for row in rows(BASE / state_source)]
        tables[output] += [{'record_type': 'OWN_CALIBRATED_CLASSIFIER', **row,
            'source_table': 'CLASSIFIER_STRESS_RESULTS_NEXT7.csv', 'classifier_alarm_interpretation': True}
            for row in individual if row['case'] in selectors[output]]
    emit('classifiers', tables, inputs)


def main(kind):
    require_guard()
    with activity('Consolidate all completed ' + kind + ' diagnostics', 'FINAL_RESEARCH_SYNTHESIS', 'FIXED_RESULT_POSTPROCESSING'):
        {'gradients': gradients, 'conditional': conditional, 'classifiers': classifiers}[kind]()
    note('Complete diagnostic family consolidated', kind + ': all required cohorts present; no new statistical draws or fits', 'FIXED_RESULT_POSTPROCESSING')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', required=True, choices=['gradients', 'conditional', 'classifiers'])
    args = parser.parse_args()
    main(args.kind)
