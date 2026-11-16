"""Read already-completed scientific evidence for the final Korean reports.

This module does not load observations, fit models or publish final acceptance.
"""
from pathlib import Path
import csv
import datetime
from src.next3.artifacts import BASE, ROOT, digest, read_json, utc_now
from src.next3.final_validation import prepared_table_proof, successful_receipt


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def exactly_one(values, **conditions):
    matched = [row for row in values if all(row.get(key) == value for key, value in conditions.items())]
    if len(matched) != 1:
        raise ValueError(f'Expected one report evidence row: {conditions}; got {len(matched)}')
    return matched[0]


def maintenance_facts():
    states = [read_json(path) for path in (BASE/'jobs').glob('*/STATE.json')]
    failures = [path for path in (BASE/'jobs').glob('*/attempt_*/RESULT.json')
                if read_json(path)['status'] == 'FAILED']
    registration_rejections = [BASE/'maintenance/CHECKPOINT_09_REGISTRATION_CORRECTION.json']
    return {
        'failed_child_attempts': len(failures),
        'prelaunch_registration_bugs': sum(record.get('attempt') == 0 and
            record.get('status') == 'QUARANTINED' and
            record.get('terminal_kind') != 'SUPERSEDED_WITHOUT_EXECUTION' for record in states),
        'additional_dependency_guard_prelaunch_rejections': sum(path.is_file() for path in registration_rejections),
        'superseded_without_execution': sum(record.get('terminal_kind') == 'SUPERSEDED_WITHOUT_EXECUTION' for record in states),
        'supervisor_errors': len(list((BASE/'runtime/errors').glob('*.json'))),
        'planned_supervisor_restarts': len(list((BASE/'runtime/maintenance_updates').glob('SUPERVISOR_*_REPLACED.json'))),
        'failed_child_receipts_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in failures},
        'additional_prelaunch_receipts_sha256': {path.relative_to(ROOT).as_posix(): digest(path)
            for path in registration_rejections if path.is_file()},
        'note': 'Prelaunch rejections and intentionally superseded jobs are not failed child executions. Intentional unit-test fault fixtures are not project failures.'}


def load_facts():
    prepared = BASE/'final_tables/prepared_v2'
    table_proof = prepared_table_proof(ROOT, prepared)
    reference_path = BASE/'preflight/FROZEN_REFERENCES.json'
    reference = read_json(reference_path)
    performance = reference['roles']['FROZEN_PERFORMANCE']
    distilled = reference['roles']['FROZEN_DISTILLED']
    closed_path = BASE/'bottlenecks/FINITE_MENU_CLOSURE_REVIEW.json'
    closed = read_json(closed_path)
    assert closed['new_recipes'] == 13 and closed['completed_full_recipes'] == 12
    assert closed['performance_promotions'] == closed['robustness_promotions'] == 0
    assert closed['DEV_RESEARCH_SATURATION'] == 'NOT_FORMALLY_TRIGGERED'
    model_rows = rows(prepared/'MODEL_RESULTS_NEXT3.csv')
    scientific = [row for row in model_rows if row['record_type'] == 'NEXT3_NEW_SCIENTIFIC_RECIPE']
    assert len(scientific) == 13 and sum(bool(row['full_mean']) for row in scientific) == 12
    assert all(row['promoted'] == 'False' for row in scientific)
    best = max((row for row in scientific if row['full_mean']), key=lambda row: float(row['full_mean']))
    paired = rows(prepared/'PAIRED_BOOTSTRAP_NEXT3.csv')
    mechanism = rows(prepared/'MECHANISM_RESULTS_NEXT3.csv')
    inherited = {}
    for name in ('SCALE_DOMINANT', 'DEPENDENCE_DOMINANT'):
        row = exactly_one(mechanism, candidate=performance['experiment_id'],
            reference=distilled['experiment_id'], scope='mechanism_'+name)
        inherited[name] = {key: float(row[key]) for key in
            ('candidate_mean', 'reference_mean', 'estimate_mean_delta', 'ci_2_5', 'ci_97_5', 'break_series')}
    recipe_pairs = {}
    for name in ('I2_HISTORICAL_RANK_PRODUCTS', 'R1_SCORE_GAP', 'R3_DEPTH2_GATE', 'S3_NEGATIVE_BAYES'):
        recipe_pairs[name] = {}
        for scope in ('all_time', 'mechanism_SCALE_DOMINANT', 'mechanism_DEPENDENCE_DOMINANT'):
            row = exactly_one(paired, candidate=name, reference=performance['experiment_id'], scope=scope)
            recipe_pairs[name][scope] = {key: float(row[key]) for key in
                ('estimate_mean_delta', 'ci_2_5', 'ci_97_5', 'probability_delta_positive')}
    null = rows(prepared/'LONG_NULL_RESULTS_NEXT3.csv')
    null_n0 = []
    for horizon in (1024, 4096, 16384):
        row = exactly_one(null, candidate=performance['experiment_id'],
            evidence_origin='NEXT2_FROZEN_NO_NEXT3_REFIT', background='N0',
            horizon=str(horizon), alpha='0.05')
        assert row['ts_auc'] == ''
        null_n0.append({'horizon': horizon, 'eventual_alarm_rate': float(row['eventual_alarm_rate'])})
    counters = rows(prepared/'COUNTEREXAMPLE_RESULTS_NEXT3.csv')
    outliers = {}
    for name in (performance['experiment_id'], distilled['experiment_id']):
        row = exactly_one(counters, candidate=name, scenario='O1',
            evidence_origin='NEXT2_FROZEN_NO_NEXT3_REFIT', alpha='0.05')
        assert row['ts_auc'] == ''
        outliers[name] = float(row['outlier_alarm_excess'])
    mass_path = BASE/'pair_contributions/SUMMARY.csv'
    mass_rows = rows(mass_path)
    masses = {name: {key: float(row[key]) for key in
        ('fraction_of_official_pair_mass', 'blend_minus_J102_contribution')}
        for name in ('DEPENDENCE_DOMINANT', 'SCALE_DOMINANT')
        if (row := exactly_one(mass_rows, scope='mechanism', group=name))}
    transfer_path = BASE/'legacy_state_transfer/run_v1/WINDOW_COUNTERFACTUAL_INTERVALS.csv'
    transfer_rows = rows(transfer_path)
    ewma = {}
    for anchor in (0, 127, 255):
        row = exactly_one(transfer_rows, anchor_offset=str(anchor), block='LEGACY_EWMA',
            direction='CONTROL_INTO_OUTLIER', metric='probability_mean',
            window_start_inclusive='256', window_end_exclusive='512')
        ewma[str(anchor)] = {key: float(row[key]) for key in
            ('delta_from_destination', 'delta_from_destination_ci_low', 'delta_from_destination_ci_high')}
    latency = read_json(prepared/'LATENCY_NEXT3.json')
    memory = read_json(prepared/'STATE_MEMORY_NEXT3.json')
    initial = read_json(ROOT/'BOTTLENECK_BOARD_INITIAL.json')
    science_lock = read_json(ROOT/'configs/next3/SCIENTIFIC_ROLE_FREEZE.json')
    assert science_lock['scientific_role_count'] == 2 and not science_lock['generation_gate_passed']
    evidence_paths = [reference_path, closed_path, prepared/'RESULTS.json', prepared/'INPUT_RETENTION.json',
        ROOT/'BOTTLENECK_BOARD_INITIAL.json', ROOT/'configs/next3/SCIENTIFIC_ROLE_FREEZE.json',
        BASE/'pair_matrix_audit/run_v1/RESULTS.json', BASE/'outlier_state_transfer/run_v1/RESULTS.json',
        BASE/'legacy_state_transfer/run_v1/RESULTS.json', BASE/'selection_uncertainty/RESULTS.json',
        BASE/'maintenance/methodology_review/CURRENT_REPORT.json', mass_path, transfer_path]
    evidence_paths += [prepared/name for name in table_proof['files']]
    return {'created_utc': utc_now(), 'scope': 'Observed frozen scientific and engineering evidence; no final acceptance asserted here.',
        'performance': performance, 'distilled': distilled, 'new_recipes': 13, 'full_recipes': 12,
        'science_fits': 129, 'scientific_rows': len(model_rows), 'prepared_ledger_rows': table_proof['files']['EXPERIMENT_LEDGER.csv']['rows'],
        'exploratory_best': {key: best[key] for key in ('experiment_id', 'full_mean', 'delta_vs_frozen_performance', 'final_disposition')},
        'selected_delta': 0.0, 'selected_ci': [0.0, 0.0], 'selected_strict_probability_positive': 0.0,
        'selected_identity_bootstrap_replicates': 0, 'new_performance_candidate': None,
        'new_robustness_candidate': None, 'generation_gate_passed': False,
        'inherited_mechanism': inherited, 'recipe_pairs': recipe_pairs,
        'null_N0_alpha05': null_n0, 'inherited_outlier_excess': outliers,
        'pair_mass_contributions': masses, 'legacy_EWMA_state_transfer': ewma,
        'latency': latency, 'memory': memory, 'finite_menu_closure': closed,
        'initial_bottleneck_statuses': {item['id']: item['status'] for item in initial['bottlenecks']},
        'maintenance': maintenance_facts(),
        'evidence_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in evidence_paths}}
