"""User-facing tables from completed frozen evidence; never score or fit again."""
from pathlib import Path
import argparse
import csv
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, note, require_guard


def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def time_bin(value):
    for start, stop, name in ((0, 10, '0_9'), (10, 25, '10_24'), (25, 50, '25_49'),
            (50, 100, '50_99'), (100, 200, '100_199'), (200, 400, '200_399')):
        if start <= value < stop:
            return name
    assert value >= 400
    return '400_plus'


def core_tables():
    require_guard()
    assert read_json(BASE / 'SCORE_EVIDENCE_CONSOLIDATED.json')['status'] == 'COMPLETE'
    sources = ['MATCHED_FIVE_FOLD_RESULTS_NEXT7.csv', 'PAIRED_UNCERTAINTY_NEXT7.csv',
        'MODEL_COMPARISON_NEXT7.csv', 'FRACTURE_RESULTS_NEXT7_G1_G2.csv', 'FRACTURE_RESULTS_NEXT7_G3.csv',
        'TARGET_ALIGNMENT_RESULTS_G1.csv', 'TARGET_ALIGNMENT_RESULTS_G2.csv',
        'EVENT_TIMING_FORENSIC_G1.csv', 'EVENT_TIMING_FORENSIC_G2.csv', 'EVENT_MEMORY_RESULTS_NEXT7.csv',
        'H_CALIBRATION_RP_G2.csv', 'SCALE_CALIBRATION_FROZEN.json']
    outputs = []
    def write(name, records):
        csv_write(name, records)
        outputs.append(name)
    with activity('Consolidate fixed predictive and state evidence tables', 'REPORT_EVIDENCE', 'F5_CAUSAL_NESTED_INTEGRATION'):
        write('CV_RESULTS_NEXT7.csv', rows(BASE / sources[0]))
        write('PAIRED_BOOTSTRAP_NEXT7.csv', rows(BASE / sources[1]))
        models = rows(BASE / 'MODEL_COMPARISON_NEXT7.csv')
        write('ALT_RESULTS_NEXT7.csv', [{'candidate': row['method'], 'status': 'NOT_RUN',
            'reason': 'Frozen strong-candidate mean-effect gate not met; no alternate split fit or score opened.',
            'original_primary_mean_delta': row['mean_matched_delta_M0'], 'new_ALT_fits': 0,
            'ALT_score': None, 'not_a_zero_score': True} for row in models if row['new_NEXT7_scientific_candidate'] == 'True'])
        fractures = rows(BASE / sources[3]) + rows(BASE / sources[4])
        assert len({(row['scope'], row['candidate'], row['reference']) for row in fractures}) == len(fractures)
        write('TIME_FRACTURE_NEXT7.csv', [row for row in fractures if row['scope'].startswith('time_')])
        write('BREAK_AGE_NEXT7.csv', [row for row in fractures if row['scope'].startswith('age_')])
        write('REGIME_RESULTS_NEXT7.csv', [row for row in fractures if row['scope'].startswith(('regime_', 'mechanism_'))])
        alignment = []
        timing = []
        for generation in ('G1', 'G2'):
            alignment += [{'generation': generation, **row, 'state_memory_not_classifier_alarm': True}
                          for row in rows(BASE / f'TARGET_ALIGNMENT_RESULTS_{generation}.csv')]
            groups = {}
            for row in rows(BASE / f'EVENT_TIMING_FORENSIC_{generation}.csv'):
                event_time, tau = int(row['event_time']), int(row['tau_offline_only'])
                truth = 'NO_BREAK' if tau < 0 else 'BEFORE_BREAK' if event_time < tau else 'AFTER_BREAK'
                key = (int(row['fold']), truth, int(row['confirmationChannel']), time_bin(event_time))
                groups.setdefault(key, []).append(row)
            for (fold, truth, channel, period), cohort in sorted(groups.items()):
                strength_key = 'confirmationStrength' if generation == 'G1' else 'raw_confirmation_ratio'
                first_key = 'firstLatchTime' if generation == 'G1' else 'firstQualifiedLatchTime'
                timing.append({'generation': generation, 'fold': fold, 'event_truth_stratum': truth,
                    'confirmation_channel': channel, 'event_time_bin': period,
                    'event_updates': len(cohort), 'distinct_series_in_group': len({row['dataset_id'] for row in cohort}),
                    'median_first_latch_time': float(np.median([float(row[first_key]) for row in cohort])),
                    'median_raw_confirmation_ratio': float(np.median([float(row[strength_key]) for row in cohort])),
                    'median_admission_cut': float(np.median([float(row['admission_cut']) for row in cohort])) if generation == 'G2' else None,
                    'pointwise_truth_used_only_offline': True, 'independent_event_sample_claim': False,
                    'scope': 'Aggregated event timing; full original per-event provenance retained in the generation-specific local forensic table.'})
        write('TARGET_ALIGNMENT_RESULTS.csv', alignment)
        write('EVENT_TIMING_FORENSIC.csv', timing)
        write('EVENT_MEMORY_RESULTS.csv', rows(BASE / 'EVENT_MEMORY_RESULTS_NEXT7.csv'))
        calibration = read_json(BASE / 'SCALE_CALIBRATION_FROZEN.json')
        cal_rows = []
        for table in calibration['tables']:
            for si, scale in enumerate(table['scale_order']):
                for ri, rule in enumerate(table['rule_order']):
                    cal_rows.append({'record_type': 'G1_FROZEN_SCALE_CUT', 'H_length': table['H_length'],
                        'scale': scale, 'rule': rule, 'cut': table['cutoffs'][si][ri],
                        'calibration_paths': table['null_series'], 'calibration_horizon': table['horizon'],
                        'order_per_background_1based': table['rank_per_background'], 'strict_greater': True,
                        'max_over_four_backgrounds': True, 'simultaneous_or_anytime_guarantee': False})
        rp = rows(BASE / 'H_CALIBRATION_RP_G2.csv')
        for length in (1000, 1500, 2048, 3000, 5000):
            def route(h):
                return max([v for v in (1000, 1500, 2048, 3000, 5000) if v <= h], default=1000)
            cohort = [row for row in rp if route(int(row['H_length'])) == length]
            for blocks in (4, 8, 16, 32, 64, 128, 256, 512, 1024):
                values = np.array([float(row[f'cut_blocks{blocks}']) for row in cohort])
                cal_rows.append({'record_type': 'G2_RP_H_ONLY_ADMISSION_DISTRIBUTION', 'H_length_route': length,
                    'completed_block_grid': blocks, 'RP_series': len(cohort),
                    'cut_min': float(values.min()) if len(values) else None,
                    'cut_median': float(np.median(values)) if len(values) else None,
                    'cut_max': float(values.max()) if len(values) else None,
                    'reference_valid_series': sum(row['reference_valid'] == 'True' for row in cohort),
                    'labels_used_to_choose_cut': False, 'simultaneous_or_anytime_guarantee': False})
        write('H_CALIBRATION_RESULTS.csv', cal_rows)
        from src.next7.scale import COLUMNS
        descriptors = []
        for index, name in enumerate(COLUMNS):
            role = 'completed_current_block' if index == 0 else 'completed_block_count' if index == 1 else 'signed_scale_evidence' if index < 4 else 'confirmation_diagnostic'
            descriptors.append({'column_index': index, 'name': name, 'role': role,
                'current_block_size': 16, 'maximum_confirming_window_blocks': 4,
                'H_only_normalization': True, 'evidence_clip_absolute': 4 if index in (2, 3) else None,
                'positive_direction': 'scale expansion', 'negative_direction': 'scale contraction',
                'S1_physical_statistic': 'log(max(median(abs(block-median(block))), H epsilon))',
                'S2_physical_statistic': 'smoothed H-only PIT logit of block log-MAD, divided by pi/sqrt(3)',
                'trained_predictive_target_used': False, 'calibrated_break_probability': False})
        write('SIGNED_SCALE_FEATURES.csv', descriptors)
        save('COMPLETED_CORE_TABLES_NEXT7.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
            'source_sha256': digest(__file__), 'inputs_sha256': {name: digest(BASE / name) for name in sources},
            'outputs_sha256': {name: digest(BASE / name) for name in outputs},
            'new_fits': 0, 'new_random_draws': 0, 'new_score_release': False,
            'event_timing_export_aggregates_raw_RP_event_rows': True,
            'pending_separate_outputs': ['confirmation rule intervals', 'complete classifier stress', 'all crossed cohorts', 'formal final audit']})
    note('Fixed core result tables consolidated', 'Required CV, intervals, timing, H calibration and feature tables created from frozen evidence', 'F5_CAUSAL_NESTED_INTEGRATION')


def confirmation_tables():
    require_guard()
    from scripts.next7.synthetic_uncertainty import metric, estimate, interval
    from scripts.next7.evaluate_state_synthetic import VARIANTS
    cases = read_json(BASE / 'NEXT7_PROTOCOL_LOCK.json')['evaluation']['cases']
    count_path = BASE / 'synthetic_stats/COUNTS_2000x512.npy'
    assert digest(count_path) == read_json(BASE / 'synthetic_stats/COUNTS_LOCK.json')['sha256']
    counts = np.load(count_path, allow_pickle=False).astype(float)
    names = ('any_latch', 'excess_latch_vs_paired_control', 'first_detection', 'restricted_first_detection_delay')
    frozen = BASE / 'CONFIRMATION_RULE_REPORT_LOCK.json'
    identity = {'status': 'FROZEN', 'created_utc': utc_now(), 'source_sha256': digest(__file__),
        'estimation_source_sha256': digest(ROOT / 'scripts/next7/synthetic_uncertainty.py'),
        'count_sha256': digest(count_path), 'draws': 2000, 'variants': [item[0] for item in VARIANTS],
        'metrics': names, 'new_fits': 0, 'new_random_draws': 0,
        'scope': 'Complete report of the original seven frozen state variants; same original512 seed units and observed-break eligibility. Does not select a new confirmation rule.'}
    if not frozen.exists():
        save(frozen.name, identity)
    assert read_json(frozen)['source_sha256'] == digest(__file__)
    result, inputs = [], {}
    with activity('All original confirmation rule uncertainty summaries', 'FIXED_RULE_REPORTING', 'F2_FRESH_CONFIRMATION'), threadpool_limits(limits=1):
        for case in cases:
            directory = BASE / 'synthetic_g1' / case
            done = read_json(directory / 'DONE.json')
            assert done['status'] == 'COMPLETE' and done['seeds'] == 512
            assert digest(directory / 'SERIES.csv') == done['SERIES_sha256']
            inputs[case] = digest(directory / 'DONE.json')
            records = rows(directory / 'SERIES.csv')
            for variant, _, _, _ in VARIANTS:
                for horizon in done['horizons']:
                    group = sorted([row for row in records if row['variant'] == variant and int(row['horizon']) == horizon], key=lambda row: int(row['seed']))
                    assert [int(row['seed']) for row in group] == list(range(512))
                    for name in names:
                        estimated = estimate([metric(row, name) for row in group], counts)
                        fields = {'estimate': None, 'ci_2_5': None, 'ci_97_5': None,
                                  'usable_draws': 0, 'requested_draws': 2000}
                        eligible = 0
                        if estimated is not None:
                            point, draws, eligible = estimated
                            fields = interval(point, draws)
                        result.append({'case': case, 'variant': variant, 'horizon': horizon,
                            'metric': name, 'eligible_seeds': eligible, **fields,
                            'unobserved_break_not_imputed_as_miss': True, 'state_not_classifier_alarm': True,
                            'pointwise_reused_seed_interval': True, 'new_rule_selected': False})
        csv_write('CONFIRMATION_RULE_RESULTS.csv', result)
        save('CONFIRMATION_RULE_REPORT_DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
            'implementation_lock_sha256': digest(frozen), 'inputs_sha256': inputs,
            'rows': len(result), 'result_sha256': digest(BASE / 'CONFIRMATION_RULE_RESULTS.csv'),
            'new_fits': 0, 'new_random_draws': 0})
    note('All original confirmation rule tables completed', 'Seven fixed variants with same2000 seed counts and observed-break support', 'F2_FRESH_CONFIRMATION')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirmation', action='store_true')
    args = parser.parse_args()
    confirmation_tables() if args.confirmation else core_tables()
