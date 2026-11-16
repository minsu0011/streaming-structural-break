"""Fixed2000 paired whole-seed intervals for state memory and H transport."""
from pathlib import Path
import csv
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from src.next7.common import BASE, save, read_json, digest, csv_write, utc_now, activity, require_guard

G1_NAME = 'S1_BLOCK_LOG_MAD_C1_SOFT_3OF4'
METHODS = ('NEXT6_HARD4_EVENT', 'G1_RAW_EVENT', 'G2_QUALIFIED_EVENT')
PAIRS = (('NEXT6_HARD4_EVENT', 'G1_RAW_EVENT'), ('NEXT6_HARD4_EVENT', 'G2_QUALIFIED_EVENT'),
         ('G1_RAW_EVENT', 'G2_QUALIFIED_EVENT'))
METRICS = ('any_latch', 'control_any_latch', 'excess_latch_vs_paired_control', 'prebreak_latch',
    'first_detection', 'restricted_first_detection_delay', 'event_score_mean',
    'memory_contamination_final', 'memory_contaminated_after128', 'memory_retained_after_recovery')


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def number(value):
    if value in ('True', True):
        return 1.
    if value in ('False', False):
        return 0.
    return float(value)


def metric(row, name):
    if name in ('first_detection', 'prebreak_latch', 'restricted_first_detection_delay') and not number(row['observed_break']):
        return None
    if name == 'restricted_first_detection_delay':
        return float(row['first_detection_delay']) if row['first_detection_delay'] != '' else int(row['horizon']) - int(row['tau_offline_only'])
    alternatives = {
        'event_score_mean': ('event_score_mean', 'event_mean'),
        'memory_contamination_final': ('event_memory_delta_final', 'memory_contamination_final'),
        'memory_contaminated_after128': ('event_memory_contamination_after_128', 'memory_contamination_after128'),
        'memory_retained_after_recovery': ('event_retained_after_recovery', 'memory_retained_after_current_recovery')}
    keys = alternatives.get(name, (name,))
    for key in keys:
        if key in row and row[key] not in ('', None):
            value = number(row[key])
            return float(abs(value) > 0.) if name == 'memory_contaminated_after128' else value
    return None


def estimate(values, counts):
    eligible = np.array([v is not None for v in values])
    if not eligible.any():
        return None
    x = np.array([0. if v is None else v for v in values])
    denominator = counts[:, eligible].sum(axis=1)
    samples = np.full(len(counts), np.nan)
    good = denominator > 0
    samples[good] = (counts[good] @ x) / denominator[good]
    return float(x[eligible].mean()), samples, int(eligible.sum())


def interval(point, draws):
    finite = draws[np.isfinite(draws)]
    return {'estimate': point, 'ci_2_5': float(np.quantile(finite, .025)) if len(finite) else None,
        'ci_97_5': float(np.quantile(finite, .975)) if len(finite) else None,
        'usable_draws': len(finite), 'requested_draws': len(draws)}


def group_statistics(groups, context, counts):
    individual, paired = [], []
    cache = {}
    raw_values = {}
    for method, rows in groups.items():
        rows.sort(key=lambda r: int(r['seed']))
        assert [int(r['seed']) for r in rows] == list(range(512))
        for name in METRICS:
            raw_values[method, name] = [metric(r, name) for r in rows]
            value = estimate(raw_values[method, name], counts)
            if value is None:
                continue
            point, samples, support = value
            cache[method, name] = (point, samples, support)
            individual.append({**context, 'method': method, 'metric': name, 'eligible_seeds': support,
                **interval(point, samples), 'pointwise_interval': True, 'state_not_classifier_alarm': True})
    for reference, candidate in PAIRS:
        for name in METRICS:
            if (reference, name) not in cache or (candidate, name) not in cache:
                continue
            common = [None if b is None or c is None else c - b
                for b, c in zip(raw_values[reference, name], raw_values[candidate, name])]
            value = estimate(common, counts)
            if value is None:
                continue
            difference, draws, support = value
            finite = draws[np.isfinite(draws)]
            paired.append({**context, 'reference': reference, 'candidate': candidate, 'metric': name,
                'eligible_seeds': support, **interval(difference, draws),
                'eligibility': 'intersection of reference and candidate observable metrics',
                'probability_delta_positive': float(np.mean(finite > 0)) if len(finite) else None,
                'pointwise_interval': True, 'paired_same_seed_and_path': True})
    return individual, paired, cache


def main():
    require_guard()
    protocol = read_json(BASE / 'NEXT7_PROTOCOL_LOCK.json')
    cases = protocol['evaluation']['cases']
    for case in cases:
        for directory in ('synthetic_g1', 'synthetic_g2_full', 'h_transport'):
            done = read_json(BASE / directory / case / 'DONE.json')
            assert done['status'] == 'COMPLETE' and done['seeds'] == 512
            assert digest(BASE / directory / case / 'SERIES.csv') == done['SERIES_sha256']
    lock = BASE / 'SYNTHETIC_UNCERTAINTY_LOCK_NEXT7.json'
    count_path = BASE / 'synthetic_stats/COUNTS_2000x512.npy'
    if not lock.exists():
        save('SYNTHETIC_UNCERTAINTY_LOCK_NEXT7.json', {'status': 'FROZEN', 'created_utc': utc_now(),
            'draws': 2000, 'seed': 20260913, 'series': 512, 'methods': METHODS, 'comparisons': PAIRS,
            'metrics': METRICS, 'source_sha256': digest(__file__), 'before_any_synthetic_CI': True,
            'resampling': 'Whole seeded paths, same512 multiplicities for each paired comparison; never individual time points.',
            'cross_case_dependence': 'Reuse of seed indices/draws is not extra independent evidence; pointwise intervals only.',
            'H_length_pairs': 'Same5000-point source H and same online path; suffix length changes only the available H.',
            'delay': 'First-event delay censored at horizon minus tau; prebreak false latch counts as no valid first-event detection.',
            'not_yet_observed_change': 'Detection and prebreak metrics are undefined when tau has not occurred by the reporting horizon.',
            'support': 'Report eligible seeds and valid draws; no new draws to fill unsupported replicates.',
            'no_draw_extension_or_threshold_changes': True})
        count_path.parent.mkdir(parents=True, exist_ok=True)
        counts = np.random.default_rng(20260913).multinomial(512, np.full(512, 1 / 512), size=2000).astype(np.int32)
        with count_path.open('xb') as f:
            np.save(f, counts, allow_pickle=False)
        save('synthetic_stats/COUNTS_LOCK.json', {'created_utc': utc_now(), 'sha256': digest(count_path),
            'shape': [2000, 512], 'all_pair_comparisons_use_exact_counts': True})
    if read_json(lock)['source_sha256'] != digest(__file__):
        repair = read_json(BASE / 'SYNTHETIC_UNCERTAINTY_BUG_REPAIR_001.json')
        assert repair['old_source_sha256'] == read_json(lock)['source_sha256']
        assert repair['new_source_sha256'] == digest(__file__)
        assert repair['published_intervals_before_repair'] == 0
        assert repair['test_xml_sha256'] == digest(BASE / 'audit/SYNTHETIC_STATISTICS_TESTS_001.xml')
    assert digest(count_path) == read_json(BASE / 'synthetic_stats/COUNTS_LOCK.json')['sha256']
    counts = np.load(count_path, allow_pickle=False).astype(float)
    individual, paired, h_individual, h_paired, h_length_comparison = [], [], [], [], []
    with activity('paired synthetic and H-transport uncertainty', 'UNCERTAINTY', 'F4_NULL_GROWTH_CORRECTION'), threadpool_limits(limits=1):
        for case in cases:
            g1 = read_rows(BASE / f'synthetic_g1/{case}/SERIES.csv')
            g2 = read_rows(BASE / f'synthetic_g2_full/{case}/SERIES.csv')
            transport = read_rows(BASE / f'h_transport/{case}/SERIES.csv')
            horizons = sorted({int(r['horizon']) for r in g2})
            for horizon in horizons:
                groups = {'NEXT6_HARD4_EVENT': [r for r in g1 if r['variant'] == 'NEXT6_HARD4_EVENT' and int(r['horizon']) == horizon],
                    'G1_RAW_EVENT': [r for r in g1 if r['variant'] == G1_NAME and int(r['horizon']) == horizon],
                    'G2_QUALIFIED_EVENT': [r for r in g2 if int(r['horizon']) == horizon]}
                a, b, _ = group_statistics(groups, {'case': case, 'horizon': horizon, 'H_length': 2048, 'partition': 31}, counts)
                individual.extend(a)
                paired.extend(b)
                caches = {}
                for length in (1000, 1500, 2048, 3000, 5000):
                    groups = {name: [r for r in transport if r['variant'] == name and int(r['H_length']) == length
                        and int(r['horizon']) == horizon] for name in METHODS}
                    a, b, caches[length] = group_statistics(groups,
                        {'case': case, 'horizon': horizon, 'H_length': length, 'partition': 33}, counts)
                    h_individual.extend(a)
                    h_paired.extend(b)
                for length in (1000, 1500, 3000, 5000):
                    for (method, name), (c, cdraws, support) in caches[length].items():
                        b, bdraws, bcount = caches[2048][method, name]
                        assert support == bcount
                        h_length_comparison.append({'case': case, 'horizon': horizon, 'method': method, 'metric': name,
                            'reference_H_length': 2048, 'candidate_H_length': length, 'eligible_seeds': support,
                            **interval(c - b, cdraws - bdraws), 'same_online_path': True, 'pointwise_interval': True})
            print(utc_now(), 'synthetic intervals', case, flush=True)
        csv_write('synthetic_stats/STATE_METRIC_INTERVALS.csv', individual)
        csv_write('synthetic_stats/PAIRED_STATE_METRIC_INTERVALS.csv', paired)
        csv_write('synthetic_stats/H_TRANSPORT_METRIC_INTERVALS.csv', h_individual)
        csv_write('synthetic_stats/PAIRED_H_TRANSPORT_METRIC_INTERVALS.csv', h_paired)
        csv_write('synthetic_stats/PAIRED_H_LENGTH_VS2048.csv', h_length_comparison)
        csv_write('LONG_NULL_RESULTS_NEXT7.csv', [r for r in individual if r['case'].startswith('N')])
        csv_write('SCALE_UP_DOWN_RESULTS_NEXT7.csv', [r for r in individual if r['case'] in ('V1', 'V2', 'V3', 'V4')])
        csv_write('SHOCK_EVENT_CONTAMINATION_NEXT7.csv', [r for r in individual if r['case'] in ('O1', 'OB16', 'REPEATED_SHOCK')])
        csv_write('EVENT_MEMORY_RESULTS_NEXT7.csv', individual)
        csv_write('H_LENGTH_CALIBRATION_NEXT7.csv', h_individual)
        save('SYNTHETIC_UNCERTAINTY_DONE_NEXT7.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
            'counts_sha256': digest(count_path), 'source_sha256': digest(__file__),
            'state_rows': len(individual), 'paired_state_rows': len(paired), 'H_rows': len(h_individual),
            'paired_H_rows': len(h_paired), 'paired_H_length_rows': len(h_length_comparison),
            'no_new_scientific_candidate': True, 'state_evidence_not_calibrated_classifier_alarm': True})
    require_guard().check_state()


if __name__ == '__main__':
    main()
