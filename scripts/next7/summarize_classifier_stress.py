"""Whole-seed paired uncertainty conditional on the five frozen fitted models."""
from pathlib import Path
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, require_guard, note
from src.next7.classifier_stress import METHODS, METRICS
from src.next7.diagnostic_statistics import fixed_fold_draws, fixed_fold_pair, interval
from src.next6_extension.synthetic import CASES
from scripts.next7.conditional_null_grid import atomic_npz

SIMPLE = ('any_alarm', 'prebreak_alarm', 'postbreak_first_detection', 'postbreak_any_detection',
    'first_detection_delay', 'first_postbreak_alarm_delay', 'score_max', 'recovery_delay',
    'recovery_censored', 'paired_effect_after_128_max', 'paired_excess_alarm_after_128',
    'control_any_alarm', 'treatment_alarm_control_never', 'restricted_first_detection_delay')
BREAK_ONLY = ('prebreak_alarm', 'postbreak_first_detection', 'postbreak_any_detection',
    'first_detection_delay', 'first_postbreak_alarm_delay', 'restricted_first_detection_delay')


def lock():
    sources = {n: digest(ROOT / n) for n in ('scripts/next7/summarize_classifier_stress.py',
        'src/next7/diagnostic_statistics.py', 'src/next7/classifier_stress.py')}
    path = BASE / 'CLASSIFIER_UNCERTAINTY_IMPLEMENTATION_LOCK.json'
    if not path.exists():
        tests = BASE / 'audit/DIAGNOSTIC_STATISTICS_TESTS_001.xml'
        xml = ET.parse(tests)
        assert len(xml.findall('.//testcase')) == 6 and not xml.findall('.//failure') and not xml.findall('.//error')
        count = BASE / 'synthetic_stats/COUNTS_2000x512.npy'
        save(path.name, {'status': 'FROZEN', 'created_utc': utc_now(), 'source_sha256': sources,
            'protocol_sha256': digest(BASE / 'CLASSIFIER_STRESS_PROTOCOL_LOCK.json'),
            'test_xml_sha256': digest(tests), 'count_file_sha256': digest(count),
            'unit': 'One generating seed; exactly the same2000 count rows for every fixed fold, method, case and horizon.',
            'estimand': 'Equal mean of five fixed-fold conditional metric means, recomputed inside each seed draw.',
            'missing': 'Require nonempty support in all five fixed folds. Otherwise draw missing; no model pool change or zero imputation.',
            'paired': 'Common finite seed/fold intersection before both conditional means.',
            'eligibility': 'Break metrics require observed break; after128 metrics require at least one observed point after intervention_end+128.',
            'first_alarm_time': 'Conditional on an alarm, plus a separate restricted time with no alarm censored at horizon.',
            'restricted_first_detection_delay': 'Original metric: timely first alarm delay; no alarm or prebreak first alarm assigned horizon minus tau. This is not first subsequent postbreak alarm delay.',
            'comparisons': 'Every nonreference method versus M0 and M3; M3 versus M0; G2_EVENT versus M_EVENT; G3 versus G2_FULL.',
            'limits': 'Conditional on frozen fitted models/cutoffs, pointwise and descriptive; no five-independent-fold, joint-coverage or calibrated-score claim.',
            'new_fits': 0, 'new_random_draws': 0, 'new_cutoffs': 0})
    assert read_json(path)['source_sha256'] == sources
    assert read_json(path)['protocol_sha256'] == digest(BASE / 'CLASSIFIER_STRESS_PROTOCOL_LOCK.json')
    return read_json(path)


def selected_metrics(values, case, horizon, suite):
    # values: generating seed x fitted fold x metric
    get = lambda name: values[:, :, METRICS.index(name)].T.copy()
    observed = get('observed_break') > .5
    out = {name: get(name) for name in SIMPLE}
    for name in BREAK_ONLY:
        out[name][~observed] = np.nan
    first = get('first_alarm')
    out['first_alarm_time_given_alarm'] = np.where(first >= 0, first, np.nan)
    out['restricted_first_alarm_time'] = np.where(first >= 0, first, horizon)
    durations = {'O1': 1, 'OB16': 16, 'REPEATED_SHOCK': 385, 'TEMP_SCALE_PULSE': 32, 'RETURN_NORMAL': 256}
    if case in durations:
        at = np.array([suite['tau_cycle'][i % len(suite['tau_cycle'])] for i in range(512)])
        observable = at + durations[case] + 128 < horizon
        for name in ('paired_effect_after_128_max', 'paired_excess_alarm_after_128'):
            out[name][:, ~observable] = np.nan
    return out


def csv_safe(row):
    result = {}
    for key, value in row.items():
        if isinstance(value, list):
            result[key] = json.dumps([float(x) if isinstance(x, (float, np.floating)) and np.isfinite(x)
                else None if isinstance(x, (float, np.floating)) else x for x in value], allow_nan=False)
        else:
            result[key] = value
    return result


def summarize(case):
    require_guard()
    protocol = lock()
    relative = f'classifier_statistics/{case}'
    dest = BASE / relative
    if (dest / 'DONE.json').exists():
        return
    source = BASE / f'classifier_stress/evaluation/{case}'
    receipt = read_json(source / 'DONE.json')
    assert receipt['status'] == 'COMPLETE' and receipt['seeds'] == 512 and receipt['fitted_folds'] == 5
    assert receipt['methods'] == list(METHODS) and receipt['metrics'] == list(METRICS)
    assert receipt['calibration_sha256'] == digest(BASE / 'CLASSIFIER_CALIBRATION_FROZEN.json')
    assert receipt['implementation_sha256'] == digest(BASE / 'CLASSIFIER_STRESS_IMPLEMENTATION_LOCK.json')
    count_path = BASE / 'synthetic_stats/COUNTS_2000x512.npy'
    assert digest(count_path) == protocol['count_file_sha256']
    counts = np.load(count_path, allow_pickle=False)
    assert counts.shape == (2000, 512) and np.all(counts.sum(axis=1) == 512)
    suite = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    with activity(f'Whole-seed fixed-model classifier uncertainty {case}', 'CLASSIFIER_PAIRED_UNCERTAINTY', 'F5_CAUSAL_NESTED_INTEGRATION'):
        values = np.empty((512, 5, len(METHODS), len(receipt['horizons']), len(METRICS)))
        for seed, row in enumerate(receipt['seed_receipts']):
            assert row['seed'] == seed
            path = source / f'seed_{seed:03d}.npz'
            assert digest(path) == row['sha256']
            with np.load(path, allow_pickle=False) as f:
                assert int(f['seed']) == seed and str(f['case']) == case
                values[seed] = f['values']
        summaries, contrasts, amplitude_rows, draws = [], [], [], {}
        comparisons = [('M3', 'M0')]
        comparisons += [(m, ref) for m in METHODS[2:] for ref in ('M0', 'M3')]
        comparisons += [('G2_EVENT', 'M_EVENT'), ('G3_PAIRWISE_FULL', 'G2_FULL')]
        for j, horizon in enumerate(receipt['horizons']):
            matrices = {method: selected_metrics(values[:, :, k, j], case, horizon, suite) for k, method in enumerate(METHODS)}
            for name in ('any_alarm', 'prebreak_alarm', 'postbreak_first_detection', 'postbreak_any_detection',
                         'first_alarm_time_given_alarm', 'restricted_first_alarm_time', 'first_detection_delay',
                         'first_postbreak_alarm_delay', 'control_any_alarm', 'treatment_alarm_control_never'):
                np.testing.assert_array_equal(matrices['M0'][name], matrices['AMPLITUDE_0_9_M0'][name])
            for method, metrics in matrices.items():
                for name, matrix in metrics.items():
                    estimate, d, means, support = fixed_fold_draws(matrix, counts)
                    key = f'{method}_{horizon}_{name}'
                    draws[key] = d
                    summaries.append(csv_safe({'case': case, 'horizon': horizon, 'method': method, 'metric': name,
                        'mean_across_fixed_folds': estimate, **interval(d), 'fold_means': means.tolist(),
                        'eligible_seeds_by_fixed_fold': support.tolist(), 'fixed_folds': 5,
                        'independent_fold_claim': False, 'conditional_on_models_and_calibration': True}))
            for candidate, reference in comparisons:
                for name in matrices[candidate]:
                    row, delta = fixed_fold_pair(matrices[candidate][name], matrices[reference][name], counts)
                    draws[f'DELTA_{candidate}_VS_{reference}_{horizon}_{name}'] = delta
                    contrasts.append(csv_safe({'case': case, 'horizon': horizon, 'candidate': candidate,
                        'reference': reference, 'metric': name, **row, 'same_seed_counts': True,
                        'common_observable_support': True, 'pointwise_only': True}))
            shared = values[:, :, METHODS.index('AMPLITUDE_0_9_M0'), j,
                METRICS.index('amplitude_control_shared_M0_cut_any_alarm')].T
            own = matrices['AMPLITUDE_0_9_M0']['any_alarm']
            assert np.all(shared <= own)
            row, delta = fixed_fold_pair(shared, own, counts)
            amplitude_rows.append(csv_safe({'case': case, 'horizon': horizon,
                'comparison': 'Amplitude0.9M0: unscaled shared M0 cutoff minus correctly scaled own cutoff',
                **row, 'structural_improvement_claim': False, 'own_cut_alarm_paths_exact_M0': True}))
        csv_write(relative + '/SUMMARY.csv', summaries)
        csv_write(relative + '/PAIRED_CONTRASTS.csv', contrasts)
        csv_write(relative + '/AMPLITUDE_CONTROL.csv', amplitude_rows)
        atomic_npz(dest / 'BOOTSTRAP_DRAWS.npz', **draws)
        save(relative + '/DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(), 'case': case,
            'raw_done_sha256': digest(source / 'DONE.json'),
            'implementation_sha256': digest(BASE / 'CLASSIFIER_UNCERTAINTY_IMPLEMENTATION_LOCK.json'),
            'draws': 2000, 'fixed_folds': 5, 'fold_resampling': False, 'undefined_support_not_imputed': True,
            'own_cut_amplitude_alarm_metrics_exact_M0': True,
            'files_sha256': {p.name: digest(p) for p in dest.iterdir() if p.is_file()}})
    note('Classifier paired uncertainty complete', f'{case}: fixed five-model pool and common observable seeds', 'F5_CAUSAL_NESTED_INTEGRATION')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock-only', action='store_true')
    parser.add_argument('--case', choices=CASES)
    args = parser.parse_args()
    if args.lock_only:
        lock()
    else:
        assert args.case is not None
        summarize(args.case)
