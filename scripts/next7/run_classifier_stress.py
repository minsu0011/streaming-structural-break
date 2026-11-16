"""Resumable fixed classifier calibration and per-case evaluation."""
from pathlib import Path
import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, require_guard, note
from src.next7.classifier_stress import (METHODS, METRICS, load_models, predict_all, old_case_rows,
    numeric_row, curve_metrics, assert_old_parity)
from src.next6_extension.synthetic import generated, CASES
from scripts.next7.conditional_null_grid import atomic_npz

SOURCES = ('src/next7/classifier_stress.py', 'scripts/next7/run_classifier_stress.py',
    'src/next7/fast_state.py', 'src/next7/features.py', 'src/next7/null_growth.py',
    'src/next7/model.py', 'src/next7/pairwise.py', 'src/next7/integrated_runtime.py',
    'src/next5/runtime.py', 'src/next6/runtime.py', 'src/next6_extension/synthetic.py', 'src/next2/counterexamples.py')


def checked():
    protocol = read_json(BASE / 'CLASSIFIER_STRESS_PROTOCOL_LOCK.json')
    implementation = read_json(BASE / 'CLASSIFIER_STRESS_IMPLEMENTATION_LOCK.json')
    assert implementation['source_sha256'] == {n: digest(ROOT / n) for n in SOURCES}
    assert implementation['protocol_sha256'] == digest(BASE / 'CLASSIFIER_STRESS_PROTOCOL_LOCK.json')
    return protocol


def preflight():
    guard = require_guard()
    xml_path = BASE / 'audit/INTEGRATED_RUNTIME_TESTS_001.xml'
    xml = ET.parse(xml_path)
    assert len(xml.findall('.//testcase')) == 19 and not xml.findall('.//failure') and not xml.findall('.//error')
    protocol = read_json(BASE / 'CLASSIFIER_STRESS_PROTOCOL_LOCK.json')
    models, provenance = load_models(guard, include_m3=True)
    suite = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    save('CLASSIFIER_STRESS_IMPLEMENTATION_LOCK.json', {'status': 'FROZEN', 'created_utc': utc_now(),
        'source_sha256': {n: digest(ROOT / n) for n in SOURCES},
        'protocol_sha256': digest(BASE / 'CLASSIFIER_STRESS_PROTOCOL_LOCK.json'),
        'integrated_runtime_tests_sha256': digest(xml_path), 'fitted_models': provenance,
        'before_classifier_calibration': True})
    case, seed = 'N0', 0
    old = old_case_rows(case, protocol)
    with activity('Classifier stress source and frozen metric parity gate', 'TEST_GATE', 'F5_CAUSAL_NESTED_INTEGRATION'):
        h, o, tau, info = generated(case, seed, partition=31, horizon=4096, control=False, lock=suite)
        p = predict_all(models, h, o, include_m3=True)
        for fold in range(5):
            cal = read_json(ROOT / f'artifacts/next6_extension/calibration/fold_{fold}/DONE.json')['thresholds']
            for method in ('M0', 'M3'):
                for horizon in (1024, 4096):
                    v = p[fold, :, METHODS.index(method)]
                    actual = curve_metrics(v, v, cal[method], tau, info, horizon)
                    assert_old_parity(actual, old[fold, seed, method, horizon], horizon)
    save('CLASSIFIER_STRESS_PREFLIGHT_GATE.json', {'status': 'PASS', 'created_utc': utc_now(),
        'fitted_folds': 5, 'frozen_M0_M3_prefix_metric_checks': 20, 'nested_membership_checked': True,
        'implementation_sha256': digest(BASE / 'CLASSIFIER_STRESS_IMPLEMENTATION_LOCK.json')})
    note('Classifier stress preflight passed', 'Frozen M0 and M3 metric parity across all five matched folds', 'F5_CAUSAL_NESTED_INTEGRATION')


def calibrate():
    guard = require_guard()
    checked()
    assert read_json(BASE / 'CLASSIFIER_STRESS_PREFLIGHT_GATE.json')['status'] == 'PASS'
    models, _ = load_models(guard)
    suite = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    old_maxima = {}
    old_cut = {}
    for fold in range(5):
        parent = ROOT / f'artifacts/next6_extension/calibration/fold_{fold}'
        old_cut[fold] = read_json(parent / 'DONE.json')['thresholds']
        with (parent / 'MAXIMA.csv').open(newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                old_maxima[fold, row['background'], int(row['seed']), row['method']] = float(row['maximum'])
    files = []
    for case in ('N0', 'N1', 'N2', 'N3'):
        relative = f'classifier_stress/calibration/{case}'
        directory = BASE / relative
        for seed in range(256):
            item = directory / f'seed_{seed:03d}.npz'
            receipt = directory / f'seed_{seed:03d}_DONE.json'
            if receipt.exists():
                assert digest(item) == read_json(receipt)['sha256']
                files.append(item)
                continue
            with activity(f'classifier calibration {case} seed{seed}', 'ROBUSTNESS_CALIBRATION', 'F5_CAUSAL_NESTED_INTEGRATION'):
                if item.exists():
                    with np.load(item, allow_pickle=False) as f:
                        maxima = f['maxima'].copy()
                        assert str(f['case']) == case and int(f['seed']) == seed
                else:
                    h, o, _, _ = generated(case, seed, partition=30, horizon=1024, control=True, lock=suite)
                    p = predict_all(models, h, o)
                    maxima = np.max(p, axis=1)
                    for fold in range(5):
                        assert abs(maxima[fold, 0] - old_maxima[fold, case, seed, 'M0']) < 1e-12
                        maxima[fold, 1] = old_maxima[fold, case, seed, 'M3']
                    assert np.isfinite(maxima).all()
                    atomic_npz(item, maxima=maxima, case=np.array(case), seed=np.array(seed))
                assert maxima.shape == (5, len(METHODS))
                save(str(receipt.relative_to(BASE)), {'status': 'COMPLETE', 'created_utc': utc_now(),
                    'sha256': digest(item), 'case': case, 'seed': seed, 'M0_exact_old_maximum_parity': True})
                files.append(item)
            if seed % 32 == 31:
                print(utc_now(), 'classifier calibration', case, seed + 1, '/256', flush=True)
    all_maxima = np.stack([np.load(p, allow_pickle=False)['maxima'] for p in files])
    cuts = np.sort(all_maxima, axis=0)[973]
    rows = []
    for fold in range(5):
        assert abs(cuts[fold, 0] - old_cut[fold]['M0']) < 1e-12
        assert cuts[fold, 1] == old_cut[fold]['M3']
        cuts[fold, 0] = old_cut[fold]['M0']
        cuts[fold, 2] = .9 * cuts[fold, 0]
        for j, method in enumerate(METHODS):
            rows.append({'fold': fold, 'method': method, 'threshold': float(cuts[fold, j]),
                'calibration_series': 1024, 'calibration_horizon': 1024, 'order_1based': 974, 'strict_greater': True})
    csv_write('CLASSIFIER_CALIBRATION_NEXT7.csv', rows)
    save('CLASSIFIER_CALIBRATION_FROZEN.json', {'status': 'FROZEN', 'created_utc': utc_now(),
        'methods': METHODS, 'thresholds': cuts.tolist(), 'all_five_before_new_evaluation': True,
        'calibration_paths': 1024, 'per_seed_five_fold_maxima_sha256': {p.relative_to(BASE).as_posix(): digest(p) for p in files},
        'implementation_sha256': digest(BASE / 'CLASSIFIER_STRESS_IMPLEMENTATION_LOCK.json')})
    note('All classifier stress cutoffs frozen', 'Same1024 null calibration paths, amplitude cutoff scaled exactly', 'F5_CAUSAL_NESTED_INTEGRATION')


def evaluate(case):
    guard = require_guard()
    protocol = checked()
    cal = read_json(BASE / 'CLASSIFIER_CALIBRATION_FROZEN.json')
    assert cal['status'] == 'FROZEN' and cal['all_five_before_new_evaluation']
    cuts = np.array(cal['thresholds'])
    null = case.startswith('N')
    horizons = (1024, 4096, 16384) if null else (1024, 4096)
    models, _ = load_models(guard, include_m3=null)
    suite = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    previous = old_case_rows(case, protocol)
    relative = f'classifier_stress/evaluation/{case}'
    directory = BASE / relative
    records = []
    identity = digest(BASE / 'CLASSIFIER_STRESS_IMPLEMENTATION_LOCK.json')
    for seed in range(512):
        item, receipt = directory / f'seed_{seed:03d}.npz', directory / f'seed_{seed:03d}_DONE.json'
        if receipt.exists():
            done = read_json(receipt)
            assert digest(item) == done['sha256'] and done['implementation_sha256'] == identity
            records.append(done)
            continue
        with activity(f'classifier stress {case} seed{seed}', 'FINAL_CLASSIFIER_ROBUSTNESS', 'F5_CAUSAL_NESTED_INTEGRATION'):
            if item.exists():
                with np.load(item, allow_pickle=False) as f:
                    values = f['values'].copy()
                    assert str(f['case']) == case and int(f['seed']) == seed and str(f['implementation_sha256']) == identity
            else:
                h, o, tau, info = generated(case, seed, partition=31, horizon=max(horizons), control=False, lock=suite)
                hc, oc, _, _ = generated(case, seed, partition=31, horizon=max(horizons), control=True, lock=suite)
                np.testing.assert_array_equal(h, hc)
                np.testing.assert_array_equal(o[:info['scheduled']], oc[:info['scheduled']])
                treatment = predict_all(models, h, o, include_m3=null)
                control = treatment if np.array_equal(o, oc) else predict_all(models, hc, oc, include_m3=null)
                values = np.full((5, len(METHODS), len(horizons), len(METRICS)), np.nan)
                for fold in range(5):
                    for paths in (treatment, control):
                        np.testing.assert_array_equal(paths[fold, :, 0] > cuts[fold, 0], paths[fold, :, 2] > cuts[fold, 2])
                    for position, method in enumerate(METHODS):
                        for j, horizon in enumerate(horizons):
                            if method == 'M3' and not null:
                                values[fold, position, j] = numeric_row(previous[fold, seed, method, horizon], horizon)
                                continue
                            v = curve_metrics(treatment[fold, :, position], control[fold, :, position], cuts[fold, position],
                                tau, info, horizon, amplitude_M0_cut=cuts[fold, 0] if method == 'AMPLITUDE_0_9_M0' else None)
                            assert v[METRICS.index('paired_prechange_max_error')] < 1e-12
                            if method in ('M0', 'M3') and horizon <= 4096:
                                assert_old_parity(v, previous[fold, seed, method, horizon], horizon)
                            values[fold, position, j] = v
                atomic_npz(item, values=values, case=np.array(case), seed=np.array(seed),
                    implementation_sha256=np.array(identity), horizons=np.array(horizons))
            assert values.shape == (5, len(METHODS), len(horizons), len(METRICS))
            assert np.isfinite(values[:, :, :, :7]).all()
            done = {'status': 'COMPLETE', 'created_utc': utc_now(), 'case': case, 'seed': seed,
                'sha256': digest(item), 'implementation_sha256': identity,
                'M0_existing_prefix_metrics_parity': True, 'M3_existing_metrics_reused_or_prefix_parity': True,
                'amplitude_own_cut_alarm_path_exact_M0': True}
            save(str(receipt.relative_to(BASE)), done)
            records.append(done)
        if seed % 32 == 31:
            print(utc_now(), 'classifier stress', case, seed + 1, '/512', flush=True)
    save(relative + '/DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(), 'case': case,
        'seeds': 512, 'fitted_folds': 5, 'methods': METHODS, 'metrics': METRICS, 'horizons': horizons,
        'calibration_sha256': digest(BASE / 'CLASSIFIER_CALIBRATION_FROZEN.json'),
        'implementation_sha256': identity, 'seed_receipts': records, 'new_fits': 0})
    guard.check_state()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['preflight', 'calibrate', 'evaluate'], required=True)
    parser.add_argument('--case', choices=CASES)
    args = parser.parse_args()
    if args.mode == 'preflight':
        preflight()
    elif args.mode == 'calibrate':
        calibrate()
    else:
        assert args.case is not None
        evaluate(args.case)
