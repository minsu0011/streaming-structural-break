"""Execute the fixed G2 screen, then full synthetic only when its gate passes."""
from pathlib import Path
import argparse
import os
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from src.next7.common import BASE, save, read_json, digest, csv_write, utc_now, activity, require_guard
from src.next7.null_growth import (QualifiedState, admission_replay, new_admission_args,
    shared_uniforms, COLUMNS)
from src.next7.event import CombinedState
from src.next7.calibration import load_calibration, cutoff
from src.next6_extension.synthetic import generated


def first(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else -1


def measure(out, control, tau, info, horizon):
    p, c = out[:horizon], control[:horizon]
    latch, clatch = first(p[:, 30] > 0), first(c[:, 30] > 0)
    raw_latch = first(p[:, 3] > 0)
    changed = 0 <= tau < horizon
    fresh_after = first((p[:, 40] > 0) & (np.arange(horizon) >= max(tau, 0)))
    row = {'horizon': horizon, 'observed_break': changed, 'tau_offline_only': tau,
        'any_latch': latch >= 0, 'first_latch': latch, 'control_any_latch': clatch >= 0,
        'excess_latch_vs_paired_control': latch >= 0 and clatch < 0,
        'prebreak_latch': changed and 0 <= latch < tau,
        'first_detection': changed and latch >= tau,
        'first_detection_delay': latch - tau if changed and latch >= tau else None,
        'any_postbreak_qualification': changed and fresh_after >= tau,
        'raw_any_latch': raw_latch >= 0, 'raw_first_latch': raw_latch,
        'raw_first_detection': changed and raw_latch >= tau,
        'event_memory_final': float(p[-1, 30]), 'raw_event_memory_final': float(p[-1, 3]),
        'event_memory_delta_final': float(p[-1, 30] - c[-1, 30]),
        'event_score_mean': float(p[:, 30].mean()), 'raw_event_score_mean': float(p[:, 3].mean()),
        'current_score_mean': float(p[:, 1].mean()), 'current_state_final': int(p[-1, 0]),
        'qualified_episodes_capped': int(p[-1, 31]), 'first_shock_time': int(p[-1, 19]),
        'raw_first_confirmation_time': int(p[-1, 20]), 'first_qualified_time': int(p[-1, 35]),
        'admission_cut_final': float(p[-1, 42]), 'admission_supported_final': bool(p[-1, 43]),
        'event_without_qualification': bool(np.any((p[:, 30] > 0) & (p[:, 35] < 0))),
        'event_monotone': bool(np.all(np.diff(p[:, 30]) >= 0)),
        'qualified_without_raw_confirmation': bool(np.any((p[:, 40] > 0) & (p[:, 18] == 0))),
        'paired_prechange_max_error': float(np.abs(p[:min(info['scheduled'], horizon)] - c[:min(info['scheduled'], horizon)]).max(initial=0))}
    end = info['intervention_end']
    if end is not None and end + 128 < horizon:
        current = [0, 1, 2, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 25, 28, 29]
        row['current_exact_coupled_after128'] = bool(np.array_equal(p[end + 128:, current], c[end + 128:, current]))
        row['memory_contamination_after128'] = float(np.abs(p[end + 128:, 30] - c[end + 128:, 30]).max())
        row['memory_retained_after_current_recovery'] = bool(np.any((p[end:, 0] == 1) & (p[end:, 30] > 0)))
    return row


def summary_rows(rows):
    result = []
    for horizon in sorted(set(r['horizon'] for r in rows)):
        cohort = [r for r in rows if r['horizon'] == horizon]
        record = {'case': cohort[0]['case'], 'horizon': horizon, 'series': len(cohort)}
        metrics = ['any_latch', 'control_any_latch', 'excess_latch_vs_paired_control', 'prebreak_latch',
            'first_detection', 'any_postbreak_qualification', 'raw_any_latch', 'raw_first_detection',
            'event_memory_final', 'event_memory_delta_final', 'event_score_mean', 'raw_event_score_mean',
            'admission_cut_final', 'admission_supported_final', 'H_reference_valid']
        for metric in metrics:
            record[metric] = float(np.mean([r[metric] for r in cohort]))
        delays = [r['first_detection_delay'] for r in cohort if r['first_detection_delay'] is not None]
        record['detected_only_delay_median'] = float(np.median(delays)) if delays else None
        record['detected_only_delay_count'] = len(delays)
        result.append(record)
    return result


def check_lock():
    require_guard()
    implementation = read_json(BASE / 'G2_IMPLEMENTATION_LOCK.json')
    for path, sha in implementation['source_sha256'].items():
        assert digest(ROOT / path) == sha, path
    assert digest(BASE / 'G2_PROTOCOL_LOCK.json') == implementation['protocol_sha256']
    assert digest(BASE / 'calibration/g2/SHARED_UNIFORMS.npy') == implementation['uniforms_sha256']
    np.testing.assert_array_equal(np.load(BASE / 'calibration/g2/SHARED_UNIFORMS.npy', allow_pickle=False), shared_uniforms())
    assert read_json(BASE / 'audit/G2_CONTRACT_TEST_GATE.json')['status'] == 'PASS'
    return implementation


def main(mode='screen', only_case=None):
    implementation = check_lock()
    protocol = read_json(BASE / 'G2_PROTOCOL_LOCK.json')
    original = read_json(BASE / 'NEXT7_PROTOCOL_LOCK.json')
    if mode == 'full':
        assert read_json(BASE / 'G2_SCREEN_GATE.json')['continue_full'], 'Frozen screen gate failed'
    allowed = protocol['screen']['cases'] if mode == 'screen' else original['evaluation']['cases']
    cases = [only_case] if only_case else allowed
    assert all(c in allowed for c in cases)
    seeds = 64 if mode == 'screen' else 512
    calibration = load_calibration()
    suite = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    for case in cases:
        relative = f'synthetic_g2_{mode}/{case}'
        directory = BASE / relative
        if (directory / 'DONE.json').exists():
            continue
        with activity(f'G2 {mode} paired state {case}', 'ROBUSTNESS_RESEARCH', 'F4_NULL_GROWTH_CORRECTION'):
            horizon = 16384 if case.startswith('N') else 4096
            horizons = [1024, 4096, 16384] if case.startswith('N') else [1024, 4096]
            rows, events, calibrations = [], [], []
            for seed in range(seeds):
                h, o, tau, info = generated(case, seed, partition=31, horizon=horizon, control=False, lock=suite)
                hc, oc, _, _ = generated(case, seed, partition=31, horizon=horizon, control=True, lock=suite)
                np.testing.assert_array_equal(h, hc)
                cut = cutoff(calibration, len(h), 0, 1)
                started = time.perf_counter()
                state = QualifiedState(h, core_cut=calibration['core_cut'], scale_cut=cut)
                initialization_seconds = time.perf_counter() - started
                out = state.replay(o)
                if np.array_equal(o, oc):
                    control = out
                else:
                    raw_control = CombinedState(h, core_cut=calibration['core_cut'], scale_cut=cut).replay(oc)
                    control = admission_replay(raw_control, new_admission_args(state.calibration))
                for hh in horizons:
                    row = {'case': case, 'seed': seed, 'mode': mode, 'variant': 'G2_H_NULL_MAXIMUM_ADMISSION',
                        'H_reference_valid': state.calibration['reference_valid'], **measure(out, control, tau, info, hh)}
                    assert row['event_monotone'] and row['paired_prechange_max_error'] == 0
                    assert not row['event_without_qualification'] and not row['qualified_without_raw_confirmation']
                    assert row.get('current_exact_coupled_after128', True)
                    rows.append(row)
                calibrations.append({'case': case, 'seed': seed, 'initialization_seconds': initialization_seconds,
                    'bounded_mutable_state_bytes': state.bounded_mutable_state_bytes, **state.calibration})
                status = out[:, 41] > .5
                for t in np.flatnonzero(status & ~np.r_[False, status[:-1]]):
                    oldest = int(out[t, 38])
                    events.append({'case': case, 'seed': seed, 'event_time': int(t),
                        'firstShockTime': int(out[t, 19]), 'rawFirstConfirmationTime': int(out[t, 20]),
                        'firstQualifiedTime': int(out[t, 35]), 'confirmationChannel': int(out[t, 37]),
                        'confirmationBlocks': ','.join(map(str, range(oldest, oldest + 4))),
                        'rawConfirmationRatio': float(out[t, 2]), 'admissionCut': float(out[t, 42]),
                        'qualifiedStrength': float(out[t, 44]), 'eventDirection': int(out[t, 34]),
                        'eventAge': int(out[t, 32]), 'tau_offline_only': tau})
                if seed % 32 == 31:
                    print(utc_now(), mode, case, seed + 1, '/', seeds, flush=True)
            csv_write(relative + '/SERIES.csv', rows)
            csv_write(relative + '/EVENTS.csv', events)
            csv_write(relative + '/SUMMARY.csv', summary_rows(rows))
            save(relative + '/H_CALIBRATIONS.json', calibrations)
            save(relative + '/DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
                'case': case, 'mode': mode, 'seeds': seeds, 'columns': COLUMNS,
                'implementation_sha256': digest(BASE / 'G2_IMPLEMENTATION_LOCK.json'),
                'SERIES_sha256': digest(directory / 'SERIES.csv'),
                'SUMMARY_sha256': digest(directory / 'SUMMARY.csv'), 'contracts': 'PASS'})
    if mode == 'screen' and all((BASE / f'synthetic_g2_screen/{c}/DONE.json').exists() for c in allowed):
        if not (BASE / 'G2_SCREEN_GATE.json').exists():
            import csv
            summaries = []
            for c in allowed:
                with (BASE / f'synthetic_g2_screen/{c}/SUMMARY.csv').open(newline='', encoding='utf-8') as f:
                    summaries.extend(csv.DictReader(f))
            values = {(r['case'], int(r['horizon'])): r for r in summaries}
            n0 = float(values['N0', 16384]['any_latch'])
            n2 = float(values['N2', 16384]['any_latch'])
            v4 = float(values['V4', 1024]['first_detection'])
            passed = n0 <= .50 and n2 <= .50 and v4 >= .10
            save('G2_SCREEN_GATE.json', {'status': 'PASS' if passed else 'FAIL',
                'created_utc': utc_now(), 'continue_full': passed,
                'N0_16384_latch': n0, 'N2_16384_latch': n2, 'V4_1024_first_detection': v4,
                'rule': protocol['screen']['fixed_rule_to_continue_full'],
                'screen_is_not_promotion': True, 'no_G2_RP_scores_used': True,
                'source_sha256': implementation['source_sha256']})
    require_guard().check_state()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['screen', 'full'], default='screen')
    parser.add_argument('--case')
    a = parser.parse_args()
    main(a.mode, a.case)
