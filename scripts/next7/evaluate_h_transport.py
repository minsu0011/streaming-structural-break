"""Fixed H-suffix transport on shared online paths; no model selection."""
from pathlib import Path
import argparse
import copy
import os
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from src.next7.common import BASE, save, read_json, digest, csv_write, utc_now, activity, require_guard
from src.next7.null_growth import QualifiedState, admission_replay, new_admission_args
from src.next7.event import CombinedState
from src.next7.calibration import load_calibration, cutoff
from src.next6_extension.synthetic import generated
from scripts.next7.evaluate_g2_synthetic import check_lock

LENGTHS = (1000, 1500, 2048, 3000, 5000)
VARIANTS = (('NEXT6_HARD4_EVENT', 26), ('G1_RAW_EVENT', 3), ('G2_QUALIFIED_EVENT', 30))


def first(mask):
    positions = np.flatnonzero(mask)
    return int(positions[0]) if len(positions) else -1


def evaluate(out, control, column, tau, info, horizon):
    p, c = out[:horizon, column], control[:horizon, column]
    latch, clatch = first(p > 0), first(c > 0)
    changed = 0 <= tau < horizon
    detected = changed and latch >= tau
    return {'horizon': horizon, 'tau_offline_only': tau, 'observed_break': changed,
        'any_latch': latch >= 0, 'first_latch': latch, 'control_any_latch': clatch >= 0,
        'prebreak_latch': changed and 0 <= latch < tau, 'first_detection': detected,
        'first_detection_delay': latch - tau if detected else None,
        'restricted_first_detection_delay': latch - tau if detected else horizon - tau if changed else None,
        'excess_latch_vs_paired_control': latch >= 0 and clatch < 0,
        'event_final': float(p[-1]), 'event_mean': float(p.mean()),
        'memory_contamination_final': float(p[-1] - c[-1]),
        'event_monotone': bool(np.all(np.diff(p) >= 0)),
        'paired_prechange_exact': bool(np.array_equal(p[:min(info['scheduled'], horizon)], c[:min(info['scheduled'], horizon)]))}


def main(only_case=None):
    require_guard()
    check_lock()
    protocol = read_json(BASE / 'NEXT7_PROTOCOL_LOCK.json')
    allowed = protocol['evaluation']['cases']
    cases = [only_case] if only_case else allowed
    assert all(c in allowed for c in cases)
    source_hashes = {n: digest(ROOT / n) for n in ('scripts/next7/evaluate_h_transport.py',
        'src/next7/null_growth.py', 'src/next7/event.py', 'src/next7/scale.py', 'src/next6_extension/synthetic.py')}
    lockfile = BASE / 'H_TRANSPORT_LOCK_NEXT7.json'
    if not lockfile.exists():
        save('H_TRANSPORT_LOCK_NEXT7.json', {'status': 'FROZEN', 'created_utc': utc_now(),
            'source_sha256': source_hashes, 'partition': 33, 'indices': [0, 511], 'H_lengths': LENGTHS,
            'cases': allowed, 'variants': [v[0] for v in VARIANTS],
            'H_coupling': 'Generate5000 H observations once. Each shorter H is its most recent suffix; the entire online path and intervention are identical across H lengths.',
            'H2048_anchor': 'Prespecified default-length anchor in the same partition33 and same online path; not reuse of the partition31 outcome.',
            'H_alignment': 'Each detector applies its own fixed60/40 split and16-block boundaries to that suffix.',
            'cutoffs': 'Frozen H routing; no re-estimation from transport outcomes',
            'purpose': 'Reference-support and distribution-transport diagnosis only; no candidate or threshold search',
            'no_extra_independent_seed_claim': True, 'state_latches_not_final_classifier_alarms': True})
    assert read_json(lockfile)['source_sha256'] == source_hashes
    suite = copy.deepcopy(read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json'))
    suite['historical_length'] = 5000
    calibration = load_calibration()
    for case in cases:
        relative = f'h_transport/{case}'
        directory = BASE / relative
        if (directory / 'DONE.json').exists():
            continue
        with activity(f'fixed H suffix transport {case}', 'REFERENCE_UNCERTAINTY', 'F4_NULL_GROWTH_CORRECTION'):
            horizon = 16384 if case.startswith('N') else 4096
            horizons = [1024, 4096, 16384] if case.startswith('N') else [1024, 4096]
            rows, tables = [], []
            for seed in range(512):
                whole_h, o, tau, info = generated(case, seed, partition=33, horizon=horizon, control=False, lock=suite)
                hc, oc, _, _ = generated(case, seed, partition=33, horizon=horizon, control=True, lock=suite)
                np.testing.assert_array_equal(whole_h, hc)
                assert len(whole_h) == 5000
                for length in LENGTHS:
                    h = whole_h[-length:]
                    cut = cutoff(calibration, length, 0, 1)
                    state = QualifiedState(h, core_cut=calibration['core_cut'], scale_cut=cut)
                    out = state.replay(o)
                    if np.array_equal(o, oc):
                        control = out
                    else:
                        c = CombinedState(h, core_cut=calibration['core_cut'], scale_cut=cut).replay(oc)
                        control = admission_replay(c, new_admission_args(state.calibration))
                    tables.append({'case': case, 'seed': seed, 'H_length': length, 'scale_cut': cut,
                        'reference_valid': state.calibration['reference_valid'],
                        'pseudo_blocks': state.calibration['pseudo_blocks'],
                        **{f'cut_blocks{g}': v for g, v in zip(state.calibration['grid_blocks'], state.calibration['admission_cuts'])}})
                    for name, column in VARIANTS:
                        for hh in horizons:
                            row = {'case': case, 'seed': seed, 'H_length': length, 'variant': name,
                                'H_reference_valid': state.calibration['reference_valid'],
                                **evaluate(out, control, column, tau, info, hh)}
                            assert row['event_monotone'] and row['paired_prechange_exact']
                            rows.append(row)
                if seed % 64 == 63:
                    print(utc_now(), 'H transport', case, seed + 1, '/512', flush=True)
            csv_write(relative + '/SERIES.csv', rows)
            csv_write(relative + '/H_TABLES.csv', tables)
            summaries = []
            for length in LENGTHS:
                for name, _ in VARIANTS:
                    for hh in horizons:
                        cohort = [r for r in rows if r['H_length'] == length and r['variant'] == name and r['horizon'] == hh]
                        record = {'case': case, 'H_length': length, 'variant': name, 'horizon': hh, 'series': len(cohort)}
                        for metric in ('any_latch', 'control_any_latch', 'prebreak_latch', 'first_detection',
                            'restricted_first_detection_delay', 'excess_latch_vs_paired_control',
                            'event_final', 'event_mean', 'memory_contamination_final', 'H_reference_valid'):
                            values = [r[metric] for r in cohort if r[metric] is not None]
                            record[metric] = float(np.mean(values)) if values else None
                        summaries.append(record)
            csv_write(relative + '/SUMMARY.csv', summaries)
            save(relative + '/DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(), 'case': case,
                'seeds': 512, 'partition': 33, 'H_lengths': LENGTHS, 'source_sha256': source_hashes,
                'SERIES_sha256': digest(directory / 'SERIES.csv'), 'SUMMARY_sha256': digest(directory / 'SUMMARY.csv'),
                'H_TABLES_sha256': digest(directory / 'H_TABLES.csv'), 'paired_prefix_contract': 'PASS'})
    require_guard().check_state()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--case')
    main(p.parse_args().case)
