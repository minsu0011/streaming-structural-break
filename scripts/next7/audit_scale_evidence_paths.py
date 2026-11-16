"""Locate fixed V1-V4 misses before raw confirmation or after G2 admission."""
from pathlib import Path
import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
from src.next7.worker_environment import environment
os.environ.update(environment())
import numpy as np
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, note, require_guard
from src.next7.scale_evidence_diagnostics import first_detection_partition, scale_window_partition
from src.next7.calibration import load_calibration, cutoff
from src.next7.fast_state import FastQualifiedState
from src.next6_extension.synthetic import generated

CASES = ('V1', 'V2', 'V3', 'V4')
CATEGORIES = ('prebreak_qualified_memory', 'correct_first_qualification',
              'postbreak_raw_confirmation_rejected', 'no_postbreak_raw_confirmation')


def csv_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def freeze():
    names = ('scripts/next7/audit_scale_evidence_paths.py', 'src/next7/scale_evidence_diagnostics.py',
             'src/next7/fast_state.py', 'src/next7/scale.py', 'src/next7/event.py',
             'src/next7/null_growth.py', 'src/next6_extension/synthetic.py')
    sources = {name: digest(ROOT / name) for name in names}
    path = BASE / 'SCALE_EVIDENCE_PATH_AUDIT_LOCK_NEXT7.json'
    if not path.exists():
        tests = BASE / 'audit/SCALE_EVIDENCE_TESTS_002.xml'
        xml = ET.parse(tests)
        assert len(xml.findall('.//testcase')) == 9 and not xml.findall('.//failure') and not xml.findall('.//error')
        assert read_json(BASE / 'audit/SCALE_EVIDENCE_TEST_ACCESS_002.json')['status'] == 'PASS'
        original = {}
        for case in CASES:
            parent = BASE / f'synthetic_g2_full/{case}'
            assert digest(parent / 'SERIES.csv') == read_json(parent / 'DONE.json')['SERIES_sha256']
            for name in ('H_CALIBRATIONS.json', 'SERIES.csv', 'DONE.json'):
                p = parent / name
                original[p.relative_to(BASE).as_posix()] = digest(p)
        save(path.name, {'status': 'FROZEN', 'created_utc': utc_now(), 'source_sha256': sources,
            'tests_sha256': digest(tests), 'original_inputs_sha256': original,
            'motivation': 'Actual V4 ceiling blockage intersects only11 of332 misses by1024; separate absent raw confirmation from admission rejection without changing inference.',
            'cases': CASES, 'seed_indices': list(range(512)), 'partition': 31, 'H_length': 2048,
            'horizon': 4096, 'reports_at': [1024, 4096], 'new_seed_draws': 0, 'new_fits': 0, 'new_candidates': 0,
            'first_detection_partition': CATEGORIES,
            'scale_window_scope': 'Four complete16-raw-point scale blocks wholly after offline tau. Intermediate signed evidence includes opposing observations, physical-ring addition order, and unchanged strict C1 cutoff. These windows are descriptive; early crossing windows can already detect.',
            'statistics': 'Original2000x512 multiplicities. First-event categories are paired seed fractions. Window categories use ratio of total window counts with seed-level resampling; windows are not independent samples.',
            'inference_scope': 'Only original H/current scalar enter the unchanged detector; tau is used after replay for forensic partitions. No all-point path exported.'})
    result = read_json(path)
    assert result['source_sha256'] == sources
    for name, sha in result['original_inputs_sha256'].items():
        assert digest(BASE / name) == sha
    return digest(path)


def main(lock_only):
    identity = freeze()
    if lock_only:
        return
    require_guard()
    calibration = load_calibration()
    suite = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    count_path = BASE / 'synthetic_stats/COUNTS_2000x512.npy'
    assert digest(count_path) == read_json(BASE / 'synthetic_stats/COUNTS_LOCK.json')['sha256']
    counts = np.load(count_path, allow_pickle=False)
    records = []
    for case in CASES:
        parent = BASE / f'synthetic_g2_full/{case}'
        old = {(int(r['seed']), int(r['horizon'])): r for r in csv_rows(parent / 'SERIES.csv')}
        old_h = {r['seed']: r for r in read_json(parent / 'H_CALIBRATIONS.json')}
        with activity('Fixed scale evidence path audit ' + case, 'SCALE_EVIDENCE_FORENSIC', 'F3_SYMMETRIC_SCALE'):
            for seed in range(512):
                relative = f'scale_evidence_paths/{case}/seed_{seed:03d}.json'
                if (BASE / relative).exists():
                    record = read_json(BASE / relative)
                    assert record['status'] == 'PASS' and record['identity'] == identity
                else:
                    h, online, tau, _ = generated(case, seed, partition=31, horizon=4096, control=False, lock=suite)
                    scale_cut = cutoff(calibration, len(h))
                    state = FastQualifiedState(h, core_cut=calibration['core_cut'], scale_cut=scale_cut)
                    assert all(state.calibration[key] == old_h[seed][key] for key in state.calibration)
                    path = state.replay(online)
                    horizons = []
                    for horizon in (1024, 4096):
                        expected = old[seed, horizon]
                        out = path[:horizon]
                        first_raw = np.flatnonzero(out[:, 3] > 0)
                        first_q = np.flatnonzero(out[:, 30] > 0)
                        assert int(first_raw[0] if len(first_raw) else -1) == int(expected['raw_first_latch'])
                        assert int(first_q[0] if len(first_q) else -1) == int(expected['first_latch'])
                        assert out[-1, 30] == float(expected['event_memory_final'])
                        assert out[-1, 42] == float(expected['admission_cut_final'])
                        category = first_detection_partition(path, tau, horizon)
                        assert (category == 'correct_first_qualification') == (expected['first_detection'] == 'True')
                        horizons.append({'case': case, 'seed': seed, 'horizon': horizon, 'category': category,
                            'original_first_event_and_final_memory_parity': True,
                            **scale_window_partition(path, tau, horizon, scale_cut)})
                    record = {'status': 'PASS', 'identity': identity, 'case': case, 'seed': seed, 'horizons': horizons}
                    save(relative, record)
                records.extend(record['horizons'])
                if seed % 64 == 0:
                    print(utc_now(), case, seed, '/512', flush=True)
    first_summary, window_summary = [], []
    window_metrics = [key for key in records[0] if key not in ('case', 'seed', 'horizon', 'category',
        'original_first_event_and_final_memory_parity', 'first_possible_fully_postbreak_window_point', 'eligible_windows')]
    for case in CASES:
        for horizon in (1024, 4096):
            selected = [r for r in records if r['case'] == case and r['horizon'] == horizon]
            assert [r['seed'] for r in selected] == list(range(512))
            for category in CATEGORIES:
                values = np.array([r['category'] == category for r in selected], dtype=np.float64)
                low, high = np.quantile(counts @ values / 512, [.025, .975])
                first_summary.append({'case': case, 'horizon': horizon, 'category': category,
                    'series': 512, 'count': int(values.sum()), 'fraction': float(values.mean()),
                    'CI_low_pointwise': float(low), 'CI_high_pointwise': float(high), 'bootstrap_draws': 2000})
            denominator = np.array([r['eligible_windows'] for r in selected], dtype=np.float64)
            for metric in window_metrics:
                values = np.array([r[metric] for r in selected], dtype=np.float64)
                draws = (counts @ values) / (counts @ denominator)
                low, high = np.quantile(draws, [.025, .975])
                window_summary.append({'case': case, 'horizon': horizon, 'metric': metric,
                    'window_count': int(values.sum()), 'total_fully_postbreak_scale_windows': int(denominator.sum()),
                    'fraction_of_fully_postbreak_scale_windows': float(values.sum() / denominator.sum()),
                    'CI_low_pointwise_seed_resampling': float(low), 'CI_high_pointwise_seed_resampling': float(high),
                    'seed_units': 512, 'bootstrap_draws': 2000, 'independent_window_assumption': False,
                    'not_an_event_detection_rate': True})
    csv_write('SCALE_FIRST_EVENT_FAILURE_PARTITION_NEXT7.csv', first_summary)
    csv_write('SCALE_POSTBREAK_EVIDENCE_WINDOWS_NEXT7.csv', window_summary)
    csv_write('SCALE_EVIDENCE_PATH_PER_SEED_NEXT7.csv', records)
    outputs = ('SCALE_FIRST_EVENT_FAILURE_PARTITION_NEXT7.csv', 'SCALE_POSTBREAK_EVIDENCE_WINDOWS_NEXT7.csv',
               'SCALE_EVIDENCE_PATH_PER_SEED_NEXT7.csv')
    save('SCALE_EVIDENCE_PATH_AUDIT_DONE_NEXT7.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
        'lock_sha256': identity, 'original_series_replayed': 2048, 'original_horizon_comparisons': 4096,
        'scalar_observations_replayed': 2048 * 4096, 'original_H_and_first_event_final_memory_parity': True,
        'original_count_sha256': digest(count_path), 'new_candidates': 0, 'new_fits': 0, 'new_draws': 0,
        'all_point_paths_exported': False, 'outputs_sha256': {name: digest(BASE / name) for name in outputs}})
    note('Fixed V1-V4 evidence audit complete', 'All2048 original seeds: first-event outcomes and fully postbreak scale evidence', 'F3_SYMMETRIC_SCALE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock-only', action='store_true')
    main(parser.parse_args().lock_only)
