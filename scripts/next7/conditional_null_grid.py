"""Resumable conditional null grid using only fixed historical/innovation indices."""
from pathlib import Path
import argparse
import copy
import csv
import json
import os
import sys
import time
import uuid
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from src.next7.common import BASE, save, read_json, digest, utc_now, activity, require_guard, note
from src.next7.fast_state import FastQualifiedState, HORIZONS, SUMMARY_COLUMNS
from src.next7.conditional_null import source_components, conditional_points, conditional_summary
from src.next7.calibration import load_calibration, cutoff
from src.next6_extension.synthetic import generated

CASES = ('N0', 'N1', 'N2', 'N3')
LENGTHS = (1000, 2048, 5000)
SOURCES = ('scripts/next7/conditional_null_grid.py', 'src/next7/fast_state.py',
    'src/next7/conditional_null.py', 'src/next7/tests/test_fast_conditional.py',
    'src/next7/null_growth.py', 'src/next7/event.py', 'src/next7/scale.py',
    'src/next6_extension/fresh_state.py', 'src/next6_extension/fresh_runtime.py',
    'src/next2/counterexamples.py', 'src/next6_extension/synthetic.py')


def source_hashes():
    return {n: digest(ROOT / n) for n in SOURCES}


def checked():
    require_guard()
    lock = read_json(BASE / 'CONDITIONAL_IMPLEMENTATION_LOCK.json')
    assert lock['source_sha256'] == source_hashes()
    assert lock['protocol_sha256'] == digest(BASE / 'CONDITIONAL_NULL_FORENSIC_LOCK.json')
    return lock


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(str(path))
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.rename(temporary, path)


def prepare():
    require_guard()
    tests = BASE / 'audit/FAST_CONDITIONAL_TESTS_001.xml'
    xml = ET.parse(tests)
    assert len(xml.findall('.//testcase')) == 20 and not xml.findall('.//failure') and not xml.findall('.//error')
    lockfile = BASE / 'CONDITIONAL_IMPLEMENTATION_LOCK.json'
    if not lockfile.exists():
        save('audit/FAST_CONDITIONAL_TEST_GATE.json', {'status': 'PASS', 'created_utc': utc_now(),
            'tests': 20, 'xml_sha256': digest(tests), 'source_sha256': source_hashes()})
        save('CONDITIONAL_IMPLEMENTATION_LOCK.json', {'status': 'FROZEN', 'created_utc': utc_now(),
            'source_sha256': source_hashes(), 'protocol_sha256': digest(BASE / 'CONDITIONAL_NULL_FORENSIC_LOCK.json'),
            'test_xml_sha256': digest(tests), 'before_crossed_grid': True})
    checked()
    cal = load_calibration()
    suite = copy.deepcopy(read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json'))
    suite['historical_length'] = 5000
    for case in CASES:
        relative = f'conditional_null/diagonal/{case}'
        directory = BASE / relative
        if (directory / 'DONE.json').exists():
            continue
        with activity('conditional generator/stream diagonal parity ' + case, 'TEST_GATE', 'F4_NULL_GROWTH_CORRECTION'):
            with (BASE / f'h_transport/{case}/SERIES.csv').open(newline='', encoding='utf-8') as f:
                previous = {(int(r['seed']), int(r['H_length']), int(r['horizon']), r['variant']): r for r in csv.DictReader(f)}
            summaries = np.empty((3, 512, 3, len(SUMMARY_COLUMNS)))
            maxima = np.empty((3, 512, 9, 2))
            hcal = []
            for index in range(512):
                h, eps, terminal, spec = source_components(case, index, suite)
                original_h, original_o, _, _ = generated(case, index, partition=33, horizon=16384, control=False, lock=suite)
                np.testing.assert_array_equal(h, original_h)
                actual = conditional_points(eps, float(spec['phi_before']), float(spec['arch_before']), *terminal)
                np.testing.assert_array_equal(actual, original_o)
                for position, length in enumerate(LENGTHS):
                    state = FastQualifiedState(h[-length:], core_cut=cal['core_cut'], scale_cut=cutoff(cal, length))
                    result, grid = conditional_summary(eps, float(spec['phi_before']), float(spec['arch_before']),
                        terminal[0], terminal[1], state.args)
                    summaries[position, index], maxima[position, index] = result, grid
                    for j, horizon in enumerate(HORIZONS):
                        for method, latch_col, first_col, strength_col in (
                            ('NEXT6_HARD4_EVENT', 5, 2, 8), ('G1_RAW_EVENT', 6, 3, 9), ('G2_QUALIFIED_EVENT', 7, 4, 10)):
                            expected = previous[index, length, int(horizon), method]
                            assert result[j, latch_col] == float(expected['any_latch'] == 'True')
                            assert result[j, first_col] == float(expected['first_latch'])
                            assert result[j, strength_col] == float(expected['event_final'])
                    hcal.append({'reference_index': index, 'H_length': length, 'calibration': state.calibration})
                if index % 64 == 63:
                    print(utc_now(), 'conditional diagonal', case, index + 1, '/512', flush=True)
            atomic_npz(directory / 'RESULTS.npz', summaries=summaries, grid_maxima=maxima)
            save(relative + '/H_CALIBRATIONS.json', hcal)
            save(relative + '/DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(), 'case': case,
                'reference_indices': 512, 'H_lengths': LENGTHS, 'raw_diagonal_exact': True,
                'all_horizon_latch_time_and_strength_match_H_transport': True,
                'RESULTS_sha256': digest(directory / 'RESULTS.npz'),
                'H_CALIBRATIONS_sha256': digest(directory / 'H_CALIBRATIONS.json')})
    if not (BASE / 'CONDITIONAL_DIAGONAL_GATE.json').exists():
        save('CONDITIONAL_DIAGONAL_GATE.json', {'status': 'PASS', 'created_utc': utc_now(),
            'backgrounds': 4, 'raw_diagonals': 2048, 'state_H_diagonals': 6144,
            'exact_raw_generator_and_existing_H_transport_results': True,
            'implementation_sha256': digest(lockfile),
            'case_receipts': {c: digest(BASE / f'conditional_null/diagonal/{c}/DONE.json') for c in CASES}})
    note('Conditional null diagonal gate PASS', 'Queue crossed fixed-innovation diagnostic with bounded workers', 'F4_NULL_GROWTH_CORRECTION')


def run(case, length):
    checked()
    assert case in CASES and length in LENGTHS
    assert read_json(BASE / 'CONDITIONAL_DIAGONAL_GATE.json')['status'] == 'PASS'
    relative = f'conditional_null/grid/{case}/H_{length}'
    directory = BASE / relative
    if (directory / 'DONE.json').exists():
        return
    cal = load_calibration()
    suite = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    components = [source_components(case, index, suite) for index in range(512)]
    histories = [v[0] for v in components]
    innovations = np.ascontiguousarray(np.stack([v[1] for v in components]))
    terminals = np.stack([v[2] for v in components])
    spec = components[0][3]
    del components
    diagonal_dir = BASE / f'conditional_null/diagonal/{case}'
    diagonal_receipt = read_json(diagonal_dir / 'DONE.json')
    assert digest(diagonal_dir / 'RESULTS.npz') == diagonal_receipt['RESULTS_sha256']
    assert digest(diagonal_dir / 'H_CALIBRATIONS.json') == diagonal_receipt['H_CALIBRATIONS_sha256']
    with np.load(diagonal_dir / 'RESULTS.npz', allow_pickle=False) as f:
        diagonal = f['summaries'][LENGTHS.index(length)].copy()
        diagonal_max = f['grid_maxima'][LENGTHS.index(length)].copy()
    tables = {r['reference_index']: r['calibration'] for r in read_json(diagonal_dir / 'H_CALIBRATIONS.json') if r['H_length'] == length}
    implementation = digest(BASE / 'CONDITIONAL_IMPLEMENTATION_LOCK.json')
    row_receipts = []
    for index in range(512):
        rowdir = directory / f'H_{index:03d}'
        outfile = rowdir / 'RESULTS.npz'
        done = rowdir / 'DONE.json'
        identity = {'case': case, 'H_length': length, 'reference_index': index,
            'innovation_indices': 512, 'implementation_sha256': implementation,
            'diagonal_receipt_sha256': digest(diagonal_dir / 'DONE.json')}
        if done.exists():
            receipt = read_json(done)
            assert receipt['identity'] == identity and digest(outfile) == receipt['result_sha256']
            row_receipts.append(receipt)
            continue
        with activity(f'conditional grid {case} H{length} reference{index}', 'REFERENCE_UNCERTAINTY', 'F4_NULL_GROWTH_CORRECTION'):
            if outfile.exists():
                # A completed atomic data file can survive a crash before its receipt.
                with np.load(outfile, allow_pickle=False) as f:
                    assert json.loads(str(f['identity_json'])) == identity
                    results, maxima = f['summaries'].copy(), f['grid_maxima'].copy()
                recovered = True
            else:
                checked()
                state = FastQualifiedState(histories[index][-length:], core_cut=cal['core_cut'], scale_cut=cutoff(cal, length))
                assert state.calibration == tables[index]
                results = np.empty((512, 3, len(SUMMARY_COLUMNS)))
                maxima = np.empty((512, 9, 2))
                started = time.perf_counter()
                for j in range(512):
                    if j == index:
                        results[j], maxima[j] = diagonal[index], diagonal_max[index]
                        continue
                    results[j], maxima[j] = conditional_summary(innovations[j], float(spec['phi_before']), float(spec['arch_before']),
                        terminals[index, 0], terminals[index, 1], state.clone().args)
                assert np.isfinite(results).all() and np.isfinite(maxima).all()
                atomic_npz(outfile, summaries=results, grid_maxima=maxima,
                    identity_json=np.array(json.dumps(identity, sort_keys=True)))
                recovered = False
                print(utc_now(), case, 'H', length, 'reference', index + 1, '/512',
                    'seconds', round(time.perf_counter() - started, 3), flush=True)
            assert results.shape == (512, 3, len(SUMMARY_COLUMNS)) and maxima.shape == (512, 9, 2)
            np.testing.assert_array_equal(results[index], diagonal[index])
            np.testing.assert_array_equal(maxima[index], diagonal_max[index])
            receipt = {'status': 'COMPLETE', 'created_utc': utc_now(), 'identity': identity,
                'result_sha256': digest(outfile), 'recovered_atomic_file_after_interruption': recovered,
                'mean_qualified_latch_by_horizon': results[:, :, 7].mean(axis=0).tolist(),
                'mean_fixed_horizon_exceedance': results[:, :, 12].mean(axis=0).tolist()}
            save(str(done.relative_to(BASE)), receipt)
            row_receipts.append(receipt)
    save(relative + '/DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(), 'case': case,
        'H_length': length, 'histories': 512, 'innovation_streams': 512, 'crossed_combinations': 512 * 512,
        'independent_seed_count_not_512_squared': True, 'row_receipts': row_receipts,
        'implementation_sha256': implementation, 'summary_columns': SUMMARY_COLUMNS})


def queue():
    checked()
    assert read_json(BASE / 'CONDITIONAL_DIAGONAL_GATE.json')['status'] == 'PASS'
    path = BASE / 'runtime/TASK_QUEUE.json'
    items = read_json(path) if path.exists() else []
    for case in CASES:
        for length in LENGTHS:
            task = {'id': f'conditional_{case}_H{length}', 'status': 'READY',
                'script': 'scripts/next7/conditional_null_grid.py', 'source_sha256': digest(__file__),
                'args': ['--mode', 'run', '--case', case, '--H-length', str(length)],
                'gates': ['PREFLIGHT_NEXT7.json', 'audit/FAST_CONDITIONAL_TEST_GATE.json', 'CONDITIONAL_DIAGONAL_GATE.json'],
                'hypothesis': 'F4_NULL_GROWTH_CORRECTION'}
            prior = [i for i in items if i['id'] == task['id']]
            if prior:
                assert prior == [task]
            else:
                items.append(task)
    save('runtime/TASK_QUEUE.json', items, False)
    note('Conditional null grid queued', 'Twelve diagnostic tasks, maximum two owned workers, fixed sources and row recovery', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['prepare', 'run', 'queue'], required=True)
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--H-length', type=int, choices=LENGTHS)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'queue':
        queue()
    else:
        run(args.case, args.H_length)
