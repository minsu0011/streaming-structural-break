"""Compute the exact empirical-bootstrap support envelope for every frozen H."""
from pathlib import Path
import os
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, require_guard, note
from src.next7.bootstrap_support import exact_support
from src.next7.conditional_null import source_components
from src.next7.calibration import load_calibration, cutoff
from src.next7.null_growth import bootstrap_maxima, shared_uniforms, GRID, ORDER_INDEX
from src.next6_extension.fresh_runtime import SafeState
from src.next7.scale import ScaleState


def main():
    guard = require_guard()
    protocol_path = BASE / 'BOOTSTRAP_SUPPORT_FORENSIC_LOCK.json'
    protocol = read_json(protocol_path)
    repair_path = BASE / 'BOOTSTRAP_SUPPORT_RING_ORDER_REPAIR_001.json'
    if repair_path.exists():
        repair = read_json(repair_path)
        assert protocol['source_sha256'] == repair['old_source_sha256']['src/next7/bootstrap_support.py']
        assert all(digest(ROOT / n) == sha for n, sha in repair['new_source_sha256'].items())
    else:
        assert protocol['source_sha256'] == digest(ROOT / 'src/next7/bootstrap_support.py')
    tests_path = BASE / ('audit/BOOTSTRAP_SUPPORT_TESTS_002.xml' if repair_path.exists() else 'audit/BOOTSTRAP_SUPPORT_TESTS_001.xml')
    xml = ET.parse(tests_path)
    assert len(xml.findall('.//testcase')) == (5 if repair_path.exists() else 4) and not xml.findall('.//failure') and not xml.findall('.//error')
    hashes = {n: digest(ROOT / n) for n in ('scripts/next7/prepare_bootstrap_support.py', 'src/next7/bootstrap_support.py',
        'src/next7/conditional_null.py', 'src/next7/null_growth.py', 'src/next7/scale.py', 'src/next6_extension/fresh_runtime.py',
        'src/next6_extension/fresh_state.py')}
    lock = BASE / 'BOOTSTRAP_SUPPORT_IMPLEMENTATION_LOCK.json'
    if not lock.exists():
        save(lock.name, {'status': 'FROZEN', 'created_utc': utc_now(), 'source_sha256': hashes,
            'protocol_sha256': digest(protocol_path), 'test_xml_sha256': digest(tests_path)})
    assert read_json(lock)['source_sha256'] == hashes
    suite = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    calibration = load_calibration()
    all_rows = []
    for case in ('N0', 'N1', 'N2', 'N3'):
        parent = BASE / f'conditional_null/diagonal/{case}'
        diagonal_done = read_json(parent / 'DONE.json')
        assert digest(parent / 'H_CALIBRATIONS.json') == diagonal_done['H_CALIBRATIONS_sha256']
        original = {(r['H_length'], r['reference_index']): r['calibration'] for r in read_json(parent / 'H_CALIBRATIONS.json')}
        for length in (1000, 2048, 5000):
            relative = f'bootstrap_support/{case}/H_{length}'
            directory = BASE / relative
            result_rows = []
            with activity(f'Exact H empirical support {case} H{length}', 'H_CALIBRATION_SUPPORT_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
                for seed in range(512):
                    receipt = directory / f'H_{seed:03d}.json'
                    if receipt.exists():
                        record = read_json(receipt)
                        assert record['implementation_sha256'] == digest(lock)
                        result_rows.append(record['result'])
                        continue
                    h, _, _, _ = source_components(case, seed, suite)
                    h = h[-length:]
                    expected = original[length, seed]
                    prefix = int(.6 * length)
                    assert expected['reference_valid'] and expected['reference_prefix_length'] == prefix
                    core = SafeState(h[:prefix], threshold=calibration['core_cut'])
                    scale = ScaleState(h[:prefix])
                    c, s = core.replay(h[prefix:]), scale.replay(h[prefix:])
                    completed = s[:, 0] > .5
                    vectors = np.column_stack((c[completed, 12:20], s[completed, 2]))
                    assert len(vectors) == expected['pseudo_blocks']
                    np.testing.assert_array_equal(vectors.min(axis=0), expected['pseudo_vector_min'])
                    np.testing.assert_array_equal(vectors.max(axis=0), expected['pseudo_vector_max'])
                    scale_cut = cutoff(calibration, length)
                    maxima = bootstrap_maxima(vectors, shared_uniforms(), GRID, calibration['core_cut'], scale_cut)
                    cuts = np.maximum(1., np.sort(maxima, axis=0)[ORDER_INDEX])
                    np.testing.assert_array_equal(cuts, expected['admission_cuts'])
                    internal, stitched, envelope = exact_support(vectors, calibration['core_cut'], scale_cut)
                    assert np.max(maxima) <= envelope and np.max(cuts) <= envelope
                    row = {'case': case, 'H_length': length, 'reference_index': seed,
                        'reference_prefix_length': prefix, 'pseudo_blocks': len(vectors),
                        'possible_four_block_chunks': len(vectors) - 3,
                        'internal_window_max_ratio': float(internal), 'stitched_window_max_ratio': float(stitched),
                        'empirical_support_envelope_with_floor1': float(envelope),
                        'maximum_original_bootstrap_ratio': float(maxima.max()),
                        'cut_at1024': float(cuts[4]), 'cut_at4096': float(cuts[6]), 'cut_at16384': float(cuts[8]),
                        'envelope_minus_final_cut': float(envelope - cuts[-1]),
                        'stitching_exceeds_every_internal_H_window': bool(stitched > internal),
                        'core_prefix_calibration_floor_nodes': len(core.calibration.get('spread_floor_nodes', [])),
                        'used_as_new_detector_cut': False}
                    save(str(receipt.relative_to(BASE)), {'status': 'COMPLETE', 'created_utc': utc_now(),
                        'implementation_sha256': digest(lock), 'original_G2_all_grid_cuts_exact': True, 'result': row})
                    result_rows.append(row)
                    if seed % 32 == 31:
                        print(utc_now(), 'bootstrap support', case, length, seed + 1, '/512', flush=True)
                if not (directory / 'DONE.json').exists():
                    csv_write(relative + '/PER_H_SUPPORT.csv', result_rows)
                    save(relative + '/DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
                        'case': case, 'H_length': length, 'references': 512,
                        'original_G2_all_grid_cuts_exact': True, 'every_bootstrap_maximum_within_exact_envelope': True,
                        'source_sha256': hashes, 'summary_sha256': digest(directory / 'PER_H_SUPPORT.csv')})
                all_rows.extend(result_rows)
    csv_write('BOOTSTRAP_EMPIRICAL_SUPPORT_NEXT7.csv', all_rows)
    save('BOOTSTRAP_SUPPORT_DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
        'H_references_by_length_and_background': len(all_rows), 'source_sha256': hashes,
        'result_sha256': digest(BASE / 'BOOTSTRAP_EMPIRICAL_SUPPORT_NEXT7.csv'),
        'scope': 'Diagnostic exact support of unchanged G2 empirical resampler; no new threshold or fit.'})
    guard.check_state()
    note('H empirical-bootstrap support envelopes complete', 'Bounded resampling support quantified; compare with complete future grids next', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    main()
