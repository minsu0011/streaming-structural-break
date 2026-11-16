"""Partition completed crossed null grids by the unchanged resampler's support."""
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
from src.next7.diagnostic_statistics import support_partition
from src.next7.crossed_statistics import seed_pair_product_draws
from scripts.next7.conditional_null_grid import atomic_npz

NAMES = ('BEYOND_EMPIRICAL_BOOTSTRAP_SUPPORT', 'WITHIN_SUPPORT_ABOVE_TERMINAL_CUT', 'EARLIER_CROSSING_RETAINED_ONLY')


def lock():
    sources = {n: digest(ROOT / n) for n in ('scripts/next7/decompose_bootstrap_support.py',
        'src/next7/diagnostic_statistics.py', 'src/next7/crossed_statistics.py')}
    path = BASE / 'SUPPORT_PARTITION_IMPLEMENTATION_LOCK.json'
    if not path.exists():
        tests = BASE / 'audit/DIAGNOSTIC_STATISTICS_TESTS_001.xml'
        xml = ET.parse(tests)
        assert len(xml.findall('.//testcase')) == 6 and not xml.findall('.//failure') and not xml.findall('.//error')
        save(path.name, {'status': 'FROZEN', 'created_utc': utc_now(), 'source_sha256': sources,
            'support_protocol_sha256': digest(BASE / 'BOOTSTRAP_SUPPORT_FORENSIC_LOCK.json'),
            'crossed_statistics_sha256': digest(BASE / 'CROSSED_STATISTICS_LOCK.json'),
            'test_xml_sha256': digest(tests), 'partition': list(NAMES),
            'strict_formula': 'M>U; T<M<=U; qualified and M<=T, with T<=U and all three disjoint summing to qualified.',
            'claim_scope': 'Unchanged empirical resampling support and original detector only. No oracle threshold deployment or distribution-free guarantee.',
            'resampling': 'Existing2000 same-seed-pair product weights; not independent262144 cells.',
            'new_fits': 0, 'new_random_draws': 0, 'new_cutoffs': 0})
    assert read_json(path)['source_sha256'] == sources
    return read_json(BASE / 'CROSSED_STATISTICS_LOCK.json')


def summarize(case, length):
    require_guard()
    protocol = lock()
    relative = f'support_partition/{case}/H_{length}'
    dest = BASE / relative
    if (dest / 'DONE.json').exists():
        return
    support_dir = BASE / f'bootstrap_support/{case}/H_{length}'
    support_done = read_json(support_dir / 'DONE.json')
    assert support_done['status'] == 'COMPLETE' and support_done['references'] == 512
    assert digest(support_dir / 'PER_H_SUPPORT.csv') == support_done['summary_sha256']
    with (support_dir / 'PER_H_SUPPORT.csv').open(newline='', encoding='utf-8') as f:
        support_rows = list(csv.DictReader(f))
    assert [int(r['reference_index']) for r in support_rows] == list(range(512))
    envelope = np.array([float(r['empirical_support_envelope_with_floor1']) for r in support_rows])
    grid_dir = BASE / f'conditional_null/grid/{case}/H_{length}'
    grid_done = read_json(grid_dir / 'DONE.json')
    assert grid_done['status'] == 'COMPLETE' and len(grid_done['row_receipts']) == 512
    counts_path = BASE / protocol['count_file']
    assert digest(counts_path) == protocol['count_file_sha256']
    counts = np.load(counts_path, allow_pickle=False)
    assert counts.shape == (2000, 512)
    with activity(f'Complete empirical-support failure partition {case} H{length}', 'NULL_SUPPORT_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
        values = np.empty((512, 512, 3, 14))
        for i, receipt in enumerate(grid_done['row_receipts']):
            assert receipt['identity']['reference_index'] == i
            path = grid_dir / f'H_{i:03d}/RESULTS.npz'
            assert digest(path) == receipt['result_sha256']
            with np.load(path, allow_pickle=False) as f:
                values[i] = f['summaries']
        pieces = support_partition(values[..., 0], values[..., 11], envelope[:, None, None], values[..., 7])
        summaries, per_h, draws = [], [], {}
        for j, horizon in enumerate((1024, 4096, 16384)):
            total_draws = seed_pair_product_draws(values[:, :, j, 7], counts)
            total = float(values[:, :, j, 7].mean())
            partition_draws = []
            for k, name in enumerate(NAMES):
                m = pieces[k, :, :, j]
                d = seed_pair_product_draws(m, counts)
                draws[f'{name}_{horizon}'] = d
                partition_draws.append(d)
                fraction = np.divide(d, total_draws, out=np.full(len(d), np.nan), where=total_draws > 0)
                good = fraction[np.isfinite(fraction)]
                lo, hi = np.quantile(d, [.025, .975])
                rate = float(m.mean())
                summaries.append({'case': case, 'H_length': length, 'horizon': horizon, 'component': name,
                    'crossed_rate': rate, 'CI_low_approx_pointwise': float(lo), 'CI_high_approx_pointwise': float(hi),
                    'fraction_of_all_G2_latches': rate / total if total else None,
                    'latch_fraction_CI_low': float(np.quantile(good, .025)) if len(good) else None,
                    'latch_fraction_CI_high': float(np.quantile(good, .975)) if len(good) else None,
                    'fraction_usable_draws': len(good), 'diagonal_original_rate': float(np.diag(m).mean()),
                    'reference_histories': 512, 'reused_innovation_streams': 512, 'independent_cell_claim': False,
                    'used_as_new_cutoff': False})
                for i in range(512):
                    per_h.append({'case': case, 'H_length': length, 'horizon': horizon, 'reference_index': i,
                        'component': name, 'conditional_rate': float(m[i].mean()),
                        'empirical_support_envelope': float(envelope[i]), 'terminal_cut': float(values[i, 0, j, 11])})
            np.testing.assert_allclose(np.sum(partition_draws, axis=0), total_draws, rtol=0, atol=2e-16)
        csv_write(relative + '/SUMMARY.csv', summaries)
        csv_write(relative + '/PER_H_PARTITION.csv', per_h)
        atomic_npz(dest / 'BOOTSTRAP_DRAWS.npz', **draws)
        atomic_npz(dest / 'PARTITION_FOR_FIGURES.npz', partition=pieces, envelope=envelope)
        save(relative + '/DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
            'grid_done_sha256': digest(grid_dir / 'DONE.json'), 'support_done_sha256': digest(support_dir / 'DONE.json'),
            'implementation_sha256': digest(BASE / 'SUPPORT_PARTITION_IMPLEMENTATION_LOCK.json'),
            'every_cell_partition_exact': True, 'every_resample_additive': True,
            'source_files_sha256': {p.name: digest(p) for p in dest.iterdir() if p.is_file()}})
    note('Empirical-support decomposition complete', f'{case} H{length}: unchanged resampler support and original cut separated', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock-only', action='store_true')
    parser.add_argument('--case', choices=['N0', 'N1', 'N2', 'N3'])
    parser.add_argument('--H-length', type=int, choices=[1000, 2048, 5000])
    args = parser.parse_args()
    if args.lock_only:
        lock()
    else:
        assert args.case is not None and args.H_length is not None
        summarize(args.case, args.H_length)
