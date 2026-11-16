"""Summarize complete conditional cohorts with fixed crossed-seed accounting."""
from pathlib import Path
import argparse
import os
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from scipy.stats import spearmanr
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, require_guard, note
from src.next7.crossed_statistics import seed_pair_product_draws, conditional_H_draws, complete_grid_decomposition
from scripts.next7.conditional_null_grid import atomic_npz


def lock():
    protocol = read_json(BASE / 'CROSSED_STATISTICS_LOCK.json')
    assert digest(BASE / protocol['count_file']) == protocol['count_file_sha256']
    assert digest(ROOT / 'src/next7/crossed_statistics.py') == protocol['statistics_source_sha256']
    sources = {n: digest(ROOT / n) for n in ('scripts/next7/aggregate_conditional_grid.py', 'src/next7/crossed_statistics.py')}
    path = BASE / 'CROSSED_AGGREGATOR_IMPLEMENTATION_LOCK.json'
    if not path.exists():
        tests = BASE / 'audit/CROSSED_STATISTICS_TESTS_001.xml'
        xml = ET.parse(tests)
        assert len(xml.findall('.//testcase')) == 4 and not xml.findall('.//failure') and not xml.findall('.//error')
        save(path.name, {'status': 'FROZEN', 'created_utc': utc_now(), 'source_sha256': sources,
            'protocol_sha256': digest(BASE / 'CROSSED_STATISTICS_LOCK.json'), 'test_xml_sha256': digest(tests)})
    else:
        assert read_json(path)['source_sha256'] == sources
    return protocol


def summarize(case, length):
    require_guard()
    protocol = lock()
    relative = f'conditional_statistics/{case}/H_{length}'
    dest = BASE / relative
    if (dest / 'DONE.json').exists():
        return
    source = BASE / f'conditional_null/grid/{case}/H_{length}'
    receipt = read_json(source / 'DONE.json')
    assert receipt['status'] == 'COMPLETE' and len(receipt['row_receipts']) == 512
    counts = np.load(BASE / protocol['count_file'], allow_pickle=False)
    assert counts.shape == (2000, 512) and np.all(counts.sum(axis=1) == 512)
    with activity(f'complete conditional grid uncertainty {case} H{length}', 'CROSSED_NULL_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
        values = np.empty((512, 512, 3, 14))
        grid = np.empty((512, 512, 9, 2))
        for i, row in enumerate(receipt['row_receipts']):
            assert row['identity']['reference_index'] == i
            path = source / f'H_{i:03d}/RESULTS.npz'
            assert digest(path) == row['result_sha256']
            with np.load(path, allow_pickle=False) as f:
                values[i], grid[i] = f['summaries'], f['grid_maxima']
        assert np.isfinite(values).all() and np.isfinite(grid).all()
        diagonal_path = BASE / f'conditional_null/diagonal/{case}'
        with np.load(diagonal_path / 'RESULTS.npz', allow_pickle=False) as f:
            expected = f['summaries'][[1000, 2048, 5000].index(length)]
            np.testing.assert_array_equal(values[np.arange(512), np.arange(512)], expected)
        raw, qualified, terminal = values[..., 6], values[..., 7], values[..., 12]
        assert np.all(values[..., 5] <= raw) and np.all(qualified <= raw) and np.all(terminal <= qualified)
        early = qualified - terminal
        np.testing.assert_array_equal(terminal + early, qualified)
        hrows, summaries, correlations, draws = [], [], [], {}
        matrices = {'CORE_EVER_LATCH': values[..., 5], 'G1_RAW_EVER_LATCH': raw,
            'G2_QUALIFIED_EVER_LATCH': qualified, 'FIXED_HORIZON_MAX_EXCEEDS_TERMINAL_CUT': terminal,
            'EARLIER_CROSSING_RETAINED_ONLY': early}
        for j, horizon in enumerate((1024, 4096, 16384)):
            for method, tensor in matrices.items():
                matrix = tensor[:, :, j]
                boot = seed_pair_product_draws(matrix, counts)
                lo, hi = np.quantile(boot, [.025, .975])
                row_mean = matrix.mean(axis=1)
                summary = {'case': case, 'H_length': length, 'horizon': horizon, 'metric': method,
                    'mean_crossed_rate': float(matrix.mean()), 'CI_low_approx_pointwise': float(lo),
                    'CI_high_approx_pointwise': float(hi), 'diagonal_original_rate': float(np.diag(matrix).mean()),
                    'reference_histories': 512, 'innovation_streams': 512, 'crossed_cells': 512 * 512,
                    'bootstrap_draws': 2000, 'cell_independence_claim': False,
                    **complete_grid_decomposition(matrix)}
                for quantile in (0., .1, .25, .5, .75, .9, .95, 1.):
                    summary[f'conditional_H_rate_quantile_{quantile:g}'] = float(np.quantile(row_mean, quantile))
                for threshold in (.05, .1, .5):
                    summary[f'fraction_H_conditional_rate_above_{threshold:g}'] = float(np.mean(row_mean > threshold))
                summaries.append(summary)
                draws[f'{method}_{horizon}'] = boot
            q = qualified[:, :, j]
            conditional = conditional_H_draws(q, counts)
            intervals = np.quantile(conditional, [.025, .975], axis=0)
            cut = values[:, 0, j, 11]
            np.testing.assert_array_equal(values[:, :, j, 11], np.broadcast_to(cut[:, None], (512, 512)))
            for i in range(512):
                row = {'case': case, 'H_length': length, 'horizon': horizon, 'reference_index': i,
                    'conditional_G2_latch_rate': float(q[i].mean()),
                    'conditional_MC_CI_low': float(intervals[0, i]), 'conditional_MC_CI_high': float(intervals[1, i]),
                    'fixed_horizon_exceedance_rate': float(terminal[i, :, j].mean()),
                    'earlier_retained_only_rate': float(early[i, :, j].mean()),
                    'core_latch_rate': float(values[i, :, j, 5].mean()), 'raw_latch_rate': float(raw[i, :, j].mean()),
                    'terminal_admission_cut': float(cut[i]),
                    'empirical_future_max_ratio_q95_diagnostic_only': float(np.quantile(values[i, :, j, 0], .95)),
                    'maximum_future_routed_ratio': float(values[i, :, j, 1].max()),
                    'new_cut_selected': False}
                hrows.append(row)
            correlations.append({'case': case, 'H_length': length, 'horizon': horizon,
                'cut_vs_conditional_G2_latch_Spearman': float(spearmanr(cut, q.mean(axis=1)).statistic) if np.ptp(cut) and np.ptp(q.mean(axis=1)) else None,
                'posthoc_descriptive_not_causal': True})
        # Exact same-draw contrasts preserve the shared histories and innovations.
        contrasts = []
        for horizon in (1024, 4096, 16384):
            for candidate, reference in [('G2_QUALIFIED_EVER_LATCH', 'G1_RAW_EVER_LATCH'),
                                         ('G2_QUALIFIED_EVER_LATCH', 'CORE_EVER_LATCH'),
                                         ('EARLIER_CROSSING_RETAINED_ONLY', 'FIXED_HORIZON_MAX_EXCEEDS_TERMINAL_CUT')]:
                delta = draws[f'{candidate}_{horizon}'] - draws[f'{reference}_{horizon}']
                lo, hi = np.quantile(delta, [.025, .975])
                selected = {r['metric']: r['mean_crossed_rate'] for r in summaries if r['horizon'] == horizon}
                contrasts.append({'case': case, 'H_length': length, 'horizon': horizon, 'candidate': candidate, 'reference': reference,
                    'mean_rate_delta': selected[candidate] - selected[reference],
                    'CI_low_approx_pointwise': float(lo), 'CI_high_approx_pointwise': float(hi),
                    'draws': 2000, 'same_seed_pair_counts': True, 'promotion_use': False})
        csv_write(relative + '/SUMMARY.csv', summaries)
        csv_write(relative + '/PER_H_CONDITIONAL_RISK.csv', hrows)
        csv_write(relative + '/PAIRED_CONTRASTS.csv', contrasts)
        csv_write(relative + '/CUTOFF_CORRELATIONS.csv', correlations)
        atomic_npz(dest / 'BOOTSTRAP_DRAWS.npz', **draws)
        atomic_npz(dest / 'GRID_FOR_FIGURES.npz', qualified=qualified.astype(np.uint8),
            terminal=terminal.astype(np.uint8), early_only=early.astype(np.uint8),
            terminal_cuts=values[:, 0, :, 11], row_grid_raw_max_q95=np.quantile(grid[..., 0], .95, axis=1))
        save(relative + '/DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(), 'case': case, 'H_length': length,
            'input_done_sha256': digest(source / 'DONE.json'), 'source_sha256': digest(__file__),
            'statistics_lock_sha256': digest(BASE / 'CROSSED_STATISTICS_LOCK.json'),
            'cellwise_retention_partition_exact': True, 'diagonal_matches_H_transport': True,
            'new_fits': 0, 'new_random_draws': 0,
            'files_sha256': {p.name: digest(p) for p in dest.iterdir() if p.is_file()}})
    note('Conditional cohort uncertainty complete', f'{case} H{length}: complete crossed grid,512 seed units, explicit retained-boundary partition', 'F4_NULL_GROWTH_CORRECTION')


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
