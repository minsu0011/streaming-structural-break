"""Audit bounded scale admission on original synthetic and cached RP cohorts."""
from pathlib import Path
import csv
import os
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
from src.next7.worker_environment import environment
os.environ.update(environment())
import numpy as np
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, note, require_guard
from src.next7.calibration import load_calibration, cutoff
from src.next7.null_growth import GRID
from src.next6_extension.synthetic import CASES


def csv_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def fraction(values, counts):
    values = np.asarray(values, dtype=np.float64)
    draws = counts @ values / 512
    low, high = np.quantile(draws, [.025, .975])
    return {'histories': len(values), 'count': int(values.sum()), 'fraction': float(values.mean()),
            'CI_low_pointwise': float(low), 'CI_high_pointwise': float(high)}


def main():
    guard = require_guard()
    calibration = load_calibration()
    count_path = BASE / 'synthetic_stats/COUNTS_2000x512.npy'
    assert digest(count_path) == read_json(BASE / 'synthetic_stats/COUNTS_LOCK.json')['sha256']
    original = {}
    for case in CASES:
        parent = BASE / f'synthetic_g2_full/{case}'
        done = read_json(parent / 'DONE.json')
        assert done['status'] == 'COMPLETE' and done['seeds'] == 512
        assert digest(parent / 'SERIES.csv') == done['SERIES_sha256']
        for name in ('H_CALIBRATIONS.json', 'SERIES.csv', 'DONE.json'):
            path = parent / name
            original[path.relative_to(BASE).as_posix()] = digest(path)
    for generation, names in [('g1', ('state.npy',)), ('g2', ('qualified.npy', 'H_CALIBRATIONS.json'))]:
        parent = BASE / f'cache/{generation}'
        manifest = read_json(parent / 'MANIFEST.json')
        assert manifest['status'] == 'COMPLETE' and manifest['series'] == 6400
        for name in names:
            assert digest(parent / name) == manifest['files_sha256'][name]
            original[(parent / name).relative_to(BASE).as_posix()] = digest(parent / name)
        original[(parent / 'MANIFEST.json').relative_to(BASE).as_posix()] = digest(parent / 'MANIFEST.json')
    directory = ROOT / 'data/next5/research_5d54e316ff5bf25d'
    manifest = read_json(directory / 'MANIFEST.json')
    assert digest(directory / 'MANIFEST.json') == read_json(BASE / 'cache/g1/MANIFEST.json')['RP_manifest_sha256']
    assert digest(directory / 'series.npz') == manifest['files_sha256']['series.npz']
    save('OBSERVED_SCALE_ADMISSION_LOCK_NEXT7.json', {
        'status': 'FROZEN', 'created_utc': utc_now(), 'source_sha256': digest(__file__),
        'parent_theorem_done_sha256': digest(BASE / 'SCALE_ADMISSION_CEILING_DONE_NEXT7.json'),
        'inputs_sha256': original, 'original_bootstrap_counts_sha256': digest(count_path),
        'RP_series_metadata_sha256': digest(directory / 'series.npz'),
        'scope': 'Posthoc read-only algebra audit: original23 synthetic512-seed cohorts and all6400 cached RP state paths. No new paths, H fitting, labels, tau, candidate, threshold or draws.',
        'synthetic_H_hash_caveat': 'Original synthetic DONE bound SERIES/SUMMARY but not H_CALIBRATIONS. Their current bytes are frozen here before analysis; original SERIES final cuts independently cross-check every evaluated horizon. RP calibration and both state caches have their original manifest hashes.',
        'estimands': ['Synthetic marginal H fraction with cutoff at least4/scale_cut at each original grid',
                      'Synthetic observed first-detection misses intersecting ceiling blockage by each original horizon; noncausal upper bound on affected histories',
                      'RP exact census of supported raw scale confirmations partitioned into hard-ceiling rejection, subceiling rejection and scale cutoff pass; core can independently qualify'],
        'uncertainty': 'Synthetic fractions use original2000 paired seed multiplicity rows. RP census has no independence assumption or new bootstrap. Export aggregate RP groups only.'})
    counts = np.load(count_path, allow_pickle=False)
    summaries, observed = [], []
    with activity('Actual synthetic and RP bounded scale-admission audit', 'SCALE_ADMISSION_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
        for case in CASES:
            parent = BASE / f'synthetic_g2_full/{case}'
            hrows = sorted(read_json(parent / 'H_CALIBRATIONS.json'), key=lambda r: r['seed'])
            assert [r['seed'] for r in hrows] == list(range(512))
            assert all(r['case'] == case and r['H_length'] == 2048 and r['reference_valid'] for r in hrows)
            cuts = np.array([r['admission_cuts'] for r in hrows])
            assert cuts.shape == (512, 9) and np.all(np.diff(cuts, axis=1) >= 0)
            cap = 4 / cutoff(calibration, 2048)
            blocked = cuts >= cap
            for j, grid in enumerate(GRID):
                summaries.append({'case': case, 'H_length': 2048, 'grid_blocks': int(grid),
                    'horizon_points': int(grid * 16), 'scale_raw_ratio_ceiling': cap,
                    **fraction(blocked[:, j], counts), 'metric': 'scale_only_qualification_impossible_at_cut',
                    'not_a_detection_rate': True, 'earlier_memory_can_exist': True})
            series = csv_rows(parent / 'SERIES.csv')
            for horizon in sorted({int(r['horizon']) for r in series}):
                selected = sorted((r for r in series if int(r['horizon']) == horizon), key=lambda r: int(r['seed']))
                assert [int(r['seed']) for r in selected] == list(range(512))
                j = int(np.searchsorted(GRID, horizon // 16, side='left'))
                np.testing.assert_array_equal([float(r['admission_cut_final']) for r in selected], cuts[:, j])
                masks = {'hard_ceiling_active_by_horizon': blocked[:, j]}
                if all(r['observed_break'] == 'True' for r in selected):
                    miss = np.array([r['first_detection'] != 'True' for r in selected])
                    masks.update(first_detection_miss=miss, miss_with_hard_ceiling=miss & blocked[:, j],
                                 miss_without_hard_ceiling=miss & ~blocked[:, j])
                for metric, values in masks.items():
                    observed.append({'case': case, 'horizon': horizon, 'metric': metric,
                        **fraction(values, counts), 'interpretation': 'Descriptive intersection, not causal attribution. Monotone cuts make terminal blockage an upper bound on histories exposed to the hard ceiling within this horizon; it does not bound subceiling sensitivity loss.'})
        csv_write('OBSERVED_SYNTHETIC_SCALE_CEILING_NEXT7.csv', summaries)
        csv_write('OBSERVED_SCALE_CEILING_MISS_PARTITION_NEXT7.csv', observed)
        with np.load(directory / 'series.npz', allow_pickle=False) as archive:
            identifiers = archive['dataset_id']
            guard.admit(identifiers, purpose='NEXT7 aggregate cached scale-admission forensic')
            assert len(identifiers) == 6400 and set(map(int, identifiers)) == guard.research_ids
            offsets = archive['online_offsets']
            lengths = np.diff(archive['historical_offsets'])
        raw = np.load(BASE / 'cache/g1/state.npy', mmap_mode='r', allow_pickle=False)
        qualified = np.load(BASE / 'cache/g2/qualified.npy', mmap_mode='r', allow_pickle=False)
        assert raw.shape == (3203157, 30) and qualified.shape == (3203157, 15)
        hcals = {r['dataset_id']: r for r in read_json(BASE / 'cache/g2/H_CALIBRATIONS.json')}
        bin_lengths = [r['H_length'] for r in calibration['tables']]
        aggregate = {}
        total_checked = 0
        for index, sid in enumerate(identifiers):
            cal = hcals[int(sid)]
            length = int(lengths[index])
            assert cal['H_length'] == length
            first, last = offsets[index:index + 2]
            r, q = raw[first:last], qualified[first:last]
            cap = 4 / cutoff(calibration, length)
            assert np.all(r[:, 11] <= cap)
            supported = bool(cal['reference_valid']) & (r[:, 16] <= 1024)
            positions = np.searchsorted(GRID, r[:, 16], side='left').clip(0, 8)
            expected_cuts = np.where(supported, np.asarray(cal['admission_cuts'])[positions], cal['unsupported_cut'])
            np.testing.assert_array_equal(q[:, 12], expected_cuts)
            np.testing.assert_array_equal(q[:, 13], supported.astype(float))
            eligible = (r[:, 15] > .5) & (r[:, 29] > .5) & supported
            hard = q[:, 12] >= cap
            fresh_q = q[:, 10] > .5
            np.testing.assert_array_equal(fresh_q, (r[:, 15] > .5) & (r[:, 18] > .5) & supported & (r[:, 2] > q[:, 12]))
            assert not np.any(fresh_q & hard & (q[:, 7] >= 8))
            assert np.all(r[fresh_q & hard, 13] > q[fresh_q & hard, 12])
            masks = {
                'raw_scale_confirmation_supported': eligible,
                'raw_scale_confirmation_hard_ceiling_rejected': eligible & hard,
                'raw_scale_confirmation_subceiling_rejected': eligible & ~hard & (r[:, 11] <= q[:, 12]),
                'raw_scale_confirmation_exceeds_admission_cut': eligible & (r[:, 11] > q[:, 12]),
                'raw_scale_rejected_but_core_qualified': eligible & (r[:, 11] <= q[:, 12]) & fresh_q,
                'raw_scale_rejected_and_no_qualification': eligible & (r[:, 11] <= q[:, 12]) & ~fresh_q,
                'fresh_qualified_with_hard_scale_ceiling': fresh_q & hard,
                'fresh_qualified_scale_winner': fresh_q & (q[:, 7] >= 8),
                'fresh_qualified_core_winner': fresh_q & (q[:, 7] < 8)}
            assert int(eligible.sum()) == sum(int(masks[k].sum()) for k in (
                'raw_scale_confirmation_hard_ceiling_rejected', 'raw_scale_confirmation_subceiling_rejected',
                'raw_scale_confirmation_exceeds_admission_cut'))
            bin_index = max(0, int(np.searchsorted(bin_lengths, length, side='right')) - 1)
            for group in ('ALL_RP', f'cut_bin_H{bin_lengths[bin_index]}'):
                acc = aggregate.setdefault(group, {'series': 0, 'rows': 0, 'eligible_blocks': 0, 'counts': {}, 'series_counts': {}})
                acc['series'] += 1
                acc['rows'] += len(r)
                acc['eligible_blocks'] += int(eligible.sum())
                for metric, mask in masks.items():
                    acc['counts'][metric] = acc['counts'].get(metric, 0) + int(mask.sum())
                    acc['series_counts'][metric] = acc['series_counts'].get(metric, 0) + int(mask.any())
            total_checked += len(r)
        report = []
        for group, acc in aggregate.items():
            for metric, count in acc['counts'].items():
                report.append({'group': group, 'metric': metric, 'block_events': count,
                    'series_with_at_least_one_event': acc['series_counts'][metric],
                    'total_series': acc['series'], 'total_online_rows': acc['rows'],
                    'total_supported_raw_scale_confirmations': acc['eligible_blocks'],
                    'fraction_of_raw_scale_confirmations': count / acc['eligible_blocks'] if acc['eligible_blocks'] and metric.startswith('raw_scale_') else None,
                    'series_fraction': acc['series_counts'][metric] / acc['series'],
                    'unit': 'Exact RP census; repeated block events are not independent samples',
                    'no_tau_labels_or_outcomes_used': True})
        csv_write('RP_SCALE_ADMISSION_PARTITION_NEXT7.csv', report)
        outputs = ['OBSERVED_SYNTHETIC_SCALE_CEILING_NEXT7.csv', 'OBSERVED_SCALE_CEILING_MISS_PARTITION_NEXT7.csv', 'RP_SCALE_ADMISSION_PARTITION_NEXT7.csv']
        save('OBSERVED_SCALE_ADMISSION_DONE_NEXT7.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
            'lock_sha256': digest(BASE / 'OBSERVED_SCALE_ADMISSION_LOCK_NEXT7.json'),
            'original_synthetic_cohorts': len(CASES), 'histories_per_cohort': 512,
            'synthetic_final_cut_crosschecks': sum(512 * (3 if case.startswith('N') else 2) for case in CASES),
            'RP_series': 6400, 'RP_rows_checked': total_checked,
            'RP_fresh_scale_winner_hard_ceiling_violations': 0,
            'RP_qualified_cut_support_and_fresh_decisions_exact': True,
            'new_candidates': 0, 'new_fits': 0, 'new_draws': 0, 'new_online_paths': 0,
            'exported_RP_series_identifiers_or_tau': False,
            'outputs_sha256': {name: digest(BASE / name) for name in outputs}})
    guard.check_state()
    note('Observed scale admission audit complete', 'All original23 synthetic cohorts and3.2M cached RP rows; aggregate-only mechanism accounting', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    main()
