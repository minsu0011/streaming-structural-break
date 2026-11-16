"""Explain frozen G2/G3 rank changes and event admission without further fitting."""
from pathlib import Path
import os
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from scipy.stats import spearmanr
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, require_guard, note
from src.next7.rank_cells import rank_cells
from src.next7.features import NAMES
from src.next7.model import Correction
from src.next6.data import source

GROUPS = ('NO_RAW_MEMORY', 'RAW_MEMORY_NOT_YET_QUALIFIED', 'QUALIFIED_MEMORY')


def load_predictions(path, rows=None):
    with np.load(path, allow_pickle=False) as f:
        if rows is not None:
            np.testing.assert_array_equal(rows, f['global_row'])
        return {k: f[k].copy() for k in f.files}


def main():
    guard = require_guard()
    tests = BASE / 'audit/RANK_CELL_TESTS_001.xml'
    xml = ET.parse(tests)
    assert len(xml.findall('.//testcase')) == 4 and not xml.findall('.//failure') and not xml.findall('.//error')
    save('FORENSIC_G2_G3_LOCK.json', {'status': 'FROZEN', 'created_utc': utc_now(),
        'groups': GROUPS, 'scope': 'Posthoc exact rank accounting, no causal identification or new candidate.',
        'source_sha256': {n: digest(ROOT / n) for n in ('scripts/next7/forensic_g2_g3.py', 'src/next7/rank_cells.py')},
        'test_xml_sha256': digest(tests), 'new_fits': 0})
    data = source()
    meta = data.rows()
    raw = np.load(BASE / 'cache/g1/state.npy', mmap_mode='r', allow_pickle=False)
    qualified = np.load(BASE / 'cache/g2/qualified.npy', mmap_mode='r', allow_pickle=False)
    hcal = {r['dataset_id']: r for r in read_json(BASE / 'cache/g2/H_CALIBRATIONS.json')}
    for generation in ('g1', 'g2'):
        manifest = read_json(BASE / f'cache/{generation}/MANIFEST.json')
        for name, sha in manifest['files_sha256'].items():
            assert digest(BASE / f'cache/{generation}' / name) == sha
    series, episodes, pairs, correlations, coefficients = [], [], [], [], []
    scores = {}
    for generation in ('G1', 'G2', 'G3'):
        scores.update(read_json(BASE / f'{generation}_OUTER_SCORES.json')['scores'])
    with activity('Qualified event timing and recovery forensic', 'BOTTLENECK_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
        for i, sid in enumerate(data.ids):
            a, b = map(int, data.series['online_offsets'][i:i + 2])
            v, q = raw[a:b], qualified[a:b]
            tau = int(data.series['tau'][i])
            first_raw, first_q = int(v[-1, 21]), int(q[-1, 5])
            assert np.all(q[:, 0] <= v[:, 3] + 1e-15)
            assert np.all(np.diff(q[:, 0]) >= 0) and not np.any((q[:, 0] > 0) & (v[:, 3] == 0))
            category = 'no_break' if tau < 0 else 'break_observed' if tau < len(v) else 'break_unobserved'
            c = hcal[int(sid)]
            record = {'dataset_id': int(sid), 'fold': guard.split.assignment(int(sid))[1],
                'category': category, 'tau_offline_only': tau, 'online_length_offline_only': len(v),
                'H_length': c['H_length'], 'reference_valid': c['reference_valid'], 'pseudo_blocks': c['pseudo_blocks'],
                'cut_blocks4': c['admission_cuts'][0], 'cut_blocks64': c['admission_cuts'][4], 'cut_blocks1024': c['admission_cuts'][-1],
                'raw_latched': first_raw >= 0, 'qualified_latched': first_q >= 0,
                'raw_first_latch': first_raw, 'qualified_first_latch': first_q,
                'raw_ever_qualified_never': first_raw >= 0 and first_q < 0,
                'admission_delay_when_both_latched': first_q - first_raw if first_q >= 0 else None,
                'raw_prebreak_latch': first_raw >= 0 and tau >= 0 and first_raw < tau,
                'qualified_prebreak_latch': first_q >= 0 and tau >= 0 and first_q < tau,
                'raw_first_latch_after_break': first_raw >= 0 and tau >= 0 and first_raw >= tau,
                'qualified_first_latch_after_break': first_q >= 0 and tau >= 0 and first_q >= tau,
                'qualified_memory_while_current_normal': bool(np.any((q[:, 0] > 0) & (v[:, 0] == 1))),
                'qualified_memory_while_recovering': bool(np.any((q[:, 0] > 0) & (v[:, 7] > 0))),
                'raw_positive_memory_points': int((v[:, 3] > 0).sum()),
                'qualified_positive_memory_points': int((q[:, 0] > 0).sum()),
                'raw_only_positive_memory_points': int(((v[:, 3] > 0) & (q[:, 0] == 0)).sum()),
                'raw_final_memory': float(v[-1, 3]), 'qualified_final_memory': float(q[-1, 0]),
                'target_final': int(meta['target'][b - 1]),
                'qualified_first_channel': int(q[first_q, 7]) if first_q >= 0 else -1}
            series.append(record)
            held = q[:, 11] > .5
            for t in np.flatnonzero(held & ~np.r_[False, held[:-1]]):
                oldest = int(q[t, 8])
                assert q[t, 10] > .5 and v[t, 18] > .5 and v[t, 15] > .5
                episodes.append({'dataset_id': int(sid), 'fold': record['fold'], 'event_time': int(t),
                    'firstShockTime': int(v[t, 19]), 'firstRawConfirmationTime': int(v[t, 20]),
                    'firstRawLatchTime': int(v[t, 21]), 'firstQualifiedLatchTime': int(q[t, 5]),
                    'confirmationChannel': int(q[t, 7]), 'confirmationBlocks': ','.join(map(str, range(oldest, oldest + 4))),
                    'same_direction_blocks': int(q[t, 9]), 'raw_confirmation_ratio': float(v[t, 2]),
                    'admission_cut': float(q[t, 12]), 'qualified_strength': float(q[t, 14]),
                    'memory_strength': float(q[t, 0]), 'direction': int(q[t, 4]),
                    'target_at_event': int(meta['target'][a + t]), 'tau_offline_only': tau})
    comparisons = [('M_EVENT', 'M0'), ('G2_EVENT', 'M0'), ('G2_FULL', 'M0'),
        ('G3_PAIRWISE_FULL', 'M0'), ('G2_EVENT', 'M_EVENT'), ('G2_FULL', 'M_FULL'), ('G3_PAIRWISE_FULL', 'G2_FULL')]
    for fold in range(5):
        with activity(f'G2 G3 exact memory rank cells and correlations fold{fold}', 'BOTTLENECK_FORENSIC', 'F5_CAUSAL_NESTED_INTEGRATION'):
            original = load_predictions(BASE / f'g1/fold_{fold}/fixed_predictions.npz')
            rows = original['global_row']
            current = load_predictions(BASE / f'g2/fold_{fold}/fixed_predictions.npz', rows)
            prediction = {k: original[k] for k in ('M0', 'M_EVENT')}
            np.testing.assert_array_equal(current['M0'], prediction['M0'])
            prediction['G2_EVENT'] = current['G2_EVENT']
            for method, relative in [('M_FULL', f'g1/fold_{fold}/M_FULL'),
                                     ('G2_FULL', f'g2/fold_{fold}/G2_FULL'),
                                     ('G3_PAIRWISE_FULL', f'g3/fold_{fold}')]:
                prediction[method] = load_predictions(BASE / relative / 'predictions.npz', rows)['prediction']
                model = Correction.load(BASE / relative / 'model.joblib')
                for j, index in enumerate(model.columns):
                    coefficients.append({'fold': fold, 'method': method, 'feature': NAMES[index],
                        'standardized_coefficient': float(model.beta[j]), 'training_center': float(model.center[j]),
                        'training_spread': float(model.spread[j]), 'baseline_logit_coefficient': 1.})
            y, at = meta['target'][rows], meta['time_online'][rows]
            group = np.where(qualified[rows, 0] > 0, 2, np.where(raw[rows, 3] > 0, 1, 0))
            partitions = {method: rank_cells(y, p, at, group, 3) for method, p in prediction.items()}
            mass = partitions['M0'][1]
            total = mass.sum()
            for method, (wins, pairmass) in partitions.items():
                np.testing.assert_array_equal(mass, pairmass)
                assert abs(wins.sum() / total - scores[method][str(fold)]) < 1e-12
            for candidate, reference in comparisons:
                delta = partitions[candidate][0] - partitions[reference][0]
                assert abs(delta.sum() / total - (scores[candidate][str(fold)] - scores[reference][str(fold)])) < 1e-12
                if candidate == 'G2_EVENT' and reference == 'M_EVENT':
                    assert delta[0, 0] == 0
                for pos in range(3):
                    for neg in range(3):
                        pairs.append({'fold': fold, 'candidate': candidate, 'reference': reference,
                            'positive_memory_group': GROUPS[pos], 'negative_memory_group': GROUPS[neg],
                            'pair_mass': int(mass[pos, neg]), 'fraction_all_pairs': float(mass[pos, neg] / total),
                            'reference_wins': float(partitions[reference][0][pos, neg]),
                            'candidate_wins': float(partitions[candidate][0][pos, neg]),
                            'delta_contribution_to_fold_TS_AUC': float(delta[pos, neg] / total),
                            'posthoc_algebraic_partition_not_causal_effect': True})
            order = np.argsort(at, kind='stable')
            boundaries = np.r_[0, np.flatnonzero(np.diff(at[order])) + 1, len(order)]
            for candidate, reference in [('G2_EVENT', 'M0'), ('G2_FULL', 'M0'), ('G3_PAIRWISE_FULL', 'M0'), ('G3_PAIRWISE_FULL', 'G2_FULL')]:
                p, b = prediction[candidate], prediction[reference]
                numerator_s = numerator_p = denominator = 0.
                supported = 0
                for a, end in zip(boundaries[:-1], boundaries[1:]):
                    r = order[a:end]
                    npos = int(y[r].sum())
                    weight = npos * (len(r) - npos)
                    if weight and np.ptp(p[r]) > 0 and np.ptp(b[r]) > 0:
                        numerator_s += weight * float(spearmanr(b[r], p[r]).statistic)
                        numerator_p += weight * float(np.corrcoef(b[r], p[r])[0, 1])
                        denominator += weight
                        supported += 1
                pearson = float(np.corrcoef(b, p)[0, 1])
                delta = scores[candidate][str(fold)] - scores[reference][str(fold)]
                correlations.append({'fold': fold, 'candidate': candidate, 'reference': reference,
                    'Pearson': pearson, 'Spearman': float(spearmanr(b, p).statistic),
                    'within_time_Spearman_pair_weighted': numerator_s / denominator,
                    'within_time_Pearson_pair_weighted': numerator_p / denominator,
                    'within_time_supported_pair_mass': int(denominator), 'within_time_supported_times': supported,
                    'mean_absolute_correction': float(np.mean(np.abs(p - b))), 'fold_delta_TS_AUC': delta,
                    'no_independent_signal_claim_under_policy': pearson > .995 and delta < .0005})
            print(utc_now(), 'G2 G3 forensic fold', fold, flush=True)
    summaries = []
    metrics = ('raw_latched', 'qualified_latched', 'raw_ever_qualified_never', 'raw_prebreak_latch',
        'qualified_prebreak_latch', 'raw_first_latch_after_break', 'qualified_first_latch_after_break',
        'qualified_memory_while_current_normal', 'qualified_memory_while_recovering',
        'raw_positive_memory_points', 'qualified_positive_memory_points', 'raw_only_positive_memory_points')
    for category in sorted({r['category'] for r in series}):
        selected = [r for r in series if r['category'] == category]
        summaries.append({'category': category, 'series': len(selected),
            **{key + '_mean': float(np.mean([r[key] for r in selected])) for key in metrics}})
    pair_summary = []
    for candidate, reference in comparisons:
        for pos in GROUPS:
            for neg in GROUPS:
                selected = [r for r in pairs if r['candidate'] == candidate and r['reference'] == reference
                    and r['positive_memory_group'] == pos and r['negative_memory_group'] == neg]
                pair_summary.append({'candidate': candidate, 'reference': reference,
                    'positive_memory_group': pos, 'negative_memory_group': neg,
                    'mean_fold_pair_share': float(np.mean([r['fraction_all_pairs'] for r in selected])),
                    'mean_fold_delta_contribution': float(np.mean([r['delta_contribution_to_fold_TS_AUC'] for r in selected]))})
    csv_write('forensic_g2_g3/SERIES_MEMORY_ADMISSION_AND_RECOVERY.csv', series)
    csv_write('EVENT_TIMING_FORENSIC_G2.csv', episodes)
    csv_write('TARGET_ALIGNMENT_RESULTS_G2.csv', summaries)
    csv_write('forensic_g2_g3/EXACT_MEMORY_PAIR_DECOMPOSITION.csv', pairs)
    csv_write('forensic_g2_g3/MEAN_MEMORY_PAIR_DECOMPOSITION.csv', pair_summary)
    csv_write('forensic_g2_g3/CORRELATION_AND_REDUNDANCY.csv', correlations)
    csv_write('forensic_g2_g3/CORRECTION_COEFFICIENTS.csv', coefficients)
    save('forensic_g2_g3/DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
        'series': len(series), 'qualified_episodes': len(episodes), 'rank_cells': len(pairs),
        'all_rank_cells_sum_to_exact_frozen_scores': True, 'current_state_source_changed': False,
        'new_fits': 0, 'new_random_draws': 0, 'target_alignment': summaries,
        'source_sha256': digest(__file__)})
    guard.check_state()
    note('G2 G3 target alignment forensic complete', 'Exact rank costs and benefits separated by qualified versus raw memory', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    main()
