"""Compare all frozen H lengths with the same original crossed seed weights."""
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
from scipy.stats import spearmanr
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, require_guard, note
from src.next7.h_length_diagnostics import discordant_latches
from src.next7.crossed_statistics import conditional_H_draws
from scripts.next7.conditional_null_grid import atomic_npz

LENGTH_PAIRS = ((2048, 1000), (5000, 2048), (5000, 1000))


def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def freeze():
    sources = {name: digest(ROOT / name) for name in ('scripts/next7/compare_crossed_H_lengths.py',
        'src/next7/h_length_diagnostics.py', 'src/next7/crossed_statistics.py')}
    path = BASE / 'CROSSED_H_LENGTH_COMPARISON_LOCK.json'
    if not path.exists():
        tests = BASE / 'audit/H_LENGTH_DIAGNOSTICS_TESTS_002.xml'
        xml = ET.parse(tests)
        assert len(xml.findall('.//testcase')) == 4 and not xml.findall('.//failure') and not xml.findall('.//error')
        save(path.name, {'status': 'FROZEN', 'created_utc': utc_now(), 'source_sha256': sources,
            'test_xml_sha256': digest(tests), 'crossed_statistics_lock_sha256': digest(BASE / 'CROSSED_STATISTICS_LOCK.json'),
            'support_partition_lock_sha256': digest(BASE / 'SUPPORT_PARTITION_IMPLEMENTATION_LOCK.json'),
            'comparisons': LENGTH_PAIRS, 'backgrounds': ['N0', 'N1', 'N2', 'N3'],
            'horizons': [1024, 4096, 16384], 'new_random_draws': 0, 'new_fits': 0, 'new_cutoffs': 0,
            'paired_unit': 'Same original seed indices, common H5000 suffixes and innovation paths; reuse the exact same2000 seed-pair count rows.',
            'conditional_path': 'Each innovation stream starts from the same generating H5000 terminal AR/ARCH state across all H suffix lengths. Generator state never enters detector input.',
            'comparisons_scope': 'Diagnostic comparison of all three frozen H lengths, including all length-dependent detector calibration and representation effects. No inference-time H choice, favorable-length selection, or isolated causal effect of sample size.',
            'statistics': 'Subtract already stored same-draw rate estimates. Separately partition qualified decisions into added, removed, both and neither false memory; verify each crossed cell and draw. Conditional-H differences use the existing innovation count rows.',
            'uncertainty': 'Approximate pointwise seed-pair product intervals, not independent crossed cells or simultaneous conditional-H coverage.',
            'development_test_erratum': 'First test receipt retained a failed equality between two separately rounded means. Expected value was corrected to the mean of materialized signed decision differences. Numerical implementation unchanged; no research data consumed before correction.'})
    assert read_json(path)['source_sha256'] == sources
    return read_json(path)


def verify_completed(directory):
    receipt = read_json(directory / 'DONE.json')
    assert receipt['status'] == 'COMPLETE'
    inventory = receipt.get('files_sha256', receipt.get('source_files_sha256'))
    assert inventory
    for name, expected in inventory.items():
        assert digest(directory / name) == expected
    return receipt


def compare(case):
    require_guard()
    freeze()
    relative = f'crossed_H_length_comparison/{case}'
    destination = BASE / relative
    if (destination / 'DONE.json').exists():
        return
    protocol = read_json(BASE / 'CROSSED_STATISTICS_LOCK.json')
    count_path = BASE / protocol['count_file']
    assert digest(count_path) == protocol['count_file_sha256']
    counts = np.load(count_path, allow_pickle=False)
    assert counts.shape == (2000, 512)
    means, draws, grids, source_hashes = {}, {}, {}, {}
    with activity(f'Matched crossed historical length diagnostic {case}', 'H_LENGTH_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
        for length in (1000, 2048, 5000):
            means[length], draws[length] = {}, {}
            for family, metric_field, mean_field in (
                    ('conditional_statistics', 'metric', 'mean_crossed_rate'),
                    ('support_partition', 'component', 'crossed_rate')):
                directory = BASE / f'{family}/{case}/H_{length}'
                verify_completed(directory)
                source_hashes[(directory / 'DONE.json').relative_to(BASE).as_posix()] = digest(directory / 'DONE.json')
                with np.load(directory / 'BOOTSTRAP_DRAWS.npz', allow_pickle=False) as data:
                    for key in data.files:
                        if key in draws[length]:
                            np.testing.assert_array_equal(draws[length][key], data[key])
                        draws[length][key] = data[key].copy()
                for row in rows(directory / 'SUMMARY.csv'):
                    key = f"{row[metric_field]}_{row['horizon']}"
                    value = float(row[mean_field])
                    if key in means[length]:
                        assert value == means[length][key]
                    means[length][key] = value
            with np.load(BASE / f'conditional_statistics/{case}/H_{length}/GRID_FOR_FIGURES.npz', allow_pickle=False) as data:
                grids[length] = {'qualified': data['qualified'].copy(), 'terminal_cuts': data['terminal_cuts'].copy()}
            assert grids[length]['qualified'].shape == (512, 512, 3)
        contrasts, discordances, per_h, risk_rank = [], [], [], []
        output_draws = {}
        for candidate_length, reference_length in LENGTH_PAIRS:
            assert set(means[candidate_length]) == set(means[reference_length])
            for key in sorted(means[candidate_length]):
                delta = draws[candidate_length][key] - draws[reference_length][key]
                assert delta.shape == (2000,) and np.isfinite(delta).all()
                metric, horizon = key.rsplit('_', 1)
                lo, hi = np.quantile(delta, [.025, .975])
                contrasts.append({'case': case, 'H_candidate': candidate_length, 'H_reference': reference_length,
                    'horizon': int(horizon), 'metric': metric,
                    'candidate_crossed_rate': means[candidate_length][key],
                    'reference_crossed_rate': means[reference_length][key],
                    'mean_rate_delta': means[candidate_length][key] - means[reference_length][key],
                    'CI_low_approx_pointwise': float(lo), 'CI_high_approx_pointwise': float(hi),
                    'draws': 2000, 'same_original_seed_pair_counts': True, 'new_H_selection': False})
                output_draws[f'H{candidate_length}_minus_H{reference_length}_{key}'] = delta
            for j, horizon in enumerate((1024, 4096, 16384)):
                candidate = grids[candidate_length]['qualified'][:, :, j]
                reference = grids[reference_length]['qualified'][:, :, j]
                rates, partition_draws, delta = discordant_latches(candidate, reference, counts)
                np.testing.assert_array_equal(delta, output_draws[
                    f'H{candidate_length}_minus_H{reference_length}_G2_QUALIFIED_EVER_LATCH_{horizon}'])
                for component, rate in rates.items():
                    lo, hi = np.quantile(partition_draws[component], [.025, .975])
                    discordances.append({'case': case, 'H_candidate': candidate_length, 'H_reference': reference_length,
                        'horizon': horizon, 'component': component, 'crossed_rate': rate,
                        'CI_low_approx_pointwise': float(lo), 'CI_high_approx_pointwise': float(hi),
                        'draws': 2000, 'cell_independence_claim': False})
                    output_draws[f'H{candidate_length}_minus_H{reference_length}_{component}_{horizon}'] = partition_draws[component]
                conditional_delta = conditional_H_draws(candidate.astype(np.int8) - reference.astype(np.int8), counts)
                low, high = np.quantile(conditional_delta, [.025, .975], axis=0)
                cand_risk, ref_risk = candidate.mean(axis=1), reference.mean(axis=1)
                for i in range(512):
                    per_h.append({'case': case, 'H_candidate': candidate_length, 'H_reference': reference_length,
                        'horizon': horizon, 'reference_index': i,
                        'candidate_conditional_G2_risk': float(cand_risk[i]),
                        'reference_conditional_G2_risk': float(ref_risk[i]),
                        'conditional_risk_delta': float(cand_risk[i] - ref_risk[i]),
                        'conditional_MC_CI_low': float(low[i]), 'conditional_MC_CI_high': float(high[i]),
                        'candidate_terminal_cut': float(grids[candidate_length]['terminal_cuts'][i, j]),
                        'reference_terminal_cut': float(grids[reference_length]['terminal_cuts'][i, j]),
                        'simultaneous_interval_claim': False})
                risk_rank.append({'case': case, 'H_candidate': candidate_length, 'H_reference': reference_length,
                    'horizon': horizon, 'conditional_risk_Spearman': float(spearmanr(cand_risk, ref_risk).statistic)
                    if np.ptp(cand_risk) and np.ptp(ref_risk) else None,
                    'histories_with_lower_risk': int(np.sum(cand_risk < ref_risk)),
                    'histories_with_equal_risk': int(np.sum(cand_risk == ref_risk)),
                    'histories_with_higher_risk': int(np.sum(cand_risk > ref_risk)),
                    'descriptive_only': True})
        csv_write(relative + '/PAIRED_H_LENGTH_CONTRASTS.csv', contrasts)
        csv_write(relative + '/QUALIFIED_DECISION_DISCORDANCE.csv', discordances)
        csv_write(relative + '/PER_H_PAIRED_RISK.csv', per_h)
        csv_write(relative + '/RISK_ORDER_STABILITY.csv', risk_rank)
        atomic_npz(destination / 'BOOTSTRAP_DRAWS.npz', **output_draws)
        save(relative + '/DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(), 'case': case,
            'implementation_sha256': digest(BASE / 'CROSSED_H_LENGTH_COMPARISON_LOCK.json'),
            'inputs_sha256': source_hashes, 'count_file_sha256': digest(count_path),
            'new_draws': 0, 'new_candidates': 0, 'new_cutoffs': 0,
            'files_sha256': {path.name: digest(path) for path in destination.iterdir() if path.is_file()}})
    note('Matched H length diagnostic complete', case + ': all lengths, common seed counts, added versus removed false memory', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock-only', action='store_true')
    parser.add_argument('--case', choices=['N0', 'N1', 'N2', 'N3'])
    args = parser.parse_args()
    if args.lock_only:
        freeze()
    else:
        assert args.case is not None
        compare(args.case)
