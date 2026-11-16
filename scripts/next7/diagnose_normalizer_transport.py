"""Locate H-reference estimation differences without altering G2 predictions."""
from pathlib import Path
import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.worker_environment import environment
os.environ.update(environment())
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from scipy.stats import spearmanr
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, require_guard, note
from src.next7.normalizer_transport import historical_parameters, paired_risk_correlations, CHANNELS, COVARIATES
from src.next7.conditional_null import source_components
from src.next7.calibration import load_calibration
from scripts.next7.conditional_null_grid import atomic_npz


def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def freeze():
    sources = {name: digest(ROOT / name) for name in ('scripts/next7/diagnose_normalizer_transport.py',
        'src/next7/normalizer_transport.py', 'src/next7/crossed_statistics.py', 'src/next7/conditional_null.py',
        'src/next7/scale.py', 'src/next7/null_growth.py', 'src/next6_extension/fresh_runtime.py')}
    path = BASE / 'NORMALIZER_TRANSPORT_FORENSIC_LOCK.json'
    if not path.exists():
        tests = BASE / 'audit/NORMALIZER_TRANSPORT_TESTS_002.xml'
        xml = ET.parse(tests)
        assert len(xml.findall('.//testcase')) == 4 and not xml.findall('.//failure') and not xml.findall('.//error')
        save(path.name, {'status': 'FROZEN', 'created_utc': utc_now(), 'source_sha256': sources,
            'tests_sha256': digest(tests), 'crossed_statistics_lock_sha256': digest(BASE / 'CROSSED_STATISTICS_LOCK.json'),
            'inquiry': 'G2 pseudo-online calibration initializes core/scale from H[:floor(.6H)] while its real online branch initializes the same algorithms from all H. Quantify their already defined reference differences and association with conditional false memory.',
            'design_origin': 'Posthoc diagnostic motivated by completed N0 H1000/H2048 empirical-support obstruction; not an untouched confirmatory hypothesis.',
            'fixed_backgrounds': ['N0', 'N1', 'N2', 'N3'], 'fixed_H_lengths': [1000, 2048, 5000],
            'fixed_horizons': [1024, 4096, 16384], 'histories_per_cohort': 512,
            'H_reconstruction': 'Original partition33 seed indices and H5000 suffixes. source_components horizon0 requests no online innovations. Only original H normalization is reconstructed; no supervised learner or new detector.',
            'channels': CHANNELS, 'covariates': COVARIATES,
            'scale_identity': 'For the same unfloored log-MAD value l, z_full=z_prefix*(spread_prefix/spread_full)+(center_prefix-center_full)/spread_full before clipping. Different epsilon floors can break the common-l premise; no universal affine identity after clipping is claimed.',
            'core_limit': 'Core center/spread differences also accompany different empirical rank maps. The scale affine identity is not asserted for core channels.',
            'analysis': 'Report all23 covariates for all12 cohorts and3 horizons. Pearson association with per-H conditional latch risk; same2000 seed-pair weights recompute risk over innovations and weighted correlation over H. Spearman is descriptive for original grid only.',
            'missing': 'Zero variation makes correlation undefined; preserve missing values and usable draw count. No zero imputation.',
            'limits': 'Approximate pointwise diagnostics with reused H and innovations. Not simultaneous inference, causal mediation, future risk guarantee, or a new H length/threshold choice.',
            'new_random_draws': 0, 'new_learned_recipes': 0, 'new_scientific_candidates': 0,
            'new_thresholds': 0, 'detector_or_classifier_outputs_changed': False})
    assert read_json(path)['source_sha256'] == sources
    return read_json(path)


def prepare_h():
    require_guard()
    freeze()
    suite = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    calibration = load_calibration()
    all_rows, cohorts = [], []
    for case in ('N0', 'N1', 'N2', 'N3'):
        for length in (1000, 2048, 5000):
            relative = f'normalizer_transport/H_parameters/{case}/H_{length}'
            directory = BASE / relative
            done_path = directory / 'DONE.json'
            if done_path.exists():
                done = read_json(done_path)
                assert done['implementation_sha256'] == digest(BASE / 'NORMALIZER_TRANSPORT_FORENSIC_LOCK.json')
                for name, sha in done['files_sha256'].items():
                    assert digest(directory / name) == sha
                all_rows += rows(directory / 'PARAMETERS.csv')
                cohorts.append(done_path.relative_to(BASE).as_posix())
                continue
            support_path = BASE / f'bootstrap_support/{case}/H_{length}/PER_H_SUPPORT.csv'
            support = rows(support_path)
            assert [int(row['reference_index']) for row in support] == list(range(512))
            assert digest(support_path) == read_json(support_path.parent / 'DONE.json')['summary_sha256']
            with activity(f'Historical normalizer transport parameters {case} H{length}', 'H_REFERENCE_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
                records, x = [], np.empty((512, len(COVARIATES)))
                for seed in range(512):
                    h, innovations, _, _ = source_components(case, seed, suite, horizon=0)
                    assert len(innovations) == 0 and len(h) == 5000
                    p = historical_parameters(h[-length:], calibration['core_cut'])
                    assert p['prefix_length'] == int(support[seed]['reference_prefix_length'])
                    assert len(p['prefix_core_calibration']['spread_floor_nodes']) == int(support[seed]['core_prefix_calibration_floor_nodes'])
                    x[seed] = p['covariates']
                    for j, channel in enumerate(CHANNELS):
                        records.append({'case': case, 'H_length': length, 'reference_index': seed,
                            'channel': channel, 'prefix_length': p['prefix_length'],
                            'prefix_center': float(p['centers'][0, j]), 'full_center': float(p['centers'][1, j]),
                            'prefix_spread': float(p['spreads'][0, j]), 'full_spread': float(p['spreads'][1, j]),
                            'center_shift_full_units': float(x[seed, j]),
                            'log_prefix_spread_over_full': float(x[seed, j + 9]),
                            **{name: float(x[seed, index]) for index, name in enumerate(COVARIATES) if index >= 18},
                            'prefix_scale_epsilon': p['prefix_scale_calibration']['epsilon'],
                            'full_scale_epsilon': p['full_scale_calibration']['epsilon'],
                            'observations_or_labels_used_from_future': 0})
                csv_write(relative + '/PARAMETERS.csv', records)
                atomic_npz(directory / 'COVARIATES.npz', values=x, names=np.array(COVARIATES), reference_index=np.arange(512))
                save(relative + '/DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
                    'case': case, 'H_length': length, 'histories': 512,
                    'implementation_sha256': digest(BASE / 'NORMALIZER_TRANSPORT_FORENSIC_LOCK.json'),
                    'original_prefix_floor_count_parity': True, 'source_support_sha256': digest(support_path),
                    'files_sha256': {path.name: digest(path) for path in directory.iterdir() if path.is_file()}})
                all_rows.extend(records)
                cohorts.append(done_path.relative_to(BASE).as_posix())
    csv_write('H_NORMALIZER_TRANSPORT_PARAMETERS_NEXT7.csv', all_rows)
    save('NORMALIZER_H_PARAMETERS_DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
        'histories': 6144, 'channel_rows': len(all_rows),
        'cohorts_sha256': {name: digest(BASE / name) for name in cohorts},
        'result_sha256': digest(BASE / 'H_NORMALIZER_TRANSPORT_PARAMETERS_NEXT7.csv'),
        'new_draws': 0, 'new_model_candidates': 0})
    note('Historical normalizer transport extracted', 'All6144 original H histories; no online points or labels used', 'F4_NULL_GROWTH_CORRECTION')


def correlate(case, length):
    require_guard()
    freeze()
    relative = f'normalizer_transport/correlations/{case}/H_{length}'
    destination = BASE / relative
    if (destination / 'DONE.json').exists():
        return
    parameters = BASE / f'normalizer_transport/H_parameters/{case}/H_{length}'
    parameter_done = read_json(parameters / 'DONE.json')
    assert parameter_done['status'] == 'COMPLETE'
    assert digest(parameters / 'COVARIATES.npz') == parameter_done['files_sha256']['COVARIATES.npz']
    with np.load(parameters / 'COVARIATES.npz', allow_pickle=False) as data:
        x = data['values'].copy()
        assert list(data['names']) == list(COVARIATES)
        np.testing.assert_array_equal(data['reference_index'], np.arange(512))
    statistics = BASE / f'conditional_statistics/{case}/H_{length}'
    statistics_done = read_json(statistics / 'DONE.json')
    assert statistics_done['status'] == 'COMPLETE'
    assert digest(statistics / 'GRID_FOR_FIGURES.npz') == statistics_done['files_sha256']['GRID_FOR_FIGURES.npz']
    with np.load(statistics / 'GRID_FOR_FIGURES.npz', allow_pickle=False) as data:
        decisions = data['qualified'].copy()
    count_lock = read_json(BASE / 'CROSSED_STATISTICS_LOCK.json')
    count_path = BASE / count_lock['count_file']
    assert digest(count_path) == count_lock['count_file_sha256']
    counts = np.load(count_path, allow_pickle=False)
    all_counts = np.vstack((np.ones(512, dtype=counts.dtype), counts))
    with activity(f'Normalizer difference versus conditional null risk {case} H{length}', 'H_REFERENCE_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
        records, output = [], {}
        for j, horizon in enumerate((1024, 4096, 16384)):
            risk = decisions[:, :, j].mean(axis=1)
            correlations = paired_risk_correlations(x, decisions[:, :, j], all_counts)
            output[f'Pearson_{horizon}'] = correlations
            for k, name in enumerate(COVARIATES):
                valid = correlations[1:, k][np.isfinite(correlations[1:, k])]
                lo, hi = np.quantile(valid, [.025, .975]) if len(valid) else (None, None)
                spearman = float(spearmanr(x[:, k], risk).statistic) if np.ptp(x[:, k]) and np.ptp(risk) else None
                records.append({'case': case, 'H_length': length, 'horizon': horizon, 'covariate': name,
                    'Pearson_conditional_risk': float(correlations[0, k]) if np.isfinite(correlations[0, k]) else None,
                    'Pearson_CI_low_approx_pointwise': float(lo) if lo is not None else None,
                    'Pearson_CI_high_approx_pointwise': float(hi) if hi is not None else None,
                    'Spearman_conditional_risk_descriptive': spearman,
                    'covariate_min': float(x[:, k].min()), 'covariate_median': float(np.median(x[:, k])),
                    'covariate_max': float(x[:, k].max()), 'usable_draws': len(valid), 'requested_draws': 2000,
                    'missing_variation_not_imputed': True, 'causal_attribution_claim': False,
                    'new_inference_feature': False, 'posthoc_diagnostic': True})
        csv_write(relative + '/CORRELATIONS.csv', records)
        atomic_npz(destination / 'BOOTSTRAP_CORRELATIONS.npz', **output)
        save(relative + '/DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(), 'case': case, 'H_length': length,
            'implementation_sha256': digest(BASE / 'NORMALIZER_TRANSPORT_FORENSIC_LOCK.json'),
            'parameters_done_sha256': digest(parameters / 'DONE.json'),
            'conditional_statistics_done_sha256': digest(statistics / 'DONE.json'),
            'count_file_sha256': digest(count_path), 'new_draws': 0,
            'files_sha256': {path.name: digest(path) for path in destination.iterdir() if path.is_file()}})
    note('Normalizer transport association complete', f'{case} H{length}: all23 diagnostics retained; no causal or promotion claim', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock-only', action='store_true')
    parser.add_argument('--prepare-H', action='store_true')
    parser.add_argument('--case', choices=['N0', 'N1', 'N2', 'N3'])
    parser.add_argument('--H-length', type=int, choices=[1000, 2048, 5000])
    args = parser.parse_args()
    if args.lock_only:
        freeze()
    elif args.prepare_H:
        prepare_h()
    else:
        assert args.case is not None and args.H_length is not None
        correlate(args.case, args.H_length)
