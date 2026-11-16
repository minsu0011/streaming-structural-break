"""Full finite-training-pair gradient forensic, without fitting another model."""
from pathlib import Path
import argparse
import os
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from threadpoolctl import threadpool_limits
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, require_guard, activity, note
from src.next7.model import Correction, logit
from src.next7.features import NAMES
from src.next7.pairwise_diagnostics import exact_stratum, sampled_stratum
from src.next6.data import source
from scripts.next7.fit_g1_corrections import verify_inner
from scripts.next7.conditional_null_grid import atomic_npz

BETA_POINTS = ('ZERO', 'G2_PARENT', 'G3_FITTED')


def freeze():
    path = BASE / 'G3_FULL_GRADIENT_FORENSIC_LOCK.json'
    sources = {n: digest(ROOT / n) for n in ('scripts/next7/full_g3_gradient.py', 'src/next7/pairwise_diagnostics.py',
        'src/next7/pairwise.py', 'src/next7/model.py', 'src/next7/features.py')}
    if path.exists():
        assert read_json(path)['source_sha256'] == sources
        return
    save(path.name, {'status': 'FROZEN', 'created_utc': utc_now(), 'source_sha256': sources,
        'motivation': 'Prespecified-time diagnostics show local sampling differences; determine whether they cancel in the complete training objective before attributing held-out failure to sampling.',
        'scope': 'Every eligible same-time positive-negative outerTRAIN pair for each of five frozen G3 folds.',
        'beta_points': BETA_POINTS, 'new_fits': 0, 'new_candidates': 0, 'new_random_draws': 0,
        'sampled_comparator': 'Original fixed1024-per-time pair plans, unchanged.',
        'reuse': 'Reuse the already computed exact diagnostic values for the seven prespecified times, with hashes and support equality.',
        'pair_weights': 'Eligible pair mass at time divided by total eligible training pair mass.',
        'penalty': .01, 'objective': 'Weighted mean pairwise logistic plus.01/2*norm(beta)^2',
        'mathematical_bound': 'Strong convexity gives F(beta)-min F <= norm(full_gradient)^2/(2*.01) and distance to the unique exact-objective minimizer <= norm(full_gradient)/.01, in exact arithmetic.',
        'bound_limit': 'Numerically evaluated training surrogate bounds, not confidence intervals, held-out TS-AUC bounds, or evidence for a new model. No exact-objective optimizer is fitted.',
        'no_response_to_results': 'Do not change sample cap, penalty, seed, features, coefficients, calibration or any outer prediction.',
        'preregistered_diagnostic_done_sha256': digest(BASE / 'G3_PAIR_DIAGNOSTIC_DONE.json'),
        'partial_results_sha256': digest(BASE / 'G3_EXACT_VS_SAMPLED_PAIR_DIAGNOSTICS.csv'),
        'partial_gradients_sha256': digest(BASE / 'G3_EXACT_VS_SAMPLED_GRADIENT_COORDINATES.csv'),
        'G3_protocol_sha256': digest(BASE / 'G3_PROTOCOL_LOCK.json'),
        'test_xml_sha256': digest(BASE / 'audit/G3_PAIR_DIAGNOSTIC_TESTS_001.xml')})
    note('Full G3 sampling-error forensic frozen', 'Training-only exhaustive objective gradients; no optimization or new candidate', 'F6_TIME_MATCHED_RANK_CORRECTION')


def csv_rows(name):
    import csv
    with (BASE / name).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def run(fold):
    guard = require_guard()
    freeze()
    protocol = read_json(BASE / 'G3_FULL_GRADIENT_FORENSIC_LOCK.json')
    assert digest(BASE / 'G3_EXACT_VS_SAMPLED_PAIR_DIAGNOSTICS.csv') == protocol['partial_results_sha256']
    assert digest(BASE / 'G3_EXACT_VS_SAMPLED_GRADIENT_COORDINATES.csv') == protocol['partial_gradients_sha256']
    prior_loss = {(int(r['time']), r['beta']): r for r in csv_rows('G3_EXACT_VS_SAMPLED_PAIR_DIAGNOSTICS.csv') if int(r['fold']) == fold}
    prior_gradient = {}
    for row in csv_rows('G3_EXACT_VS_SAMPLED_GRADIENT_COORDINATES.csv'):
        if int(row['fold']) == fold:
            prior_gradient.setdefault((int(row['time']), row['beta']), {})[row['feature']] = row
    data = source()
    meta = data.rows()
    features = np.load(BASE / 'cache/g2/head_features.npy', mmap_mode='r', allow_pickle=False)
    train = {int(s) for s in data.ids if guard.split.assignment(int(s))[1] != fold}
    valid = set(map(int, data.ids)) - train
    train_rows, inner_base, _ = verify_inner(data, fold, train, valid)
    model = Correction.load(BASE / f'g3/fold_{fold}/model.joblib')
    parent = Correction.load(BASE / f'g2/fold_{fold}/G2_FULL/model.joblib')
    frozen_plan = read_json(BASE / f'g3/fold_{fold}/PAIR_PLAN_FROZEN.json')
    pair_path = BASE / f'cache/g3_pairs/fold_{fold}.npz'
    assert digest(pair_path) == frozen_plan['file_sha256']
    with np.load(pair_path, allow_pickle=False) as f:
        pairs = {k: f[k].copy() for k in f.files}
    lookup = np.full(len(meta['target']), -1, dtype=np.int32)
    lookup[train_rows] = np.arange(len(train_rows))
    assert np.all(lookup[pairs['positive_global_row']] >= 0) and np.all(lookup[pairs['negative_global_row']] >= 0)
    times = meta['time_online'][train_rows]
    betas = np.stack((np.zeros(10), parent.beta, model.beta))
    relative = f'g3_full_gradient/fold_{fold}'
    directory = BASE / relative
    if (directory / 'DONE.json').exists():
        return
    total_mass = frozen_plan['total_eligible_pair_mass']
    total = np.zeros((3, 2, 11))  # beta x (exact,sampled) x (loss,gradient10)
    per_time = []
    for time_record in frozen_plan['time_records']:
        at = time_record['time']
        output = directory / f'time_{at:04d}.npz'
        mass = time_record['eligible_pair_mass']
        if output.exists():
            with np.load(output, allow_pickle=False) as f:
                values = f['values'].copy()
                assert int(f['time']) == at and int(f['pair_mass']) == mass
                assert str(f['protocol_sha256']) == digest(BASE / 'G3_FULL_GRADIENT_FORENSIC_LOCK.json')
        elif (at, 'ZERO') in prior_loss:
            values = np.empty((3, 2, 11))
            for b, name in enumerate(BETA_POINTS):
                row = prior_loss[at, name]
                assert int(row['pair_mass']) == mass and int(row['sampled_pairs']) == time_record['sampled_pairs']
                values[b, 0, 0], values[b, 1, 0] = float(row['exact_mean_loss']), float(row['sampled_mean_loss'])
                for j, feature in enumerate(NAMES):
                    row = prior_gradient[at, name][feature]
                    values[b, 0, j + 1], values[b, 1, j + 1] = float(row['exact_unpenalized_gradient']), float(row['sampled_unpenalized_gradient'])
            atomic_npz(output, values=values, time=np.array(at), pair_mass=np.array(mass),
                protocol_sha256=np.array(digest(BASE / 'G3_FULL_GRADIENT_FORENSIC_LOCK.json')), reused_prespecified_time=np.array(True))
        else:
            with activity(f'G3 exhaustive gradient fold{fold} time{at}', 'TRAINING_SAMPLING_FORENSIC', 'F6_TIME_MATCHED_RANK_CORRECTION'):
                local = np.flatnonzero(times == at)
                global_rows = train_rows[local]
                y = meta['target'][global_rows]
                x = np.asarray(features[global_rows], dtype=np.float64)[:, model.columns]
                z = np.clip((x - model.center) / model.spread, -8., 8.)
                offset = logit(inner_base[local])
                positive, negative = y == 1, y == 0
                assert int(positive.sum()) * int(negative.sum()) == mass
                selected = pairs['time'] == at
                p, n = pairs['positive_global_row'][selected], pairs['negative_global_row'][selected]
                assert len(p) == time_record['sampled_pairs']
                zp = np.clip((np.asarray(features[p])[:, model.columns] - model.center) / model.spread, -8., 8.)
                zn = np.clip((np.asarray(features[n])[:, model.columns] - model.center) / model.spread, -8., 8.)
                difference = zp - zn
                offsets = logit(inner_base[lookup[p]]) - logit(inner_base[lookup[n]])
                values = np.empty((3, 2, 11))
                for b, beta in enumerate(betas):
                    with threadpool_limits(limits=1):
                        exact_loss, exact_gradient = exact_stratum(beta, z[positive], z[negative], offset[positive], offset[negative])
                        sample_loss, sample_gradient, _ = sampled_stratum(beta, difference, offsets)
                    values[b, 0, 0], values[b, 1, 0] = exact_loss, sample_loss
                    values[b, 0, 1:], values[b, 1, 1:] = exact_gradient, sample_gradient
                atomic_npz(output, values=values, time=np.array(at), pair_mass=np.array(mass),
                    protocol_sha256=np.array(digest(BASE / 'G3_FULL_GRADIENT_FORENSIC_LOCK.json')), reused_prespecified_time=np.array(False))
        assert np.isfinite(values).all() and values.shape == (3, 2, 11)
        total += (mass / total_mass) * values
        per_time.append({'time': at, 'pair_mass': mass, 'file_sha256': digest(output)})
        if at % 64 == 0:
            print(utc_now(), 'G3 exhaustive training gradient fold', fold, 'time', at, flush=True)
    assert sum(r['pair_mass'] for r in per_time) == total_mass
    report = []
    coordinates = []
    for b, name in enumerate(BETA_POINTS):
        beta = betas[b]
        total[b, :, 0] += .005 * (beta @ beta)
        total[b, :, 1:] += .01 * beta
        exact_gradient, sample_gradient = total[b, 0, 1:], total[b, 1, 1:]
        norm = float(np.linalg.norm(exact_gradient))
        report.append({'fold': fold, 'beta': name, 'exact_penalized_objective': float(total[b, 0, 0]),
            'sampled_penalized_objective': float(total[b, 1, 0]), 'sample_minus_exact_objective': float(total[b, 1, 0] - total[b, 0, 0]),
            'exact_gradient_L2': norm, 'sampled_gradient_L2': float(np.linalg.norm(sample_gradient)),
            'sampling_gradient_error_L2': float(np.linalg.norm(sample_gradient - exact_gradient)),
            'strong_convexity_training_suboptimality_bound_numeric': norm * norm / .02,
            'distance_to_exact_training_minimizer_bound_numeric': norm / .01,
            'eligible_training_pair_mass': total_mass, 'sampled_training_pairs': frozen_plan['sampled_pairs'],
            'no_heldout_AUC_bound_or_new_fitted_model': True})
        for j, feature in enumerate(NAMES):
            coordinates.append({'fold': fold, 'beta': name, 'feature': feature,
                'exact_penalized_gradient': float(exact_gradient[j]), 'sampled_penalized_gradient': float(sample_gradient[j]),
                'sampling_gradient_error': float(sample_gradient[j] - exact_gradient[j])})
    assert abs(total[0, 1, 0] - model.training['initial_objective']) < 1e-12
    assert abs(total[2, 1, 0] - model.training['objective']) < 1e-12
    assert abs(float(np.max(np.abs(total[2, 1, 1:]))) - model.training['gradient_max_abs']) < 1e-12
    csv_write(relative + '/OBJECTIVE_SUMMARY.csv', report)
    csv_write(relative + '/GRADIENT_COORDINATES.csv', coordinates)
    save(relative + '/DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(), 'fold': fold,
        'total_eligible_pairs': total_mass, 'times': len(per_time), 'per_time_receipts': per_time,
        'original_sampled_optimizer_objective_and_gradient_parity': True, 'new_fits': 0, 'new_random_draws': 0,
        'source_sha256': digest(__file__), 'protocol_sha256': digest(BASE / 'G3_FULL_GRADIENT_FORENSIC_LOCK.json')})
    guard.check_state()
    note('Full G3 training gradient complete', f'fold{fold}: sampling error and training surrogate bound, no refit', 'F6_TIME_MATCHED_RANK_CORRECTION')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock-only', action='store_true')
    parser.add_argument('--fold', type=int, choices=range(5))
    args = parser.parse_args()
    if args.lock_only:
        freeze()
    else:
        assert args.fold is not None
        run(args.fold)
