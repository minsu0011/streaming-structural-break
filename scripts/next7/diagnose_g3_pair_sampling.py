"""Compare fixed G3 sampled training pairs with all pairs at preregistered times."""
from pathlib import Path
import os
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from threadpoolctl import threadpool_limits
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, require_guard, note
from src.next7.model import Correction, logit
from src.next7.features import NAMES
from src.next7.pairwise_diagnostics import exact_stratum, sampled_stratum
from src.next6.data import source
from scripts.next7.fit_g1_corrections import verify_inner


def main():
    guard = require_guard()
    protocol = read_json(BASE / 'G3_PROTOCOL_LOCK.json')
    times = protocol['diagnostics_only']['exact_pair_gradient_times']
    tests = BASE / 'audit/G3_PAIR_DIAGNOSTIC_TESTS_001.xml'
    xml = ET.parse(tests)
    assert len(xml.findall('.//testcase')) == 5 and not xml.findall('.//failure') and not xml.findall('.//error')
    save('G3_PAIR_DIAGNOSTIC_LOCK.json', {'status': 'FROZEN', 'created_utc': utc_now(),
        'times': times, 'times_preregistered_before_G3_fit': True,
        'beta_points': ['ZERO', 'G2_PARENT', 'G3_FITTED'],
        'purpose': 'Training-only sampling approximation forensic; no fits, new draws or tuning.',
        'loss_and_gradient': 'Within-time unpenalized mean pairwise logistic; global mass contribution also reported.',
        'sampling_SE_limit': 'Descriptive for the fixed sample; fitted beta depends on these pairs, so no significance claim.',
        'source_sha256': {name: digest(ROOT / name) for name in (
            'scripts/next7/diagnose_g3_pair_sampling.py', 'src/next7/pairwise_diagnostics.py',
            'src/next7/tests/test_pairwise_diagnostics.py')},
        'test_xml_sha256': digest(tests), 'protocol_sha256': digest(BASE / 'G3_PROTOCOL_LOCK.json')})
    data = source()
    meta = data.rows()
    features = np.load(BASE / 'cache/g2/head_features.npy', mmap_mode='r', allow_pickle=False)
    rows = []
    coefficients = []
    repair_checks = []
    for fold in range(5):
        with activity(f'G3 fixed sample versus exhaustive pair gradient fold{fold}', 'BOTTLENECK_FORENSIC', 'F6_TIME_MATCHED_RANK_CORRECTION'):
            train = {int(s) for s in data.ids if guard.split.assignment(int(s))[1] != fold}
            valid = set(map(int, data.ids)) - train
            train_rows, inner_base, _ = verify_inner(data, fold, train, valid)
            model = Correction.load(BASE / f'g3/fold_{fold}/model.joblib')
            parent = Correction.load(BASE / f'g2/fold_{fold}/G2_FULL/model.joblib')
            frozen = read_json(BASE / f'g3/fold_{fold}/PAIR_PLAN_FROZEN.json')
            pair_path = BASE / f'cache/g3_pairs/fold_{fold}.npz'
            assert digest(pair_path) == frozen['file_sha256']
            with np.load(pair_path, allow_pickle=False) as f:
                pairs = {key: f[key].copy() for key in f.files}
            lookup = np.full(len(meta['target']), -1, dtype=np.int32)
            lookup[train_rows] = np.arange(len(train_rows))
            assert np.all(lookup[pairs['positive_global_row']] >= 0) and np.all(lookup[pairs['negative_global_row']] >= 0)
            if fold == 0:
                old = BASE / 'audit/G3_STANDARDIZATION_PARTIAL'
                old_plans = list(old.rglob('PAIR_PLAN_FROZEN.json'))
                assert len(old_plans) == 1
                old_plan = read_json(old_plans[0])
                assert old_plan['pair_plan_sha256'] == frozen['pair_plan_sha256']
                assert old_plan['file_sha256'] == frozen['file_sha256']
                repair_checks.append({'fold': 0, 'pre_repair_pair_plan_bitwise_unchanged': True,
                    'plan_sha256': frozen['pair_plan_sha256'], 'pair_file_sha256': frozen['file_sha256']})
            ttrain = meta['time_online'][train_rows]
            time_records = {r['time']: r for r in frozen['time_records']}
            for at in times:
                local = np.flatnonzero(ttrain == at)
                global_rows = train_rows[local]
                y = meta['target'][global_rows]
                x = np.asarray(features[global_rows], dtype=np.float64)[:, model.columns]
                z = np.clip((x - model.center) / model.spread, -8., 8.)
                offset = logit(inner_base[local])
                pos, neg = y == 1, y == 0
                mass = int(pos.sum()) * int(neg.sum())
                if not mass:
                    rows.append({'fold': fold, 'time': at, 'status': 'NO_ELIGIBLE_PAIRS', 'pair_mass': 0})
                    continue
                assert mass == time_records[at]['eligible_pair_mass']
                selected = pairs['time'] == at
                pglobal, nglobal = pairs['positive_global_row'][selected], pairs['negative_global_row'][selected]
                assert np.all(meta['time_online'][pglobal] == at) and np.all(meta['time_online'][nglobal] == at)
                pz = np.clip((np.asarray(features[pglobal])[:, model.columns] - model.center) / model.spread, -8., 8.)
                nz = np.clip((np.asarray(features[nglobal])[:, model.columns] - model.center) / model.spread, -8., 8.)
                difference = pz - nz
                offsets = logit(inner_base[lookup[pglobal]]) - logit(inner_base[lookup[nglobal]])
                for name, beta in [('ZERO', np.zeros(10)), ('G2_PARENT', parent.beta), ('G3_FITTED', model.beta)]:
                    with threadpool_limits(limits=1):
                        exact_loss, exact_gradient = exact_stratum(beta, z[pos], z[neg], offset[pos], offset[neg])
                        sample_loss, sample_gradient, se = sampled_stratum(beta, difference, offsets)
                    fraction = mass / frozen['total_eligible_pair_mass']
                    norm1, norm2 = np.linalg.norm(exact_gradient), np.linalg.norm(sample_gradient)
                    rows.append({'fold': fold, 'time': at, 'beta': name, 'status': 'COMPLETE',
                        'positive_rows': int(pos.sum()), 'negative_rows': int(neg.sum()), 'pair_mass': mass,
                        'sampled_pairs': int(selected.sum()), 'fraction_global_training_pair_mass': fraction,
                        'exact_mean_loss': exact_loss, 'sampled_mean_loss': sample_loss,
                        'sample_minus_exact_loss': sample_loss - exact_loss, 'sample_mean_loss_SE_descriptive': se,
                        'exact_gradient_L2': float(norm1), 'sampled_gradient_L2': float(norm2),
                        'gradient_error_L2': float(np.linalg.norm(sample_gradient - exact_gradient)),
                        'gradient_cosine': float(exact_gradient @ sample_gradient / norm1 / norm2) if norm1 > 1e-14 and norm2 > 1e-14 else None,
                        'global_weighted_loss_error': fraction * (sample_loss - exact_loss),
                        'diagnostic_only_no_model_change': True})
                    for j, feature in enumerate(NAMES):
                        coefficients.append({'fold': fold, 'time': at, 'beta': name, 'feature': feature,
                            'exact_unpenalized_gradient': float(exact_gradient[j]),
                            'sampled_unpenalized_gradient': float(sample_gradient[j]),
                            'gradient_error': float(sample_gradient[j] - exact_gradient[j]),
                            'global_mass_weighted_error': float(fraction * (sample_gradient[j] - exact_gradient[j]))})
            print(utc_now(), 'G3 exact training pair diagnostic fold', fold, flush=True)
    csv_write('G3_EXACT_VS_SAMPLED_PAIR_DIAGNOSTICS.csv', rows)
    csv_write('G3_EXACT_VS_SAMPLED_GRADIENT_COORDINATES.csv', coefficients)
    save('G3_PAIR_DIAGNOSTIC_DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
        'stratum_beta_rows': len(rows), 'gradient_coordinates': len(coefficients),
        'standardization_repair_plan_parity': repair_checks, 'new_fits': 0, 'new_random_draws': 0,
        'outer_validation_rows_used': 0, 'results_sha256': digest(BASE / 'G3_EXACT_VS_SAMPLED_PAIR_DIAGNOSTICS.csv')})
    guard.check_state()
    note('G3 fixed sampling forensic completed', 'Exact training pairs compared; no cap, seed or fitted model changes', 'F6_TIME_MATCHED_RANK_CORRECTION')


if __name__ == '__main__':
    main()
