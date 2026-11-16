"""Fixed G2 paired bootstrap, reusing compatible G1 draws and reference results."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from numba import set_num_threads
from src.next7.common import BASE, save, read_json, digest, csv_write, utc_now, activity, require_guard
from src.next6.data import source
from src.next5.bootstrap import PreparedAUC, paired_summary
from scripts.next7.bootstrap_g1 import prediction as g1_prediction


def prediction(method, fold, metadata, guard):
    if method == 'M0' or method.startswith('G1_'):
        return g1_prediction(method.removeprefix('G1_'), fold, metadata, guard)
    directory = BASE / f'g2/fold_{fold}'
    path = directory / 'G2_FULL/predictions.npz' if method == 'G2_FULL' else directory / 'fixed_predictions.npz'
    with np.load(path, allow_pickle=False) as values:
        rows = values['global_row'].copy()
        guard.admit(np.unique(values['dataset_id']), purpose='G2 bootstrap admission')
        np.testing.assert_array_equal(metadata['dataset_id'][rows], values['dataset_id'])
        return rows, values['prediction' if method == 'G2_FULL' else method].copy()


def main():
    guard = require_guard()
    selection = read_json(BASE / 'G2_BOOTSTRAP_CANDIDATES_FROZEN.json')
    n = selection['draws']
    assert n in (2000, 5000)
    methods = ['M0', 'G1_M_EVENT', 'G1_M_FULL', 'G2_EVENT', 'G2_FULL']
    g1lock = read_json(BASE / 'G1_BOOTSTRAP_COUNTS_LOCK.json')
    compatible = n == g1lock['draws']
    data = source()
    metadata = data.rows()
    path = BASE / 'G2_BOOTSTRAP_COUNTS_LOCK.json'
    source_hashes = {n: digest(ROOT / n) for n in ('scripts/next7/bootstrap_g2.py', 'src/next5/bootstrap.py', 'src/next/bootstrap.py')}
    if not path.exists():
        if compatible:
            records = g1lock['counts']
        else:
            records = {}
            rng = np.random.default_rng(read_json(BASE / 'NEXT7_RANDOMNESS_LOCK.json')['bootstrap_seed'])
            for fold in range(5):
                ids = np.array(sorted(int(s) for s in data.ids if guard.split.assignment(int(s))[1] == fold))
                assert len(set(guard.split.groups[int(i)] for i in ids)) == len(ids)
                counts = np.vstack((np.ones((1, len(ids)), dtype=np.int32),
                    rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=n))).astype(np.int32)
                dest = BASE / f'bootstrap/g2/counts/fold_{fold}.npz'
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open('xb') as f:
                    np.savez(f, dataset_id=ids, multiplicities=counts)
                records[str(fold)] = {'path': dest.relative_to(BASE).as_posix(), 'sha256': digest(dest),
                    'groups': len(ids), 'identity_row_plus_draws': n + 1}
        save('G2_BOOTSTRAP_COUNTS_LOCK.json', {'status': 'FROZEN', 'created_utc': utc_now(),
            'draws': n, 'counts': records, 'methods': methods, 'source_sha256': source_hashes,
            'reuse_existing_G1_draws': compatible, 'no_post_interval_extension': True,
            'all_methods_share_exact_series_multiplicities': True})
    lock = read_json(path)
    assert lock['source_sha256'] == source_hashes and lock['draws'] == n
    set_num_threads(4)
    point = read_json(BASE / 'G2_OUTER_SCORES.json')['scores']
    old_point = read_json(BASE / 'G1_OUTER_SCORES.json')['scores']
    matrices = {}
    for method in methods:
        result = np.empty((n + 1, 5))
        for fold in range(5):
            if compatible and (method == 'M0' or method.startswith('G1_')):
                old_method = method.removeprefix('G1_')
                old = BASE / f'bootstrap/g1/{old_method}/fold_{fold}.npy'
                assert digest(old) == read_json(old.with_suffix('.json'))['result_sha256']
                result[:, fold] = np.load(old, allow_pickle=False)
                continue
            out = BASE / f'bootstrap/g2/{method}/fold_{fold}.npy'
            done = out.with_suffix('.json')
            if done.exists():
                assert digest(out) == read_json(done)['result_sha256']
                result[:, fold] = np.load(out, allow_pickle=False)
                continue
            with activity(f'G2 paired {method} fold{fold} draws{n}', 'UNCERTAINTY', 'F5_CAUSAL_NESTED_INTEGRATION'):
                count_path = BASE / lock['counts'][str(fold)]['path']
                assert digest(count_path) == lock['counts'][str(fold)]['sha256']
                with np.load(count_path, allow_pickle=False) as v:
                    ids, counts = v['dataset_id'].copy(), v['multiplicities'].copy()
                    guard.admit(ids, purpose='G2 bootstrap multiplicities')
                rows, p = prediction(method, fold, metadata, guard)
                actual, codes = np.unique(metadata['dataset_id'][rows], return_inverse=True)
                np.testing.assert_array_equal(actual, ids)
                prepared = PreparedAUC.create(metadata['target'][rows], p, metadata['time_online'][rows], codes, len(ids))
                values = prepared.evaluate(counts)
                expected = (point[method] if method in point else old_point[method.removeprefix('G1_')])[str(fold)]
                assert abs(values[0] - expected) < 1e-12
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open('xb') as f:
                    np.save(f, values, allow_pickle=False)
                save(str(done.relative_to(BASE)), {'status': 'COMPLETE', 'created_utc': utc_now(),
                    'method': method, 'fold': fold, 'draws': n, 'result_sha256': digest(out),
                    'counts_sha256': digest(count_path), 'unweighted_scorer_parity': True})
                result[:, fold] = values
        matrices[method] = result
    rows = []
    for method in selection['selected']:
        for reference in methods[:3]:
            rows.append({'candidate': method, 'reference': reference, 'fixed_fitted_OOF_conditional': True,
                'adaptive_selection_not_corrected': True, **paired_summary(matrices[reference], matrices[method])})
    if not (BASE / 'G2_BOOTSTRAP_DONE.json').exists():
        csv_write('PAIRED_BOOTSTRAP_NEXT7_G2.csv', rows)
        save('G2_BOOTSTRAP_DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(), 'draws': n,
            'candidates': selection['selected'], 'reuse_existing_G1_reference_draws': compatible,
            'counts_lock_sha256': digest(path), 'all_unweighted_fold_scores_match': True})
    guard.check_state()


if __name__ == '__main__':
    main()
