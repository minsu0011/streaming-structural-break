"""Fixed G3 whole-series bootstrap, with unchanged reference draws when compatible."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from numba import set_num_threads
from src.next7.common import BASE, save, read_json, digest, csv_write, utc_now, activity, require_guard
from src.next7.uncertainty import paired_summary
from src.next5.bootstrap import PreparedAUC
from src.next6.data import source
from scripts.next7.bootstrap_g2 import prediction as reference_prediction


def main():
    guard = require_guard()
    policy = read_json(BASE / 'G3_BOOTSTRAP_CANDIDATE_FROZEN.json')
    n = policy['draws']
    assert n in (2000, 5000)
    data = source()
    metadata = data.rows()
    original = read_json(BASE / 'G1_BOOTSTRAP_COUNTS_LOCK.json')
    compatible = n == original['draws']
    hashes = {p: digest(ROOT / p) for p in ('scripts/next7/bootstrap_g3.py', 'src/next7/uncertainty.py', 'src/next5/bootstrap.py')}
    lockfile = BASE / 'G3_BOOTSTRAP_COUNTS_LOCK.json'
    if not lockfile.exists():
        records = original['counts'] if compatible else {}
        if not compatible:
            rng = np.random.default_rng(20260913)
            for fold in range(5):
                ids = np.array(sorted(int(s) for s in data.ids if guard.split.assignment(int(s))[1] == fold))
                counts = np.vstack((np.ones((1, len(ids)), dtype=np.int32),
                    rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=n))).astype(np.int32)
                path = BASE / f'bootstrap/g3/counts/fold_{fold}.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as f:
                    np.savez(f, dataset_id=ids, multiplicities=counts)
                records[str(fold)] = {'path': path.relative_to(BASE).as_posix(), 'sha256': digest(path)}
        save('G3_BOOTSTRAP_COUNTS_LOCK.json', {'status': 'FROZEN', 'created_utc': utc_now(),
            'draws': n, 'counts': records, 'source_sha256': hashes, 'reuse_existing_draws': compatible,
            'no_extension_after_intervals': True, 'whole_series_within_original_fold': True})
    lock = read_json(lockfile)
    assert lock['source_sha256'] == hashes and lock['draws'] == n
    set_num_threads(4)
    matrices = {}
    for method in ('M0', 'G2_FULL', 'G3_PAIRWISE_FULL'):
        matrix = np.empty((n + 1, 5))
        for fold in range(5):
            if compatible and method != 'G3_PAIRWISE_FULL':
                generation = 'g1' if method == 'M0' else 'g2'
                path = BASE / f'bootstrap/{generation}/{method}/fold_{fold}.npy'
                assert digest(path) == read_json(path.with_suffix('.json'))['result_sha256']
                matrix[:, fold] = np.load(path, allow_pickle=False)
                continue
            out = BASE / f'bootstrap/g3/{method}/fold_{fold}.npy'
            done = out.with_suffix('.json')
            if done.exists():
                assert digest(out) == read_json(done)['result_sha256']
                matrix[:, fold] = np.load(out, allow_pickle=False)
                continue
            with activity(f'G3 paired {method} fold{fold} draws{n}', 'UNCERTAINTY', 'F6_TIME_MATCHED_RANK_CORRECTION'):
                cp = BASE / lock['counts'][str(fold)]['path']
                assert digest(cp) == lock['counts'][str(fold)]['sha256']
                with np.load(cp, allow_pickle=False) as v:
                    ids, counts = v['dataset_id'].copy(), v['multiplicities'].copy()
                guard.admit(ids, purpose='G3 bootstrap count admission')
                if method == 'G3_PAIRWISE_FULL':
                    with np.load(BASE / f'g3/fold_{fold}/predictions.npz', allow_pickle=False) as v:
                        rows, p = v['global_row'].copy(), v['prediction'].copy()
                        np.testing.assert_array_equal(metadata['dataset_id'][rows], v['dataset_id'])
                    expected = read_json(BASE / 'G3_OUTER_SCORES.json')['scores'][method][str(fold)]
                else:
                    rows, p = reference_prediction(method, fold, metadata, guard)
                    generation = 'G1' if method == 'M0' else 'G2'
                    expected = read_json(BASE / f'{generation}_OUTER_SCORES.json')['scores'][method][str(fold)]
                actual, codes = np.unique(metadata['dataset_id'][rows], return_inverse=True)
                np.testing.assert_array_equal(actual, ids)
                values = PreparedAUC.create(metadata['target'][rows], p, metadata['time_online'][rows], codes, len(ids)).evaluate(counts)
                assert abs(values[0] - expected) < 1e-12
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open('xb') as f:
                    np.save(f, values, allow_pickle=False)
                save(str(done.relative_to(BASE)), {'status': 'COMPLETE', 'created_utc': utc_now(),
                    'result_sha256': digest(out), 'counts_sha256': digest(cp), 'unweighted_scorer_parity': True})
                matrix[:, fold] = values
        matrices[method] = matrix
    if not (BASE / 'G3_BOOTSTRAP_DONE.json').exists():
        rows = [{'candidate': 'G3_PAIRWISE_FULL', 'reference': reference,
            'fixed_fitted_OOF_conditional': True, 'adaptive_selection_not_corrected': True,
            **paired_summary(matrices[reference], matrices['G3_PAIRWISE_FULL'])} for reference in ('M0', 'G2_FULL')]
        csv_write('PAIRED_BOOTSTRAP_NEXT7_G3.csv', rows)
        save('G3_BOOTSTRAP_DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(), 'draws': n,
            'counts_lock_sha256': digest(lockfile), 'source_sha256': hashes, 'all_fold_scores_match': True})
    guard.check_state()


if __name__ == '__main__':
    main()
