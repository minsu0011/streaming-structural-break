"""Extend the already fixed fracture accounting to the final G3 definition."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from numba import set_num_threads
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, require_guard, note
from src.next7.uncertainty import prepared_totals, paired_summary
from src.next5.bootstrap import PreparedAUC
from src.next5.fractures import masks
from src.next6.data import source
from scripts.next7.fracture_g1_g2 import diagnostics


def main():
    guard = require_guard()
    source_hashes = {n: digest(ROOT / n) for n in ('scripts/next7/fracture_g3.py',
        'scripts/next7/fracture_g1_g2.py', 'src/next7/uncertainty.py', 'src/next5/fractures.py', 'src/next5/bootstrap.py')}
    counts_lock = read_json(BASE / 'G3_BOOTSTRAP_COUNTS_LOCK.json')
    assert counts_lock['draws'] == 2000
    assert counts_lock['counts'] == read_json(BASE / 'G1_BOOTSTRAP_COUNTS_LOCK.json')['counts']
    lockfile = BASE / 'FRACTURE_G3_LOCK.json'
    if not lockfile.exists():
        save(lockfile.name, {'status': 'FROZEN', 'created_utc': utc_now(), 'candidate': 'G3_PAIRWISE_FULL',
            'references': ['M0', 'G2_FULL'], 'source_sha256': source_hashes,
            'parent_scope_lock_sha256': digest(BASE / 'FRACTURE_G1_G2_LOCK.json'),
            'counts_lock_sha256': digest(BASE / 'G3_BOOTSTRAP_COUNTS_LOCK.json'),
            'scope': 'Exactly prior time, positive-age, H regime and posthoc observed-mechanism masks; no new bins or learned gates.',
            'missing_support': 'Zero-pair draws are undefined; use common supported folds and report available draw counts.',
            'purpose': 'Final-candidate diagnosis only; no fits, new random draws or selection changes.',
            'interval_limit': 'Pointwise, conditional on fitted OOF; adaptive selection and multiple comparisons not corrected.'})
    assert read_json(lockfile)['source_sha256'] == source_hashes
    data = source()
    meta = data.rows()
    desc = diagnostics(data)
    set_num_threads(2)
    completed = []
    scopes = None
    for fold in range(5):
        relative = f'fractures_g3/fold_{fold}'
        directory = BASE / relative
        if (directory / 'DONE.json').exists():
            receipt = read_json(directory / 'DONE.json')
            assert digest(directory / 'G3_PAIRWISE_FULL.npz') == receipt['result_sha256']
            completed.append(receipt)
            scopes = receipt['scopes']
            continue
        with activity(f'G3 fixed paired fractures fold{fold}', 'BOTTLENECK_FORENSIC', 'F6_TIME_MATCHED_RANK_CORRECTION'):
            path = BASE / f'g3/fold_{fold}/predictions.npz'
            assert digest(path) == read_json(BASE / f'g3/fold_{fold}/DONE.json')['prediction_sha256']
            with np.load(path, allow_pickle=False) as f:
                rows, prediction = f['global_row'].copy(), f['prediction'].copy()
                np.testing.assert_array_equal(f['dataset_id'], meta['dataset_id'][rows])
            ids, codes = np.unique(meta['dataset_id'][rows], return_inverse=True)
            guard.admit(ids, purpose='G3 frozen outer fracture diagnostics')
            assert all(guard.split.assignment(int(s))[1] == fold for s in ids)
            count_path = BASE / counts_lock['counts'][str(fold)]['path']
            assert digest(count_path) == counts_lock['counts'][str(fold)]['sha256']
            with np.load(count_path, allow_pickle=False) as f:
                np.testing.assert_array_equal(ids, f['dataset_id'])
                counts = f['multiplicities'].copy()
            y, times = meta['target'][rows], meta['time_online'][rows]
            scope = masks(y, times, desc['tau'][rows], desc['historical_regime'][rows],
                desc['signature'][rows], desc['observed_scale_direction'][rows])
            scopes = list(scope)
            parent = BASE / f'fractures_g1_g2/fold_{fold}'
            parent_done = read_json(parent / 'DONE.json')
            assert scopes == parent_done['scopes']
            for name in ('M0.npz', 'G2_FULL.npz'):
                assert digest(parent / name) == parent_done['files_sha256'][name]
            totals = {}
            for name, selected in scope.items():
                if name == 'all_time':
                    continue
                totals[name] = (prepared_totals(PreparedAUC.create(y[selected], prediction[selected], times[selected],
                    codes[selected], len(ids)), counts) if np.any(selected) else np.zeros((len(counts), 2)))
            total_time = sum(v for name, v in totals.items() if name.startswith('time_'))
            total_age = sum(v for name, v in totals.items() if name.startswith('age_'))
            np.testing.assert_array_equal(total_time, total_age)
            path = BASE / f'bootstrap/g3/G3_PAIRWISE_FULL/fold_{fold}.npy'
            assert digest(path) == read_json(path.with_suffix('.json'))['result_sha256']
            np.testing.assert_allclose(total_time[:, 0] / total_time[:, 1], np.load(path, allow_pickle=False), rtol=0., atol=1e-12)
            totals['all_time'] = total_time
            with np.load(parent / 'M0.npz', allow_pickle=False) as f:
                for name in scopes:
                    np.testing.assert_array_equal(totals[name][:, 1], f[name][:, 1])
            directory.mkdir(parents=True, exist_ok=True)
            output = directory / 'G3_PAIRWISE_FULL.npz'
            with output.open('xb') as f:
                np.savez(f, **totals)
            receipt = {'status': 'COMPLETE', 'created_utc': utc_now(), 'fold': fold, 'scopes': scopes,
                'result_sha256': digest(output), 'counts_sha256': digest(count_path),
                'time_age_all_draws_exact_partition': True, 'all_time_matches_frozen_bootstrap': True,
                'pair_support_exact_parent_parity': True}
            save(relative + '/DONE.json', receipt)
            completed.append(receipt)
    tables = {method: {} for method in ('M0', 'G2_FULL', 'G3_PAIRWISE_FULL')}
    for method in tables:
        for fold in range(5):
            relative = f'fractures_g3/fold_{fold}' if method == 'G3_PAIRWISE_FULL' else f'fractures_g1_g2/fold_{fold}'
            with np.load(BASE / relative / (method + '.npz'), allow_pickle=False) as f:
                for name in scopes:
                    tables[method].setdefault(name, []).append(f[name].copy())
    report = []
    for name in scopes:
        candidate = np.stack(tables['G3_PAIRWISE_FULL'][name], axis=1)
        for reference in ('M0', 'G2_FULL'):
            base = np.stack(tables[reference][name], axis=1)
            np.testing.assert_array_equal(candidate[:, :, 1], base[:, :, 1])
            supported = base[0, :, 1] > 0
            eligible = np.all(base[:, supported, 1] > 0, axis=1) if supported.any() else np.zeros(len(base), dtype=bool)
            selected = np.flatnonzero(eligible)
            row = {'scope': name, 'candidate': 'G3_PAIRWISE_FULL', 'reference': reference,
                'identity_supported_folds': int(supported.sum()), 'identity_pair_mass': int(base[0, :, 1].sum()),
                'requested_draws': 2000, 'usable_bootstrap_draws': int(eligible[1:].sum()),
                'missing_pair_draws': int((~eligible[1:]).sum()), 'coverage_status': 'ALL5' if supported.all() else 'PARTIAL_FOLDS',
                'pointwise_conditional_interval': True, 'adaptive_selection_not_corrected': True}
            if eligible[0] and len(selected) > 1:
                bs = base[selected][:, supported, 0] / base[selected][:, supported, 1]
                cs = candidate[selected][:, supported, 0] / candidate[selected][:, supported, 1]
                row.update(paired_summary(bs, cs))
                row['mean_reference_TS_AUC'], row['mean_candidate_TS_AUC'] = float(bs[0].mean()), float(cs[0].mean())
            report.append(row)
    csv_write('FRACTURE_RESULTS_NEXT7_G3.csv', report)
    save('FRACTURE_G3_DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(), 'folds': completed,
        'scopes': scopes, 'new_fits': 0, 'new_random_draws': 0, 'source_sha256': source_hashes,
        'result_sha256': digest(BASE / 'FRACTURE_RESULTS_NEXT7_G3.csv')})
    guard.check_state()
    note('G3 fixed fracture extension complete', 'All original time/age/diagnostic scopes with exact shared support', 'F6_TIME_MATCHED_RANK_CORRECTION')


if __name__ == '__main__':
    main()
