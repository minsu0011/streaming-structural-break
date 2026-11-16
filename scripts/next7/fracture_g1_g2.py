"""Fixed time, age, H-regime and observed-mechanism diagnosis with paired support."""
from pathlib import Path
import gc
import os
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from numba import set_num_threads
from src.next7.common import BASE, save, read_json, digest, csv_write, utc_now, activity, require_guard, note
from src.next7.uncertainty import prepared_totals, paired_summary
from src.next5.bootstrap import PreparedAUC
from src.next5.fractures import masks, pair_mass
from src.next6.data import source
from scripts.next7.bootstrap_g1 import prediction as g1_prediction
from scripts.next7.bootstrap_g2 import prediction as g2_prediction

METHODS = ('M0', 'I3_CURRENT_EVENT', 'M_FULL', 'I4_EVENT_SCALE_TREE', 'G2_EVENT', 'G2_FULL')


def diagnostics(data):
    directory = ROOT / 'artifacts/next5/diagnostics'
    lock = read_json(directory / 'RESEARCH_DIAGNOSTIC_LOCK.json')
    assert lock['series'] == 6400 and lock['training_feature_use_forbidden']
    path = directory / 'RESEARCH_SIGNATURES.npz'
    assert digest(path) == lock['files_sha256'][path.name]
    with np.load(path, allow_pickle=False) as v:
        ids = v['dataset_id'].copy()
        data.guard.admit(ids, purpose='Frozen RP-only posthoc fracture descriptors')
        assert set(map(int, ids)) == set(map(int, data.ids))
        lookup = {int(s): i for i, s in enumerate(ids)}
        order = np.array([lookup[int(s)] for s in data.ids])
        tau = v['tau'][order].copy()
        np.testing.assert_array_equal(tau, data.series['tau'])
        result = {name: v[name][order].copy() for name in ('tau', 'historical_regime', 'signature', 'observed_scale_direction')}
    lengths = np.diff(data.series['online_offsets'])
    return {key: np.repeat(value, lengths) for key, value in result.items()}


def cached_all_bootstrap(method, fold):
    generation = 'g2' if method.startswith('G2_') else 'g1'
    path = BASE / f'bootstrap/{generation}/{method}/fold_{fold}.npy'
    assert digest(path) == read_json(path.with_suffix('.json'))['result_sha256']
    return np.load(path, allow_pickle=False)


def main():
    guard = require_guard()
    xml = BASE / 'audit/UNCERTAINTY_TESTS_001.xml'
    tree = ET.parse(xml)
    assert not tree.findall('.//failure') and not tree.findall('.//error')
    assert len(tree.findall('.//testcase')) == 4
    assert read_json(BASE / 'G2_BOOTSTRAP_DONE.json')['draws'] == 2000
    assert read_json(BASE / 'G1_BOOTSTRAP_DONE.json')['draws'] == 2000
    counts_lock = read_json(BASE / 'G1_BOOTSTRAP_COUNTS_LOCK.json')
    assert read_json(BASE / 'G2_BOOTSTRAP_COUNTS_LOCK.json')['counts'] == counts_lock['counts']
    source_hashes = {n: digest(ROOT / n) for n in ('scripts/next7/fracture_g1_g2.py',
        'src/next7/uncertainty.py', 'src/next5/fractures.py', 'src/next5/bootstrap.py')}
    lockfile = BASE / 'FRACTURE_G1_G2_LOCK.json'
    if not lockfile.exists():
        save('FRACTURE_G1_G2_LOCK.json', {'status': 'FROZEN', 'created_utc': utc_now(),
            'methods': METHODS, 'source_sha256': source_hashes, 'test_xml_sha256': digest(xml),
            'draws': 2000, 'reuse_exact_existing_whole_series_counts': True,
            'purpose': 'Diagnosis of five already-scored candidates; no new model, blend, cutoff or promotion selection',
            'scope_definitions': 'Exact fixed NEXT5 time/positive-age/regime/observed-mechanism masks',
            'age_comparators': 'All negative examples at original online times; age restricts positive examples only',
            'missing_pairs': 'Zero eligible pair mass is missing, never AUC=.5. A paired bootstrap mean requires every included fold supported; report discarded draws.',
            'multiple_intervals': 'Pointwise conditional intervals; not simultaneous or adaptive-selection corrected',
            'mechanism_limit': 'Existing posthoc observed signatures, not latent DGP labels or training inputs',
            'historical_regime_limit': 'Existing historical transform is a descriptive transductive RP grouping, not an outer-fitted predictive feature',
            'median_clarity': 'Median of matched fold deltas and difference of fold-score medians reported separately'})
    assert read_json(lockfile)['source_sha256'] == source_hashes
    data = source()
    metadata = data.rows()
    descriptors = diagnostics(data)
    set_num_threads(4)
    completed = []
    scope_names = None
    for fold in range(5):
        relative = f'fractures_g1_g2/fold_{fold}'
        directory = BASE / relative
        done = directory / 'DONE.json'
        if done.exists():
            record = read_json(done)
            for name, sha in record['files_sha256'].items():
                assert digest(directory / name) == sha
            completed.append(record)
            scope_names = record['scopes']
            continue
        with activity(f'fixed paired fractures G1/G2 fold{fold}', 'BOTTLENECK_FORENSIC', 'F5_CAUSAL_NESTED_INTEGRATION'):
            rows, baseline = g1_prediction('M0', fold, metadata, guard)
            y, times = metadata['target'][rows], metadata['time_online'][rows]
            ids, codes = np.unique(metadata['dataset_id'][rows], return_inverse=True)
            cp = BASE / counts_lock['counts'][str(fold)]['path']
            assert digest(cp) == counts_lock['counts'][str(fold)]['sha256']
            with np.load(cp, allow_pickle=False) as v:
                np.testing.assert_array_equal(ids, v['dataset_id'])
                counts = v['multiplicities'].copy()
            scope = masks(y, times, descriptors['tau'][rows], descriptors['historical_regime'][rows],
                descriptors['signature'][rows], descriptors['observed_scale_direction'][rows])
            scope_names = list(scope)
            supports = []
            for name, select in scope.items():
                supports.append({'fold': fold, 'scope': name, 'rows': int(select.sum()),
                    'series': len(np.unique(codes[select])),
                    'positive_series': len(np.unique(codes[select & (y == 1)])),
                    'negative_series': len(np.unique(codes[select & (y == 0)])),
                    'positive_rows': int(y[select].sum()), 'negative_rows': int(np.sum(y[select] == 0)),
                    'pair_mass': pair_mass(y[select], times[select])})
            csv_write(relative + '/SUPPORT.csv', supports)
            files = {'SUPPORT.csv': digest(directory / 'SUPPORT.csv')}
            for method in METHODS:
                r, p = (g2_prediction(method, fold, metadata, guard) if method.startswith('G2_')
                    else g1_prediction(method, fold, metadata, guard))
                np.testing.assert_array_equal(r, rows)
                totals = {}
                for name, select in scope.items():
                    if name == 'all_time':
                        continue
                    if not np.any(select):
                        totals[name] = np.zeros((len(counts), 2))
                    else:
                        prepared = PreparedAUC.create(y[select], p[select], times[select], codes[select], len(ids))
                        totals[name] = prepared_totals(prepared, counts)
                time_total = sum(v for name, v in totals.items() if name.startswith('time_'))
                age_total = sum(v for name, v in totals.items() if name.startswith('age_'))
                np.testing.assert_array_equal(time_total, age_total)
                assert np.all(time_total[:, 1] > 0)
                np.testing.assert_allclose(time_total[:, 0] / time_total[:, 1], cached_all_bootstrap(method, fold), atol=1e-12, rtol=0)
                totals['all_time'] = time_total
                output = directory / (method + '.npz')
                with output.open('xb') as f:
                    np.savez(f, **totals)
                files[output.name] = digest(output)
                print(utc_now(), 'fracture fold', fold, method, 'complete', flush=True)
                del p, totals
                gc.collect()
            record = {'status': 'COMPLETE', 'created_utc': utc_now(), 'fold': fold,
                'scopes': scope_names, 'files_sha256': files, 'counts_sha256': digest(cp),
                'time_and_age_totals_exactly_partition_every_bootstrap_draw': True,
                'all_time2001_draws_match_existing_bootstrap': True}
            save(relative + '/DONE.json', record)
            completed.append(record)
    all_tables = {m: {} for m in METHODS}
    for method in METHODS:
        for fold in range(5):
            with np.load(BASE / f'fractures_g1_g2/fold_{fold}/{method}.npz', allow_pickle=False) as v:
                for name in scope_names:
                    all_tables[method].setdefault(name, []).append(v[name].copy())
    report = []
    for name in scope_names:
        base = np.stack(all_tables['M0'][name], axis=1)  # draw x fold x (wins,pairs)
        supported_folds = base[0, :, 1] > 0
        for method in METHODS[1:]:
            cand = np.stack(all_tables[method][name], axis=1)
            np.testing.assert_array_equal(cand[:, :, 1], base[:, :, 1])
            eligible = np.all(base[:, supported_folds, 1] > 0, axis=1) if supported_folds.any() else np.zeros(len(base), dtype=bool)
            selected_draws = np.flatnonzero(eligible)
            record = {'scope': name, 'candidate': method, 'reference': 'M0',
                'identity_supported_folds': int(supported_folds.sum()),
                'identity_pair_mass': int(base[0, :, 1].sum()), 'requested_draws': 2000,
                'usable_bootstrap_draws': int(eligible[1:].sum()), 'missing_pair_draws': int((~eligible[1:]).sum()),
                'coverage_status': 'ALL5' if supported_folds.all() else 'PARTIAL_FOLDS',
                'pointwise_conditional_interval': True, 'adaptive_selection_not_corrected': True}
            if eligible[0] and len(selected_draws) > 1:
                ref_score = base[selected_draws][:, supported_folds, 0] / base[selected_draws][:, supported_folds, 1]
                cand_score = cand[selected_draws][:, supported_folds, 0] / cand[selected_draws][:, supported_folds, 1]
                record.update(paired_summary(ref_score, cand_score))
                record['mean_reference_TS_AUC'] = float(ref_score[0].mean())
                record['mean_candidate_TS_AUC'] = float(cand_score[0].mean())
            report.append(record)
    csv_write('FRACTURE_RESULTS_NEXT7_G1_G2.csv', report)
    save('FRACTURE_G1_G2_DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
        'methods': METHODS, 'scopes': scope_names, 'folds': completed, 'new_fits': 0,
        'source_sha256': source_hashes, 'no_candidate_changes_from_fractures': True})
    guard.check_state()
    note('Fixed G1/G2 paired fractures complete', 'Review structural causes without threshold search', 'F5_CAUSAL_NESTED_INTEGRATION')


if __name__ == '__main__':
    main()
