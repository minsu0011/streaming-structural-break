"""Freeze G3 sampled pair plans and all five strict nested outer predictions."""
from pathlib import Path
import gc
import os
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from src.next7.common import BASE, save, read_json, digest, ids_hash, utc_now, activity, require_guard, note
from src.next7.pairwise import fit_pairwise, make_pair_plan, plan_hash, implementation_hash
from src.next7.model import Correction
from src.next6.data import source
from src.next5.references import primary
from scripts.next7.fit_g1_corrections import verify_inner
from scripts.next7.evaluate_g2_synthetic import check_lock


def main():
    guard = require_guard()
    check_lock()
    protocol = read_json(BASE / 'G3_PROTOCOL_LOCK.json')
    assert protocol['candidate'] == 'G3_PAIRWISE_FULL'
    repair_path = BASE / 'G3_STANDARDIZATION_BUG_REPAIR_001.json'
    tests_path = BASE / ('audit/G3_PAIRWISE_TESTS_002.xml' if repair_path.exists() else 'audit/G3_PAIRWISE_TESTS_001.xml')
    tree = ET.parse(tests_path)
    assert len(tree.findall('.//testcase')) == (8 if repair_path.exists() else 7) and not tree.findall('.//failure') and not tree.findall('.//error')
    cache = BASE / 'cache/g2'
    manifest = read_json(cache / 'MANIFEST.json')
    for name, sha in manifest['files_sha256'].items():
        assert digest(cache / name) == sha
    for name, sha in manifest['source_sha256'].items():
        assert digest(ROOT / name) == sha
    sources = {n: digest(ROOT / n) for n in ('scripts/next7/fit_g3_pairwise.py',
        'src/next7/pairwise.py', 'src/next7/tests/test_pairwise.py', 'src/next7/model.py',
        'src/next7/features.py', 'scripts/next7/fit_g1_corrections.py')}
    lockfile = BASE / 'G3_IMPLEMENTATION_LOCK.json'
    if lockfile.exists():
        if read_json(lockfile)['source_sha256'] != sources:
            repair = read_json(repair_path)
            assert repair['old_source_sha256'] == read_json(lockfile)['source_sha256']
            assert repair['new_source_sha256'] == sources
            assert repair['serialized_models_before_repair'] == 0 and repair['outer_predictions_before_repair'] == 0
            assert repair['test_xml_sha256'] == digest(tests_path)
    else:
        save('G3_IMPLEMENTATION_LOCK.json', {'status': 'FROZEN', 'created_utc': utc_now(),
            'source_sha256': sources, 'model_implementation_sha256': implementation_hash(),
            'protocol_sha256': digest(BASE / 'G3_PROTOCOL_LOCK.json'), 'test_xml_sha256': digest(tests_path),
            'before_any_G3_fit_or_score': True})
    data = source()
    metadata = data.rows()
    features = np.load(cache / 'head_features.npy', mmap_mode='r', allow_pickle=False)
    baseline, _, proof = primary('full', 'PRIMARY', metadata, guard)
    rowfolds = np.repeat(np.array([guard.split.assignment(int(s))[1] for s in data.ids], dtype=np.int8),
        np.diff(data.series['online_offsets']))
    receipts = []
    for fold in range(5):
        relative = f'g3/fold_{fold}'
        directory = BASE / relative
        final = directory / 'OUTER_PREDICTIONS_FROZEN.json'
        if final.exists():
            receipt = read_json(final)
            assert digest(directory / 'predictions.npz') == receipt['prediction_sha256']
            receipts.append(receipt)
            continue
        valid_rows = np.flatnonzero(rowfolds == fold)
        train = set(map(int, np.unique(metadata['dataset_id'][rowfolds != fold])))
        valid = set(map(int, np.unique(metadata['dataset_id'][valid_rows])))
        guard.partition(train, valid)
        train_rows, inner_base, inner_receipt = verify_inner(data, fold, train, valid)
        assert not set(train_rows) & set(valid_rows)
        with activity(f'G3 time-matched pairwise nested fit fold{fold}', 'SCIENTIFIC_FIT', 'F6_TIME_MATCHED_RANK_CORRECTION'):
            model_path = directory / 'model.joblib'
            if model_path.exists():
                raise RuntimeError('Partial G3 fit requires explicit repair receipt')
            x = np.asarray(features[train_rows], dtype=np.float64)
            y, times = metadata['target'][train_rows], metadata['time_online'][train_rows]
            ids = metadata['dataset_id'][train_rows]
            plan = make_pair_plan(y, times, ids)
            p, n = plan['positive_row'], plan['negative_row']
            guard.partition(np.unique(ids[p]), valid)
            guard.partition(np.unique(ids[n]), valid)
            assert np.all(times[p] == times[n]) and np.all(ids[p] != ids[n])
            assert np.all(y[p] == 1) and np.all(y[n] == 0)
            pair_path = BASE / f'cache/g3_pairs/fold_{fold}.npz'
            pair_path.parent.mkdir(parents=True, exist_ok=True)
            with pair_path.open('xb') as f:
                np.savez(f, positive_global_row=train_rows[p], negative_global_row=train_rows[n],
                    weights=plan['weights'], time=plan['time'])
            save(relative + '/PAIR_PLAN_FROZEN.json', {'status': 'FROZEN', 'created_utc': utc_now(),
                'pair_plan_sha256': plan_hash(plan), 'file_sha256': digest(pair_path),
                'sampled_pairs': len(p), 'total_eligible_pair_mass': plan['total_pair_mass'],
                'all_positive_members_outerTRAIN_only': True, 'all_negative_members_outerTRAIN_only': True,
                'outer_validation_ids_sha256': ids_hash(valid), 'training_ids_sha256': ids_hash(train),
                'same_online_time_only': True, 'no_same_series_opposite_label_pairs': True,
                'time_records': plan['time_records'], 'before_G3_fit': True})
            provenance = {'fold': fold, 'strict_nested_OOF': True, 'outer_validation_excluded': True,
                'training_ids_sha256': ids_hash(train), 'validation_ids_sha256': ids_hash(valid),
                'inner_OOF_sha256': inner_receipt['BL80_INNER_OOF_sha256'], 'training_rows': len(train_rows),
                'training_series': len(train), 'validation_series': len(valid), 'validation_labels_used': False,
                'H_only_state_calibration': True, 'research_split_sha256': guard.split.digest,
                'feature_cache_manifest_sha256': digest(cache / 'MANIFEST.json'), 'candidate': 'G3_PAIRWISE_FULL',
                'pair_plan_file_sha256': digest(pair_path), 'G3_protocol_sha256': digest(BASE / 'G3_PROTOCOL_LOCK.json')}
            model = fit_pairwise(x, inner_base, y, times, ids, provenance, frozen_plan=plan)
            parent = Correction.load(BASE / f'g2/fold_{fold}/G2_FULL/model.joblib')
            np.testing.assert_array_equal(model.center, parent.center)
            np.testing.assert_array_equal(model.spread, parent.spread)
            model.save(model_path)
            prediction = np.empty(len(valid_rows))
            for start in range(0, len(valid_rows), 50000):
                rows = valid_rows[start:start + 50000]
                prediction[start:start + len(rows)] = model.predict(baseline[rows], features[rows])
            assert np.isfinite(prediction).all()
            output = directory / 'predictions.npz'
            with output.open('xb') as f:
                np.savez(f, global_row=valid_rows, dataset_id=metadata['dataset_id'][valid_rows],
                    prediction=prediction, M0=baseline[valid_rows])
            save(relative + '/DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
                'candidate': 'G3_PAIRWISE_FULL', 'fold': fold, 'model_sha256': digest(model_path),
                'prediction_sha256': digest(output), 'training': model.training,
                'standardization_exact_G2_parent_parity': True, 'outer_score_released': False})
            receipt = {'status': 'FROZEN', 'created_utc': utc_now(), 'fold': fold,
                'candidate': 'G3_PAIRWISE_FULL', 'prediction_sha256': digest(output),
                'model_sha256': digest(model_path), 'outer_score_released': False,
                'baseline_authority': proof, 'training_ids_sha256': ids_hash(train),
                'validation_ids_sha256': ids_hash(valid), 'source_sha256': sources}
            save(relative + '/OUTER_PREDICTIONS_FROZEN.json', receipt)
            receipts.append(receipt)
            del x, y, times, ids, plan, model, parent, prediction
            gc.collect()
    if not (BASE / 'G3_ALL_OUTER_PREDICTIONS_FROZEN.json').exists():
        save('G3_ALL_OUTER_PREDICTIONS_FROZEN.json', {'status': 'FROZEN', 'created_utc': utc_now(),
            'folds': receipts, 'all_five_completed_before_G3_scores': True,
            'scientific_candidates': ['G3_PAIRWISE_FULL'], 'source_sha256': sources})
    guard.check_state()
    note('G3 five outer predictions frozen', 'Same current/event state, objective change only', 'F6_TIME_MATCHED_RANK_CORRECTION')


if __name__ == '__main__':
    main()
