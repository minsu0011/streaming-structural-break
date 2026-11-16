"""One fixed G2 meta recipe, unchanged M_FULL optimizer and strict inner OOF."""
from pathlib import Path
import gc
import os
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from src.next7.common import BASE, save, read_json, digest, ids_hash, utc_now, activity, require_guard, note
from src.next7.model import fit_correction, implementation_hash
from src.next.fitting import weights_from_training
from src.next6.data import source
from src.next5.references import primary
from scripts.next7.fit_g1_corrections import verify_inner
from scripts.next7.evaluate_g2_synthetic import check_lock


def main():
    guard = require_guard()
    check_lock()
    assert read_json(BASE / 'G2_SCREEN_GATE.json')['continue_full']
    for fold in range(5):
        assert read_json(BASE / f'nested_base/fold_{fold}/DONE.json')['status'] == 'COMPLETE'
    cache = BASE / 'cache/g2'
    manifest = read_json(cache / 'MANIFEST.json')
    for name, sha in manifest['files_sha256'].items():
        assert digest(cache / name) == sha
    for name, sha in manifest['source_sha256'].items():
        assert digest(ROOT / name) == sha
    # G1 mathematical recipe remains source-identical after its reviewed zero-W3 validator repair.
    original = read_json(BASE / 'G1_CORRECTION_BUG_REPAIR_001.json')['new_source_sha256']
    assert digest(ROOT / 'src/next7/model.py') == original['src/next7/model.py']
    assert digest(ROOT / 'src/next7/features.py') == original['src/next7/features.py']
    sources = {n: digest(ROOT / n) for n in ('scripts/next7/fit_g2_corrections.py',
        'scripts/next7/fit_g1_corrections.py', 'src/next7/model.py', 'src/next7/features.py', 'src/next7/null_growth.py')}
    lock = BASE / 'G2_CORRECTION_IMPLEMENTATION_LOCK.json'
    if lock.exists():
        assert read_json(lock)['source_sha256'] == sources
    else:
        save('G2_CORRECTION_IMPLEMENTATION_LOCK.json', {'status': 'FROZEN', 'created_utc': utc_now(),
            'source_sha256': sources, 'model_source_sha256': implementation_hash(),
            'G2_protocol_sha256': digest(BASE / 'G2_PROTOCOL_LOCK.json'),
            'G2_feature_manifest_sha256': digest(cache / 'MANIFEST.json'),
            'before_all_G2_outer_scores': True, 'method': 'M_FULL recipe; qualified memory replacement only'})
    data = source()
    metadata = data.rows()
    features = np.load(cache / 'head_features.npy', mmap_mode='r', allow_pickle=False)
    qualified = np.load(cache / 'qualified.npy', mmap_mode='r', allow_pickle=False)
    baseline, _, proof = primary('full', 'PRIMARY', metadata, guard)
    rowfolds = np.repeat(np.array([guard.split.assignment(int(s))[1] for s in data.ids], dtype=np.int8),
        np.diff(data.series['online_offsets']))
    receipts = []
    for fold in range(5):
        directory = BASE / f'g2/fold_{fold}'
        final = directory / 'OUTER_PREDICTIONS_FROZEN.json'
        if final.exists():
            receipt = read_json(final)
            for name, sha in receipt['prediction_sha256'].items():
                assert digest(directory / name) == sha
            receipts.append(receipt)
            continue
        valid_rows = np.flatnonzero(rowfolds == fold)
        train = set(map(int, np.unique(metadata['dataset_id'][rowfolds != fold])))
        valid = set(map(int, np.unique(metadata['dataset_id'][valid_rows])))
        guard.partition(train, valid)
        train_rows, inner_base, inner_receipt = verify_inner(data, fold, train, valid)
        assert not set(train_rows) & set(valid_rows)
        assert set(map(int, np.unique(metadata['dataset_id'][train_rows]))) == train
        with activity(f'G2_FULL nested correction fold{fold}', 'SCIENTIFIC_FIT', 'F5_CAUSAL_NESTED_INTEGRATION'):
            model_path = directory / 'G2_FULL/model.joblib'
            if model_path.exists():
                raise RuntimeError('Partial G2 correction needs explicit repair receipt')
            x = np.asarray(features[train_rows], dtype=np.float64)
            y = metadata['target'][train_rows]
            weights = weights_from_training(y, metadata['time_online'][train_rows], metadata['dataset_id'][train_rows], 'W3')
            provenance = {'fold': fold, 'strict_nested_OOF': True, 'outer_validation_excluded': True,
                'training_ids_sha256': ids_hash(train), 'validation_ids_sha256': ids_hash(valid),
                'inner_OOF_sha256': inner_receipt['BL80_INNER_OOF_sha256'], 'training_rows': len(train_rows),
                'training_series': len(train), 'validation_series': len(valid), 'validation_labels_used': False,
                'H_only_state_calibration': True, 'research_split_sha256': guard.split.digest,
                'feature_cache_manifest_sha256': digest(cache / 'MANIFEST.json'), 'candidate': 'G2_FULL',
                'one_change': 'G1 M_FULL current inputs preserved; qualified memory inputs replace raw memory'}
            model = fit_correction('M_FULL', x, inner_base, y, weights, provenance,
                read_json(BASE / 'G1_CANDIDATE_RESERVATION.json')['I4_tree'])
            model.save(model_path)
            prediction = np.empty(len(valid_rows))
            for start in range(0, len(valid_rows), 50000):
                rows = valid_rows[start:start + 50000]
                prediction[start:start + len(rows)] = model.predict(baseline[rows], features[rows])
            assert np.isfinite(prediction).all()
            output = directory / 'G2_FULL/predictions.npz'
            with output.open('xb') as f:
                np.savez(f, global_row=valid_rows, dataset_id=metadata['dataset_id'][valid_rows], prediction=prediction)
            save(f'g2/fold_{fold}/G2_FULL/DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
                'candidate': 'G2_FULL', 'fold': fold, 'model_sha256': digest(model_path),
                'prediction_sha256': digest(output), 'outer_score_released': False, 'training': model.training})
            del x, y, weights, prediction, model
            gc.collect()
        with activity(f'G2 fixed event freeze fold{fold}', 'PREDICTION_FREEZE', 'F5_CAUSAL_NESTED_INTEGRATION'):
            fixed = directory / 'fixed_predictions.npz'
            with fixed.open('xb') as f:
                np.savez(f, global_row=valid_rows, dataset_id=metadata['dataset_id'][valid_rows],
                    M0=baseline[valid_rows], AMPLITUDE_0_9_M0=.9 * baseline[valid_rows],
                    G2_EVENT=.9 * baseline[valid_rows] + .1 * qualified[valid_rows, 0])
            hashes = {'fixed_predictions.npz': digest(fixed), 'G2_FULL/predictions.npz': digest(output)}
            receipt = {'status': 'FROZEN', 'created_utc': utc_now(), 'fold': fold,
                'methods': ['M0', 'AMPLITUDE_0_9_M0', 'G2_EVENT', 'G2_FULL'],
                'prediction_sha256': hashes, 'outer_score_released': False, 'baseline_authority': proof,
                'training_ids_sha256': ids_hash(train), 'validation_ids_sha256': ids_hash(valid),
                'source_sha256': sources}
            save(f'g2/fold_{fold}/OUTER_PREDICTIONS_FROZEN.json', receipt)
            receipts.append(receipt)
    if not (BASE / 'G2_ALL_OUTER_PREDICTIONS_FROZEN.json').exists():
        save('G2_ALL_OUTER_PREDICTIONS_FROZEN.json', {'status': 'FROZEN', 'created_utc': utc_now(),
            'folds': receipts, 'all_five_completed_before_G2_scores': True,
            'scientific_candidates': ['G2_EVENT', 'G2_FULL'], 'fitted_meta_candidates': ['G2_FULL'],
            'source_sha256': sources})
    guard.check_state()
    note('G2 five outer predictions frozen', 'One score release; no G2 threshold or recipe changes', 'F5_CAUSAL_NESTED_INTEGRATION')


if __name__ == '__main__':
    main()
