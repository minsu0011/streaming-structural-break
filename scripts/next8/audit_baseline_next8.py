"""Reuse existing fitted baseline evidence only after exact membership audit."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
import datetime as dt
import hashlib
import numpy as np
from src.next8.common import BASE, read_json, save, digest, ids_hash, utc_now, activity, note
from src.next8.data import source, outer_scope, inner_scope, rows_for
from src.next5.data import admitted_npz
from src.next5.model import FeatureModel


def main():
    with activity('Verify all inherited outer and inner BL80 training exclusions',
                  'REFERENCE_PROVENANCE', 'B9_NESTED_SELECTION'):
        data = source()
        metadata = data.rows()
        ids = metadata['dataset_id']
        old = ROOT/'artifacts/next7'
        protocol = read_json(old/'NEXT7_PROTOCOL_LOCK.json')
        files, folds = {}, []

        def bind(path):
            files[path.relative_to(ROOT).as_posix()] = digest(path)

        for fold in range(5):
            outer = outer_scope(fold, data)
            train_rows = rows_for(metadata, outer.training_ids)
            valid_rows = rows_for(metadata, outer.validation_ids)
            directory = old/f'nested_base/fold_{fold}'
            contract = read_json(directory/'CONTRACT.json')
            done = read_json(directory/'DONE.json')
            assert contract['protocol_sha256'] == digest(old/'NEXT7_PROTOCOL_LOCK.json')
            assert set(contract['training_ids']) == outer.training_ids
            assert set(contract['outer_validation_ids']) == outer.validation_ids
            assert contract['inner_assignment'] == protocol['inner_mapping'][str(fold)]
            for record in (contract, done):
                for name, sha in record['source_sha256'].items():
                    assert digest(ROOT/name) == sha, name
            joined = np.full(len(train_rows), np.nan)
            for inner in range(3):
                scope = inner_scope(fold, inner, data)
                expected = rows_for(metadata, scope.validation_ids)
                pieces = []
                for kind in ('J102', 'C205'):
                    p = directory/f'inner_{inner}/{kind}'
                    start, finish = read_json(p/'START.json'), read_json(p/'DONE.json')
                    assert finish['status'] == 'COMPLETE'
                    assert not finish['validation_labels_used_in_fit']
                    assert finish['training_group_identity'] == start['training_ids_sha256'] == ids_hash(scope.training_ids)
                    assert finish['validation_group_identity'] == start['validation_ids_sha256'] == ids_hash(scope.validation_ids)
                    assert finish['outer_validation_identity'] == ids_hash(outer.validation_ids)
                    assert finish['research_split_sha256'] == data.guard.split.digest
                    assert digest(p/'model.joblib') == finish['model_sha256']
                    assert digest(p/'predictions.npz') == finish['predictions_sha256']
                    assert dt.datetime.fromisoformat(contract['created_utc']) <= dt.datetime.fromisoformat(start['created_utc']) <= dt.datetime.fromisoformat(finish['created_utc']) <= dt.datetime.fromisoformat(done['created_utc'])
                    a = admitted_npz(p/'predictions.npz', data.guard, 'NEXT8 inner reference provenance')
                    np.testing.assert_array_equal(a['global_row'], expected)
                    np.testing.assert_array_equal(a['dataset_id'], ids[expected])
                    assert np.isfinite(a['prediction']).all()
                    pieces.append(a['prediction'])
                    for name in ('START.json', 'DONE.json', 'model.joblib', 'predictions.npz'):
                        bind(p/name)
                positions = np.searchsorted(train_rows, expected)
                assert np.isnan(joined[positions]).all()
                joined[positions] = .8*pieces[0]+.2*pieces[1]
            path = directory/'BL80_INNER_OOF.npz'
            assert digest(path) == done['BL80_INNER_OOF_sha256']
            a = admitted_npz(path, data.guard, 'NEXT8 exact complete nested OOF join')
            np.testing.assert_array_equal(a['global_row'], train_rows)
            np.testing.assert_array_equal(a['dataset_id'], ids[train_rows])
            np.testing.assert_array_equal(a['prediction'], joined)
            bind(path)
            for kind in ('J102', 'C205', 'D2_expert'):
                p = ROOT/f'artifacts/next5/references/fits/PRIMARY/full/{kind}/fold_{fold}'
                finish = read_json(p/'RESULTS.json')
                assert finish['status'] == 'COMPLETE' and not finish['validation_labels_used_in_fit']
                assert finish['research_split_sha256'] == data.guard.split.digest
                expected_hash = hashlib.sha256(np.asarray(sorted(outer.training_ids), dtype=data.ids.dtype).tobytes()).hexdigest()
                assert finish['training_ids_sha256'] == expected_hash
                assert finish['training_rows'] == len(train_rows) and finish['validation_rows'] == len(valid_rows)
                assert digest(p/'model.joblib') == finish['model_sha256']
                assert digest(p/'predictions.npz') == finish['prediction_file_sha256']
                model = FeatureModel.load(p/'model.joblib')
                assert model.training['training_ids_sha256'] == expected_hash
                assert model.training['research_split_sha256'] == data.guard.split.digest
                a = admitted_npz(p/'predictions.npz', data.guard, 'NEXT8 outer reference exclusion audit')
                np.testing.assert_array_equal(a['global_row'], valid_rows)
                np.testing.assert_array_equal(a['dataset_id'], ids[valid_rows])
                assert np.isfinite(a['prediction']).all()
                for name in ('FIT_START.json', 'RESULTS.json', 'model.joblib', 'predictions.npz'):
                    bind(p/name)
            folds.append({**outer.provenance(), 'inner_reference_models': 6,
                          'outer_reference_models': 3, 'exact_join': 'PASS'})
            print(utc_now(), 'BASELINE PROVENANCE', fold, flush=True)
        data.guard.check_state()
        save('BASELINE_PROVENANCE_NEXT8.json', {
            'status': 'PASS', 'timestamp': utc_now(), 'folds': folds,
            'files_sha256': files, 'new_fits': 0, 'new_scores': 0,
            'inner_models': 30, 'outer_models': 15,
            'full_DEV_deployment_used_for_outer_evaluation': False,
            'limitations': 'Recorded training membership, source identities, model metadata, saved row indices and arithmetic independently checked; not an OS information-flow proof.',
            'source_sha256': digest(__file__)})
    note('Baseline reuse gate PASS', '30 inner and15 outer reference fits validated without refitting', 'B9_NESTED_SELECTION')


if __name__ == '__main__':
    main()
