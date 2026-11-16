"""Strictly nested BL80 OOF components, with original frozen learner recipes.

This creates new inner reference fits only under NEXT7. Existing canonical and
outer reference models are immutable. No outer score is computed here.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import os
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next7/numba_cache')
import numpy as np
from threadpoolctl import threadpool_limits
from src.next7.common import BASE, save, read_json, digest, ids_hash, utc_now, activity, require_guard
from src.next6.data import source, reference_matrix
from src.next5.model import FeatureModel, feature_module, source_hash
from src.models.learners import make_estimator
from src.models.compact_trees import export_lightgbm
from src.next.fitting import weights_from_training


def rows_for(data, ids):
    return np.concatenate([np.arange(*data.series['online_offsets'][data.id_to_index[int(s)]:data.id_to_index[int(s)]+2],
                                    dtype=np.int64) for s in sorted(ids)])


def fit_one(x, metadata, training_rows, validation_rows, guard, registration, outer_valid_ids):
    train_ids=np.unique(metadata['dataset_id'][training_rows])
    valid_ids=np.unique(metadata['dataset_id'][validation_rows])
    guard.partition(train_ids,valid_ids)
    guard.partition(train_ids,outer_valid_ids)
    assert not set(valid_ids)&set(outer_valid_ids)
    module=feature_module(registration['feature_module'])
    settings=registration['feature_settings']
    assert module.implementation_hash(settings)==registration['feature_sha256']
    assert source_hash()==registration['model_source_sha256']
    names=tuple(module.feature_names(settings))
    matrix=np.asarray(x[training_rows],dtype=np.float32)
    assert matrix.shape[1]==len(names) and np.isfinite(matrix).all()
    columns=np.flatnonzero(np.ptp(matrix,axis=0)>1e-12).astype(np.int32)
    matrix=matrix[:,columns]
    y=np.asarray(metadata['target'][training_rows],dtype=np.uint8)
    weights=weights_from_training(y,metadata['time_online'][training_rows],metadata['dataset_id'][training_rows],'W3')
    params=registration['parameters']
    assert params['learner']=='M4_lightgbm' and params['weighting']=='W3'
    estimator=make_estimator('M4_lightgbm',params['learner_params'])
    with threadpool_limits(limits=4):
        estimator.fit(matrix,y,sample_weight=weights)
    compact=export_lightgbm(estimator)
    with threadpool_limits(limits=1):
        native=estimator.booster_.predict(matrix[:2048],num_threads=1)
    error=float(np.max(np.abs(compact.predict(matrix[:2048])-native)))
    assert error<1e-12
    training={'training_rows':len(training_rows),'training_series':len(train_ids),
        'validation_rows':len(validation_rows),'validation_series':len(valid_ids),
        'training_ids_sha256':hashlib.sha256(np.sort(train_ids).tobytes()).hexdigest(),
        'training_group_identity':ids_hash(train_ids),'validation_group_identity':ids_hash(valid_ids),
        'outer_validation_identity':ids_hash(outer_valid_ids),'outer_validation_excluded':True,
        'training_only_columns_weights':True,'validation_labels_used_in_fit':False,
        'research_split_sha256':guard.split.digest,'shadow_lock_sha256':guard.shadow_lock_sha,
        'native_compact_max_abs_difference':error,'original_fixed_recipe':True,
        'NEXT5_session_mutated':False,'new_inner_reference_fit_not_scientific_candidate':True}
    model=FeatureModel(registration['feature_module'],settings,names,columns,compact,estimator,
        registration['feature_sha256'],source_hash(),training)
    return model,training


def main(fold):
    guard=require_guard()
    assert read_json(BASE/'runtime/CORE_TEST_GATE.json')['status']=='PASS'
    protocol=read_json(BASE/'NEXT7_PROTOCOL_LOCK.json')
    data=source()
    metadata=data.rows()
    assignments=protocol['inner_mapping'][str(fold)]
    sets=[{int(i) for i,j in assignments.items() if j==k} for k in range(3)]
    train=set.union(*sets)
    valid=set(data.ids)-train
    assert len(train)==5120 and len(valid)==1280
    for sid in valid:
        assert data.guard.split.assignment(int(sid))[1]==fold
    guard.partition(train,valid)
    execution_sources={n:digest(ROOT/n) for n in ('scripts/next7/fit_nested_baseline.py','src/next5/model.py',
        'src/next5/reference_features.py','src/models/learners.py','src/models/compact_trees.py','src/next/fitting.py')}
    root=BASE/f'nested_base/fold_{fold}'
    contract={'status':'FROZEN','created_utc':utc_now(),'fold':fold,'training_ids':sorted(map(int,train)),
        'outer_validation_ids':sorted(map(int,valid)),'inner_assignment':assignments,
        'source_sha256':execution_sources,'protocol_sha256':digest(BASE/'NEXT7_PROTOCOL_LOCK.json'),
        'no_outer_score_release':True}
    if not (root/'CONTRACT.json').exists():
        save(f'nested_base/fold_{fold}/CONTRACT.json',contract)
    else:
        previous=read_json(root/'CONTRACT.json')
        assert previous['source_sha256']==execution_sources and previous['inner_assignment']==assignments
    all_receipts=[]
    for kind in ('J102','C205'):
        with activity(f'load guarded {kind} reference design fold{fold}','REFERENCE_PREPARATION','F5_CAUSAL_NESTED_INTEGRATION'):
            x=reference_matrix(kind,data)
        registration=read_json(ROOT/f'artifacts/next5/references/registrations/REF_{kind}.json')
        for inner in range(3):
            directory=root/f'inner_{inner}/{kind}'
            if (directory/'DONE.json').exists():
                receipt=read_json(directory/'DONE.json')
                assert digest(directory/'model.joblib')==receipt['model_sha256']
                assert digest(directory/'predictions.npz')==receipt['predictions_sha256']
                all_receipts.append(receipt)
                continue
            tr=rows_for(data,train-sets[inner])
            va=rows_for(data,sets[inner])
            with activity(f'nested BL80 fold{fold} inner{inner} {kind}','REFERENCE_FIT','F5_CAUSAL_NESTED_INTEGRATION'):
                if directory.exists() and any(directory.iterdir()):
                    raise RuntimeError('Partial reference fit requires explicit repair directory; never overwrite')
                directory.mkdir(parents=True,exist_ok=True)
                save(str((directory/'START.json').relative_to(BASE)),{
                    'created_utc':utc_now(),'fold':fold,'inner':inner,'kind':kind,
                    'training_ids_sha256':ids_hash(train-sets[inner]),'validation_ids_sha256':ids_hash(sets[inner]),
                    'outer_validation_excluded':True,'registration_sha256':digest(ROOT/f'artifacts/next5/references/registrations/REF_{kind}.json')})
                model,training=fit_one(x,metadata,tr,va,guard,registration,valid)
                model.save(directory/'model.joblib')
                predictions=np.empty(len(va))
                for start in range(0,len(va),50000):
                    predictions[start:start+50000]=model.predict(x[va[start:start+50000]])
                assert np.isfinite(predictions).all()
                np.savez(directory/'predictions.npz',global_row=va,dataset_id=metadata['dataset_id'][va],prediction=predictions)
                receipt={'status':'COMPLETE','created_utc':utc_now(),'fold':fold,'inner':inner,'kind':kind,
                    'model_sha256':digest(directory/'model.joblib'),'predictions_sha256':digest(directory/'predictions.npz'),
                    'outer_score_computed':False,**training}
                save(str((directory/'DONE.json').relative_to(BASE)),receipt)
                all_receipts.append(receipt)
                del model,predictions
                gc.collect()
        del x
        gc.collect()
    trainrows=rows_for(data,train)
    joined=np.full(len(trainrows),np.nan)
    for inner in range(3):
        parts=[]
        indices=None
        for kind in ('J102','C205'):
            path=root/f'inner_{inner}/{kind}/predictions.npz'
            with np.load(path,allow_pickle=False) as values:
                guard.admit(np.unique(values['dataset_id']),purpose='NEXT7 inner OOF admission')
                rows=values['global_row']
                np.testing.assert_array_equal(metadata['dataset_id'][rows],values['dataset_id'])
                if indices is not None:
                    np.testing.assert_array_equal(indices,rows)
                indices=rows.copy()
                parts.append(values['prediction'].copy())
        positions=np.searchsorted(trainrows,indices)
        np.testing.assert_array_equal(trainrows[positions],indices)
        assert np.isnan(joined[positions]).all()
        joined[positions]=.8*parts[0]+.2*parts[1]
    assert np.isfinite(joined).all()
    joined_path=root/'BL80_INNER_OOF.npz'
    if not joined_path.exists():
        np.savez(joined_path,global_row=trainrows,dataset_id=metadata['dataset_id'][trainrows],prediction=joined)
    if not (root/'DONE.json').exists():
        save(f'nested_base/fold_{fold}/DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'fold':fold,
            'BL80_INNER_OOF_sha256':digest(joined_path),'inner_fits':len(all_receipts),
            'all_meta_training_rows_predicted_outside_their_training_groups':True,
            'all_outer_validation_groups_excluded':True,'outer_labels_used':False,
            'outer_scores_released':False,'source_sha256':execution_sources})
    guard.check_state()
    print(utc_now(),'NESTED BASE FROZEN',fold,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',type=int,choices=range(5),required=True)
    main(parser.parse_args().fold)
