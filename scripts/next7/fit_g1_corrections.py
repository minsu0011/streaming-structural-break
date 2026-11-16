"""Outer-blind correction fits and frozen G1 predictions for all five folds."""
from pathlib import Path
import gc
import os
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next7/numba_cache')
import numpy as np
from src.next7.common import BASE,save,read_json,digest,ids_hash,utc_now,activity,require_guard,note
from src.next7.model import Correction,fit_correction,implementation_hash
from src.next.fitting import weights_from_training
from src.next6.data import source
from src.next5.references import primary

METHODS=('I3_CURRENT_EVENT','M_FULL','I4_EVENT_SCALE_TREE')


def verify_inner(data,fold,train,valid):
    directory=BASE/f'nested_base/fold_{fold}'
    contract=read_json(directory/'CONTRACT.json')
    done=read_json(directory/'DONE.json')
    assert done['status']=='COMPLETE' and done['all_outer_validation_groups_excluded']
    assert set(contract['training_ids'])==train and set(contract['outer_validation_ids'])==valid
    assert digest(directory/'BL80_INNER_OOF.npz')==done['BL80_INNER_OOF_sha256']
    for inner in range(3):
        held={int(i) for i,k in contract['inner_assignment'].items() if k==inner}
        for kind in ('J102','C205'):
            d=directory/f'inner_{inner}/{kind}'
            r=read_json(d/'DONE.json')
            assert r['training_group_identity']==ids_hash(train-held)
            assert r['validation_group_identity']==ids_hash(held)
            assert r['outer_validation_identity']==ids_hash(valid)
            assert r['outer_validation_excluded'] and not r['validation_labels_used_in_fit']
            assert digest(d/'model.joblib')==r['model_sha256']
            assert digest(d/'predictions.npz')==r['predictions_sha256']
            data.guard.partition(train-held,held)
            data.guard.partition(train-held,valid)
    with np.load(directory/'BL80_INNER_OOF.npz',allow_pickle=False) as values:
        data.guard.admit(np.unique(values['dataset_id']),purpose='G1 nested correction OOF input')
        rows=values['global_row'].copy()
        pred=values['prediction'].copy()
        assert set(map(int,np.unique(values['dataset_id'])))==train
        assert np.isfinite(pred).all()
    return rows,pred,done


def main():
    guard=require_guard()
    for fold in range(5):
        assert read_json(BASE/f'nested_base/fold_{fold}/DONE.json')['status']=='COMPLETE'
    repair_file=BASE/'G1_CORRECTION_BUG_REPAIR_001.json'
    test_name='MODEL_TESTS_002.xml' if repair_file.exists() else 'MODEL_TESTS_001.xml'
    tests=ET.parse(BASE/'audit'/test_name)
    assert not tests.findall('.//failure') and not tests.findall('.//error')
    assert len(tests.findall('.//testcase'))==(6 if repair_file.exists() else 5)
    reservation=read_json(BASE/'G1_CANDIDATE_RESERVATION.json')
    cache=BASE/'cache/g1'
    cache_receipt=read_json(cache/'MANIFEST.json')
    for name,sha in cache_receipt['files_sha256'].items():
        assert digest(cache/name)==sha
    for name,sha in cache_receipt['source_sha256'].items():
        assert digest(ROOT/name)==sha
    sources={name:digest(ROOT/name) for name in ('scripts/next7/fit_g1_corrections.py','src/next7/model.py',
        'src/next7/features.py','src/next7/event.py','src/next7/scale.py')}
    lockfile=BASE/'G1_CORRECTION_IMPLEMENTATION_LOCK.json'
    if not lockfile.exists():
        save('G1_CORRECTION_IMPLEMENTATION_LOCK.json',{'status':'FROZEN','created_utc':utc_now(),
            'source_sha256':sources,'model_source_sha256':implementation_hash(),
            'test_xml_sha256':digest(BASE/'audit'/test_name),
            'protocol_sha256':digest(BASE/'NEXT7_PROTOCOL_LOCK.json'),
            'candidate_reservation_sha256':digest(BASE/'G1_CANDIDATE_RESERVATION.json'),
            'before_outer_score_release':True})
    else:
        original=read_json(lockfile)['source_sha256']
        if original!=sources:
            repair=read_json(repair_file)
            assert repair['old_source_sha256']==original
            assert repair['new_source_sha256']==sources
            assert repair['fitted_models_before_repair']==0
            assert repair['test_xml_sha256']==digest(BASE/'audit'/test_name)
    data=source()
    metadata=data.rows()
    features=np.load(cache/'head_features.npy',mmap_mode='r',allow_pickle=False)
    states=np.load(cache/'state.npy',mmap_mode='r',allow_pickle=False)
    baseline,_,proof=primary('full','PRIMARY',metadata,guard)
    rowfolds=np.array([guard.split.assignment(int(s))[1] for s in data.ids],dtype=np.int8)
    rowfolds=np.repeat(rowfolds,np.diff(data.series['online_offsets']))
    all_outer=[]
    for fold in range(5):
        directory=BASE/f'g1/fold_{fold}'
        final=directory/'OUTER_PREDICTIONS_FROZEN.json'
        if final.exists():
            receipt=read_json(final)
            for name,sha in receipt['prediction_sha256'].items():
                assert digest(directory/name)==sha
            all_outer.append(receipt)
            continue
        valid_rows=np.flatnonzero(rowfolds==fold)
        train=set(map(int,np.unique(metadata['dataset_id'][rowfolds!=fold])))
        valid=set(map(int,np.unique(metadata['dataset_id'][valid_rows])))
        guard.partition(train,valid)
        train_rows,inner_base,inner_receipt=verify_inner(data,fold,train,valid)
        assert set(map(int,np.unique(metadata['dataset_id'][train_rows])))==train
        assert not set(train_rows)&set(valid_rows)
        train_x=np.asarray(features[train_rows],dtype=np.float64)
        # The only target slice used here is outer TRAIN. No scorer is imported.
        y=metadata['target'][train_rows]
        weights=weights_from_training(y,metadata['time_online'][train_rows],metadata['dataset_id'][train_rows],'W3')
        provenance={'fold':fold,'strict_nested_OOF':True,'outer_validation_excluded':True,
            'training_ids_sha256':ids_hash(train),'validation_ids_sha256':ids_hash(valid),
            'inner_OOF_sha256':inner_receipt['BL80_INNER_OOF_sha256'],
            'training_rows':len(train_rows),'training_series':len(train),'validation_series':len(valid),
            'H_only_state_calibration':True,'validation_labels_used':False,
            'research_split_sha256':guard.split.digest,'feature_cache_manifest_sha256':digest(cache/'MANIFEST.json')}
        for method in METHODS:
            model_path=directory/method/'model.joblib'
            done_path=directory/method/'DONE.json'
            if done_path.exists():
                assert digest(model_path)==read_json(done_path)['model_sha256']
                continue
            with activity(f'G1 nested correction {method} fold{fold}','SCIENTIFIC_FIT','F5_CAUSAL_NESTED_INTEGRATION'):
                if model_path.exists():
                    raise RuntimeError('Partial fitted correction requires explicit repair receipt')
                model=fit_correction(method,train_x,inner_base,y,weights,provenance,reservation['I4_tree'])
                model.save(model_path)
                prediction=np.empty(len(valid_rows))
                for a in range(0,len(valid_rows),50000):
                    rows=valid_rows[a:a+50000]
                    prediction[a:a+len(rows)]=model.predict(baseline[rows],features[rows])
                assert np.isfinite(prediction).all()
                pred_path=directory/method/'predictions.npz'
                np.savez(pred_path,global_row=valid_rows,dataset_id=metadata['dataset_id'][valid_rows],prediction=prediction)
                save(str(done_path.relative_to(BASE)),{'status':'COMPLETE','created_utc':utc_now(),
                    'method':method,'fold':fold,'model_sha256':digest(model_path),
                    'prediction_sha256':digest(pred_path),'outer_score_released':False,'training':model.training})
                del model,prediction
                gc.collect()
        del train_x,y,weights
        gc.collect()
        with activity(f'G1 fixed ablation prediction freeze fold{fold}','PREDICTION_FREEZE','F5_CAUSAL_NESTED_INTEGRATION'):
            v=np.asarray(states[valid_rows])
            b=baseline[valid_rows]
            predictions={'M0':b,'AMPLITUDE_0.9_M0':.9*b,
                'M_CURRENT':.9*b+.1*v[:,1],'M_EVENT':.9*b+.1*v[:,3],
                'M_SCALE':.9*b+.1*v[:,10],'M_EVENT_SCALE':.9*b+.05*v[:,3]+.05*v[:,10],
                'M_FULL_FIXED':.9*b+(.1/3.)*(v[:,1]+v[:,3]+v[:,10]),
                'M_CORE_EVENT':.9*b+.1*v[:,26],
                'CURRENT_ONLY':v[:,1],'EVENT_ONLY':v[:,3],'SCALE_ONLY':v[:,10]}
            output=directory/'fixed_predictions.npz'
            assert not output.exists()
            np.savez(output,global_row=valid_rows,dataset_id=metadata['dataset_id'][valid_rows],**predictions)
            hashes={'fixed_predictions.npz':digest(output)}
            hashes.update({m+'/predictions.npz':digest(directory/m/'predictions.npz') for m in METHODS})
            receipt={'status':'FROZEN','created_utc':utc_now(),'fold':fold,'methods':list(predictions)+list(METHODS),
                'prediction_sha256':hashes,'outer_score_released':False,
                'base_is_original_matched_outer_fit':True,'baseline_authority':proof,
                'training_ids_sha256':ids_hash(train),'validation_ids_sha256':ids_hash(valid),
                'source_sha256':sources}
            save(str(final.relative_to(BASE)),receipt)
            all_outer.append(receipt)
    if not (BASE/'G1_ALL_OUTER_PREDICTIONS_FROZEN.json').exists():
        save('G1_ALL_OUTER_PREDICTIONS_FROZEN.json',{'status':'FROZEN','created_utc':utc_now(),
            'folds':all_outer,'all_five_folds_completed_before_outer_score_release':True,
            'scientific_candidates':reservation['full_scientific_candidates'],
            'fitted_meta_candidates':list(METHODS),'source_sha256':sources})
    guard.check_state()
    note('G1 all five outer predictions frozen','Ready for one score release; no threshold or weight changes permitted','F5_CAUSAL_NESTED_INTEGRATION')


if __name__=='__main__':
    main()
