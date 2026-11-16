"""Independently verify nested prediction membership, joins and frozen head lineage."""
from pathlib import Path
import datetime as dt
import os
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next7.worker_environment import environment
os.environ.update(environment())
import numpy as np
from src.next7.common import BASE,read_json,save,digest,ids_hash,utc_now,activity,note,require_guard


def timestamp(record):return dt.datetime.fromisoformat(record['created_utc'])


def unchanged_sources(record):
    for name,sha in record.get('source_sha256',{}).items():assert digest(ROOT/name)==sha,name


def main():
    from scripts.next7.benchmark_runtime import audit_gate
    audit_gate()
    guard=require_guard()
    from src.next6.data import source
    data=source()
    metadata=data.rows()
    ids=np.asarray(metadata['dataset_id'])
    protocol=read_json(BASE/'NEXT7_PROTOCOL_LOCK.json')
    all_ids=set(map(int,data.ids))
    assert len(all_ids)==6400 and len({guard.split.groups[i] for i in all_ids})==6400
    folds,receipts=[],{}
    note('Final nested training lineage audit','Original30 inner references,25 heads and all frozen outer predictions; no refit','FINAL_AUDIT',phase='FINAL_AUDIT')
    with activity('Final nested row membership and exact BL80 OOF join audit','FINAL_AUDIT','FINAL_AUDIT'):
        for fold in range(5):
            root=BASE/f'nested_base/fold_{fold}'
            contract=read_json(root/'CONTRACT.json')
            done=read_json(root/'DONE.json')
            unchanged_sources(contract);unchanged_sources(done)
            assert contract['protocol_sha256']==digest(BASE/'NEXT7_PROTOCOL_LOCK.json')
            assignments={int(k):int(v) for k,v in contract['inner_assignment'].items()}
            assert assignments=={int(k):int(v) for k,v in protocol['inner_mapping'][str(fold)].items()}
            outer_valid={sid for sid in all_ids if guard.split.assignment(sid)[1]==fold}
            train=all_ids-outer_valid
            assert set(contract['training_ids'])==train and set(contract['outer_validation_ids'])==outer_valid
            guard.partition(train,outer_valid)
            train_rows=np.flatnonzero(np.isin(ids,list(train)))
            joined=np.full(len(train_rows),np.nan)
            model_count=0
            for inner in range(3):
                held={sid for sid,k in assignments.items() if k==inner}
                fitted=train-held
                assert not ({guard.split.groups[sid] for sid in fitted}&{guard.split.groups[sid] for sid in held|outer_valid})
                expected_rows=np.flatnonzero(np.isin(ids,list(held)))
                predictions=[]
                for kind in ('J102','C205'):
                    directory=root/f'inner_{inner}/{kind}'
                    start=read_json(directory/'START.json');finished=read_json(directory/'DONE.json')
                    assert finished['status']=='COMPLETE' and not finished['validation_labels_used_in_fit']
                    assert finished['training_group_identity']==start['training_ids_sha256']==ids_hash(fitted)
                    assert finished['validation_group_identity']==start['validation_ids_sha256']==ids_hash(held)
                    assert finished['outer_validation_identity']==ids_hash(outer_valid)
                    assert finished['research_split_sha256']==guard.split.digest
                    assert digest(directory/'model.joblib')==finished['model_sha256']
                    assert digest(directory/'predictions.npz')==finished['predictions_sha256']
                    assert timestamp(contract)<=timestamp(start)<=timestamp(finished)<=timestamp(done)
                    with np.load(directory/'predictions.npz',allow_pickle=False) as values:
                        np.testing.assert_array_equal(values['global_row'],expected_rows)
                        np.testing.assert_array_equal(values['dataset_id'],ids[expected_rows])
                        prediction=values['prediction'].copy()
                        assert prediction.shape==expected_rows.shape and np.isfinite(prediction).all()
                        predictions.append(prediction)
                    for p in (directory/'START.json',directory/'DONE.json'):receipts[p.relative_to(BASE).as_posix()]=digest(p)
                    model_count+=1
                positions=np.searchsorted(train_rows,expected_rows)
                assert np.isnan(joined[positions]).all()
                joined[positions]=.8*predictions[0]+.2*predictions[1]
            assert model_count==6 and np.isfinite(joined).all()
            joined_path=root/'BL80_INNER_OOF.npz'
            assert digest(joined_path)==done['BL80_INNER_OOF_sha256']
            with np.load(joined_path,allow_pickle=False) as values:
                np.testing.assert_array_equal(values['global_row'],train_rows)
                np.testing.assert_array_equal(values['dataset_id'],ids[train_rows])
                np.testing.assert_array_equal(values['prediction'],joined)
            heads=[]
            for group,methods in (('g1',('I3_CURRENT_EVENT','M_FULL','I4_EVENT_SCALE_TREE')),
                                  ('g2',('G2_FULL',)),('g3',('G3_PAIRWISE_FULL',))):
                generation=group.upper()
                outer_path=BASE/f'{group}/fold_{fold}/OUTER_PREDICTIONS_FROZEN.json'
                outer=read_json(outer_path)
                all_outer_path=BASE/f'{generation}_ALL_OUTER_PREDICTIONS_FROZEN.json'
                all_outer=read_json(all_outer_path)
                release=read_json(BASE/f'{generation}_OUTER_SCORE_RELEASE.json')
                unchanged_sources(outer);unchanged_sources(all_outer)
                assert outer==next(r for r in all_outer['folds'] if r['fold']==fold)
                assert outer['training_ids_sha256']==ids_hash(train) and outer['validation_ids_sha256']==ids_hash(outer_valid)
                assert not outer['outer_score_released']
                assert timestamp(outer)<=timestamp(all_outer)<timestamp(release)
                assert release['prediction_freeze_sha256']==digest(all_outer_path)
                pmap=outer['prediction_sha256']
                if isinstance(pmap,str):pmap={'predictions.npz':pmap}
                for name,sha in pmap.items():assert digest(outer_path.parent/name)==sha
                for method in methods:
                    directory=outer_path.parent if group=='g3' else outer_path.parent/method
                    fit_done=read_json(directory/'DONE.json')
                    training=fit_done['training']
                    assert fit_done['status']=='COMPLETE'
                    assert training['strict_nested_OOF'] and training['outer_validation_excluded']
                    assert not training['validation_labels_used']
                    assert training['training_ids_sha256']==ids_hash(train)
                    assert training['validation_ids_sha256']==ids_hash(outer_valid)
                    assert training['inner_OOF_sha256']==done['BL80_INNER_OOF_sha256']
                    assert training['training_rows']==len(train_rows)
                    assert digest(directory/'model.joblib')==fit_done['model_sha256']
                    assert digest(directory/'predictions.npz')==fit_done['prediction_sha256']
                    assert timestamp(done)<timestamp(fit_done)<=timestamp(outer)
                    receipts[(directory/'DONE.json').relative_to(BASE).as_posix()]=digest(directory/'DONE.json')
                    heads.append(method)
                receipts[outer_path.relative_to(BASE).as_posix()]=digest(outer_path)
            folds.append({'fold':fold,'outer_train_series':len(train),'outer_validation_series':len(outer_valid),
                'inner_reference_fits':model_count,'head_recipes':heads,'inner_OOF_rows_checked':len(train_rows),
                'every_inner_row_from_excluded_group':True,'all_training_rows_covered_once':True,
                'reconstructed_BL80_inner_OOF_bitwise_equal':True,
                'all_head_and_outer_prediction_bytes_unchanged':True,'all_five_outer_folds_frozen_before_score_release':True})
        guard.check_state()
        save('audit/FINAL_TRAINING_LINEAGE.json',{'status':'PASS','created_utc':utc_now(),
            'source_sha256':digest(__file__),'folds':folds,'inner_reference_models':30,'fitted_heads':25,
            'new_fits_in_audit':0,'new_score_release':False,'original_RP_historical_groups':6400,
            'series_bootstrap_matches_group_bootstrap_for_this_RP':True,
            'receipts_sha256':receipts,'scope':'Independent membership/row-index/mixture arithmetic checks, frozen source and artifact hashes, and recorded training chronology. This audits the saved execution evidence; it is not a new fit or an OS-level information-flow proof.'})
    note('Final nested training lineage passed','All30 inner reference and25 head artifacts; exact original OOF reconstruction','FINAL_AUDIT',phase='FINAL_AUDIT')


if __name__=='__main__':main()
