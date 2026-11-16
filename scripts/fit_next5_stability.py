"""Refit unchanged candidates for diagnostic ALT and historical H exclusion."""
from pathlib import Path
import argparse
import hashlib
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.next5.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next5.data import prepare_research
from src.next5.cache import build_design
from src.next5.model import fit_registered
from src.next5.stability import assignments,heldout_rows,reference_root,load_reference,training_hash
from src.scoring.ts_auc import ts_auc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate');parser.add_argument('--reference',choices=('J102','C205'))
    parser.add_argument('--split',choices=('ALT_A','ALT_B','H_EXCLUSION'),required=True);args=parser.parse_args()
    if bool(args.candidate)==bool(args.reference):raise ValueError('Exactly one diagnostic candidate or reference required')
    policy=read_json(BASE/'stability/POLICY_LOCK.json')
    for name,sha in policy['execution_sources_sha256'].items():assert digest(ROOT/name)==sha
    session=read_json(BASE/'SESSION.json');assert not session.get('shadow_opened') and not session.get('audit_started_utc')
    if (BASE/'shadow/FINALIST_MANIFEST_NEXT5.json').exists():raise RuntimeError('No new diagnostic fit after finalist freeze')
    if args.reference:
        if args.split!='H_EXCLUSION':raise ValueError('ALT references use the already-frozen reference driver')
        cid='REF_'+args.reference;registration_path=BASE/'references/registrations'/(cid+'.json');out=reference_root(args.reference,args.split)
    else:
        cid=args.candidate;assert cid in policy['diagnostic_nominees'];registration_path=BASE/'registrations'/(cid+'.json')
        assert digest(registration_path)==policy['diagnostic_nominees'][cid]['registration_sha256']
        out=BASE/'stability/candidates'/cid/args.split
    registration=read_json(registration_path)
    for name,sha in registration['execution_source_sha256'].items():assert digest(ROOT/name)==sha
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'candidate':args.candidate,'reference':args.reference,'split':args.split,
        'registration_sha256':digest(registration_path),'stability_policy_sha256':digest(BASE/'stability/POLICY_LOCK.json'),
        'diagnostic_refit_not_new_candidate_or_primary_rescue':True},immutable=True)
    data=prepare_research();matrix,metadata,cache=build_design(data,registration);mapping=assignments(args.split,data.guard)
    if args.candidate:
        jp,jr=load_reference('J102',args.split,metadata,data.guard);cp,cr=load_reference('C205',args.split,metadata,data.guard)
        baseline=.8*jp+.2*cp;reference_receipts=[jr,cr]
    rows_all=[];predictions_all=[];experts_all=[];results=[]
    for fold in range(5):
        train,valid=heldout_rows(metadata['dataset_id'],mapping,fold);data.guard.partition(np.unique(metadata['dataset_id'][train]),np.unique(metadata['dataset_id'][valid]))
        directory=out/('fold_'+str(fold));directory.mkdir()
        expected_training=training_hash(metadata,train)
        if args.candidate:
            for kind in ('J102','C205'):
                ref=read_json(reference_root(kind,args.split)/('fold_'+str(fold))/'RESULTS.json')
                assert ref['training_ids_sha256']==expected_training
        atomic_json(directory/'FIT_START.json',{'created_utc':utc_now(),'candidate':args.candidate,'reference':args.reference,
            'split':args.split,'fold':fold,'registration_sha256':digest(registration_path),'training_rows':len(train),'validation_rows':len(valid)},immutable=True)
        progress('Fixed held-out stability refit',cid+' '+args.split+' fold '+str(fold));started=time.perf_counter()
        print('NEXT5 STABILITY FIT',cid,args.split,fold,len(train),len(valid),flush=True)
        model,training=fit_registered(matrix,metadata,train,valid,guard=data.guard,registration=registration)
        assert training['training_ids_sha256']==expected_training
        model.save(directory/'model.joblib');expert=model.predict(matrix[valid]);prediction=expert
        if args.candidate:
            weights=registration['combination'];prediction=weights.get('ResearchPool_BL80',0.)*baseline[valid]+weights.get('C205',0.)*cp[valid]+weights['new_model']*expert
        assert np.isfinite(prediction).all()
        values={'global_row':valid,'dataset_id':metadata['dataset_id'][valid],'prediction':prediction}
        score=float(ts_auc(metadata['target'][valid],prediction,metadata['time_online'][valid]))
        result={'status':'COMPLETE','created_utc':utc_now(),'candidate':args.candidate,'reference':args.reference,'split':args.split,'fold':fold,
            'TS_AUC':score,'model_sha256':digest(directory/'model.joblib'),'registration_sha256':digest(registration_path),
            'prediction_sha256':hashlib.sha256(prediction.tobytes()).hexdigest(),'elapsed_seconds':time.perf_counter()-started,
            'cache_key':cache['key'],**training}
        if args.candidate:
            values.update(expert=expert,baseline=baseline[valid]);base_score=float(ts_auc(metadata['target'][valid],baseline[valid],metadata['time_online'][valid]))
            result.update(BL80_TS_AUC=base_score,delta=score-base_score,standalone_TS_AUC=float(ts_auc(metadata['target'][valid],expert,metadata['time_online'][valid])))
        np.savez(directory/'predictions.npz',**values);result['prediction_file_sha256']=digest(directory/'predictions.npz')
        atomic_json(directory/'RESULTS.json',result,immutable=True);results.append(result);rows_all.append(valid);predictions_all.append(prediction);experts_all.append(expert)
        print('NEXT5 STABILITY SCORE',cid,args.split,fold,score,result.get('delta'),flush=True)
    rows=np.concatenate(rows_all);order=np.argsort(rows,kind='stable');predictions=np.concatenate(predictions_all)
    values={'global_row':rows[order],'dataset_id':metadata['dataset_id'][rows[order]],'prediction':predictions[order]}
    if args.candidate:values.update(expert=np.concatenate(experts_all)[order],baseline=baseline[rows[order]])
    np.savez(out/'OOF.npz',**values);scores=np.asarray([r['TS_AUC'] for r in results])
    result={'status':'COMPLETE','created_utc':utc_now(),'candidate':args.candidate,'reference':args.reference,'split':args.split,'fold_results':results,
        'mean_TS_AUC':float(scores.mean()),'median_TS_AUC':float(np.median(scores)),'worst_TS_AUC':float(scores.min()),
        'new_fits':5,'shadow_rows':0,'seal_rows':0,'OOF_sha256':digest(out/'OOF.npz'),'registration_sha256':digest(registration_path),
        'selection_authority':'Secondary held-out diagnostic; cannot rescue a failed PRIMARY eligibility gate.'}
    if args.candidate:
        base=np.asarray([r['BL80_TS_AUC'] for r in results]);result.update(mean_delta=float(np.mean(scores-base)),minimum_delta=float(np.min(scores-base)),
            positive_folds=int(np.sum(scores>base)),reference_receipts=reference_receipts)
    atomic_json(out/'RESULTS.json',result,immutable=True);data.guard.check_state()
    progress('Fixed held-out stability refit complete',cid+' '+args.split+'; exact same recipe and matched training membership')


if __name__=='__main__':main()
