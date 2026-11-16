"""Prospective screen/full evaluation with fixed ResearchPool-only OOF blends."""
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
from src.next5.references import primary
from src.next5.saturation import assert_budget
from src.scoring.ts_auc import ts_auc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True)
    parser.add_argument('--scope',choices=('screen','full'),required=True)
    args=parser.parse_args();cid=args.candidate;scope=args.scope
    session=read_json(BASE/'SESSION.json')
    if session.get('modeling_blocked') or session.get('shadow_opened') or session.get('audit_started_utc'):
        raise RuntimeError('Research fit phase is blocked')
    if (BASE/'shadow/FINALIST_MANIFEST_NEXT5.json').exists():raise RuntimeError('No new scientific fitting after finalist freeze')
    if read_json(BASE/'preflight/NEXT5_PREFLIGHT.json')['status']!='PASS':raise RuntimeError('Preflight required')
    registration=read_json(BASE/'registrations'/(cid+'.json'));registration_sha=digest(BASE/'registrations'/(cid+'.json'))
    for name,sha in registration['execution_source_sha256'].items():
        if digest(ROOT/name)!=sha:raise RuntimeError('Candidate execution source changed before score')
    policy=read_json(ROOT/'configs/next5/RESEARCH_POLICY_LOCK.json');out=BASE/'experiments'/cid/scope
    if out.exists():raise RuntimeError('Preserve candidate attempt and all existing scores')
    if scope=='full':
        screened=read_json(BASE/'experiments'/cid/'screen/RESULTS.json')
        if not screened['screen_pass']:
            atomic_json(out/'SKIPPED.json',{'created_utc':utc_now(),'candidate':cid,'reason':'Frozen screen gate failed','new_fits':0},immutable=True)
            print('NEXT5 FULL CLOSED BY SCREEN',cid,flush=True);return
    budget=assert_budget();limit=35 if scope=='screen' else 20
    if budget[scope+'_count']>=limit:raise RuntimeError('Hard candidate budget exhausted before start')
    atomic_json(out/'START.json',{'created_utc':utc_now(),'candidate':cid,'family':registration['family'],'scope':scope,
        'registration_sha256':registration_sha,'created_before_score':True},immutable=True)
    data=prepare_research();matrix,metadata,cache=build_design(data,registration)
    baseline,dependence,reference_receipts=primary(scope,'PRIMARY',metadata,data.guard)
    split=read_json(ROOT/'NEXT5_SPLIT_LOCK.json');eligible=np.ones(len(metadata['target']),dtype=bool)
    if scope=='screen':eligible=np.isin(metadata['dataset_id'],split['screen_ids'])
    folds=[0,4] if scope=='screen' else list(range(5));results=[];all_rows=[];all_expert=[];all_prediction=[]
    for fold in folds:
        directory=out/('fold_'+str(fold));directory.mkdir()
        train=np.flatnonzero(eligible&(metadata['fold']!=fold));valid=np.flatnonzero(eligible&(metadata['fold']==fold))
        if not len(valid) or not np.isfinite(baseline[valid]).all() or not np.isfinite(dependence[valid]).all():
            raise RuntimeError('Matched baseline coverage incomplete')
        atomic_json(directory/'FIT_START.json',{'created_utc':utc_now(),'candidate':cid,'scope':scope,'fold':fold,
            'registration_sha256':registration_sha,'training_rows':len(train),'validation_rows':len(valid)},immutable=True)
        progress('ResearchPool candidate fit',cid+' '+scope+' fold '+str(fold));started=time.perf_counter()
        print('NEXT5 CANDIDATE FIT',cid,scope,fold,len(train),len(valid),flush=True)
        model,training=fit_registered(matrix,metadata,train,valid,guard=data.guard,registration=registration)
        model.save(directory/'model.joblib');expert=model.predict(matrix[valid]);blend=registration['combination']
        prediction=blend.get('ResearchPool_BL80',0.)*baseline[valid]+blend.get('C205',0.)*dependence[valid]+blend['new_model']*expert
        if not np.isfinite(prediction).all() or np.min(prediction)<0 or np.max(prediction)>1:raise RuntimeError('Invalid candidate probability')
        target=metadata['target'][valid];time_online=metadata['time_online'][valid]
        score=float(ts_auc(target,prediction,time_online));base_score=float(ts_auc(target,baseline[valid],time_online))
        standalone=float(ts_auc(target,expert,time_online))
        np.savez(directory/'predictions.npz',global_row=valid,dataset_id=metadata['dataset_id'][valid],prediction=prediction,expert=expert,baseline=baseline[valid])
        result={'status':'COMPLETE','created_utc':utc_now(),'candidate':cid,'family':registration['family'],'scope':scope,'fold':fold,
            'TS_AUC':score,'BL80_TS_AUC':base_score,'delta':score-base_score,'standalone_TS_AUC':standalone,
            'model_sha256':digest(directory/'model.joblib'),'predictions_file_sha256':digest(directory/'predictions.npz'),
            'prediction_sha256':hashlib.sha256(prediction.tobytes()).hexdigest(),'elapsed_seconds':time.perf_counter()-started,
            'registration_sha256':registration_sha,'cache_key':cache['key'],**training}
        atomic_json(directory/'RESULTS.json',result,immutable=True);results.append(result)
        all_rows.append(valid);all_expert.append(expert);all_prediction.append(prediction)
        print('NEXT5 CANDIDATE SCORE',cid,scope,fold,score,'DELTA',score-base_score,'STANDALONE',standalone,flush=True)
    rows=np.concatenate(all_rows);expert=np.concatenate(all_expert);prediction=np.concatenate(all_prediction);order=np.argsort(rows,kind='stable')
    np.savez(out/'OOF.npz',global_row=rows[order],dataset_id=metadata['dataset_id'][rows[order]],
        prediction=prediction[order],expert=expert[order],baseline=baseline[rows[order]])
    scores=np.asarray([r['TS_AUC'] for r in results]);base=np.asarray([r['BL80_TS_AUC'] for r in results]);deltas=scores-base
    standalone=float(np.mean([r['standalone_TS_AUC'] for r in results]));gate=policy['screen']
    screen_pass=bool(deltas.mean()>=gate['mean_delta_min'] and deltas.min()>=gate['each_fold_delta_min'] and
        (registration['family']!='M' or standalone>=gate['M_standalone_min'])) if scope=='screen' else None
    result={'status':'COMPLETE','created_utc':utc_now(),'candidate':cid,'family':registration['family'],'scope':scope,
        'mean':float(scores.mean()),'median':float(np.median(scores)),'worst':float(scores.min()),'mean_delta':float(deltas.mean()),
        'median_delta':float(np.median(scores)-np.median(base)),'worst_delta':float(scores.min()-base.min()),
        'minimum_fold_delta':float(deltas.min()),'positive_folds':int((deltas>0).sum()),'standalone_mean':standalone,
        'baseline_mean':float(base.mean()),'screen_pass':screen_pass,'fold_results':results,'new_fits':len(results),
        'shadow_rows':0,'seal_rows':0,'reference_receipts':reference_receipts,'registration_sha256':registration_sha,
        'OOF_sha256':digest(out/'OOF.npz'),'selection_status':'Screen only' if scope=='screen' else 'Await fixed bootstrap, ALT, H/mechanism, causality and runtime gates'}
    atomic_json(out/'RESULTS.json',result,immutable=True);data.guard.check_state();assert_budget()
    progress('ResearchPool candidate evaluation complete',cid+' '+scope+' mean delta '+str(result['mean_delta']))
    print('NEXT5 CANDIDATE COMPLETE',cid,scope,result['mean'],result['mean_delta'],screen_pass,flush=True)


if __name__=='__main__':main()
