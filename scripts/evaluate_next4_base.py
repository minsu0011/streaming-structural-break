"""Registered causal expert, locked screen/full CV and one fixed BL80 combination."""
from pathlib import Path
import sys
import argparse
import hashlib
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,digest,read_json,utc_now,progress
from src.next4.cache import build_design
from src.next4.model import fit_base,FeatureModel,source_hash
from src.next4.frozen import fold_predictions
from src.next.data import prepare_dev
from src.next2.blend import combine
from src.scoring.ts_auc import ts_auc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True)
    parser.add_argument('--scope',choices=['screen','full'],required=True);args=parser.parse_args()
    hyp=BASE/'registrations'/(args.candidate+'.json');registration=read_json(hyp)
    policy_path=ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json';policy=read_json(policy_path)
    assert registration['created_before_score'] is True and registration['policy_sha256']==digest(policy_path)
    assert registration['model_source_sha256']==source_hash()
    for path,sha in registration['execution_source_sha256'].items():
        if digest(ROOT/path)!=sha:raise RuntimeError('Registered execution source changed: '+path)
    family=read_json(BASE/'families'/(registration['family']+'_MENU_LOCK.json'))
    assert family['candidate_ids'].count(args.candidate)==1 and digest(BASE/'families'/(registration['family']+'_MENU_LOCK.json'))==registration['family_menu_sha256']
    parent=BASE/'experiments'/args.candidate;out=parent/args.scope
    if args.scope=='full':
        screen=read_json(parent/'screen/RESULTS.json')
        if not screen['screen_pass']:raise RuntimeError('Failed screen cannot reach full CV')
        existing=list((BASE/'experiments').glob('*/full/START.json'))
        if len(existing)>=read_json(ROOT/'configs/next4/CANDIDATE_BUDGET.json')['new_full_candidates_max']:
            raise RuntimeError('Full candidate budget exhausted')
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'candidate':args.candidate,'family':registration['family'],
        'scope':args.scope,'registration_sha256':digest(hyp),'policy_sha256':digest(policy_path),
        'runner_sha256':digest(Path(__file__)),'new_fits_at_start':0},immutable=True)
    data=prepare_dev(ROOT);x,metadata,names,global_rows,cache=build_design(data,registration,args.scope)
    folds=policy['screen']['folds'] if args.scope=='screen' else list(range(5));records=[]
    for fold in folds:
        train=np.flatnonzero(metadata['fold']!=fold);valid=np.flatnonzero(metadata['fold']==fold)
        baseline,baseline_sources=fold_predictions(data,fold,args.scope,global_rows[valid])
        atomic_json(out/f'fold_{fold}_FIT_START.json',{'created_utc':utc_now(),'intended_new_fits':1,
            'training_rows_sha256':hashlib.sha256(global_rows[train].tobytes()).hexdigest(),
            'validation_rows_sha256':hashlib.sha256(global_rows[valid].tobytes()).hexdigest(),
            'registration_sha256':digest(hyp)},immutable=True)
        progress('NEXT4 registered base fit',args.candidate+' '+args.scope+' fold '+str(fold))
        started=time.perf_counter();model,expert,detail=fit_base(x,metadata,train,valid,guard=data.guard,registration=registration)
        model_path=out/f'fold_{fold}_expert.joblib';model.save(model_path);loaded=FeatureModel.load(model_path)
        np.testing.assert_array_equal(np.asarray(loaded.predict(x[valid[:2048]]),dtype=np.float32),expert[:2048])
        for sid in np.unique(metadata['dataset_id'][valid])[:2]:
            h,o,_=data.get(int(sid));state=loaded.make_state(h)
            live=np.asarray([loaded.predict_one(state.update(point)) for point in o],dtype=np.float32)
            positions=metadata['dataset_id'][valid]==sid
            np.testing.assert_array_equal(live,expert[positions])
        prediction=combine(baseline['BL80'],expert,registration['parameters']['frozen_BL80_weight'])
        target=metadata['target'][valid];times=metadata['time_online'][valid]
        score=ts_auc(target,prediction,times);reference=ts_auc(target,baseline['BL80'],times)
        for name,values in [(f'fold_{fold}.npy',prediction),(f'fold_{fold}_expert.npy',expert),(f'fold_{fold}_global_rows.npy',global_rows[valid])]:
            np.save(out/name,values,allow_pickle=False)
        record={'status':'PASS','created_utc':utc_now(),'fold':fold,'ts_auc':score,'reference_ts_auc':reference,
            'delta':score-reference,'standalone_ts_auc':ts_auc(target,expert,times),
            'J102_ts_auc':ts_auc(target,baseline['J102'],times),'C205_ts_auc':ts_auc(target,baseline['C205'],times),
            'training':detail,'new_fits':1,'frozen_base_new_fits':0,'elapsed_seconds':time.perf_counter()-started,
            'streaming_cached_parity':True,'save_load_parity':True,'seal_rows':0,'frozen_OOF_sources':baseline_sources,
            'files_sha256':{n:digest(out/n) for n in [f'fold_{fold}.npy',f'fold_{fold}_expert.npy',f'fold_{fold}_global_rows.npy',model_path.name]}}
        atomic_json(out/f'fold_{fold}.json',record,immutable=True);records.append(record)
        print('NEXT4 SCORE',args.candidate,args.scope,fold,score,'delta',score-reference,flush=True)
    scores=np.asarray([r['ts_auc'] for r in records]);refs=np.asarray([r['reference_ts_auc'] for r in records])
    result={'status':'COMPLETE','created_utc':utc_now(),'candidate':args.candidate,'family':registration['family'],
        'scope':args.scope,'fold_scores':scores.tolist(),'reference_fold_scores':refs.tolist(),'fold_records':records,
        'mean':float(scores.mean()),'median':float(np.median(scores)),'worst_fold':float(scores.min()),
        'mean_delta':float((scores-refs).mean()),'median_delta':float(np.median(scores)-np.median(refs)),
        'worst_delta':float(scores.min()-refs.min()),'new_fits':len(folds),'frozen_base_new_fits':0,
        'standalone_mean':float(np.mean([r['standalone_ts_auc'] for r in records])),
        'registration_sha256':digest(hyp),'cache_identity':cache['identity'],'cache_key':cache['key'],
        'feature_count':len(names),'feature_names':names,'seal_rows':0,'promotion':False}
    if args.scope=='screen':
        result['screen_pass']=bool(result['mean_delta']>=registration['screen_gate']['mean_delta_min'] and
            np.all(scores-refs>=registration['screen_gate']['each_fold_delta_min']))
        result['disposition']='SCREEN_PASS_FULL_ELIGIBLE' if result['screen_pass'] else 'SCREEN_FAIL_STOP'
    else:
        gate=policy['performance']
        result['performance_numeric_admission']=bool(result['mean_delta']>=gate['mean_delta_min'] and
            result['median_delta']>=gate['median_delta_min'] and result['worst_delta']>=gate['worst_delta_min'])
        result['disposition']='FULL_COMPLETE_PAIRED_BOOTSTRAP_REQUIRED'
    atomic_json(out/'RESULTS.json',result,immutable=True);atomic_json(parent/'LATEST.json',result)
    data.guard.check_state();progress('NEXT4 '+args.scope+' completed',args.candidate+': '+result['disposition'])
    print('NEXT4 RESULT',args.candidate,args.scope,result['disposition'],result['mean_delta'],flush=True)


if __name__=='__main__':main()
