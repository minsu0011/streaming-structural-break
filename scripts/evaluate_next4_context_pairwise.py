"""H-only modulation of a true nested same-time pairwise combination."""
from pathlib import Path
import sys
import argparse
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,digest,read_json,utc_now,progress
from src.next4.nested_reuse import load_outer,source_hash as nested_source_hash
from src.next4.context_pairwise import fit,ContextPairwiseModel as PairwiseModel,source_hash,load_historical_cache,row_context
from src.next4.historical_context import context
from src.next4.frozen import fold_predictions
from src.next3.frozen import fold_artifact,J102,C205
from src.next.data import prepare_dev
from src.scoring.ts_auc import ts_auc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True)
    parser.add_argument('--scope',choices=['screen','full'],required=True);args=parser.parse_args()
    hyp_path=BASE/'registrations'/(args.candidate+'.json');hyp=read_json(hyp_path)
    policy_path=ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json';policy=read_json(policy_path)
    assert hyp['created_before_score'] and hyp['family']=='G' and hyp['policy_sha256']==digest(policy_path)
    assert hyp['model_source_sha256']==source_hash() and hyp['nested_source_sha256']==nested_source_hash()
    for path,sha in hyp['execution_source_sha256'].items():assert digest(ROOT/path)==sha
    menu_path=BASE/'families/G_MENU_LOCK.json';assert digest(menu_path)==hyp['family_menu_sha256']
    audit=read_json(BASE/'nested_reuse_audit/RESULTS.json')
    assert audit['status']=='PASS' and audit['source_sha256']==nested_source_hash() and audit['inner_base_fits_available_for_reuse']==56
    parent=BASE/'experiments'/args.candidate;out=parent/args.scope
    if args.scope=='full':
        assert read_json(parent/'screen/RESULTS.json')['screen_pass']
        if len(list((BASE/'experiments').glob('*/full/START.json')))>=read_json(ROOT/'configs/next4/CANDIDATE_BUDGET.json')['new_full_candidates_max']:
            raise RuntimeError('Full candidate budget exhausted')
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'candidate':args.candidate,'family':'G','scope':args.scope,
        'registration_sha256':digest(hyp_path),'runner_sha256':digest(Path(__file__)),
        'nested_reuse_audit_sha256':digest(BASE/'nested_reuse_audit/RESULTS.json'),'new_fits_at_start':0},immutable=True)
    data=prepare_dev(ROOT);records=[];folds=policy['screen']['folds'] if args.scope=='screen' else list(range(5))
    context_ids,context_values,context_proof=load_historical_cache(data)
    for fold in folds:
        train_scores,valid_scores,meta,train,valid,global_rows,proof=load_outer(data,args.scope,fold)
        train_context=row_context(context_ids,context_values,meta['dataset_id'][train])
        valid_context=row_context(context_ids,context_values,meta['dataset_id'][valid])
        data.guard.partition(np.unique(meta['dataset_id'][train]),np.unique(meta['dataset_id'][valid]))
        assert not np.intersect1d(global_rows[train],global_rows[valid]).size
        atomic_json(out/f'fold_{fold}_FIT_START.json',{'created_utc':utc_now(),'intended_new_fits':1,
            'intended_new_base_fits':0,'nested_reuse_proof':proof,'registration_sha256':digest(hyp_path)},immutable=True)
        progress('Fitting nested pairwise readout',args.candidate+' '+args.scope+' outer '+str(fold)+'; only inner OOF enters the optimizer')
        start=time.perf_counter()
        model,detail=fit(train_scores,train_context,meta['target'][train],meta['time_online'][train],meta['dataset_id'][train],hyp['parameters'])
        prediction=model.predict(valid_scores,valid_context)
        model_path=out/f'fold_{fold}_expert.joblib';model.save(model_path);loaded=PairwiseModel.load(model_path)
        np.testing.assert_array_equal(prediction[:2048],loaded.predict(valid_scores[:2048],valid_context[:2048]))
        baseline,baseline_sources=fold_predictions(data,fold,args.scope,global_rows[valid])
        jm,_,_,_=fold_artifact(data,J102,fold,args.scope);cm,_,_,_=fold_artifact(data,C205,fold,args.scope)
        for sid in np.unique(meta['dataset_id'][valid])[:2]:
            h,o,_=data.get(int(sid));js=jm.make_state(h);cs=cm.make_state(h);live=[]
            h_context=context(h)
            for point in o:
                j=float(np.float32(jm.predict_one(js.update(point))));c=float(np.float32(cm.predict_one(cs.update(point))))
                live.append(loaded.predict_one(j,c,h_context))
            np.testing.assert_array_equal(np.array(live,dtype=np.float32),prediction[meta['dataset_id'][valid]==sid])
        target=meta['target'][valid];times=meta['time_online'][valid]
        score=ts_auc(target,prediction,times);reference=ts_auc(target,baseline['BL80'],times)
        for name,values in [(f'fold_{fold}.npy',prediction),(f'fold_{fold}_expert.npy',prediction),(f'fold_{fold}_global_rows.npy',global_rows[valid])]:
            np.save(out/name,values,allow_pickle=False)
        record={'status':'PASS','created_utc':utc_now(),'fold':fold,'ts_auc':score,'reference_ts_auc':reference,'delta':score-reference,
            'standalone_ts_auc':score,'standalone_interpretation':'This is the H-modulated learned combination; immutable H context changes relative expert use',
            'J102_ts_auc':ts_auc(target,baseline['J102'],times),'C205_ts_auc':ts_auc(target,baseline['C205'],times),
            'training':detail,'weights':model.weights.tolist(),'new_fits':1,'new_meta_fits':1,'frozen_base_new_fits':0,
            'reused_inner_base_fits':proof['inner_base_fits_reused'],'elapsed_seconds':time.perf_counter()-start,
            'streaming_cached_parity':True,'save_load_parity':True,'seal_rows':0,'frozen_OOF_sources':baseline_sources,
            'files_sha256':{name:digest(out/name) for name in [f'fold_{fold}.npy',f'fold_{fold}_expert.npy',f'fold_{fold}_global_rows.npy',model_path.name]}}
        atomic_json(out/f'fold_{fold}.json',record,immutable=True);records.append(record)
        print('NEXT4 PAIRWISE SCORE',args.candidate,args.scope,fold,score,'delta',score-reference,flush=True)
    scores=np.array([r['ts_auc'] for r in records]);refs=np.array([r['reference_ts_auc'] for r in records])
    result={'status':'COMPLETE','created_utc':utc_now(),'candidate':args.candidate,'family':'G','scope':args.scope,
        'fold_scores':scores.tolist(),'reference_fold_scores':refs.tolist(),'fold_records':records,
        'mean':float(scores.mean()),'median':float(np.median(scores)),'worst_fold':float(scores.min()),
        'mean_delta':float((scores-refs).mean()),'median_delta':float(np.median(scores)-np.median(refs)),
        'worst_delta':float(scores.min()-refs.min()),'new_fits':len(folds),'new_meta_fits':len(folds),'frozen_base_new_fits':0,
        'reused_inner_base_fits':8*len(folds),'standalone_mean':float(scores.mean()),
        'standalone_interpretation':'H-only modulation of J102/C205 score gap, fitted on honest inner OOF; no outcome-derived context',
        'feature_count':12,'H_context_cache_proof':context_proof,'registration_sha256':digest(hyp_path),
        'nested_reuse_audit_sha256':digest(BASE/'nested_reuse_audit/RESULTS.json'),'seal_rows':0,'promotion':False}
    if args.scope=='screen':
        result['screen_pass']=bool(result['mean_delta']>=hyp['screen_gate']['mean_delta_min'] and np.all(scores-refs>=hyp['screen_gate']['each_fold_delta_min']))
        result['disposition']='SCREEN_PASS_FULL_ELIGIBLE' if result['screen_pass'] else 'SCREEN_FAIL_STOP'
    else:
        gate=policy['performance'];result['performance_numeric_admission']=bool(result['mean_delta']>=gate['mean_delta_min'] and
            result['median_delta']>=gate['median_delta_min'] and result['worst_delta']>=gate['worst_delta_min'])
        result['disposition']='FULL_COMPLETE_PAIRED_BOOTSTRAP_REQUIRED'
    atomic_json(out/'RESULTS.json',result,immutable=True);atomic_json(parent/'LATEST.json',result)
    data.guard.check_state();progress('Nested pairwise evaluation completed',args.candidate+' '+args.scope+' '+result['disposition'])
    print('NEXT4 PAIRWISE RESULT',args.candidate,args.scope,result['mean_delta'],flush=True)


if __name__=='__main__':main()
