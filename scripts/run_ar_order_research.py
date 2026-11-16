"""Two fixed AR-order comparisons on the same AR(4) feature/model anchor."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import argparse,gc,json,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.features.config import CONFIG
from src.features.streaming import feature_names
from src.features.ar_cache import build_ar_cache
from src.features.ar_order_input import implementation_hash
from src.models.ar_order_calibrated import AROrderCalibratedBundle
from src.models.representations import calibration_policy
from src.validation.adaptive import ResearchBudget,promotion_decision
from src.validation.splits import FrozenSplit
from src.validation.research import fit_fold,summarize
from src.validation.diagnostics import profile
from src.data.loader import load_training
from src.utils.artifacts import write_json,utc_now,sha256,ledger
from scripts.run_real_research import admission
from scripts.verify_engineering import benchmark


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--order',type=int,choices=[2,8],required=True)
    parser.add_argument('--model',choices=['M4_lightgbm','M3_histgb'],default='M4_lightgbm');args=parser.parse_args()
    if args.model=='M3_histgb' and args.order!=8:raise ValueError('The bounded model follow-up is limited to the completed order-8 representation')
    if not ResearchBudget(ROOT/'artifacts/validation/REAL_SESSION.json').can_start(1200):raise RuntimeError('Final audit buffer protected')
    admission(raw_directory=ROOT/'data/raw');split=FrozenSplit(ROOT/'artifacts/validation')
    settings=json.loads((ROOT/'configs/ar_order_variants.json').read_text());state=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text())
    anchor=next(r for r in state['results'] if r['experiment_id']==settings['comparison_anchor']);candidate={**anchor['candidate'],'model':args.model}
    config=CONFIG.variant(**{k:tuple(v) if isinstance(v,list) else v for k,v in candidate['feature_changes'].items()});policy=calibration_policy(candidate['representation'])
    version=implementation_hash(policy,config,args.order);eid=f'{args.model[:2]}_ar_order{args.order}_ABCD_S3_{version[:12]}'
    result_path=ROOT/'artifacts/real'/f'{eid}.json'
    if result_path.exists():raise RuntimeError('This prospective AR-order candidate already has a result')
    frozen={'settings':settings,'anchor_candidate':candidate,'split_sha256':split.digest,'order_feature_hashes':{str(order):implementation_hash(policy,config,order) for order in settings['candidate_orders']},
        'comparison':'change only historical AR input order','frozen_before_order_variant_scores':True}
    policy_path=ROOT/'artifacts/real/AR_ORDER_REFINEMENT_POLICY.json'
    if args.model=='M3_histgb':
        reference_path=ROOT/'artifacts/real'/f'M4_ar_order8_ABCD_S3_{version[:12]}.json'
        order8=json.loads(reference_path.read_text())
        if order8['status']!='COMPLETE' or not all(a>b for a,b in zip(order8['fold_scores'],anchor['fold_scores'])):raise RuntimeError('The fixed model follow-up requires the observed all-fold AR8 improvement')
        frozen={'comparison_anchor':order8['experiment_id'],'candidate_model':args.model,'candidate_configuration':candidate,
            'input_order':8,'representation_hash':version,'split_sha256':split.digest,'maximum_followup_candidates':1,
            'hypothesis':'AR8 improved all five folds over AR4, but median gain fell below promotion threshold; change only the small tree learner to HistGB.',
            'frozen_before_histgb_followup_score':True}
        policy_path=ROOT/'artifacts/real/AR8_HISTGB_FOLLOWUP_POLICY.json'
    if policy_path.exists() and json.loads(policy_path.read_text())!=frozen:raise RuntimeError('AR order policy changed')
    write_json(policy_path,frozen)
    report={'status':'RUNNING','experiment_id':eid,'model':args.model[:2]+'_ar_order','families':candidate['families'],'sampling':candidate['sampling'],'phase':'ar_order_refinement',
        'representation':f'ar_order{args.order}_historical_median_mad','input_order':args.order,'representation_hash':version,
        'timestamp':utc_now(),'anchor':anchor['experiment_id'],'anchor_candidate':candidate,'fold_scores':[],'cv_rows':[],'fractures':[],'latencies':[],
        'save_load_max_abs_diff':0.,'live_cached_prediction_max_abs_diff':None,'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    started=time.perf_counter()
    try:
        cache=build_ar_cache(ROOT/'data/raw',ROOT/'data/cache',split,config,policy,input_order=args.order)
        folds=[pd.read_parquet(cache/f'fold_{k}.parquet') for k in range(5)]
        ids=split.frame.loc[split.frame.role=='DEVELOPMENT_POOL','dataset_id'].tolist()
        metadata=pd.read_parquet(ROOT/'artifacts/audits/SERIES_METADATA.parquet',filters=[('dataset_id','in',ids)])
        sample=next(load_training(ROOT/'data/raw',split=split));h=sample.historical.astype(np.float32);o=sample.online.astype(np.float32);points=np.tile(o,max(1,5000//len(o)+1))[:5000]
        first_by_fold={}
        for item in load_training(ROOT/'data/raw',split=split):
            key=split.assignment(item.dataset_id)[1]
            if key not in first_by_fold:first_by_fold[key]=item
            if len(first_by_fold)==5:break
        names=feature_names(config)
        for fold,frame in enumerate(folds):
            model,prediction,score=fit_fold(folds,fold,candidate['model'],candidate['families'],candidate['sampling'],split=split,params=candidate['params'],config=config)
            bundle=AROrderCalibratedBundle(model,policy,config,args.order);path=ROOT/'artifacts/models/real'/eid/f'fold_{fold}.joblib';bundle.save(path);loaded=AROrderCalibratedBundle.load(path)
            matrix=frame.loc[:,names].to_numpy(dtype=np.float32);np.testing.assert_array_equal(loaded.predict(matrix).astype(np.float32),prediction)
            item=first_by_fold[fold];stream=loaded.make_state(item.historical.astype(np.float32))
            actual=np.array([loaded.predict_one(stream.update_and_get(p)) for p in item.online.astype(np.float32)],dtype=np.float32)
            np.testing.assert_array_equal(actual,prediction[frame.dataset_id.to_numpy()==item.dataset_id])
            report['live_cached_prediction_max_abs_diff']=0.
            def factory():
                stream=loaded.make_state(h)
                return lambda point:loaded.predict_one(stream.update_and_get(point))
            latency=benchmark(factory,points,repeats=3);latency['model_bytes']=path.stat().st_size
            report['latencies'].append(latency);report['fold_scores'].append(score)
            report['cv_rows'].append({'experiment_id':eid,'model':report['model'],'families':candidate['families'],'sampling':candidate['sampling'],'fold':fold,'ts_auc':score})
            report['fractures'].extend([{'experiment_id':eid,'fold':fold,**r} for r in profile(frame.target,prediction,frame.time_online,frame.dataset_id,metadata)])
            oof=ROOT/'data/processed/oof'/eid;oof.mkdir(parents=True,exist_ok=True);np.save(oof/f'fold_{fold}.npy',prediction,allow_pickle=False)
            write_json(oof/'MANIFEST.json',{'cache_directory':str(cache.relative_to(ROOT)),'split_sha256':split.digest,'feature_hash':version,'input_order':args.order,'wire_dtype':'float32'})
            write_json(ROOT/'artifacts/real'/f'{eid}_CHECKPOINT.json',report)
            print('AR ORDER',args.order,'fold',fold,score,'native_us',latency['throughput_median_us_per_point'],flush=True)
            del model,bundle,loaded,matrix;gc.collect()
        report.update(summarize(report['fold_scores']));report['status']='COMPLETE';report['latency_us']=float(np.median([r['throughput_median_us_per_point'] for r in report['latencies']]))
        report['versus_frozen_ar4_anchor']=promotion_decision(report,anchor)
    except Exception as exc:
        report['status']='FAILED';report['exception_type']=type(exc).__name__
        (ROOT/'experiments/logs'/f'{eid}_failure.log').write_text(traceback.format_exc(),encoding='utf-8')
        ledger(experiment_id=eid,model=report['model'],status='FAILED',notes=f'AR order {args.order}; {type(exc).__name__}; no partial-fold promotion')
    report['elapsed_seconds']=time.perf_counter()-started;write_json(result_path,report)
    print('AR ORDER',args.order,report['status'],eid,report.get('mean'),report.get('median'),report.get('worst_fold'),flush=True)


if __name__=='__main__':main()
