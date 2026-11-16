"""Two predeclared historical-context comparisons on the immutable AR4 anchor."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='4'
import gc,json,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.features.config import CONFIG
from src.features.streaming import feature_names
from src.features.historical_calibration import CalibrationPolicy
from src.features.ar_residual_input import historical_filter
from src.features.historical_context import POLICY,CONTEXT_NAMES,historical_context,names,model_families,implementation_hash
from src.models.context_calibrated import ContextCalibratedBundle
from src.models.learners import fit_candidate
from src.validation.research import contribution_weights,summarize
from src.validation.adaptive import ResearchBudget,promotion_decision
from src.validation.splits import FrozenSplit,assert_development_ids
from src.validation.diagnostics import profile
from src.scoring.ts_auc import ts_auc
from src.data.loader import load_training
from src.utils.artifacts import write_json,sha256,utc_now,ledger
from scripts.run_real_research import admission
from scripts.verify_engineering import benchmark


def context_cache(anchor,split,config,calibration):
    source_manifest=json.loads((ROOT/'data/processed/oof'/anchor['experiment_id']/'MANIFEST.json').read_text());source=ROOT/source_manifest['cache_directory']
    version=implementation_hash(calibration,config);directory=ROOT/'data/cache'/('ar_context_'+version[:16]);manifest_path=directory/'MANIFEST.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['status']=='COMPLETE' and manifest['representation_hash']==version and manifest['split_sha256']==split.digest:
            if not all(sha256(directory/f'fold_{k}.parquet')==manifest['fold_sha256'][str(k)] for k in range(5)):raise RuntimeError('Context cache bytes changed')
            return directory
    context={}
    for item in load_training(ROOT/'data/raw',split=split):
        reference,coefficients,_,_=historical_filter(item.historical.astype(np.float32),config)
        context[item.dataset_id]=historical_context(item.historical,reference,coefficients,config)
    manifest={'status':'BUILDING','representation_hash':version,'split_sha256':split.digest,'source_cache_manifest_sha256':sha256(source/'MANIFEST.json'),
        'source_cache_directory':str(source.relative_to(ROOT)),'policy':POLICY,'feature_names':names(config),'fold_sha256':{},'seal_rows':0}
    write_json(manifest_path,manifest)
    original=json.loads((source/'MANIFEST.json').read_text())
    for fold in range(5):
        path=source/f'fold_{fold}.parquet'
        if sha256(path)!=original['fold_sha256'][str(fold)]:raise RuntimeError('Original AR cache changed')
        frame=pd.read_parquet(path);assert_development_ids(frame.dataset_id.unique(),split)
        frame=frame.drop(columns=list(feature_names(config)[51:]))
        for j,name in enumerate(CONTEXT_NAMES):frame[name]=frame.dataset_id.map({sid:values[j] for sid,values in context.items()}).astype(np.float32)
        target=directory/f'fold_{fold}.parquet';frame.to_parquet(target,index=False);manifest['fold_sha256'][str(fold)]=sha256(target)
        write_json(manifest_path,manifest);print('CONTEXT CACHE',fold,'complete',flush=True);del frame;gc.collect()
    manifest['status']='COMPLETE';write_json(manifest_path,manifest);return directory


def main():
    budget=ResearchBudget(ROOT/'artifacts/validation/REAL_SESSION.json')
    if not budget.can_start(1200):raise RuntimeError('Final audit buffer protected')
    admission(raw_directory=ROOT/'data/raw');split=FrozenSplit(ROOT/'artifacts/validation')
    registry=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text());anchor=next(r for r in registry['results'] if r['experiment_id']==POLICY['anchor'])
    spec=anchor['candidate'];config=CONFIG.variant(**{k:tuple(v) if isinstance(v,list) else v for k,v in spec['feature_changes'].items()})
    calibration=CalibrationPolicy(method='historical_median_mad');version=implementation_hash(calibration,config)
    policy={'timestamp':utc_now(),'settings':POLICY,'anchor_candidate':spec,'split_sha256':split.digest,'representation_hash':version,
        'hypothesis':'After historical AR whitening, dependence and tail regimes still differ in reliability. Give the existing small learner six available historical statistics beside unchanged ABCD evidence; use a G-only control to identify static prior effects.',
        'frozen_before_both_context_scores':True,'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    policy_path=ROOT/'artifacts/real/HISTORICAL_CONTEXT_POLICY.json'
    if policy_path.exists():
        stored=json.loads(policy_path.read_text());policy['timestamp']=stored['timestamp']
        if stored!=policy:raise RuntimeError('Context policy changed')
    else:write_json(policy_path,policy)
    cache=context_cache(anchor,split,config,calibration);folds=[pd.read_parquet(cache/f'fold_{k}.parquet') for k in range(5)]
    ids=split.frame.loc[split.frame.role=='DEVELOPMENT_POOL','dataset_id'].tolist()
    metadata=pd.read_parquet(ROOT/'artifacts/audits/SERIES_METADATA.parquet',filters=[('dataset_id','in',ids)])
    first={}
    for item in load_training(ROOT/'data/raw',split=split):
        fold=split.assignment(item.dataset_id)[1]
        if fold not in first:first[fold]=item
        if len(first)==5:break
    columns=names(config)
    for families in POLICY['candidate_families']:
        if not budget.can_start(300):break
        eid=f'M4_context_{families}_S3_{version[:12]}';path=ROOT/'artifacts/real'/f'{eid}.json'
        if path.exists():
            print('CONTEXT ALREADY RECORDED',eid,flush=True);continue
        report={'status':'RUNNING','experiment_id':eid,'model':'M4_historical_context','families':families,'sampling':'S3','phase':'historical_context_refinement',
            'representation':'ar_context_historical_median_mad','representation_hash':version,'feature_names':columns,'anchor':anchor['experiment_id'],'anchor_candidate':spec,
            'timestamp':utc_now(),'fold_scores':[],'cv_rows':[],'fractures':[],'latencies':[],'save_load_max_abs_diff':0.,'live_cached_prediction_max_abs_diff':0.,
            'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
        started=time.perf_counter()
        try:
            for fold,validation in enumerate(folds):
                training=pd.concat([f for k,f in enumerate(folds) if k!=fold],ignore_index=True)
                train_ids=set(training.dataset_id);validation_ids=set(validation.dataset_id)
                assert_development_ids(train_ids|validation_ids,split)
                if {split.groups[i] for i in train_ids}&{split.groups[i] for i in validation_ids}:raise RuntimeError('Frozen groups overlap train and validation')
                matrix=training.loc[:,columns].to_numpy(dtype=np.float32);target=training.target.to_numpy()
                excluded=tuple(np.flatnonzero(np.ptp(matrix,axis=0)<=1e-12))
                model=fit_candidate('M4_lightgbm',matrix,target,families=model_families(families),sample_weight=contribution_weights(target,training.time_online,'S3'),
                    params={},config=config,excluded_columns=excluded)
                del training,matrix,target;gc.collect()
                bundle=ContextCalibratedBundle(model,calibration,config);model_path=ROOT/'artifacts/models/real'/eid/f'fold_{fold}.joblib';bundle.save(model_path);loaded=ContextCalibratedBundle.load(model_path)
                matrix=validation.loc[:,columns].to_numpy(dtype=np.float32);prediction=bundle.predict(matrix).astype(np.float32)
                np.testing.assert_array_equal(loaded.predict(matrix).astype(np.float32),prediction)
                sample=first[fold];state=loaded.make_state(sample.historical);live=np.array([loaded.predict_one(state.update_and_get(p)) for p in sample.online],dtype=np.float32)
                np.testing.assert_array_equal(live,prediction[validation.dataset_id.to_numpy()==sample.dataset_id])
                h=sample.historical.astype(np.float32);points=np.tile(sample.online.astype(np.float32),max(1,3000//len(sample.online)+1))[:3000]
                def factory():
                    stream=loaded.make_state(h)
                    return lambda point:loaded.predict_one(stream.update_and_get(point))
                latency=benchmark(factory,points,repeats=3);latency['model_bytes']=model_path.stat().st_size;report['latencies'].append(latency)
                value=ts_auc(validation.target,prediction,validation.time_online);report['fold_scores'].append(value)
                report['cv_rows'].append({'experiment_id':eid,'model':report['model'],'families':families,'sampling':'S3','fold':fold,'ts_auc':value})
                report['fractures'].extend([{'experiment_id':eid,'fold':fold,**r} for r in profile(validation.target,prediction,validation.time_online,validation.dataset_id,metadata)])
                oof=ROOT/'data/processed/oof'/eid;oof.mkdir(parents=True,exist_ok=True);np.save(oof/f'fold_{fold}.npy',prediction,allow_pickle=False)
                write_json(oof/'MANIFEST.json',{'cache_directory':str(cache.relative_to(ROOT)),'split_sha256':split.digest,'representation':'ar_context_historical_median_mad','feature_hash':version,'feature_names':columns,'wire_dtype':'float32','seal_rows':0})
                write_json(ROOT/'artifacts/real'/f'{eid}_CHECKPOINT.json',report);print('CONTEXT',families,'fold',fold,value,flush=True)
                del model,bundle,loaded,matrix,prediction;gc.collect()
            report.update(summarize(report['fold_scores']));report['status']='COMPLETE';report['latency_us']=float(np.median([r['throughput_median_us_per_point'] for r in report['latencies']]))
            report['versus_original_ar4_anchor']=promotion_decision(report,anchor)
        except Exception as exc:
            report['status']='FAILED';report['exception_type']=type(exc).__name__
            (ROOT/'experiments/logs'/f'{eid}_failure.log').write_text(traceback.format_exc(),encoding='utf-8')
            ledger(experiment_id=eid,model=report['model'],status='FAILED',notes=f'Historical context; {type(exc).__name__}; no partial-fold promotion')
        report['elapsed_seconds']=time.perf_counter()-started;write_json(path,report)
        print('CONTEXT RESULT',report['status'],eid,report.get('mean'),report.get('median'),report.get('worst_fold'),flush=True)


if __name__=='__main__':main()
