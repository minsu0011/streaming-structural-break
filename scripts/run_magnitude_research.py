"""Two bounded tests of dynamic magnitude dependence, retaining AR8 ABCD bytes."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import gc,json,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.features.config import CONFIG
from src.features.streaming import feature_names
from src.features.historical_calibration import CalibrationPolicy
from src.features.ar_cache import build_ar_cache
from src.features.magnitude_dependence import MAGNITUDE_POLICY,MagnitudeARState,names,model_families,implementation_hash
from src.models.magnitude_dependence import MagnitudeDependenceBundle
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


def magnitude_cache(split,config,policy):
    source=build_ar_cache(ROOT/'data/raw',ROOT/'data/cache',split,config,policy,input_order=8);version=implementation_hash(policy,config)
    directory=ROOT/'data/cache'/('ar8_magnitude_'+version[:16]);manifest_path=directory/'MANIFEST.json';source_hash=sha256(source/'MANIFEST.json')
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['status']=='COMPLETE' and manifest['representation_hash']==version and manifest['split_sha256']==split.digest and manifest['source_cache_manifest_sha256']==source_hash:
            if not all(sha256(directory/f'fold_{k}.parquet')==manifest['fold_sha256'][str(k)] for k in range(5)):raise RuntimeError('Magnitude cache changed')
            return directory
    series={s.dataset_id:s for s in load_training(ROOT/'data/raw',split=split)}
    manifest={'status':'BUILDING','representation_hash':version,'split_sha256':split.digest,'source_cache_manifest_sha256':source_hash,
        'magnitude_policy':MAGNITUDE_POLICY,'calibration_policy':policy.__dict__,'feature_names':names(config),'fold_sha256':{},'seal_rows':0,'first_51_ABCD_columns_unchanged':True}
    write_json(manifest_path,manifest)
    for fold in range(5):
        frame=pd.read_parquet(source/f'fold_{fold}.parquet');matrix=np.empty((len(frame),6),dtype=np.float32);times=frame.time_online.to_numpy()
        for sid,positions in frame.groupby('dataset_id',sort=False).indices.items():
            item=series[int(sid)];stream=MagnitudeARState(item.historical,config=config,policy=policy,include_abcd=False)
            if not np.array_equal(times[positions],np.arange(len(item.online))):raise RuntimeError('Magnitude cache row coordinates differ')
            for position,point in zip(positions,item.online):matrix[position]=stream.update_and_get(point)
        if not np.isfinite(matrix).all():raise RuntimeError('Nonfinite magnitude feature')
        frame=frame.drop(columns=list(feature_names(config)[51:]))
        for j,name in enumerate(names(config)[51:]):frame[name]=matrix[:,j]
        path=directory/f'fold_{fold}.parquet';frame.to_parquet(path,index=False);manifest['fold_sha256'][str(fold)]=sha256(path)
        write_json(manifest_path,manifest);print('MAGNITUDE CACHE',fold,'complete; ABCD copied unchanged',flush=True);del frame,matrix;gc.collect()
    manifest['status']='COMPLETE';write_json(manifest_path,manifest);return directory


def main():
    budget=ResearchBudget(ROOT/'artifacts/validation/REAL_SESSION.json')
    if not budget.can_start(1800):raise RuntimeError('Final audit buffer protected')
    admission(raw_directory=ROOT/'data/raw');split=FrozenSplit(ROOT/'artifacts/validation')
    registry=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text());anchor=next(r for r in registry['results'] if r['experiment_id']==MAGNITUDE_POLICY['anchor'])
    spec=anchor['candidate'];config=CONFIG.variant(**{k:tuple(v) if isinstance(v,list) else v for k,v in spec['feature_changes'].items()})
    calibration=CalibrationPolicy(method='historical_median_mad');version=implementation_hash(calibration,config)
    frozen={'timestamp':utc_now(),'settings':MAGNITUDE_POLICY,'anchor_candidate':spec,'split_sha256':split.digest,'representation_hash':version,
        'hypothesis':'Historical AR8 innovations retain magnitude dependence. Conditional-variance normalization improved worst-fold evidence, but existing D features mainly track signed products. Add six historical-null-normalized lag1 products of centered absolute and winsorized-squared innovations, and evaluate a V-only control.',
        'first_51_ABCD_columns_unchanged':True,'scopes_frozen_before_scores':True,'source_sha256':sha256(__file__),'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    policy_path=ROOT/'artifacts/real/MAGNITUDE_RESEARCH_POLICY.json'
    if policy_path.exists():raise RuntimeError('The magnitude policy has already been recorded; explicit recovery/reproduction required')
    write_json(policy_path,frozen);cache=magnitude_cache(split,config,calibration);folds=[pd.read_parquet(cache/f'fold_{k}.parquet') for k in range(5)]
    columns=names(config);ids=split.frame.loc[split.frame.role=='DEVELOPMENT_POOL','dataset_id'].tolist()
    metadata=pd.read_parquet(ROOT/'artifacts/audits/SERIES_METADATA.parquet',filters=[('dataset_id','in',ids)]);first={}
    for item in load_training(ROOT/'data/raw',split=split):
        fold=split.assignment(item.dataset_id)[1]
        if fold not in first:first[fold]=item
        if len(first)==5:break
    for families in MAGNITUDE_POLICY['family_scopes']:
        if not budget.can_start(600):break
        eid=f'M4_AR8_magnitude_{families}_S3_{version[:12]}';path=ROOT/'artifacts/real'/f'{eid}.json'
        report={'status':'RUNNING','experiment_id':eid,'model':'M4_magnitude_dependence','families':families,'sampling':'S3','phase':'magnitude_dependence_features',
            'representation':'ar8_magnitude_dependence','feature_hash':version,'feature_names':columns,'anchor':anchor['experiment_id'],'anchor_candidate':spec,'hypothesis':frozen['hypothesis'],
            'fold_scores':[],'cv_rows':[],'fractures':[],'latencies':[],'save_load_max_abs_diff':0.,'live_cached_prediction_max_abs_diff':0.,'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
        if path.exists():raise RuntimeError('Magnitude candidate was already evaluated')
        started=time.perf_counter()
        try:
            for fold,validation in enumerate(folds):
                training=pd.concat([frame for k,frame in enumerate(folds) if k!=fold],ignore_index=True)
                train_ids=set(training.dataset_id);validation_ids=set(validation.dataset_id);assert_development_ids(train_ids|validation_ids,split)
                if {split.groups[i] for i in train_ids}&{split.groups[i] for i in validation_ids}:raise RuntimeError('Magnitude training group leakage')
                matrix=training.loc[:,columns].to_numpy(dtype=np.float32);target=training.target.to_numpy()
                base=fit_candidate('M4_lightgbm',matrix,target,families=model_families(families),sample_weight=contribution_weights(target,training.time_online,'S3'),
                    params={},config=config,excluded_columns=tuple(np.flatnonzero(np.ptp(matrix,axis=0)<=1e-12)))
                del training,matrix,target;gc.collect()
                bundle=MagnitudeDependenceBundle(base,calibration,config);model_path=ROOT/'artifacts/models/real'/eid/f'fold_{fold}.joblib';bundle.save(model_path);loaded=MagnitudeDependenceBundle.load(model_path)
                matrix=validation.loc[:,columns].to_numpy(dtype=np.float32);prediction=bundle.predict(matrix).astype(np.float32);np.testing.assert_array_equal(loaded.predict(matrix).astype(np.float32),prediction)
                sample=first[fold];stream=loaded.make_state(sample.historical);live=np.array([loaded.predict_one(stream.update_and_get(p)) for p in sample.online],dtype=np.float32)
                np.testing.assert_array_equal(live,prediction[validation.dataset_id.to_numpy()==sample.dataset_id])
                def factory():
                    stream=loaded.make_state(sample.historical)
                    return lambda p:loaded.predict_one(stream.update_and_get(p))
                points=np.tile(sample.online.astype(np.float32),max(1,3000//len(sample.online)+1))[:3000]
                latency=benchmark(factory,points,repeats=3);latency['model_bytes']=model_path.stat().st_size;report['latencies'].append(latency)
                value=ts_auc(validation.target,prediction,validation.time_online);report['fold_scores'].append(value)
                report['cv_rows'].append({'experiment_id':eid,'model':report['model'],'families':families,'sampling':'S3','fold':fold,'ts_auc':value})
                report['fractures'].extend([{'experiment_id':eid,'fold':fold,**r} for r in profile(validation.target,prediction,validation.time_online,validation.dataset_id,metadata)])
                oof=ROOT/'data/processed/oof'/eid;oof.mkdir(parents=True,exist_ok=True);np.save(oof/f'fold_{fold}.npy',prediction,allow_pickle=False)
                write_json(oof/'MANIFEST.json',{'cache_directory':str(cache.relative_to(ROOT)),'split_sha256':split.digest,'representation':report['representation'],'feature_hash':version,'feature_names':columns,'wire_dtype':'float32','seal_rows':0})
                write_json(ROOT/'artifacts/real'/f'{eid}_CHECKPOINT.json',report);print('MAGNITUDE',families,'fold',fold,value,flush=True)
                del base,bundle,loaded,matrix,prediction;gc.collect()
            report.update(summarize(report['fold_scores']));report['status']='COMPLETE';report['latency_us']=float(np.median([r['throughput_median_us_per_point'] for r in report['latencies']]))
            report['versus_original_ar8_anchor']=promotion_decision(report,anchor)
        except Exception as exc:
            report['status']='FAILED';report['exception_type']=type(exc).__name__
            (ROOT/'experiments/logs'/f'{eid}_failure.log').write_text(traceback.format_exc(),encoding='utf-8')
            ledger(experiment_id=eid,model=report['model'],status='FAILED',notes=f'Magnitude dependence; {type(exc).__name__}; no partial-fold ranking')
        report['elapsed_seconds']=time.perf_counter()-started;write_json(path,report)
        print('MAGNITUDE RESULT',report['status'],eid,report.get('mean'),report.get('median'),report.get('worst_fold'),flush=True)


if __name__=='__main__':main()
