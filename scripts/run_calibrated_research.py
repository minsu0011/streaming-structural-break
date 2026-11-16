"""Bounded historical-normalization hypothesis; separate results from the base ladder."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import argparse,json,sys,time,gc
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.features.streaming import FEATURE_NAMES,feature_version,feature_names
from src.features.cache import build_cache,read_fold
from src.features.config import CONFIG
from src.features.historical_calibration import POLICY,CalibrationPolicy,transform_cached_features
from src.models.calibrated import CalibratedBundle,representation_version as implementation_hash
from src.validation.splits import FrozenSplit
from src.validation.research import fit_fold,summarize
from src.validation.diagnostics import profile
from src.validation.adaptive import ResearchBudget
from src.data.loader import load_training
from src.utils.artifacts import write_json,utc_now,sha256
from scripts.run_real_research import admission
from scripts.verify_engineering import benchmark

def calibrated_cache(raw,split,policy,config=CONFIG):
    source=build_cache(raw,ROOT/'data/cache',split=split,config=config)
    names=feature_names(config)
    transform=transform_cached_features
    if policy.method.endswith('_startup'):
        from src.features.startup_calibration import transform_cached_features as transform
    directory=ROOT/'data/cache'/('historical_calibration_'+implementation_hash(policy,config)[:16])
    manifest_path=directory/'MANIFEST.json'
    previous_fold_hashes=None
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest.get('status')=='COMPLETE' and manifest.get('representation_hash')==implementation_hash(policy,config) and manifest.get('split_sha256')==split.digest:
            previous_fold_hashes=manifest.get('fold_sha256')
        if manifest['status']=='COMPLETE' and manifest['split_sha256']==split.digest and manifest['source_manifest_sha256']==sha256(source/'MANIFEST.json'):
            for fold in range(5):
                if sha256(directory/f'fold_{fold}.parquet')!=manifest['fold_sha256'][str(fold)]:raise RuntimeError('Calibrated cache tampered')
            return directory
    directory.mkdir(parents=True,exist_ok=True)
    histories={series.dataset_id:series.historical.astype(np.float32).copy() for series in load_training(raw,split=split)}
    manifest={'status':'BUILDING','representation_hash':implementation_hash(policy,config),'base_feature_hash':feature_version(config),
        'policy':policy.__dict__,'split_sha256':split.digest,'source_manifest_sha256':sha256(source/'MANIFEST.json'),
        'calibration_data':'historical values only','seal_rows':0,'fold_sha256':{}}
    write_json(manifest_path,manifest)
    for fold in range(5):
        frame=read_fold(source,fold,split=split,config=config);matrix=frame.loc[:,names].to_numpy(dtype=np.float32,copy=True)
        for sid,positions in frame.groupby('dataset_id',sort=False).indices.items():
            matrix[positions]=transform(matrix[positions],histories[int(sid)],policy=policy,config=config)
        frame.loc[:,names]=matrix
        path=directory/f'fold_{fold}.parquet';frame.to_parquet(path,index=False)
        if previous_fold_hashes is not None and sha256(path)!=previous_fold_hashes[str(fold)]:
            raise RuntimeError('Unchanged representation produced different cached bytes; old OOF requires investigation')
        manifest['fold_sha256'][str(fold)]=sha256(path);write_json(manifest_path,manifest)
        print(f'Historical calibration cache fold {fold} complete',flush=True);del frame,matrix;gc.collect()
    manifest['status']='COMPLETE';manifest['prior_representation_bytes_revalidated']=previous_fold_hashes is not None
    write_json(manifest_path,manifest);return directory

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',default='M4_lightgbm');parser.add_argument('--sampling',default='S3')
    parser.add_argument('--families',default='ABCD');parser.add_argument('--method',default='historical_mean_std');parser.add_argument('--cache-only',action='store_true')
    args=parser.parse_args();admission(raw_directory=ROOT/'data/raw')
    budget=ResearchBudget(ROOT/'artifacts/validation/REAL_SESSION.json')
    if not budget.can_start(900):raise RuntimeError('Final audit buffer protected')
    split=FrozenSplit(ROOT/'artifacts/validation');policy=CalibrationPolicy(method=args.method)
    cache=calibrated_cache(ROOT/'data/raw',split,policy)
    if args.cache_only:return
    eid=f'{args.model}_{args.families}_{args.sampling}_{args.method}_{implementation_hash(policy)[:12]}'
    result_path=ROOT/'artifacts/real'/f'{eid}.json'
    if result_path.exists():raise RuntimeError('Completed calibrated candidate already exists; do not repeat without a reproduction designation')
    folds=[pd.read_parquet(cache/f'fold_{k}.parquet') for k in range(5)]
    dev_ids={i for f in folds for i in f.dataset_id.unique()}
    metadata=pd.read_parquet(ROOT/'artifacts/audits/SERIES_METADATA.parquet',filters=[('dataset_id','in',sorted(dev_ids))])
    sample=next(load_training(ROOT/'data/raw',split=split));h=sample.historical.astype(np.float32);points=np.tile(sample.online,5).astype(np.float32)[:1000]
    scores=[];rows=[];fractures=[];latencies=[];start=time.perf_counter()
    for fold,frame in enumerate(folds):
        model,prediction,score=fit_fold(folds,fold,args.model,args.families,args.sampling,split=split)
        bundle=CalibratedBundle(model,policy);path=ROOT/'artifacts/models/real'/eid/f'fold_{fold}.joblib';bundle.save(path);loaded=CalibratedBundle.load(path)
        probe=frame.loc[:,FEATURE_NAMES].to_numpy(dtype=np.float32)[:1000]
        np.testing.assert_array_equal(bundle.predict(probe),loaded.predict(probe))
        def factory():
            state=loaded.make_state(h)
            return lambda point:loaded.predict_one(state.update_and_get(point))
        latency=benchmark(factory,points,repeats=3);latency['model_bytes']=path.stat().st_size
        init_start=time.perf_counter();loaded.make_state(h);latency['historical_init_seconds']=time.perf_counter()-init_start
        latencies.append(latency);scores.append(score);rows.append({'experiment_id':eid,'model':args.model,'families':args.families,'sampling':args.sampling,'fold':fold,'ts_auc':score})
        fractures.extend([{'experiment_id':eid,'fold':fold,**r} for r in profile(frame.target,prediction,frame.time_online,frame.dataset_id,metadata)])
        oof=ROOT/'data/processed/oof'/eid;oof.mkdir(parents=True,exist_ok=True);np.save(oof/f'fold_{fold}.npy',prediction,allow_pickle=False)
        write_json(oof/'MANIFEST.json',{'representation_hash':implementation_hash(policy),'cache_directory':str(cache.relative_to(ROOT)),'split_sha256':split.digest,'policy':policy.__dict__})
        print(f'{eid} fold {fold}: {score:.8f}; {latency["throughput_median_us_per_point"]:.3f} us/point',flush=True)
    report={'status':'COMPLETE','experiment_id':eid,'model':args.model,'families':args.families,'sampling':args.sampling,
        'phase':'historical_null_normalization','timestamp':utc_now(),'hypothesis':'Heterogeneous historical dependence/tails distort iid-style feature normalization; standardize using each historical null replay.',
        'representation_hash':implementation_hash(policy),'policy':policy.__dict__,'split_sha256':split.digest,
        'elapsed_seconds':time.perf_counter()-start,'latencies':latencies,'latency_us':float(np.median([x['throughput_median_us_per_point'] for x in latencies])),
        'cv_rows':rows,'fractures':fractures,'LOCAL_FINAL_SEAL_OPENED':False,'reduced_scored_uses':0,**summarize(scores)}
    write_json(result_path,report);print('COMPLETE',eid,report['mean'],report['median'],report['worst_fold'],flush=True)

if __name__=='__main__':main()
