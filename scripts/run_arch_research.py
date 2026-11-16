"""Two fixed small learners test historical conditional-variance normalization."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='4'
import gc,json,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.features.config import CONFIG
from src.features.historical_calibration import CalibrationPolicy
from src.features.arch_input import ARCH_POLICY,implementation_hash
from src.features.arch_cache import build_arch_cache
from src.features.streaming import feature_names
from src.models.arch_calibrated import ARCHCalibratedBundle
from src.validation.adaptive import ResearchBudget,promotion_decision
from src.validation.splits import FrozenSplit
from src.validation.research import fit_fold,summarize
from src.validation.diagnostics import profile
from src.data.loader import load_training
from src.utils.artifacts import write_json,sha256,utc_now,ledger
from scripts.run_real_research import admission
from scripts.verify_engineering import benchmark


def main():
    budget=ResearchBudget(ROOT/'artifacts/validation/REAL_SESSION.json')
    if not budget.can_start(1800):raise RuntimeError('Final audit buffer protected')
    admission(raw_directory=ROOT/'data/raw');split=FrozenSplit(ROOT/'artifacts/validation')
    registry=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text());anchor=next(r for r in registry['results'] if r['experiment_id']==ARCH_POLICY['anchor'])
    spec=anchor['candidate'];config=CONFIG.variant(**{k:tuple(v) if isinstance(v,list) else v for k,v in spec['feature_changes'].items()})
    calibration=CalibrationPolicy(method='historical_median_mad');version=implementation_hash(calibration,config)
    frozen={'timestamp':utc_now(),'settings':ARCH_POLICY,'anchor_candidate':spec,'representation_hash':version,'split_sha256':split.digest,
        'hypothesis':'Across all DEV historical segments, 17.225 percent of AR8 squared residual sequences retain lag1 correlation above 0.1. Divide each innovation by a frozen historical ARCH1 forecast using only its previous innovation, then retain the original calibrated feature bank and S3 learner.',
        'risk':'Conditional scaling can absorb a true scale shift; report all folds and early slices, reject if the fixed promotion gates fail.',
        'models_fixed_before_scores':True,'source_sha256':sha256(__file__),'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    policy_path=ROOT/'artifacts/real/ARCH_RESEARCH_POLICY.json'
    if policy_path.exists():raise RuntimeError('ARCH policy already recorded; use explicit reproduction or recovery after inspecting the ledger')
    write_json(policy_path,frozen);cache=build_arch_cache(ROOT/'data/raw',ROOT/'data/cache',split,config,calibration)
    folds=[pd.read_parquet(cache/f'fold_{k}.parquet') for k in range(5)];names=feature_names(config)
    ids=split.frame.loc[split.frame.role=='DEVELOPMENT_POOL','dataset_id'].tolist();metadata=pd.read_parquet(ROOT/'artifacts/audits/SERIES_METADATA.parquet',filters=[('dataset_id','in',ids)])
    first={}
    for item in load_training(ROOT/'data/raw',split=split):
        fold=split.assignment(item.dataset_id)[1]
        if fold not in first:first[fold]=item
        if len(first)==5:break
    for model_name in ARCH_POLICY['models']:
        if not budget.can_start(600):break
        eid=f'{model_name}_AR8_ARCH1_ABCD_S3_{version[:12]}';path=ROOT/'artifacts/real'/f'{eid}.json'
        if path.exists():raise RuntimeError('ARCH candidate has already been evaluated')
        report={'status':'RUNNING','experiment_id':eid,'model':model_name+'_AR8_ARCH1','families':'ABCD','sampling':'S3','phase':'conditional_variance_input',
            'representation':'ar8_arch1_historical_median_mad','feature_hash':version,'anchor':anchor['experiment_id'],'anchor_candidate':spec,'learner_name':model_name,
            'hypothesis':frozen['hypothesis'],'fold_scores':[],'cv_rows':[],'fractures':[],'latencies':[],'save_load_max_abs_diff':0.,'live_cached_prediction_max_abs_diff':0.,
            'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False};started=time.perf_counter()
        try:
            for fold,validation in enumerate(folds):
                base,prediction,value=fit_fold(folds,fold,model_name,'ABCD','S3',split=split,params={},config=config)
                bundle=ARCHCalibratedBundle(base,calibration,config);model_path=ROOT/'artifacts/models/real'/eid/f'fold_{fold}.joblib';bundle.save(model_path);loaded=ARCHCalibratedBundle.load(model_path)
                np.testing.assert_array_equal(loaded.predict(validation.loc[:,names].to_numpy(dtype=np.float32)).astype(np.float32),prediction)
                sample=first[fold];stream=loaded.make_state(sample.historical);live=np.array([loaded.predict_one(stream.update_and_get(p)) for p in sample.online],dtype=np.float32)
                np.testing.assert_array_equal(live,prediction[validation.dataset_id.to_numpy()==sample.dataset_id])
                def factory():
                    stream=loaded.make_state(sample.historical)
                    return lambda p:loaded.predict_one(stream.update_and_get(p))
                points=np.tile(sample.online.astype(np.float32),max(1,3000//len(sample.online)+1))[:3000]
                latency=benchmark(factory,points,repeats=3);latency['model_bytes']=model_path.stat().st_size;report['latencies'].append(latency)
                report['fold_scores'].append(value);report['cv_rows'].append({'experiment_id':eid,'model':report['model'],'families':'ABCD','sampling':'S3','fold':fold,'ts_auc':value})
                report['fractures'].extend([{'experiment_id':eid,'fold':fold,**r} for r in profile(validation.target,prediction,validation.time_online,validation.dataset_id,metadata)])
                oof=ROOT/'data/processed/oof'/eid;oof.mkdir(parents=True,exist_ok=True);np.save(oof/f'fold_{fold}.npy',prediction,allow_pickle=False)
                write_json(oof/'MANIFEST.json',{'cache_directory':str(cache.relative_to(ROOT)),'split_sha256':split.digest,'representation':report['representation'],
                    'feature_hash':version,'wire_dtype':'float32','seal_rows':0})
                write_json(ROOT/'artifacts/real'/f'{eid}_CHECKPOINT.json',report);print('ARCH',model_name,'fold',fold,value,flush=True)
                del base,bundle,loaded,prediction;gc.collect()
            report.update(summarize(report['fold_scores']));report['status']='COMPLETE';report['latency_us']=float(np.median([r['throughput_median_us_per_point'] for r in report['latencies']]))
            report['versus_original_ar8_anchor']=promotion_decision(report,anchor)
        except Exception as exc:
            report['status']='FAILED';report['exception_type']=type(exc).__name__
            (ROOT/'experiments/logs'/f'{eid}_failure.log').write_text(traceback.format_exc(),encoding='utf-8')
            ledger(experiment_id=eid,model=report['model'],status='FAILED',notes=f'Conditional variance input; {type(exc).__name__}; no partial-fold ranking')
        report['elapsed_seconds']=time.perf_counter()-started;write_json(path,report)
        print('ARCH RESULT',report['status'],eid,report.get('mean'),report.get('median'),report.get('worst_fold'),flush=True)


if __name__=='__main__':main()
