"""One prospectively fixed equal average of AR4 LightGBM and AR8 HistGB."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import gc,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.representations import bundle_class
from src.features.streaming import feature_names
from src.validation.adaptive import ResearchBudget,promotion_decision
from src.validation.splits import FrozenSplit,assert_development_ids
from src.validation.research import summarize
from src.validation.diagnostics import profile
from src.scoring.ts_auc import ts_auc
from src.data.loader import load_training
from src.utils.artifacts import write_json,sha256,utc_now
from scripts.verify_engineering import benchmark

COMPONENTS=('M4_ar_residual_input_ABCD_S3_9f68be316ea3','M3_ar_order8_ABCD_S3_242216e87a34')
EID='M7_cross_order_AR4_LGB_AR8_HGB_equal'


def main():
    if not ResearchBudget(ROOT/'artifacts/validation/REAL_SESSION.json').can_start(600):raise RuntimeError('New experiment cutoff protected')
    split=FrozenSplit(ROOT/'artifacts/validation');directory=ROOT/'artifacts/real/cross_order_blend'
    if (directory/'RESULTS.json').exists():raise RuntimeError('The one cross-order blend has already been evaluated')
    registry=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text())
    refs=[next(r for r in registry['results'] if r['experiment_id']==eid) for eid in COMPONENTS]
    manifests=[json.loads((ROOT/'data/processed/oof'/eid/'MANIFEST.json').read_text()) for eid in COMPONENTS]
    if any(m['split_sha256']!=split.digest for m in manifests):raise RuntimeError('Component split differs')
    caches=[ROOT/m['cache_directory'] for m in manifests]
    policy={'timestamp':utc_now(),'components':COMPONENTS,'weights':[.5,.5],'maximum_candidates':1,
        'formula':'float32(0.5*float64(float32(AR4_probability)) + 0.5*float64(float32(AR8_probability)))',
        'hypothesis':'AR8 HistGB improved every fold while a different AR order and tree learner may leave complementary errors. Test one equal average; do not fit weights or a stacking model.',
        'split_sha256':split.digest,'source_sha256':sha256(__file__),'frozen_before_blend_score':True,
        'component_oof_sha256':{eid:{str(f):sha256(ROOT/'data/processed/oof'/eid/f'fold_{f}.npy') for f in range(5)} for eid in COMPONENTS}}
    if (directory/'POLICY.json').exists():raise RuntimeError('Existing policy requires an explicit reproduction command')
    write_json(directory/'POLICY.json',policy)
    ids=split.frame.loc[split.frame.role=='DEVELOPMENT_POOL','dataset_id'].tolist()
    metadata=pd.read_parquet(ROOT/'artifacts/audits/SERIES_METADATA.parquet',filters=[('dataset_id','in',ids)])
    first={}
    for series in load_training(ROOT/'data/raw',split=split):
        fold=split.assignment(series.dataset_id)[1]
        if fold not in first:first[fold]=series
        if len(first)==5:break
    result={'experiment_id':EID,'model':'M7_fixed_average','families':'ABCD','sampling':'S3','phase':'cross_order_fixed_average',
        'representation':'two_frozen_AR_representations','components':list(COMPONENTS),'partner_weight':.5,
        'fold_scores':[],'cv_rows':[],'fractures':[],'latencies':[],'live_oof_max_abs_diff':0.,'reloaded_component_oof_max_abs_diff':0.,
        'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False,'deployment_status':'RESEARCH_ONLY_PENDING_SCORE_GATES'}
    started=time.perf_counter()
    for fold in range(5):
        frames=[];models=[];predictions=[]
        for eid,ref,cache in zip(COMPONENTS,refs,caches):
            manifest=json.loads((cache/'MANIFEST.json').read_text());path=cache/f'fold_{fold}.parquet'
            if sha256(path)!=manifest['fold_sha256'][str(fold)]:raise RuntimeError('Component feature cache changed')
            frame=pd.read_parquet(path);assert_development_ids(frame.dataset_id.unique(),split)
            model=bundle_class(ref['representation']).load(ROOT/'artifacts/models/real'/eid/f'fold_{fold}.joblib')
            matrix=frame.loc[:,feature_names(model.feature_config)].to_numpy(dtype=np.float32)
            original=np.load(ROOT/'data/processed/oof'/eid/f'fold_{fold}.npy',allow_pickle=False)
            np.testing.assert_array_equal(model.predict(matrix).astype(np.float32),original)
            frames.append(frame);models.append(model);predictions.append(original);del matrix
        for column in ('dataset_id','time_online','target'):np.testing.assert_array_equal(frames[0][column],frames[1][column])
        prediction=(.5*predictions[0].astype(np.float64)+.5*predictions[1].astype(np.float64)).astype(np.float32)
        sample=first[fold];h=sample.historical.astype(np.float32)
        def factory():
            streams=[model.make_state(h) for model in models]
            def predict_one(point):
                probabilities=[float(np.float32(model.predict_one(stream.update_and_get(point)))) for model,stream in zip(models,streams)]
                return float(np.float32(.5*probabilities[0]+.5*probabilities[1]))
            return predict_one
        detector=factory();actual=np.array([detector(p) for p in sample.online],dtype=np.float32)
        np.testing.assert_array_equal(actual,prediction[frames[0].dataset_id.to_numpy()==sample.dataset_id])
        points=np.tile(sample.online,max(1,3000//len(sample.online)+1)).astype(np.float32)[:3000]
        latency=benchmark(factory,points,repeats=3);latency['model_bytes']=sum((ROOT/'artifacts/models/real'/eid/f'fold_{fold}.joblib').stat().st_size for eid in COMPONENTS)
        result['latencies'].append(latency)
        value=ts_auc(frames[0].target,prediction,frames[0].time_online);result['fold_scores'].append(value)
        result['cv_rows'].append({'experiment_id':EID,'model':result['model'],'families':'ABCD','sampling':'S3','fold':fold,'ts_auc':value})
        result['fractures'].extend([{'experiment_id':EID,'fold':fold,**r} for r in profile(frames[0].target,prediction,frames[0].time_online,frames[0].dataset_id,metadata)])
        oof=ROOT/'data/processed/oof'/EID;oof.mkdir(parents=True,exist_ok=True);np.save(oof/f'fold_{fold}.npy',prediction,allow_pickle=False)
        write_json(oof/'MANIFEST.json',{'cache_directory':str(caches[0].relative_to(ROOT)),'component_cache_directories':[str(c.relative_to(ROOT)) for c in caches],
            'split_sha256':split.digest,'components':list(COMPONENTS),'weights':[.5,.5],'wire_dtype':'float32','formula':policy['formula'],'seal_rows':0})
        print('CROSS ORDER BLEND',fold,value,flush=True)
        del frames,models,predictions,prediction;gc.collect()
    result.update(summarize(result['fold_scores']));result['status']='COMPLETE'
    result['latency_us']=float(np.median([v['throughput_median_us_per_point'] for v in result['latencies']]))
    result['elapsed_seconds']=time.perf_counter()-started;result['versus_frozen_primary']=promotion_decision(result,refs[0])
    write_json(directory/'RESULTS.json',{'status':'COMPLETE','timestamp':utc_now(),'policy':policy,'elapsed_seconds':result['elapsed_seconds'],'results':[result]})
    print('CROSS ORDER COMPLETE',result['mean'],result['median'],result['worst_fold'],result['versus_frozen_primary'],flush=True)


if __name__=='__main__':main()
