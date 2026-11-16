"""Three predeclared sequential mechanisms and fixed-weight OOF complements."""
import sys,json,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.sequential_detectors import SequentialDetectors,POLICY,NAMES,implementation_hash
from src.models.calibrated import CalibratedBundle
from src.data.loader import load_training
from src.data.labels import labels_from_tau
from src.validation.splits import FrozenSplit
from src.validation.research import summarize
from src.validation.diagnostics import profile
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import write_json,utc_now,sha256
from scripts.verify_engineering import benchmark

ANCHOR='M4_lightgbm_ABCD_S3_historical_mean_std_f4ddb57ad049'
BLEND_WEIGHTS=(.05,.15)

def main():
    output=ROOT/'artifacts/real/COMPLEMENTARY_DETECTORS_REAL.json'
    if output.exists():raise RuntimeError('This frozen mechanism comparison already completed')
    freeze={'timestamp':utc_now(),'anchor':ANCHOR,'detector_policy':POLICY.__dict__,'detector_hash':implementation_hash(),
        'blend_weights':list(BLEND_WEIGHTS),'inputs':'historical-only initialization and current scalar point',
        'blend_rule':'float32-quantize each component, then float64 weighted mean, then official float32 wire quantization',
        'score_seen_when_frozen':False}
    frozen_path=ROOT/'artifacts/real/COMPLEMENTARY_POLICY.json'
    if frozen_path.exists():
        prior=json.loads(frozen_path.read_text());assert prior['anchor']==ANCHOR and prior['detector_hash']==implementation_hash()
    else:write_json(frozen_path,freeze)
    split=FrozenSplit(ROOT/'artifacts/validation');buffers={k:[] for k in range(5)};sample=None;start=time.perf_counter()
    for number,series in enumerate(load_training(ROOT/'data/raw',split=split),1):
        h=series.historical.astype(np.float32);o=series.online.astype(np.float32)
        if sample is None:sample=(h,o)
        state=SequentialDetectors(h);pred=np.array([state.update_and_get(x).copy() for x in o],dtype=np.float32)
        frame=pd.DataFrame(pred,columns=NAMES);frame['dataset_id']=series.dataset_id;frame['time_online']=np.arange(len(o),dtype=np.int32);frame['target']=labels_from_tau(len(o),series.tau)
        buffers[split.assignment(series.dataset_id)[1]].append(frame)
        if number%2000==0:print(f'Complementary extraction: {number} DEV series',flush=True)
    folds=[pd.concat(buffers[k],ignore_index=True) for k in range(5)]
    metadata=pd.read_parquet(ROOT/'artifacts/audits/SERIES_METADATA.parquet')
    metadata=metadata.loc[metadata.dataset_id.isin({sid for f in folds for sid in f.dataset_id.unique()})]
    h,o=sample;points=np.tile(o,max(1,3000//len(o)+1))[:3000]
    anchor_oof=ROOT/'data/processed/oof'/ANCHOR
    anchor_manifest=json.loads((anchor_oof/'MANIFEST.json').read_text());anchor_cache=ROOT/anchor_manifest['cache_directory']
    anchor_models=[CalibratedBundle.load(ROOT/'artifacts/models/real'/ANCHOR/f'fold_{k}.joblib') for k in range(5)]
    anchor_predictions=[]
    for fold,frame in enumerate(folds):
        alignment=pd.read_parquet(anchor_cache/f'fold_{fold}.parquet',columns=['dataset_id','time_online','target'])
        if not np.array_equal(alignment.to_numpy(),frame[['dataset_id','time_online','target']].to_numpy()):raise RuntimeError('OOF row alignment differs')
        anchor_predictions.append(np.load(anchor_oof/f'fold_{fold}.npy',allow_pickle=False))
    summaries=[];cv=[];fractures=[];latencies={}
    detector_oof=ROOT/'data/processed/oof'/('sequential_'+implementation_hash()[:12]);detector_oof.mkdir(parents=True,exist_ok=True)
    for fold,frame in enumerate(folds):np.save(detector_oof/f'fold_{fold}.npy',frame.loc[:,NAMES].to_numpy(dtype=np.float32),allow_pickle=False)
    for j,name in enumerate(NAMES):
        for weight in (1.,*BLEND_WEIGHTS):
            eid=name if weight==1 else f'{ANCHOR}_plus_{name}_w{weight}'
            scores=[]
            for fold,frame in enumerate(folds):
                prediction=frame[name].to_numpy(dtype=np.float32)
                if weight!=1:prediction=((1-weight)*anchor_predictions[fold].astype(np.float64)+weight*prediction.astype(np.float64)).astype(np.float32)
                score=ts_auc(frame.target,prediction,frame.time_online);scores.append(score)
                cv.append({'experiment_id':eid,'fold':fold,'ts_auc':score})
                fractures.extend([{'experiment_id':eid,'fold':fold,**r} for r in profile(frame.target,prediction,frame.time_online,frame.dataset_id,metadata)])
            def factory():
                detector=SequentialDetectors(h)
                if weight==1:return lambda point:float(detector.update_and_get(point)[j])
                model=anchor_models[0];state=model.make_state(h)
                def predict(point):
                    a=float(np.float32(model.predict_one(state.update_and_get(point))))
                    b=float(detector.update_and_get(point)[j])
                    return float(np.float32((1-weight)*a+weight*b))
                return predict
            latency=benchmark(factory,points,repeats=3);latencies[eid]=latency
            summaries.append({'experiment_id':eid,'model':name,'anchor':None if weight==1 else ANCHOR,'weight':weight,'latency_us':latency['throughput_median_us_per_point'],**summarize(scores)})
            print(eid, 'mean',summaries[-1]['mean'],'worst',summaries[-1]['worst_fold'],flush=True)
    report={'status':'COMPLETE','timestamp':utc_now(),'policy_sha256':sha256(frozen_path),'detector_hash':implementation_hash(),'split_sha256':split.digest,
        'elapsed_seconds':time.perf_counter()-start,'summaries':summaries,'cv_rows':cv,'fractures':fractures,'latencies':latencies,
        'trained_on_validation_labels':False,'all_blend_weights_predeclared':True,'LOCAL_FINAL_SEAL_OPENED':False,'reduced_scored_uses_in_this_script':0}
    write_json(output,report)
    pd.DataFrame(summaries).to_csv(ROOT/'artifacts/real/COMPLEMENTARY_RESULTS_REAL.csv',index=False)
    print('COMPLEMENTARY COMPLETE',flush=True)

if __name__=='__main__':main()
