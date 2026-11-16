"""Separate historical initialization from point cost over every real DEV series."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,gc,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.loader import series_frames
from src.validation.splits import FrozenSplit
from src.models.representations import bundle_class
from src.streaming.prepared import prepare_model,source_hashes
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',action='append',required=True);parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.repeats<=3:raise ValueError('At most three fixed runtime repetitions')
    names={p.name for p in (ROOT/'scripts').glob('*.py')};ancestors={p.pid for p in psutil.Process().parents()};active=[]
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            command=process.info['cmdline'] or []
            if process.pid!=os.getpid() and process.pid not in ancestors and (any(Path(v).name in names for v in command) or 'pytest' in command):active.append(process.pid)
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    if active:raise RuntimeError('Another project worker is active; isolated full-DEV runtime refused')
    directory=ROOT/'artifacts/real/latency/full_dev_distribution'
    if (directory/'POLICY.json').exists():raise RuntimeError('Runtime policy already exists')
    split=FrozenSplit(ROOT/'artifacts/validation');examples=[];offsets=[0]*5;ids=[[] for _ in range(5)]
    for sid,frame in series_frames(ROOT/'data/raw/X_train.parquet'):
        role,fold=split.assignment(sid)
        if role!='DEVELOPMENT_POOL':continue
        historical=frame.loc[frame.period==1,'value'].to_numpy(dtype=np.float32);online=frame.loc[frame.period==2,'value'].to_numpy(dtype=np.float32)
        examples.append((sid,fold,historical,online,offsets[fold]));offsets[fold]+=len(online);ids[fold].append(sid)
    if {row[0] for row in examples}!=set(split.frame.loc[split.frame.role=='DEVELOPMENT_POOL','dataset_id']):raise AssertionError('Incomplete DEV input coverage')
    state=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text());prepared={};models={};expected={};receipts={}
    for eid in args.candidate:
        result=next(r for r in state['results'] if r['experiment_id']==eid);loader=bundle_class(result['candidate'].get('representation','base'))
        models[eid]=[loader.load(ROOT/'artifacts/models/real'/eid/f'fold_{k}.joblib') for k in range(5)]
        prepared[eid]=[prepare_model(model,fast_initialization=True) for model in models[eid]]
        oof=ROOT/'data/processed/oof'/eid;manifest=json.loads((oof/'MANIFEST.json').read_text());cache=ROOT/manifest['cache_directory']
        if manifest['split_sha256']!=split.digest:raise AssertionError('OOF split differs')
        expected[eid]=[];receipts[eid]=[]
        for fold in range(5):
            coords=pd.read_parquet(cache/f'fold_{fold}.parquet',columns=['dataset_id','time_online'])
            if coords.dataset_id.drop_duplicates().tolist()!=ids[fold]:raise AssertionError('OOF series order differs')
            for _,part in coords.groupby('dataset_id',sort=False):
                if not np.array_equal(part.time_online.to_numpy(),np.arange(len(part))):raise AssertionError('OOF online coordinates differ')
            values=np.load(oof/f'fold_{fold}.npy',allow_pickle=False)
            if len(values)!=offsets[fold]:raise AssertionError('OOF point count differs')
            expected[eid].append(values)
            receipts[eid].append({'fold':fold,'model_sha256':sha256(ROOT/'artifacts/models/real'/eid/f'fold_{fold}.joblib'),**source_hashes(models[eid][fold],fast_initialization=True)})
            prepared[eid][fold].new_state(examples[0][2]).predict_one(0.)
    policy={'timestamp':utc_now(),'candidates':args.candidate,'repeats':args.repeats,'series':len(examples),'points_per_candidate_pass':sum(offsets),
        'ordering':'All actual DEV series per candidate per repetition; candidate order reverses on alternate repetitions. Every series uses its own frozen validation-fold model.',
        'timing':'Input arrays and models loaded before timing; kernels warmed. Historical state initialization and scalar-point prediction/collection timed separately for each series.',
        'isolation':'No other known project script or pytest process active at start. No external application activity or clock-frequency lock is assumed.',
        'not_final_deployment_timing':True,'no_performance_metric_computed':True,'split_sha256':split.digest,'script_sha256':sha256(__file__),
        'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    write_json(directory/'POLICY.json',policy);rows=[];process=psutil.Process();rss=[];start=time.perf_counter()
    for repetition in range(args.repeats):
        order=args.candidate if repetition%2==0 else list(reversed(args.candidate))
        for eid in order:
            for number,(sid,fold,historical,online,offset) in enumerate(examples,1):
                started=time.perf_counter_ns();detector=prepared[eid][fold].new_state(historical);init=(time.perf_counter_ns()-started)/1e9
                before=detector.state_array_bytes
                started=time.perf_counter_ns();prediction=np.array([detector.predict_one(p) for p in online],dtype=np.float32);point=(time.perf_counter_ns()-started)/1e9
                if before!=detector.state_array_bytes:raise AssertionError('Online state grew')
                np.testing.assert_array_equal(prediction,expected[eid][fold][offset:offset+len(online)])
                rows.append({'candidate':eid,'repetition':repetition,'dataset_id':sid,'fold':fold,'historical_length':len(historical),'online_length':len(online),
                    'initialization_seconds':init,'prediction_seconds':point,'amortized_us_per_point':(init+point)/len(online)*1e6})
                if number%2000==0:print('FULL DEV RUNTIME',eid,repetition,number,'series',flush=True)
            gc.collect();rss.append({'candidate':eid,'repetition':repetition,'rss_bytes':process.memory_info().rss})
            pd.DataFrame(rows).to_csv(directory/'CHECKPOINT_TIMINGS.csv',index=False)
    frame=pd.DataFrame(rows);summaries=[]
    for eid,part in frame.groupby('candidate',sort=False):
        median=part.groupby(['dataset_id','fold','historical_length','online_length'],as_index=False)[['initialization_seconds','prediction_seconds','amortized_us_per_point']].median()
        median.to_csv(directory/(eid+'_SERIES_MEDIAN.csv'),index=False)
        passes=[]
        for repetition,batch in part.groupby('repetition'):
            init=float(batch.initialization_seconds.sum());point=float(batch.prediction_seconds.sum());n=int(batch.online_length.sum())
            passes.append({'repetition':int(repetition),'historical_seconds':init,'prediction_seconds':point,'points':n,'amortized_us_per_point':(init+point)/n*1e6,'initialization_fraction':init/(init+point)})
        summaries.append({'candidate':eid,'series':len(median),'passes':passes,'model_receipts':receipts[eid],
            'historical_initialization_seconds_quantiles':dict(zip(['min','p50','p90','p95','p99','max'],np.quantile(median.initialization_seconds,[0,.5,.9,.95,.99,1]).tolist())),
            'per_series_amortized_us_quantiles':dict(zip(['min','p50','p90','p95','p99','max'],np.quantile(median.amortized_us_per_point,[0,.5,.9,.95,.99,1]).tolist())),
            'median_pass_amortized_us_per_point':float(np.median([r['amortized_us_per_point'] for r in passes])),
            'median_pass_initialization_fraction':float(np.median([r['initialization_fraction'] for r in passes]))})
    report={'status':'PASS','timestamp':utc_now(),'policy':policy,'summaries':summaries,'rss_after_pass':rss,'total_elapsed_seconds':time.perf_counter()-start,
        'different_float32_rows':0,'timed_prediction_rows':len(args.candidate)*args.repeats*sum(offsets),'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    write_json(directory/'RESULTS.json',report);print('FULL DEV RUNTIME COMPLETE',report['timed_prediction_rows'],'points',report['total_elapsed_seconds'],'s',flush=True)


if __name__=='__main__':main()
