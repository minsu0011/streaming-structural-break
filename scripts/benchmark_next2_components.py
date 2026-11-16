"""Isolated diagnostic costs for existing full-CV component models on fixed held-out IDs."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import argparse
import json
import time
import hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_next2_runtime import assert_isolated,distribution
from src.next.data import prepare_dev
from src.next2.evaluation import load_full_oof
from src.next2.io import model_class
from src.next2.runtime import prepare,PreparedBankPredictor
from src.next2.profiling_parts import Parts
from src.next2.numeric_memory import accounting
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);args=parser.parse_args()
    assert_isolated()
    path=ROOT/'configs/next2/COMPONENT_MEASUREMENT_LOCK.json';policy=json.loads(path.read_text())
    assert policy['driver_sha256']==sha256(Path(__file__))
    for name,digest in policy['source_sha256'].items():assert sha256(ROOT/name)==digest
    ref=next(r for r in policy['models'] if r['experiment_id']==args.candidate)
    model_path=ROOT/ref['model_path'];assert sha256(model_path)==ref['model_sha256']
    out=ROOT/'artifacts/next2/component_latency'/args.candidate;out.mkdir(parents=True,exist_ok=True)
    identity={'candidate':args.candidate,'policy_sha256':sha256(path),'model_sha256':ref['model_sha256']}
    if (out/'RESULTS.json').exists():
        r=json.loads((out/'RESULTS.json').read_text());assert r['status']=='PASS' and r['identity']==identity
        print('NEXT2 COMPONENT LATENCY RETAINED',args.candidate,flush=True);return
    if (out/'START.json').exists():raise RuntimeError('Interrupted timing requires an explicit new measurement revision')
    write_json(out/'START.json',{'status':'RUNNING','identity':identity,'created_utc':utc_now()})
    data=prepare_dev(ROOT);data.guard.admit(policy['series_ids'],purpose='Fixed held-out component runtime probes')
    assert all(data.guard.split.assignment(i)[1]==0 for i in policy['series_ids'])
    candidate,original,metadata,expected,sources=load_full_oof(data,args.candidate)
    positions={int(sid):j for j,sid in enumerate(data.ids)};oo=data.series['online_offsets']
    examples=[(sid,*data.get(sid)[:2]) for sid in policy['series_ids']]
    tick=time.perf_counter();model=model_class(candidate).load(model_path)
    engine=PreparedBankPredictor(model) if args.candidate=='D218_FULL_PIPELINE_NO_ARCH' else prepare(model)
    load_seconds=time.perf_counter()-tick
    _,h0,o0=examples[0];tick=time.perf_counter();warm=engine.new_state(h0);warm.predict_one(o0[0]);warmup=time.perf_counter()-tick
    point_times=[];init_times=[];features=[];trees=[];records=[];sampled={};sizes=[]
    for sid,h,o in examples:
        assert_isolated();tick=time.perf_counter_ns();state=engine.new_state(h);init=time.perf_counter_ns()-tick
        actual=np.empty(len(o),dtype=np.float32);durations=np.empty(len(o),dtype=np.int64)
        start=time.perf_counter_ns()
        for j,point in enumerate(o):
            tick=time.perf_counter_ns();actual[j]=state.predict_one(point);durations[j]=time.perf_counter_ns()-tick
        elapsed=time.perf_counter_ns()-start
        index=positions[sid];np.testing.assert_array_equal(actual,expected[oo[index]:oo[index+1]])
        point_times.extend(durations);init_times.append(init);sampled[sid]=actual;sizes.append(state.state_array_bytes)
        records.append({'candidate':args.candidate,'dataset_id':sid,'historical_points':len(h),'online_points':len(o),
            'historical_init_ns':init,'online_loop_ns':elapsed,'point_median_ns':float(np.median(durations)),'state_array_bytes':state.state_array_bytes})
    parts=Parts(engine.new_state(h0));parts.feature_update(o0[0]);parts.model_predict()
    for sid,h,o in examples:
        assert_isolated();parts=Parts(engine.new_state(h))
        for j,point in enumerate(o):
            tick=time.perf_counter_ns();parts.feature_update(point);middle=time.perf_counter_ns();p=parts.model_predict();end=time.perf_counter_ns()
            features.append(middle-tick);trees.append(end-middle);assert p==sampled[sid][j]
    buffer=np.concatenate([o for _,h,o in examples])[:10000];assert len(buffer)==10000
    state=engine.new_state(h0);initial=state.state_array_bytes;elapsed=0.;digest=hashlib.sha256()
    for block in range(100):
        tick=time.perf_counter_ns();p=np.asarray([state.predict_one(v) for v in buffer],dtype=np.float32);elapsed+=(time.perf_counter_ns()-tick)/1e9
        assert np.isfinite(p).all();digest.update(p.tobytes())
        if block%10==0:assert_isolated()
    assert state.state_array_bytes==initial
    memory,_,_=accounting(state)
    table=pd.DataFrame(records);table.to_csv(out/'SERIES_TIMING.csv',index=False)
    amortized=(table.historical_init_ns.sum()+table.online_loop_ns.sum())/table.online_points.sum()/1000.
    write_json(out/'RESULTS.json',{'status':'PASS','identity':identity,'created_utc':utc_now(),
        'artifact_kind':'existing full-CV fold-0 model; isolated component cost diagnostic','full_cv_status':original['status'],
        'point_distribution':distribution(point_times),'historical_initialization':distribution(init_times),
        'feature_state_update':distribution(features),'model_prediction':distribution(trees),
        'h_plus_online_amortized_us_per_point':float(amortized),'model_load_seconds':load_seconds,'first_history_plus_jit_seconds':warmup,
        'direct_1M_seconds':elapsed,'extrapolated_10M_seconds':10.*elapsed,'direct_10M_measured':False,
        'continuous_output_sha256':digest.hexdigest(),'state_array_bytes_min':min(sizes),'state_array_bytes_max':max(sizes),
        'numerical_memory':memory,'model_file_bytes':model_path.stat().st_size,'heldout_probe_series':len(examples),
        'heldout_probe_points':len(point_times),'heldout_oof_probe_exact':True,'separate_parts_exact':True,
        'source_oof_sha256':sources,'table_sha256':sha256(out/'SERIES_TIMING.csv'),'new_fits':0,'seal_rows':0,
        'limitations':'One fixed real-prefix pass; no timing search. 10M component estimate is extrapolated from directly measured 1M. Selected all-DEV production variants have a separate direct 10M protocol. Ordinary OS/power load uncontrolled.'})
    checkpoint('Isolated component cost '+args.candidate,['Remaining fixed component costs','Synthesize runtime and scientific tradeoffs'])
    print('NEXT2 COMPONENT LATENCY PASS',args.candidate,distribution(point_times)['median_us'],flush=True)


if __name__=='__main__':
    main()
