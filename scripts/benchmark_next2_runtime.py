"""Isolated fixed-protocol scalar/H/component and directly measured 10M profiling."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import argparse
import json
import time
import gc
import hashlib
import numpy as np
import pandas as pd
import psutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next2.io import model_class
from src.next2.canonical_model import load as canonical_load
from src.next2.pruned_runtime import PreparedCompactPredictor
from src.next2.used_feature_runtime import PreparedUsedPredictor
from src.next2.runtime import prepare
from src.next.serial_variance_runtime import PreparedSerialVariancePredictor
from src.next.fast_variance_initialization import PreparedFastVariancePredictor
from src.next2.profiling_parts import Parts
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def assert_isolated():
    excluded={os.getpid(),*(p.pid for p in psutil.Process().parents())}
    active=[]
    for process in psutil.process_iter(['pid','name']):
        if process.pid in excluded or not (process.info['name'] or '').lower().startswith('python'):
            continue
        try:
            cwd=Path(process.cwd()).resolve()
            if cwd==ROOT or ROOT in cwd.parents:
                active.append(process.pid)
        except (psutil.AccessDenied,psutil.NoSuchProcess):
            continue
    if active:
        raise RuntimeError('Isolated NEXT2 timing refused while project Python workers are active: '+str(active))


def distribution(ns):
    values=np.asarray(ns,dtype=np.float64)/1000.
    return {'samples':len(values),'median_us':float(np.median(values)),'p95_us':float(np.quantile(values,.95)),
        'p99_us':float(np.quantile(values,.99)),'mean_us':float(values.mean())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',required=True)
    args=parser.parse_args()
    assert_isolated()
    policy_path=ROOT/'configs/next2/RUNTIME_MEASUREMENT_LOCK.json'
    policy=json.loads(policy_path.read_text())
    assert policy['driver_sha256']==sha256(Path(__file__))
    for name,digest in policy['source_sha256'].items():
        assert sha256(ROOT/name)==digest
    recipe=policy['variants'][args.variant]
    path=ROOT/recipe['model_path']
    assert sha256(path)==recipe['model_sha256']
    identity={'variant':args.variant,'recipe':recipe,'policy_sha256':sha256(policy_path),'driver_sha256':sha256(Path(__file__))}
    out=ROOT/'artifacts/next2/latency'/args.variant
    out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULTS.json').exists():
        r=json.loads((out/'RESULTS.json').read_text())
        assert r['status']=='PASS' and r['identity']==identity
        print('NEXT2 LATENCY RETAINED',args.variant,flush=True)
        return
    if (out/'START.json').exists():
        raise RuntimeError('Interrupted benchmark requires a documented new measurement revision, not silent timing selection')
    write_json(out/'START.json',{'identity':identity,'created_utc':utc_now(),'status':'RUNNING'})
    data=prepare_dev(ROOT)
    data.guard.admit(policy['series_ids'],purpose='Fixed NEXT2 scalar runtime probes; no quality scores')
    examples=[(sid,*data.get(sid)[:2]) for sid in policy['series_ids']]
    tick=time.perf_counter()
    model=canonical_load(path) if recipe['model_format']=='NEXT2_CANONICAL_V1' else model_class(recipe['candidate']).load(path)
    load_seconds=time.perf_counter()-tick
    constructors={'original_serial':PreparedSerialVariancePredictor,'original_variance':PreparedFastVariancePredictor,
        'compact':PreparedCompactPredictor,'tree_used':PreparedUsedPredictor,'prepared':prepare}
    prepared=constructors[recipe['runtime']](model)
    _,first_h,first_o=examples[0]
    tick=time.perf_counter()
    warm=prepared.new_state(first_h);warm.predict_one(first_o[0])
    first_state_seconds=time.perf_counter()-tick
    del warm
    sampled={};point_times=[];h_times=[];series_rows=[];sizes=[]
    for repeat in range(policy['real_prefix_repeats']):
        for sid,h,o in examples:
            assert_isolated()
            before=time.perf_counter_ns();state=prepared.new_state(h);initialized=time.perf_counter_ns()
            actual=np.empty(len(o),dtype=np.float32);durations=np.empty(len(o),dtype=np.int64)
            loop_start=time.perf_counter_ns()
            for j,v in enumerate(o):
                tick=time.perf_counter_ns();actual[j]=state.predict_one(v);durations[j]=time.perf_counter_ns()-tick
            loop_ns=time.perf_counter_ns()-loop_start
            if repeat==0:
                sampled[sid]=actual
            else:
                np.testing.assert_array_equal(actual,sampled[sid])
            init_ns=initialized-before
            h_times.append(init_ns);point_times.extend(durations);sizes.append(state.state_array_bytes)
            series_rows.append({'variant':args.variant,'repeat':repeat,'dataset_id':sid,'historical_points':len(h),
                'online_points':len(o),'historical_init_ns':init_ns,'online_loop_ns':loop_ns,
                'point_median_ns':float(np.median(durations)),'state_array_bytes':state.state_array_bytes})
        print('NEXT2 LATENCY PREFIX',args.variant,repeat+1,flush=True)
    # Independent split feature/tree calls include an extra dispatch boundary.
    parts=Parts(prepared.new_state(first_h));parts.feature_update(first_o[0]);parts.model_predict()
    feature_times=[];tree_times=[]
    for sid,h,o in examples:
        assert_isolated()
        parts=Parts(prepared.new_state(h))
        for j,v in enumerate(o):
            tick=time.perf_counter_ns();parts.feature_update(v);middle=time.perf_counter_ns()
            value=parts.model_predict();end=time.perf_counter_ns()
            feature_times.append(middle-tick);tree_times.append(end-middle)
            assert value==sampled[sid][j]
        assert parts.model_predict()==value and parts.model_predict()==value
    # Direct continuous-stream measurement, with the same finite real-value buffer.
    buffer=np.concatenate([o for _,h,o in examples])[:policy['block_points']]
    assert len(buffer)==policy['block_points']
    state=prepared.new_state(first_h)
    initial_bytes=state.state_array_bytes
    gc.collect();process=psutil.Process();rss_before=peak_rss=process.memory_info().rss
    elapsed=0.;processed=0;first_million=None;digest=hashlib.sha256();block_rows=[]
    wall=time.perf_counter()
    while processed<policy['continuous_points']:
        tick=time.perf_counter_ns()
        prediction=np.asarray([state.predict_one(v) for v in buffer],dtype=np.float32)
        duration=(time.perf_counter_ns()-tick)/1e9
        elapsed+=duration;processed+=len(buffer)
        assert np.isfinite(prediction).all() and np.all((prediction>=0)&(prediction<=1))
        digest.update(prediction.tobytes())
        block_rows.append({'processed_points':processed,'block_seconds':duration,'cumulative_prediction_seconds':elapsed})
        if processed==1_000_000:
            first_million=elapsed
        if processed%1_000_000==0:
            assert_isolated();peak_rss=max(peak_rss,process.memory_info().rss)
            print('NEXT2 LATENCY CONTINUOUS',args.variant,processed,round(elapsed,3),flush=True)
    assert state.state_array_bytes==initial_bytes
    assert first_million is not None
    # Mathematical bundle check after measurement avoids contaminating first-JIT cost.
    for sid,h,o in examples[:2]:
        original=model.make_state(h)
        expected=np.asarray([model.predict_one(original.update(v)) for v in o],dtype=np.float32)
        np.testing.assert_array_equal(expected,sampled[sid])
    table=pd.DataFrame(series_rows);table.to_csv(out/'SERIES_TIMING.csv',index=False)
    pd.DataFrame(block_rows).to_csv(out/'CONTINUOUS_BLOCKS.csv',index=False)
    total_points=int(table.online_points.sum())
    amortized=(table.historical_init_ns.sum()+table.online_loop_ns.sum())/1e9/total_points*1e6
    report={'status':'PASS','created_utc':utc_now(),'identity':identity,'isolated_from_workspace_research_workers':True,
        'model_load_seconds':load_seconds,'first_history_plus_jit_seconds_fresh_process':first_state_seconds,
        'point_distribution':distribution(point_times),'historical_initialization':distribution(h_times),
        'feature_state_update':distribution(feature_times),'model_prediction':distribution(tree_times),
        'real_prefix_repeats':policy['real_prefix_repeats'],'real_series_per_repeat':len(examples),
        'h_plus_online_amortized_us_per_point':float(amortized),'h_plus_online_extrapolated_1M_seconds':float(amortized),
        'h_plus_online_extrapolated_10M_seconds':float(amortized*10),
        'continuous_stream':{'points':processed,'measured_prediction_seconds':elapsed,'measured_first_million_seconds':first_million,
            'wall_seconds_including_checks':time.perf_counter()-wall,'output_sha256':digest.hexdigest(),
            'state_array_bytes_before':initial_bytes,'state_array_bytes_after':state.state_array_bytes,
            'rss_before_bytes':rss_before,'rss_peak_bytes':peak_rss,'rss_after_bytes':process.memory_info().rss},
        'state_array_bytes_min':min(sizes),'state_array_bytes_max':max(sizes),'model_file_bytes':path.stat().st_size,
        'sample_prediction_sha256':hashlib.sha256(b''.join(sampled[sid].tobytes() for sid,_,_ in examples)).hexdigest(),
        'sampled_bundle_parity':True,'separate_parts_parity':True,'repeated_real_prefix_parity':True,
        'files_sha256':{name:sha256(out/name) for name in ('SERIES_TIMING.csv','CONTINUOUS_BLOCKS.csv')},
        'quality_scores_computed':False,'seal_rows':0,'new_fits':0,
        'limitations':'Fixed 128 DEV probes inherited from old policy; repeated input for direct 10M cost, not a quality test. OS load and power state uncontrolled. Separate calls add dispatch/timer overhead. H amortized 1M/10M costs are extrapolations; continuous 1M/10M are measured.'}
    write_json(out/'RESULTS.json',report)
    checkpoint('Isolated runtime '+args.variant,['Compare paired runtime and numerical state ownership','Final audit after research minimum'],
        artifacts=[(out/'RESULTS.json').relative_to(ROOT).as_posix()])
    print('NEXT2 LATENCY PASS',args.variant,report['point_distribution']['median_us'],amortized,flush=True)


if __name__=='__main__':
    main()
