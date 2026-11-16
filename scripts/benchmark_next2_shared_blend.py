"""One isolated exact-runtime trial; compare with the already frozen BL80_COMPACT trial."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys
import json
import time
import hashlib
import gc
import numpy as np
import pandas as pd
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.benchmark_next2_runtime import assert_isolated,distribution
from src.next.data import prepare_dev
from src.next2.canonical_model import load
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor,SharedParts as Parts,source_hash
from src.next2.numeric_memory import accounting
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    assert_isolated()
    policy_path=ROOT/'configs/next2/RUNTIME_MEASUREMENT_LOCK.json';policy=json.loads(policy_path.read_text())
    baseline_path=ROOT/'artifacts/next2/latency/BL80_COMPACT/RESULTS.json';baseline=json.loads(baseline_path.read_text())
    assert baseline['status']=='PASS'
    memory_path=ROOT/'artifacts/next2/numeric_memory/MEMORY_BY_H.csv'
    memory_receipt=json.loads((memory_path.parent/'RESULTS.json').read_text())
    assert sha256(memory_path)==memory_receipt['files_sha256'][memory_path.name]
    memory_table=pd.read_csv(memory_path,float_precision='round_trip')
    baseline_private=int(memory_table.loc[memory_table.variant=='BL80_COMPACT','private_per_series_numeric_bytes'].max())
    proofs={}
    for stage in ('all_dev','oof','alternate'):
        path=ROOT/'artifacts/next2/shared_blend_runtime'/stage/'RESULTS.json';r=json.loads(path.read_text())
        assert r['status']=='PASS' and r['identity']['runtime_sha256']==source_hash()
        proofs[str(path.relative_to(ROOT))]=sha256(path)
    artifact=ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_compact_v1/model.sbnext2'
    identity={'variant':'BL80_SHARED_V1','inherited_policy_sha256':sha256(policy_path),'runtime_sha256':source_hash(),
        'script_sha256':sha256(Path(__file__)),'protocol_sha256':sha256(ROOT/'docs/NEXT2_SHARED_BLEND_RUNTIME_PROTOCOL.md'),
        'baseline_receipt_sha256':sha256(baseline_path),'baseline_memory_sha256':sha256(memory_path),'proofs_sha256':proofs,'model_sha256':sha256(artifact)}
    out=ROOT/'artifacts/next2/latency/BL80_SHARED_V1';out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULTS.json').exists():
        r=json.loads((out/'RESULTS.json').read_text());assert r['status']=='PASS' and r['identity']==identity
        print('NEXT2 SHARED BLEND MEASUREMENT RETAINED',flush=True);return
    if (out/'START.json').exists():raise RuntimeError('Interrupted trial needs a documented new revision')
    write_json(out/'START.json',{'status':'RUNNING','created_utc':utc_now(),'identity':identity})
    data=prepare_dev(ROOT);data.guard.admit(policy['series_ids'],purpose='Fixed shared-blend exact runtime probes')
    examples=[(sid,*data.get(sid)[:2]) for sid in policy['series_ids']]
    tick=time.perf_counter();model=load(artifact);engine=PreparedSharedBlendPredictor(model);load_seconds=time.perf_counter()-tick
    _,h0,o0=examples[0];tick=time.perf_counter();warm=engine.new_state(h0);warm.predict_one(o0[0]);warm_seconds=time.perf_counter()-tick
    sampled={};points=[];histories=[];records=[];features=[];trees=[]
    for repeat in range(policy['real_prefix_repeats']):
        for sid,h,o in examples:
            assert_isolated();tick=time.perf_counter_ns();state=engine.new_state(h);init=time.perf_counter_ns()-tick
            actual=np.empty(len(o),dtype=np.float32);durations=np.empty(len(o),dtype=np.int64);begin=time.perf_counter_ns()
            for k,point in enumerate(o):
                tick=time.perf_counter_ns();actual[k]=state.predict_one(point);durations[k]=time.perf_counter_ns()-tick
            elapsed=time.perf_counter_ns()-begin
            if repeat==0:sampled[sid]=actual
            else:np.testing.assert_array_equal(actual,sampled[sid])
            points.extend(durations);histories.append(init)
            records.append({'variant':'BL80_SHARED_V1','repeat':repeat,'dataset_id':sid,'historical_points':len(h),'online_points':len(o),
                'historical_init_ns':init,'online_loop_ns':elapsed,'point_median_ns':float(np.median(durations)),'state_array_bytes':state.state_array_bytes})
        print('NEXT2 SHARED BLEND PREFIX',repeat+1,flush=True)
    parts=Parts(engine.new_state(h0));parts.feature_update(o0[0]);parts.model_predict()
    for sid,h,o in examples:
        assert_isolated();parts=Parts(engine.new_state(h))
        for k,point in enumerate(o):
            tick=time.perf_counter_ns();parts.feature_update(point);middle=time.perf_counter_ns();value=parts.model_predict();end=time.perf_counter_ns()
            features.append(middle-tick);trees.append(end-middle);assert value==sampled[sid][k]
        assert parts.model_predict()==value
    assert hashlib.sha256(b''.join(sampled[sid].tobytes() for sid,_,_ in examples)).hexdigest()==baseline['sample_prediction_sha256']
    buffer=np.concatenate([o for _,_,o in examples])[:policy['block_points']];assert len(buffer)==10000
    state=engine.new_state(h0);initial=state.state_array_bytes;memory,_,_=accounting(state)
    gc.collect();process=psutil.Process();rss_before=peak_rss=process.memory_info().rss
    elapsed=0.;count=0;blocks=[];digest=hashlib.sha256();first_million=None;wall=time.perf_counter()
    while count<10_000_000:
        tick=time.perf_counter_ns();p=np.asarray([state.predict_one(v) for v in buffer],dtype=np.float32);duration=(time.perf_counter_ns()-tick)/1e9
        elapsed+=duration;count+=len(buffer);assert np.isfinite(p).all();digest.update(p.tobytes())
        blocks.append({'processed_points':count,'block_seconds':duration,'cumulative_prediction_seconds':elapsed})
        if count==1_000_000:first_million=elapsed
        if count%1_000_000==0:
            assert_isolated();peak_rss=max(peak_rss,process.memory_info().rss)
            print('NEXT2 SHARED BLEND CONTINUOUS',count,round(elapsed,3),flush=True)
    assert state.state_array_bytes==initial and digest.hexdigest()==baseline['continuous_stream']['output_sha256']
    table=pd.DataFrame(records);table.to_csv(out/'SERIES_TIMING.csv',index=False);pd.DataFrame(blocks).to_csv(out/'CONTINUOUS_BLOCKS.csv',index=False)
    point_summary=distribution(points);h_summary=distribution(histories)
    amortized=(table.historical_init_ns.sum()+table.online_loop_ns.sum())/table.online_points.sum()/1000.
    criteria={'direct_10M_at_least_10_percent_faster':elapsed<=.9*baseline['continuous_stream']['measured_prediction_seconds'],
        'point_median_nonworse':point_summary['median_us']<=baseline['point_distribution']['median_us'],
        'h_median_within_5_percent':h_summary['median_us']<=1.05*baseline['historical_initialization']['median_us'],
        'all_dev_oof_and_alternate_exact':True,'private_state_smaller_than_baseline':memory['private_per_series_numeric_bytes']<baseline_private,'separate_parts_exact':True,'direct_10M_exact':True,'constant_state':True}
    write_json(out/'RESULTS.json',{'status':'PASS','identity':identity,'created_utc':utc_now(),'point_distribution':point_summary,
        'historical_initialization':h_summary,'feature_state_update':distribution(features),'model_prediction':distribution(trees),
        'model_load_seconds':load_seconds,'first_history_plus_jit_seconds_fresh_process':warm_seconds,
        'h_plus_online_amortized_us_per_point':float(amortized),'h_plus_online_extrapolated_1M_seconds':float(amortized),
        'h_plus_online_extrapolated_10M_seconds':float(amortized*10),'model_file_bytes':artifact.stat().st_size,
        'state_array_bytes_min':initial,'state_array_bytes_max':initial,'numerical_memory':memory,
        'continuous_stream':{'points':count,'measured_prediction_seconds':elapsed,'measured_first_million_seconds':first_million,
            'wall_seconds_including_checks':time.perf_counter()-wall,'output_sha256':digest.hexdigest(),'exact_equal_to_BL80_COMPACT':True,
            'state_array_bytes_before':initial,'state_array_bytes_after':state.state_array_bytes,'rss_before_bytes':rss_before,
            'rss_peak_bytes':peak_rss,'rss_after_bytes':process.memory_info().rss},
        'sample_prediction_sha256':hashlib.sha256(b''.join(sampled[sid].tobytes() for sid,_,_ in examples)).hexdigest(),
        'engineering_replacement_criteria':criteria,'engineering_replacement_eligible':all(criteria.values()),
        'files_sha256':{f:sha256(out/f) for f in ('SERIES_TIMING.csv','CONTINUOUS_BLOCKS.csv')},
        'new_fits':0,'seal_rows':0,'final_role_selection':False,
        'limitations':'One registered isolated sequential trial; OS/power uncontrolled. Compares against earlier BL80_COMPACT trial, with all repetitions retained. Identical new-branch AR/ARCH work is shared; legacy tree-used initialization remains separate. Two frozen tree ensembles and float32 component rounding are retained. Scientific predictions exactly unchanged.'})
    checkpoint('Measured shared fixed-blend exact runtime',['Apply prospective engineering replacement criteria','Preserve all unsuccessful trials'])
    print('NEXT2 SHARED BLEND MEASURED',elapsed,criteria,flush=True)


if __name__=='__main__':main()
