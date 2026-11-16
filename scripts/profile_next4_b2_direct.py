"""Five fixed paired direct 10M scalar profiles after causal equivalence is proven."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys
import hashlib
import time
import psutil
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.b2_inference import load_prepared
from src.next4.synthetic import load_reference
from src.next.data import prepare_dev


def isolation():
    own=psutil.Process();allowed={own.pid,*[p.pid for p in own.parents()]};found=[]
    for process in psutil.process_iter(['pid','name']):
        if process.pid in allowed or not str(process.info['name']).lower().startswith('python'):continue
        try:
            if Path(process.cwd()).resolve()==ROOT:found.append(process.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    if found:raise RuntimeError('Timing isolation violated by workspace Python processes: '+str(found))


def measure(prepared,records,buffer,policy,label,repeat):
    point_ns=[];initialization=[];elapsed_loops=[];point_count=0;sizes=[]
    isolation();start_cpu=psutil.Process().cpu_times()
    for h,o in records:
        tick=time.perf_counter();state=prepared.new_state(h);initialization.append(time.perf_counter()-tick)
        sizes.append(state.state_array_bytes);tick=time.perf_counter()
        for value in o:
            t=time.perf_counter_ns();p=state.predict_one(value);point_ns.append(time.perf_counter_ns()-t)
            assert np.isfinite(p) and 0<=p<=1
        elapsed_loops.append(time.perf_counter()-tick);point_count+=len(o)
    state=prepared.new_state(records[0][0]);initial_size=state.state_array_bytes
    outputs=np.empty(len(buffer),dtype=np.float32);fingerprint=hashlib.sha256();checkpoints=[]
    count=0;started=time.perf_counter();continuous=policy['continuous_points']
    while count<continuous:
        amount=min(len(buffer),continuous-count)
        for j in range(amount):outputs[j]=state.predict_one(buffer[j])
        assert np.isfinite(outputs[:amount]).all() and ((outputs[:amount]>=0)&(outputs[:amount]<=1)).all()
        fingerprint.update(outputs[:amount].tobytes());count+=amount
        if count%policy['measured_checkpoint_points']==0:
            elapsed=time.perf_counter()-started;isolation();assert state.state_array_bytes==initial_size
            checkpoints.append({'points':count,'elapsed_seconds':elapsed})
            progress('Isolated direct scalar timing',f'{label} paired repeat {repeat}: {count}/10000000 actual updates')
            print('NEXT4 DIRECT TIMING',label,repeat,count,round(elapsed,3),flush=True)
    total=time.perf_counter()-started;cpu=psutil.Process().cpu_times()
    return {'model':label,'paired_repeat':repeat,'sampled_series':len(records),'sampled_points':point_count,
        'single_point_median_us':float(np.median(point_ns)/1000),'single_point_p95_us':float(np.quantile(point_ns,.95)/1000),
        'single_point_p99_us':float(np.quantile(point_ns,.99)/1000),
        'historical_initialization_median_seconds':float(np.median(initialization)),
        'amortized_real_sequence_us':1e6*(sum(initialization)+sum(elapsed_loops))/point_count,
        'ten_million_seconds':total,'actual_continuous_points':count,'state_array_bytes':initial_size,
        'real_H_state_array_bytes_min':min(sizes),'real_H_state_array_bytes_max':max(sizes),
        'output_fingerprint_sha256':fingerprint.hexdigest(),'checkpoints':checkpoints,
        'process_RSS_bytes':psutil.Process().memory_info().rss,'process_CPU_seconds':(cpu.user+cpu.system)-(start_cpu.user+start_cpu.system)}


def main():
    path=BASE/'diagnostic_policy/B2_RUNTIME_POLICY.json';lock=read_json(path)
    for name,sha in lock['sources_sha256'].items():assert digest(ROOT/name)==sha
    proof=read_json(BASE/'runtime_validation/B2_DIRECT/RESULTS.json');assert proof['status']=='PASS' and proof['prefix_cases']>=300
    out=BASE/'latency/B2_DIRECT';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'policy_sha256':digest(path),'causal_proof_sha256':digest(BASE/'runtime_validation/B2_DIRECT/RESULTS.json')},immutable=True)
    policy=read_json(ROOT/'configs/next_latency_policy.json');assert digest(ROOT/'configs/next_latency_policy.json')==lock['latency_policy_sha256']
    data=prepare_dev(ROOT);data.guard.admit(policy['series_ids'],purpose='NEXT4 fixed existing timing sample')
    records=[]
    for sid in policy['series_ids']:
        h,o,_=data.get(sid);records.append((np.asarray(h,dtype=np.float32),np.asarray(o,dtype=np.float32)))
    buffer=np.concatenate([o for h,o in records])[:policy['block_points']].copy();assert len(buffer)==policy['block_points']
    tick=time.perf_counter();frozen,reference_proof=load_reference('all_dev');baseline_load=time.perf_counter()-tick
    tick=time.perf_counter();b2=load_prepared(ROOT/lock['model_directory']);candidate_load=time.perf_counter()-tick
    warm={}
    for label,prepared in [('BL80',frozen),('B2',b2)]:
        tick=time.perf_counter();prepared.new_state(records[0][0]).predict_one(float(buffer[0]));warm[label]=time.perf_counter()-tick
    results=[]
    for repeat,order in enumerate(lock['paired_run_order'],1):
        for label in order:
            r=measure(frozen if label=='BL80' else b2,records,buffer,policy,label,repeat)
            atomic_json(out/f'repeat_{repeat}_{label}.json',r,immutable=True);results.append(r)
    for label in ('BL80','B2'):
        assert len({r['output_fingerprint_sha256'] for r in results if r['model']==label})==1
    ratios=[]
    for repeat in range(1,len(lock['paired_run_order'])+1):
        a=next(r for r in results if r['model']=='BL80' and r['paired_repeat']==repeat)
        b=next(r for r in results if r['model']=='B2' and r['paired_repeat']==repeat)
        ratios.append({'paired_repeat':repeat,'ten_million_ratio':b['ten_million_seconds']/a['ten_million_seconds'],
            'scalar_median_ratio':b['single_point_median_us']/a['single_point_median_us'],
            'amortized_ratio':b['amortized_real_sequence_us']/a['amortized_real_sequence_us']})
    pd.DataFrame(ratios).to_csv(out/'PAIRED_RATIOS.csv',index=False);data.guard.check_state()
    summary={key:float(np.median([r[key] for r in ratios])) for key in ('ten_million_ratio','scalar_median_ratio','amortized_ratio')}
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'candidate':'B2_LOG_ENERGY_SEMITAIL','promotion':False,
        'paired_repeats':len(ratios),'summary_median_ratios':summary,'direct_runtime_within_3x':all(v<=3 for v in summary.values()),
        'new_fits':0,'seal_rows':0,'load_seconds':{'BL80':baseline_load,'B2':candidate_load},'warmup_seconds':warm,
        'reference_proof':reference_proof,'source_and_policy_sha256':digest(path),
        'scope':'Contemporaneous all-DEV fitted models, unchanged direct scalar implementation, same existing 128-series sample and actual 10M continuous updates per run. No shared-state optimization; timing does not override primary statistical gate.',
        'files_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}},immutable=True)
    print('NEXT4 B2 DIRECT TIMING COMPLETE',summary,flush=True)


if __name__=='__main__':main()
