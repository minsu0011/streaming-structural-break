"""Finite contemporaneous scalar timing with 10M actual updates per artifact."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys
import hashlib
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import psutil
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5.data import prepare_research
from src.next5.streaming import load_prepared
from src.next5.synthetic import load_reference
from src.next5.runtime import SharedResearchBaseline
from src.next2.shared_inference import load_artifact
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor


def isolation():
    own=psutil.Process();allowed={own.pid,*[p.pid for p in own.parents()]};found=[]
    for process in psutil.process_iter(['pid','name']):
        if process.pid in allowed or not str(process.info['name']).lower().startswith('python'):continue
        try:
            if Path(process.cwd()).resolve()==ROOT:found.append(process.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    if found:raise RuntimeError('Workspace Python timing isolation violated: '+str(found))


def measure(prepared,records,buffer,policy,label,repeat):
    isolation();point_ns=[];initialization=[];loops=[];sizes=[];points=0
    for h,o in records:
        tick=time.perf_counter();state=prepared.new_state(h);initialization.append(time.perf_counter()-tick)
        sizes.append(state.state_array_bytes);tick=time.perf_counter()
        for value in o:
            start=time.perf_counter_ns();prediction=state.predict_one(value);point_ns.append(time.perf_counter_ns()-start)
            assert np.isfinite(prediction) and 0<=prediction<=1
        loops.append(time.perf_counter()-tick);points+=len(o)
    state=prepared.new_state(records[0][0]);initial_size=state.state_array_bytes;output=np.empty(len(buffer),dtype=np.float64)
    checkpoints=[];fingerprint=hashlib.sha256();count=0;started=time.perf_counter();cpu_start=psutil.Process().cpu_times()
    while count<policy['continuous_points']:
        amount=min(len(buffer),policy['continuous_points']-count)
        for j in range(amount):output[j]=state.predict_one(buffer[j])
        assert np.isfinite(output[:amount]).all() and np.all((output[:amount]>=0)&(output[:amount]<=1))
        fingerprint.update(output[:amount].tobytes());count+=amount
        if count%policy['measured_checkpoint_points']==0:
            isolation();assert state.state_array_bytes==initial_size;elapsed=time.perf_counter()-started
            checkpoints.append({'points':count,'elapsed_seconds':elapsed})
            progress('Isolated NEXT5 scalar timing',label+' block '+str(repeat)+' '+str(count)+' actual current-point updates')
            print('NEXT5 ISOLATED RUNTIME',label,repeat,count,round(elapsed,3),flush=True)
    elapsed=time.perf_counter()-started;cpu_end=psutil.Process().cpu_times()
    return {'model':label,'repeat':repeat,'sampled_series':len(records),'sampled_points':points,
        'single_point_median_us':float(np.median(point_ns)/1000),'single_point_p95_us':float(np.quantile(point_ns,.95)/1000),
        'single_point_p99_us':float(np.quantile(point_ns,.99)/1000),'historical_initialization_median_seconds':float(np.median(initialization)),
        'amortized_real_sequence_us':1e6*(sum(initialization)+sum(loops))/points,'ten_million_seconds':elapsed,
        'actual_continuous_points':count,'state_array_bytes':initial_size,'real_H_state_array_bytes_min':min(sizes),'real_H_state_array_bytes_max':max(sizes),
        'immutable_tree_array_bytes':state.immutable_tree_array_bytes,'output_fingerprint_sha256':fingerprint.hexdigest(),'fingerprint_dtype':'float64',
        'checkpoints':checkpoints,'process_RSS_bytes':psutil.Process().memory_info().rss,
        'continuous_CPU_seconds':(cpu_end.user+cpu_end.system)-(cpu_start.user+cpu_start.system)}


def main():
    policy_path=BASE/'runtime_validation/LATENCY_POLICY_LOCK.json';policy=read_json(policy_path)
    for name,sha in policy['sources_sha256'].items():assert digest(ROOT/name)==sha
    out=BASE/'latency/fixed_v1';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'policy_sha256':digest(policy_path),'new_fits':0},immutable=True)
    data=prepare_research();data.guard.admit(policy['series_ids'],purpose='Fixed timing sample after ResearchPool intersection')
    records=[]
    for sid in policy['series_ids']:
        h,o,_=data.get(sid);records.append((np.asarray(h,dtype=np.float32),np.asarray(o,dtype=np.float32)))
    buffer=np.concatenate([o for h,o in records])[:policy['block_points']].copy();assert len(buffer)==policy['block_points']
    prepared={};load_times={};warm={};proofs={}
    tick=time.perf_counter();model,descriptor=load_artifact(ROOT/policy['frozen_primary_directory'])
    assert descriptor['model_sha256']==policy['frozen_primary_model_sha256'];prepared['FROZEN_PRIMARY']=PreparedSharedBlendPredictor(model)
    load_times['FROZEN_PRIMARY']=time.perf_counter()-tick
    tick=time.perf_counter();refs,proof=load_reference('full',0);prepared['ResearchPool_BL80']=SharedResearchBaseline(refs['J102'],refs['C205'])
    load_times['ResearchPool_BL80']=time.perf_counter()-tick;proofs['ResearchPool_BL80']=proof
    api=read_json(BASE/'runtime_validation/API_POLICY_LOCK.json')
    for cid in policy['candidates']:
        path=BASE/'runtime_validation/api'/cid/'RESULTS.json';record=read_json(path)
        assert record['status']=='PASS' and record['prefix_cases']>=300 and record['official_TS_AUC_parity']
        proofs[cid]={'API_result_sha256':digest(path),'all_primary_OOF_bitwise_float64_equal':record['bitwise_float64_equal']}
        tick=time.perf_counter();prepared[cid]=load_prepared(ROOT/api['candidates'][cid]['descriptors']['0']);load_times[cid]=time.perf_counter()-tick
    for name,value in prepared.items():
        tick=time.perf_counter();value.new_state(records[0][0]).predict_one(float(buffer[0]));warm[name]=time.perf_counter()-tick
    results=[]
    for repeat,order in enumerate(policy['block_orders'],1):
        for label in order:
            result=measure(prepared[label],records,buffer,policy,label,repeat)
            atomic_json(out/('repeat_'+str(repeat)+'_'+label+'.json'),result,immutable=True);results.append(result)
    for label in prepared:assert len({row['output_fingerprint_sha256'] for row in results if row['model']==label})==1
    ratios=[];summaries=[]
    for cid in policy['candidates']:
        for repeat in range(1,len(policy['block_orders'])+1):
            candidate=next(r for r in results if r['model']==cid and r['repeat']==repeat)
            for reference in ('FROZEN_PRIMARY','ResearchPool_BL80'):
                baseline=next(r for r in results if r['model']==reference and r['repeat']==repeat)
                ratios.append({'candidate':cid,'repeat':repeat,'reference':reference,'scalar_median_ratio':candidate['single_point_median_us']/baseline['single_point_median_us'],
                    'amortized_ratio':candidate['amortized_real_sequence_us']/baseline['amortized_real_sequence_us'],
                    'ten_million_ratio':candidate['ten_million_seconds']/baseline['ten_million_seconds']})
        rows=[r for r in ratios if r['candidate']==cid and r['reference']=='FROZEN_PRIMARY'];medians={name:float(np.median([r[name] for r in rows])) for name in ('scalar_median_ratio','amortized_ratio','ten_million_ratio')}
        summaries.append({'candidate':cid,**medians,'all_ratios_within_3x':all(value<=3 for value in medians.values()),
            'exact_primary_OOF_requirement_met':proofs[cid]['all_primary_OOF_bitwise_float64_equal'],
            'promotion':False})
    import csv
    for name,rows in (('BLOCK_RATIOS.csv',ratios),('SUMMARY.csv',summaries)):
        with (out/name).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    data.guard.check_state();atomic_json(out/'RESULTS.json',{'created_utc':utc_now(),'status':'PASS','new_fits':0,'shadow_rows':0,'seal_rows':0,
        'model_count':len(prepared),'blocks':len(policy['block_orders']),'actual_total_continuous_points':sum(r['actual_continuous_points'] for r in results),
        'summaries':summaries,'load_seconds':load_times,'warmup_seconds':warm,'proofs':proofs,
        'all_repeat_fingerprints_identical':True,'policy_sha256':digest(policy_path),'files_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()},
        'scope':'Finite balanced-order engineering timing of already-fitted artifacts; no quality score from repeated buffers and no primary gate relaxation.'},immutable=True)
    progress('Isolated scalar runtime study complete','Three fixed order blocks, nine fitted artifacts, actual 10M updates each, stable state and repeated fingerprints')


if __name__=='__main__':main()
