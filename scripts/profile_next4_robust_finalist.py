"""Finite paired scalar timings for the unchanged nominated robustness artifact."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.robust_finalist import load_prepared
from src.next4.synthetic import load_reference
from src.next.data import prepare_dev
import scripts.profile_next4_b2_direct as original_timing


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    cid=args.candidate;nompath=BASE/'finalist_nominations'/(cid+'.json');nom=read_json(nompath)
    for name,sha in nom['execution_sources_sha256'].items():assert digest(ROOT/name)==sha
    if args.audit:assert read_json(BASE/'SESSION.json').get('audit_started_utc')
    phase='FINAL_AUDIT' if args.audit else 'RESEARCH'
    # Preserve the original scalar timing implementation. Only its work-signal phase changes.
    original_timing.progress=lambda task,detail:progress(task,detail,phase=phase,counts_as_research=not args.audit)
    proofpath=BASE/'finalist_validation'/cid/('API_'+cid)/'RESULTS.json';proof=read_json(proofpath)
    assert proof['status']=='PASS' and proof['prefix_cases']==300
    out=BASE/'latency'/(cid+('_AUDIT' if args.audit else '_DIRECT'));out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'nomination_sha256':digest(nompath),'API_proof_sha256':digest(proofpath),
        'audit_phase':args.audit,'timing_implementation_unchanged':True},immutable=True)
    pp=ROOT/'configs/next_latency_policy.json';policy=read_json(pp);assert digest(pp)==nom['latency_policy_sha256']
    data=prepare_dev(ROOT);data.guard.admit(policy['series_ids'],purpose='Fixed existing finalist runtime sample')
    records=[tuple(np.asarray(v,dtype=np.float32) for v in data.get(sid)[:2]) for sid in policy['series_ids']]
    buffer=np.concatenate([o for h,o in records])[:policy['block_points']].copy()
    baseline,reference_proof=load_reference('all_dev');candidate=load_prepared(BASE/'refits'/cid/'nominated_all_dev_v1')
    prepared={'BL80':baseline,'ROBUSTNESS':candidate};warm={}
    for label,model in prepared.items():
        tick=time.perf_counter();model.new_state(records[0][0]).predict_one(float(buffer[0]));warm[label]=time.perf_counter()-tick
    order=nom['audit_paired_run_order'] if args.audit else nom['research_paired_run_order'];results=[]
    for repeat,labels in enumerate(order,1):
        for label in labels:
            result=original_timing.measure(prepared[label],records,buffer,policy,label,repeat)
            atomic_json(out/f'repeat_{repeat}_{label}.json',result,immutable=True);results.append(result)
    ratios=[]
    for repeat in range(1,len(order)+1):
        a=next(r for r in results if r['model']=='BL80' and r['paired_repeat']==repeat)
        b=next(r for r in results if r['model']=='ROBUSTNESS' and r['paired_repeat']==repeat)
        ratios.append({'paired_repeat':repeat,'ten_million_ratio':b['ten_million_seconds']/a['ten_million_seconds'],
            'scalar_median_ratio':b['single_point_median_us']/a['single_point_median_us'],
            'amortized_ratio':b['amortized_real_sequence_us']/a['amortized_real_sequence_us']})
    for label in prepared:assert len({r['output_fingerprint_sha256'] for r in results if r['model']==label})==1
    if args.audit:
        previous=read_json(BASE/'latency'/(cid+'_DIRECT')/'repeat_1_ROBUSTNESS.json')
        assert previous['output_fingerprint_sha256']==next(r for r in results if r['model']=='ROBUSTNESS')['output_fingerprint_sha256']
    summary={key:float(np.median([r[key] for r in ratios])) for key in ('ten_million_ratio','scalar_median_ratio','amortized_ratio')}
    data.guard.check_state()
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'candidate':cid,'audit_phase':args.audit,
        'paired_repeats':len(order),'ratios':ratios,'summary_median_ratios':summary,'direct_runtime_within_3x':all(v<=3 for v in summary.values()),
        'reference_proof':reference_proof,'warmup_seconds':warm,'actual_scalar_updates':len(results)*10000000,
        'timing_source_sha256':digest(ROOT/'scripts/profile_next4_b2_direct.py'),'nomination_sha256':digest(nompath),
        'new_fits':0,'seal_rows':0,'files_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}},immutable=True)
    print('NEXT4 ROBUST FINALIST DIRECT TIMING COMPLETE',cid,summary,flush=True)


if __name__=='__main__':main()
