"""Fixed label-free RP sample parity and latency; no performance selection."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next8
import numpy as np
from src.next8.common import BASE,read_json,atomic,digest,utc_now,activity,note
from src.next8.cache import FeatureCache,immutable_npz
from src.next8.data import outer_scope
from src.next8.nested_auxiliary import early_scope
from src.next8.runtime_audit import fixed_cases,load_scope_runtime,audit_case,observations,latency_summary
from src.next8.streaming_runtime import IntegratedState


def main(fold,early):
    assert read_json(BASE/'STREAMING_RUNTIME_TEST_GATE_NEXT8.json')['status']=='PASS'
    lock=read_json(BASE/'RUNTIME_AUDIT_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    cache=FeatureCache();scope=early_scope(outer_scope(fold,cache.data),early)
    directory=BASE/'runtime_validation'/scope.name;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'DONE.json').exists():raise RuntimeError('Completed runtime sample cannot be silently rerun')
    cases=fixed_cases(cache,scope,2);ids=[sid for group in cases for sid in group['selected_ids']]
    if not (directory/'CASES.json').exists():atomic(directory/'CASES.json',{'status':'FROZEN','scope':scope.provenance(),
        'selection':'First2 original IDs in each fixed H-length bin within this outer fold; no labels/outcomes','cases':cases})
    else:assert read_json(directory/'CASES.json')['cases']==cases
    heads,baseline,reference,saved,states,integration=load_scope_runtime(cache,scope,fold,early=early)
    eligible=integration['eligible']
    # Every specialization is warmed on a fixed prefix before measured paths.
    with activity('Warm runtime specializations '+scope.name,'RUNTIME_COMPILATION','B2_STATE_OBSERVABILITY'):
        h,online,_=observations(cache,ids[0])
        baseline.new_state(h).predict_one(online[0])
        for p in heads.values():p.new_state(h).predict_one(online[0])
        if eligible:IntegratedState(baseline,heads['M5'],h,True).predict_one(online[0])
    records=[];timing={};files={}
    for sid in ids:
        p=directory/f'ID{sid}';p.mkdir(parents=True,exist_ok=True)
        if (p/'DONE.json').exists():
            record=read_json(p/'DONE.json');assert digest(p/'TIMING.npz')==record['timing_sha256']
            with np.load(p/'TIMING.npz',allow_pickle=False) as a:times={k:a[k].copy() for k in a.files}
        else:
            if any(p.iterdir()):raise RuntimeError('Partial runtime case preserved for explicit repair')
            with activity(f'Real RP causal execution {scope.name} ID{sid}','RUNTIME_PARITY','B2_STATE_OBSERVABILITY'):
                record,times=audit_case(cache,sid,heads,baseline,reference,saved,states,eligible,timings=True)
                immutable_npz(p/'TIMING.npz',**times)
                record={'status':'PASS','timestamp':utc_now(),**record,'timing_sha256':digest(p/'TIMING.npz')}
                atomic(p/'DONE.json',record)
        records.append(record);files[f'ID{sid}/DONE.json']=digest(p/'DONE.json')
        for name,a in times.items():timing.setdefault(name,[]).append(a)
    summary={name:latency_summary(np.concatenate(parts)) for name,parts in timing.items()}
    for name,r in summary.items():
        r['median_ratio_BL80']=r['median_microseconds']/summary['M0']['median_microseconds']
        r['p99_ratio_BL80']=r['p99_microseconds']/summary['M0']['p99_microseconds']
        r['runtime_ratio_gate']=r['median_ratio_BL80']<=3 and r['p99_ratio_BL80']<=5
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),
        'parity_status':'PASS','cases':cases,'series':len(records),'points':sum(r['online_points'] for r in records),
        'latency':summary,'files_sha256':files,'records':records,'initialization_in_latency_gate':False,
        'latency_gate_changes_candidate_settings':False,'outer_scores_or_labels_used':False})
    note('Complete actual RP execution sample',scope.name+'; scientific scores unopened','B2_STATE_OBSERVABILITY')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True);p.add_argument('--early',action='store_true')
    a=p.parse_args();main(a.fold,a.early)
