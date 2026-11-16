"""One actual final-audit 10M replay of the retained frozen artifact, no refit."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next.data import prepare_dev
from src.next4.synthetic import load_reference
import scripts.profile_next4_b2_direct as original


def main():
    assert read_json(BASE/'SESSION.json').get('audit_started_utc')
    assert 'NEXT4_ROBUSTNESS' not in read_json(ROOT/'DEV_CANDIDATES_NEXT4.json')['roles']
    pp=ROOT/'configs/next_latency_policy.json';policy=read_json(pp);data=prepare_dev(ROOT)
    data.guard.admit(policy['series_ids'],purpose='Selected frozen artifact final timing')
    records=[tuple(np.asarray(v,dtype=np.float32) for v in data.get(sid)[:2]) for sid in policy['series_ids']]
    buffer=np.concatenate([o for h,o in records])[:policy['block_points']].copy();prepared,proof=load_reference('all_dev')
    out=BASE/'latency/FROZEN_FINAL_AUDIT';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'actual_scalar_updates_intended':10000000,
        'repetitions':1,'policy_sha256':digest(pp),'source_sha256':digest(Path(__file__))},immutable=True)
    original.progress=lambda task,detail:progress(task,detail,phase='FINAL_AUDIT',counts_as_research=False)
    prepared.new_state(records[0][0]).predict_one(float(buffer[0]))
    r=original.measure(prepared,records,buffer,policy,'BL80',1)
    assert r['output_fingerprint_sha256']==read_json(BASE/'latency/B2_DIRECT/repeat_1_BL80.json')['output_fingerprint_sha256']
    data.guard.check_state()
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'timing':r,'reference_proof':proof,
        'original_fingerprint_exact':True,'new_fits':0,'seal_rows':0,
        'source_sha256':digest(Path(__file__)),'original_timing_source_sha256':digest(ROOT/'scripts/profile_next4_b2_direct.py')},immutable=True)
    print('NEXT4 FROZEN FINAL AUDIT 10M COMPLETE',flush=True)


if __name__=='__main__':main()
