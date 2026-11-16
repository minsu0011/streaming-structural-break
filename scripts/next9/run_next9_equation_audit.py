"""Four disjoint153-feature verification shards; at most two child processes."""
from pathlib import Path
import sys
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now

def main():
    import psutil
    from src.next9.queue import identity_alive
    assert read_json(BASE/'runtime/FINAL_AUDIT_ADMISSION_NEXT9.json')['status']=='PASS'
    owner=read_json(BASE/'runtime/SCHEDULER_OWNER.json');assert not identity_alive(owner['pid'],owner['process_created'])
    # Run after the dedicated RP audit, never alongside its two workers.
    assert read_json(BASE/'INTEGRATED_RP_FINAL_AUDIT_NEXT9.json')['status']=='PASS'
    capacity=1 if psutil.virtual_memory().percent>80 else 2
    assert psutil.virtual_memory().percent<=90 and psutil.disk_usage(ROOT).free>=40*1024**3
    out=BASE/'audit/equation_driver';out.mkdir(parents=True,exist_ok=False)
    atomic(out/'OWNER.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),timestamp=utc_now(),maximum_workers=capacity))
    def run(part):
        name=f'PART_{part}';command=[str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-B',str(ROOT/'scripts/next9/audit_next9_movement_equations.py'),'--part',str(part)]
        with (out/(name+'.log')).open('x',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            atomic(out/(name+'_START.json'),dict(timestamp=utc_now(),pid=process.pid,created=psutil.Process(process.pid).create_time(),command=command))
            code=process.wait()
        atomic(out/(name+'_DONE.json'),dict(timestamp=utc_now(),returncode=code,log_sha256=digest(out/(name+'.log'))))
        return part,code
    with ThreadPoolExecutor(max_workers=capacity) as pool:
        futures=[pool.submit(run,p) for p in range(4)];results=[]
        for future in as_completed(futures):
            part,code=future.result();results.append(code);print(utc_now(),'Equation audit',part,code,flush=True)
    assert results==[0]*4
    receipts=[read_json(BASE/'audit/movement_equations'/f'part_{p}'/'DONE.json') for p in range(4)]
    features=[j for r in receipts for j in r['features']]
    assert sorted(features)==list(range(153)) and sum(r['checked_models'] for r in receipts)==18360
    atomic(BASE/'MOVEMENT_EQUATION_FINAL_AUDIT_NEXT9.json',dict(status='PASS',timestamp=utc_now(),features=153,checked_models=18360,
        frozen_normal_equations_verified=True,heldout_group_MSE_rebuilt=True,new_fit_calls=0,parent_wait_not_credited=True,
        receipts_sha256={str(p):digest(BASE/'audit/movement_equations'/f'part_{p}'/'DONE.json') for p in range(4)}))

if __name__=='__main__':main()
