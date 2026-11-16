"""Bounded two-process final dedicated-M7 audit, parent waiting uncredited."""
from pathlib import Path
import sys
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now

def main():
    import psutil
    from src.next9.queue import identity_alive
    admission=read_json(BASE/'runtime/FINAL_AUDIT_ADMISSION_NEXT9.json');assert admission['status']=='PASS'
    owner=read_json(BASE/'runtime/SCHEDULER_OWNER.json');assert not identity_alive(owner['pid'],owner['process_created'])
    workers=1 if psutil.virtual_memory().percent>80 else 2
    assert psutil.virtual_memory().percent<=90 and psutil.disk_usage(ROOT).free>=40*1024**3
    out=BASE/'audit/integrated_driver';out.mkdir(parents=True,exist_ok=False)
    atomic(out/'OWNER.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),timestamp=utc_now(),maximum_workers=workers))
    def run(fold,mode,part):
        name=f'F{fold}_{mode}_P{part}'
        command=[str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-B',str(ROOT/'scripts/next9/audit_next9_integrated_rp_v4.py'),
            '--fold',str(fold),'--mode',mode,'--part',str(part)]
        with (out/(name+'.log')).open('x',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            atomic(out/(name+'_START.json'),dict(timestamp=utc_now(),pid=process.pid,created=psutil.Process(process.pid).create_time(),command=command))
            code=process.wait()
        atomic(out/(name+'_DONE.json'),dict(timestamp=utc_now(),returncode=code,log_sha256=digest(out/(name+'.log'))))
        return name,code
    results=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run,fold,mode,part) for fold in range(5) for mode in ['NORMAL','EARLY'] for part in [0,1]]
        for future in as_completed(futures):
            name,code=future.result();results.append(dict(task=name,returncode=code))
            print(utc_now(),name,'returncode',code,flush=True)
    assert all(r['returncode']==0 for r in results),'Failures preserved; no silent retry or padding'
    coverage=[]
    for mode in ['NORMAL','EARLY']:
        ids=[];points=0
        for fold in range(5):
            for part in [0,1]:
                path=BASE/'audit/integrated_RP'/f'outer_{fold}'/mode/f'part_{part}'/'DONE.json'
                r=read_json(path);assert r['status']=='PASS' and r['all_dedicated_M7_bitwise'];ids.extend(r['series_ids']);points+=r['points']
        assert len(ids)==len(set(ids))==6400
        coverage.append(dict(mode=mode,series=6400,points=points))
    atomic(BASE/'INTEGRATED_RP_FINAL_AUDIT_NEXT9.json',dict(status='PASS',timestamp=utc_now(),coverage=coverage,
        dedicated_M7_raw_scalar_vs_shared_research_scalar_bitwise=True,save_restore_at31=True,
        actual_model_path_not_merely_cached_prediction_comparison=True,parent_wait_not_credited=True,maximum_workers=workers))

if __name__=='__main__':main()
