"""Small extension receipts using the existing scheduler, not a new orchestrator."""
from pathlib import Path
from contextlib import contextmanager
import time,os,datetime
from src.next3.artifacts import atomic_json,digest,read_json,utc_now
from src.next5.scheduler import append_csv
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/next6_extension'
BASE.mkdir(parents=True,exist_ok=True)
def save(name,value,immutable=True):atomic_json(BASE/name,value,immutable=immutable)
def checkpoint(task,detail=None):
    now=utc_now();record={'timestamp':now,'wall_timestamp':time.time(),'task':task,'detail':detail,'pid':os.getpid(),'status':'IN_PROGRESS','Shadow_Seal_opened':False}
    save('runtime/ACTIVITY.json',record,False)
    old=BASE/'runtime/LAST_CHECKPOINT.json'
    if not old.exists() or time.time()-read_json(old)['wall_timestamp']>=1800:
        save('checkpoints/'+now.replace(':','')+'.json',record);save('runtime/LAST_CHECKPOINT.json',record,False)
    print(now,task,detail or '',flush=True)
@contextmanager
def activity(task,kind='BOTTLENECK_RESEARCH'):
    start=time.perf_counter();cpu=time.process_time();stamp=utc_now();status='COMPLETE';checkpoint(task,'START '+kind)
    try:yield
    except BaseException:status='FAILED';raise
    finally:
        append_csv(BASE/'EXPERIMENT_LEDGER.csv',{'started_utc':stamp,'ended_utc':utc_now(),'task':task,'kind':kind,'status':status,'wall_seconds':time.perf_counter()-start,'process_cpu_seconds':time.process_time()-cpu,'pid':os.getpid()});checkpoint(task,status)
def csv_write(name,rows):
    import csv
    path=BASE/name;path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(),str(path)
    if not rows:path.write_text('',encoding='utf-8');return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
