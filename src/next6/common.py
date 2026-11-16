"""Small receipts/checkpoints, reusing the existing scheduler's append primitive."""
from pathlib import Path
from contextlib import contextmanager
import hashlib, json, os, time
from src.next3.artifacts import atomic_json, digest, read_json, utc_now
from src.next5.scheduler import append_csv

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/next6'
BASE.mkdir(exist_ok=True)

def save(name,obj,immutable=True):
    atomic_json(BASE/name,obj,immutable=immutable)

def ids_hash(ids):
    return hashlib.sha256(','.join(map(str,sorted(map(int,ids)))).encode()).hexdigest()

def checkpoint(task,detail=None):
    now=utc_now()
    save('runtime/ACTIVITY.json',{'timestamp':now,'pid':os.getpid(),'task':task,'detail':detail},False)
    previous=BASE/'runtime/LAST_CHECKPOINT.json'
    import datetime
    if not previous.exists() or time.time()-read_json(previous)['wall_timestamp']>=1800:
        stamp=now.replace(':','').replace('.','_')
        record={'timestamp':now,'wall_timestamp':time.time(),'task':task,'detail':detail,
            'status':'IN_PROGRESS','shadow_opened':False,'seal_opened':False,
            'resume':'Run the same NEXT6 stage command; completed artifacts are reused, never overwritten.'}
        save('checkpoints/'+stamp+'.json',record)
        save('runtime/LAST_CHECKPOINT.json',record,False)
    print(now,task,detail or '',flush=True)

@contextmanager
def activity(name,kind='MODEL_RESEARCH'):
    start=time.perf_counter(); cpu=time.process_time(); stamp=utc_now()
    checkpoint(name,'START '+kind)
    status='COMPLETE'
    try: yield
    except BaseException:
        status='FAILED'; raise
    finally:
        append_csv(BASE/'EXPERIMENT_LEDGER.csv',{'started_utc':stamp,'ended_utc':utc_now(),
            'task':name,'kind':kind,'status':status,'wall_seconds':time.perf_counter()-start,
            'process_cpu_seconds':time.process_time()-cpu,'pid':os.getpid()})
        checkpoint(name,status)
