"""Recoverable queue dispatch with ownership checks and one bounded retry."""
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from src.next8.common import BASE,ROOT,read_json,save,digest,utc_now


def identity_alive(pid,created):
    try:
        return abs(psutil.Process(pid).create_time()-created)<1e-3
    except (psutil.NoSuchProcess,psutil.AccessDenied):
        return False


def progress(record):
    cpu=0.
    if identity_alive(record['pid'],record['process_created']):
        parent=psutil.Process(record['pid'])
        for process in [parent,*parent.children(recursive=True)]:
            try:
                cpu+=sum(process.cpu_times()[:2])
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                pass
    log=BASE/record['log']
    return cpu,log.stat().st_size if log.exists() else 0


def terminate_owned(record):
    """Only the runner identity created by this queue and its current children."""
    if not identity_alive(record['pid'],record['process_created']):
        return
    parent=psutil.Process(record['pid'])
    command=parent.cmdline()
    runner=(ROOT/'scripts/next8/queue_runner.py').resolve()
    if not any(Path(arg).resolve()==runner for arg in command if arg.endswith('.py')):
        raise RuntimeError('Refuse to terminate a process not owned by the NEXT8 queue')
    processes=parent.children(recursive=True)+[parent]
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _,alive=psutil.wait_procs(processes,timeout=3)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass


def dispatch(items,capacity,observations):
    now=time.time()
    active=[]
    warnings=[]
    # Reserve every live owner before considering any READY item. A gate may
    # open before an already-running item later in queue order is visited.
    reserved=set()
    for item in items:
        if not item['id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in item['id']):
            raise ValueError('Queue ID must be a safe filename token')
        parent=BASE/f'runtime/queue/{item["id"]}'
        if (parent/'TERMINAL.json').exists():
            continue
        starts=sorted(parent.glob('attempt_*/START.json')) if parent.exists() else []
        if starts and not (starts[-1].parent/'DONE.json').exists():
            record=read_json(starts[-1])
            if identity_alive(record['pid'],record['process_created']):
                reserved.add(item['id'])
    for item in items:
        taskid=item['id']
        if not taskid or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in taskid):
            raise ValueError('Queue ID must be a safe filename token')
        parent=BASE/f'runtime/queue/{taskid}'
        terminal=parent/'TERMINAL.json'
        if terminal.exists():
            continue
        attempts=sorted(parent.glob('attempt_*/START.json')) if parent.exists() else []
        current=read_json(attempts[-1]) if attempts else None
        retry=False
        if current:
            attemptdir=attempts[-1].parent
            done=attemptdir/'DONE.json'
            recovered=attemptdir/'RECOVERY.json'
            if done.exists():
                result=read_json(done)
                save(str(terminal.relative_to(BASE)),{**result,
                    'status':'COMPLETE' if result['returncode']==0 else 'QUARANTINE',
                    'reason':'normal completion' if result['returncode']==0 else 'nonzero exit requires software/scientific classification'})
                continue
            if recovered.exists():
                retry=True
            elif identity_alive(current['pid'],current['process_created']):
                cpu,size=progress(current)
                key=taskid+'_'+str(current['attempt'])
                previous=observations.get(key,{})
                changed=cpu>previous.get('cpu',cpu)+.01 or size!=previous.get('size',size)
                last=now if changed else previous.get('last_progress',now)
                observations[key]={'cpu':cpu,'size':size,'last_progress':last}
                if now-last>1800:
                    terminate_owned(current)
                    reserved.discard(taskid)
                    save(str((attemptdir/'RECOVERY.json').relative_to(BASE)),{
                        'status':'HANG_SUSPECTED','timestamp':utc_now(),'no_progress_seconds':now-last,
                        'pid_identity_checked':True,'CPU_and_output_stalled':True,'safe_retry_limit':1})
                    warnings.append({'id':taskid,'status':'HANG_SUSPECTED'})
                    retry=True
                else:
                    active.append(current)
                    continue
            else:
                save(str((attemptdir/'RECOVERY.json').relative_to(BASE)),{
                    'status':'CRASH_OR_INTERRUPTION','timestamp':utc_now(),'safe_retry_limit':1,
                    'partial_artifacts_preserved':True})
                retry=True
            if retry and current['attempt']>=2:
                save(str(terminal.relative_to(BASE)),{'status':'QUARANTINE','timestamp':utc_now(),
                    'reason':'One safe retry exhausted','attempts':current['attempt']})
                continue
        if item.get('status')!='READY' or len(reserved)>=capacity:
            continue
        if not all((BASE/g).exists() and read_json(BASE/g).get('status') in ('PASS','FROZEN','COMPLETE') for g in item.get('gates',[])):
            continue
        script=(ROOT/item['script']).resolve()
        if not script.is_relative_to(ROOT/'scripts/next8') or digest(script)!=item['source_sha256']:
            save(str(terminal.relative_to(BASE)),{'status':'QUARANTINE','timestamp':utc_now(),
                'reason':'Script ownership or source hash mismatch; not executed'})
            continue
        attempt=1 if current is None else current['attempt']+1
        attemptdir=parent/f'attempt_{attempt}'
        attemptdir.mkdir(parents=True,exist_ok=True)
        save(str((attemptdir/'TASK.json').relative_to(BASE)),item)
        logpath=attemptdir/'runner.log'
        with logpath.open('a',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,'-B',str(ROOT/'scripts/next8/queue_runner.py'),
                '--task-id',taskid,'--attempt',str(attempt)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        record={'id':taskid,'attempt':attempt,'pid':process.pid,
            'process_created':psutil.Process(process.pid).create_time(),'created_utc':utc_now(),
            'log':(attemptdir/'child.log').relative_to(BASE).as_posix(),'task':item}
        save(str((attemptdir/'START.json').relative_to(BASE)),record)
        active.append(record)
        reserved.add(taskid)
    return active,warnings
