"""NEXT7 monitor and bounded queue dispatcher; idle monitor time is not work."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import psutil
from src.next7.common import BASE, read_json, save, csv_write, utc_now, union_seconds, digest
from src.next7.queue import dispatch
from src.next7.runtime_accounting import audit_admitted,allocate_interval


def monitor(once=False):
    session = read_json(BASE / 'SESSION_NEXT7.json')
    state_path = BASE / 'runtime/SCHEDULER_STATE.json'
    state = read_json(state_path) if state_path.exists() else {
        'intervals': [], 'audit_intervals': [], 'last_epoch': time.time(),
        'last_checkpoint': 0, 'observations': {}, 'checkpoint_count': 0,
    }
    lock = BASE / 'runtime/SCHEDULER_OWNER.json'
    if lock.exists():
        old = read_json(lock)
        if old['pid'] != os.getpid() and psutil.pid_exists(old['pid']):
            p = psutil.Process(old['pid'])
            if abs(p.create_time() - old['process_created']) < 1:
                raise RuntimeError('A live NEXT7 scheduler already owns the clock')
    save('runtime/SCHEDULER_OWNER.json', {'pid': os.getpid(),
         'process_created': psutil.Process().create_time(), 'created_utc': utc_now()}, False)
    queue_observations=state.setdefault('queue_observations',{})
    executing_source_sha256=digest(__file__)
    session_sha256=digest(BASE/'SESSION_NEXT7.json')
    while True:
        now = time.time()
        admission_path=BASE/'runtime/FINAL_AUDIT_ADMISSION.json'
        admission=read_json(admission_path) if admission_path.exists() else {}
        admitted=audit_admitted(admission,union_seconds(state['intervals']),session_sha256,
            session['research_minimum_seconds'])
        accounting_warnings=[]
        def record_interval(start,stop,phase):
            allocation=allocate_interval(start,stop,phase,admitted=admitted,
                admission_epoch=admission.get('admitted_epoch'))
            if allocation is not None:
                target,interval=allocation
                state[target].append(interval)
            elif phase=='FINAL_AUDIT' and not admitted:
                accounting_warnings.append('Premature FINAL_AUDIT evidence excluded; research minimum and admission gate required.')
        controller_path = BASE / 'runtime/CONTROLLER_WORK.json'
        controller = read_json(controller_path) if controller_path.exists() else {}
        controller_live = now - controller.get('epoch', 0) <= controller.get('lease_seconds', 0)
        # Retain short work leases even when a later note replaced the latest
        # controller file between monitor ticks. No unobserved gaps are filled.
        for path in (BASE/'runtime/controller_notes').glob('*.json'):
            n=read_json(path)
            start=max(state['last_epoch'],n['epoch'],now-60)
            stop=min(now,n['epoch']+n['lease_seconds'])
            if stop>start:
                record_interval(start,stop,n.get('phase','PRODUCTIVE_RESEARCH'))
        jobs = [read_json(p) for p in (BASE / 'runtime/jobs').glob('*.json')]
        active = []
        live = []
        # A live job alone is insufficient: CPU must progress or a fresh completion
        # receipt must cover this interval. This excludes stalled/idle children.
        for job in jobs:
            previous = state['observations'].get(job['id'], {})
            if job['status'] == 'RUNNING' and psutil.pid_exists(job['pid']):
                try:
                    proc = psutil.Process(job['pid'])
                    if proc.create_time()>job['started_epoch']+1:
                        continue # Reused PID is not this recorded work unit.
                    cpu = sum(proc.cpu_times()[:2])
                    for child in proc.children(recursive=True):
                        try:
                            cpu+=sum(child.cpu_times()[:2])
                        except (psutil.NoSuchProcess,psutil.AccessDenied):
                            pass
                except (psutil.NoSuchProcess,psutil.AccessDenied):
                    continue
                if now-job['heartbeat_epoch']<100:
                    live.append(job)
                progressed = cpu > previous.get('cpu', cpu) + 0.01
                if progressed and now - job['heartbeat_epoch'] < 100:
                    active.append(job)
                    record_interval(max(state['last_epoch'],job['started_epoch'],now-60),now,
                        'FINAL_AUDIT' if job['kind']=='FINAL_AUDIT' else 'PRODUCTIVE_RESEARCH')
                state['observations'][job['id']] = {'cpu': cpu, 'epoch': now,
                    'last_progress': now if progressed else previous.get('last_progress', now)}
            elif job['status'] in ('COMPLETE', 'FAILED') and job.get('ended_epoch', 0) > state['last_epoch']:
                if job.get('cpu_seconds', 0) > 0.01:
                    stop = min(now, job['ended_epoch'])
                    start = max(state['last_epoch'], job['started_epoch'],now-60)
                    if stop >= start:
                        record_interval(start,stop,'FINAL_AUDIT' if job['kind']=='FINAL_AUDIT' else 'PRODUCTIVE_RESEARCH')
        productive = union_seconds(state['intervals'])
        audit = union_seconds(state['audit_intervals'])
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage(str(ROOT))
        queue_path = BASE / 'runtime/TASK_QUEUE.json'
        queue = read_json(queue_path) if queue_path.exists() else []
        # Four single-thread jobs fit the measured eight-core host and RAM.
        # The original high-memory admission reduction remains in force.
        capacity = 1 if vm.percent > 85 else 4
        running,queue_warnings=dispatch(queue,capacity,queue_observations)
        owned_pids={job['pid'] for job in live}
        for owner in running:
            try:
                process=psutil.Process(owner['pid'])
                if abs(process.create_time()-owner['process_created'])<.001:
                    owned_pids.add(process.pid)
                    owned_pids.update(p.pid for p in process.children(recursive=True))
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                pass
        hang = []
        for job in jobs:
            obs = state['observations'].get(job['id'], {})
            if job['status'] == 'RUNNING' and now - obs.get('last_progress', now) > 1800:
                hang.append({'task': job['task'], 'pid': job['pid'], 'status': 'HANG_SUSPECTED',
                             'action': 'Controller inspects own child; one safe retry then quarantine.'})
        best_path = BASE / 'runtime/CURRENT_RESEARCH_STATUS.json'
        best = read_json(best_path) if best_path.exists() else {}
        heartbeat = {'timestamp': utc_now(), 'wall_seconds': now - session['started_epoch'],
            'productive_seconds': productive, 'audit_seconds': audit,
            'current_phase': 'FINAL_AUDIT' if admitted else 'PRODUCTIVE_RESEARCH',
            'current_hypothesis': controller.get('hypothesis'),
            'current_experiment': ([j['task'] for j in live] or controller.get('task')) if controller_live or live else None,
            'completed_count': sum(j['status'] == 'COMPLETE' for j in jobs),
            'failed_count': sum(j['status'] == 'FAILED' for j in jobs),
            'current_best': best.get('best', 'FROZEN_PRIMARY_BL80'),
            'current_bottleneck': best.get('bottleneck', 'B1_SCALE_CONTRACTION'),
            'next_task': best.get('next_task', 'guard then protocol freeze'),
            'RAM_percent': vm.percent, 'RAM_available_bytes': vm.available,
            'disk_free_bytes': disk.free, 'child_pids': sorted(owned_pids),
            'queue_runner_pids': [r['pid'] for r in running],
            'queue_capacity': capacity,'scientific_full_count':best.get('full_count',0),
            'audit_admission_valid':admitted,'accounting_warnings':sorted(set(accounting_warnings)),
            'hang_suspected': hang+queue_warnings, 'research_minimum_met': productive >= 36000,
            'idle_monitor_time_counted': False, 'parallel_work_not_double_counted': True}
        save('runtime/HEARTBEAT.json', heartbeat, False)
        state['last_epoch'] = now
        if now - state['last_checkpoint'] >= 1800:
            state['checkpoint_count'] += 1
            save(f'checkpoints/NEXT7_CHECKPOINT_{state["checkpoint_count"]:03d}.json', {
                **heartbeat, 'screen_count': best.get('screen_count', 0),
                'full_count': best.get('full_count', 0), 'seal_status': 'CLOSED', 'shadow_status': 'CLOSED',
                'source_sha256': executing_source_sha256})
            state['last_checkpoint'] = now
        save('runtime/SCHEDULER_STATE.json', state, False)
        ledger = [read_json(p) for p in (BASE / 'ledger').glob('*.json')]
        csv_write('EXPERIMENT_LEDGER_NEXT7.csv', sorted(ledger, key=lambda r: r['started_epoch']), immutable=False)
        if once or (BASE / 'runtime/STOP_SCHEDULER.json').exists():
            break
        time.sleep(45)  # A lightweight monitor timer, never credited by itself.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    monitor(parser.parse_args().once)
