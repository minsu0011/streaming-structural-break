"""Owned bounded queue with conservative clock and 35-minute checkpoints."""
from pathlib import Path
import os
import sys
import time
import psutil
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next9
from src.next9.common import BASE, atomic, read_json, digest, utc_now
from src.next9.queue import dispatch, identity_alive
from src.next9.work_time import snapshot


def main():
    owner_path = BASE / 'runtime/SCHEDULER_OWNER.json'
    if owner_path.exists():
        owner = read_json(owner_path)
        if identity_alive(owner['pid'], owner['process_created']):
            raise RuntimeError('An existing live NEXT9 scheduler owns this run')
    atomic(owner_path, dict(pid=os.getpid(), process_created=psutil.Process().create_time(), timestamp=utc_now()), immutable=False)
    observations = {}
    completed = {}
    last_checkpoint = 0.
    while not (BASE / 'runtime/STOP_SCHEDULER.json').exists():
        now = time.time()
        resource = read_json(BASE / 'RESOURCE_POLICY_NEXT9.json')
        vm = psutil.virtual_memory()
        free_disk = psutil.disk_usage(str(ROOT)).free
        capacity = 1 if vm.percent > resource['reduce_to_one_above_RAM_percent'] else resource['maximum_workers']
        if vm.percent > resource['pause_new_workers_above_RAM_percent'] or free_disk < resource['minimum_free_disk_bytes']:
            capacity = 0
        queue = read_json(BASE / 'runtime/TASK_QUEUE.json')
        active, warnings = dispatch(queue, capacity, observations)
        clock = snapshot(completed)
        status_path = BASE / 'runtime/CURRENT_RESEARCH_STATUS.json'
        status = read_json(status_path) if status_path.exists() else {}
        remaining = []
        failed = []
        for item in queue:
            p = BASE / 'runtime/queue' / item['id'] / 'TERMINAL.json'
            if not p.exists():
                remaining.append(item['id'])
            elif read_json(p)['status'] != 'COMPLETE':
                failed.append(item['id'])
        heartbeat = dict(timestamp=utc_now(), phase=status.get('phase', 'RESTORE'),
                         productive_seconds=clock['productive_seconds'], audit_seconds=clock['audit_seconds'],
                         clock_basis='Completed work with process CPU evidence; no note leases or idle',
                         active_tasks=[r['id'] for r in active], pending_tasks=remaining, failed_tasks=failed,
                         active_bottleneck=status.get('bottleneck', 'B9_NORMALIZATION_STATE_AVAILABILITY'),
                         RAM_percent=vm.percent, disk_free_bytes=free_disk, queue_capacity=capacity,
                         ownership_warnings=warnings, source_sha256=digest(__file__),
                         target_free_candidates=status.get('target_free_candidates'),
                         certified_variable_count=status.get('certified_variable_count'),
                         cross_edge_count=status.get('cross_edge_count'), inner_best=status.get('inner_best'),
                         outer_released=(BASE / 'OUTER_RELEASE_NEXT9.json').exists(), next_task=status.get('next_task'))
        atomic(BASE / 'runtime/HEARTBEAT_NEXT9.json', heartbeat, immutable=False)
        atomic(BASE / 'runtime/QUEUE_OBSERVATIONS_NEXT9.json', observations, immutable=False)
        if now - last_checkpoint >= 2100:
            atomic(BASE / 'checkpoints' / ('NEXT9_CHECKPOINT_' + str(time.time_ns()) + '.json'), heartbeat)
            last_checkpoint = now
        time.sleep(30)  # Excluded completely from the research/audit clocks.


if __name__ == '__main__':
    main()
