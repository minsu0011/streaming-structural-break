"""Create NEXT9 without mutating any earlier run, source, model or user diff."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    locator = ROOT / 'artifacts/next9/CURRENT_RUN.json'
    assert not locator.exists(), 'Existing NEXT9 requires explicit isolated resume; never overwrite'
    prior = json.loads((ROOT / 'artifacts/next8/CURRENT_RUN.json').read_text(encoding='utf-8'))
    oldbase = ROOT / prior['base']
    assert json.loads((oldbase / 'runtime/FINAL_COMPLETION_NEXT8.json').read_text())['status'] == 'COMPLETE'
    owner = json.loads((oldbase / 'runtime/SCHEDULER_OWNER.json').read_text())
    if psutil.pid_exists(owner['pid']):
        assert abs(psutil.Process(owner['pid']).create_time() - owner['process_created']) > 1
    run = 'run_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    base = ROOT / 'artifacts/next9' / run
    base.mkdir(parents=True, exist_ok=False)
    prompt = Path(r'C:\Users\minsu\.codex\attachments\d508c199-2147-428f-b408-409dd4d633cd\pasted-text.txt')
    (base / 'USER_SPECIFICATION_NEXT9.txt').write_bytes(prompt.read_bytes())
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    write(base / 'SESSION_NEXT9.json', dict(status='START_RECEIPT', run_id=run,
          started_utc=stamp, started_epoch=time.time(), prompt_sha256=sha(prompt),
          research_name='NEXT9_MOVEMENT_STATE_ORIENTATION', research_minimum_seconds=36000,
          final_audit_minimum_seconds=3600, pre_initialization_work_credited_seconds=0,
          time_basis='Union of successfully completed work with process CPU evidence; no idle, note lease or repeated worker-time sum',
          previous_NEXT8_base=prior['base'], prior_NEXT8_locator_sha256=sha(ROOT / 'artifacts/next8/CURRENT_RUN.json')))
    write(locator, dict(run_id=run, base=base.relative_to(ROOT).as_posix(), session_sha256=sha(base / 'SESSION_NEXT9.json')))
    for name, args in [('GIT_STATUS_BEFORE.bin', ['status', '--porcelain=v1', '-z']),
                       ('STAGED_PATCH_BEFORE.diff', ['diff', '--cached', '--binary']),
                       ('WORKTREE_PATCH_BEFORE.diff', ['diff', '--binary'])]:
        (base / name).write_bytes(subprocess.check_output(['git', *args], cwd=ROOT))
    vm = psutil.virtual_memory()
    write(base / 'RESOURCE_POLICY_NEXT9.json', dict(status='FROZEN', timestamp=stamp,
          maximum_workers=2, math_threads_per_worker=1, reduce_to_one_above_RAM_percent=80,
          pause_new_workers_above_RAM_percent=90, minimum_free_disk_bytes=40 * 1024**3,
          physical_cpus=psutil.cpu_count(logical=False), logical_cpus=psutil.cpu_count(),
          inherited_cpu_affinity=psutil.Process().cpu_affinity(), memory_total_bytes=vm.total,
          GPU_used=False, environment_installs_or_global_resource_changes=False))
    derived = []
    for old, new in [('src/next8/common.py', 'src/next9/common.py'),
                     ('src/next8/queue.py', 'src/next9/queue.py'),
                     ('src/next8/worker_environment.py', 'src/next9/worker_environment.py'),
                     ('scripts/next8/queue_runner.py', 'scripts/next9/queue_runner.py')]:
        before = (ROOT / old).read_text(encoding='utf-8')
        after = before.replace('next8', 'next9').replace('NEXT8', 'NEXT9')
        if new.endswith('queue.py'):
            after = after.replace("key=taskid+'_'+str(current['attempt'])", "key=taskid+'_'+str(current['attempt'])+'_'+str(current['process_created'])")
        if new.endswith(('queue.py', 'queue_runner.py')):
            after = after.replace("[sys.executable,'-B',", "[sys.executable,'-X','utf8','-B',")
        if new.endswith('worker_environment.py'):
            after = after.replace('    return result', "    result['PYTHONUTF8'] = '1'\n    result['PYTHONIOENCODING'] = 'utf-8'\n    return result")
        write(ROOT / new, after)
        derived.append(dict(original=old, original_sha256=sha(ROOT / old), derived=new, derived_sha256=sha(ROOT / new)))
    write(base / 'INFRASTRUCTURE_REUSE_NEXT9.json', dict(status='FROZEN', timestamp=stamp, records=derived,
          original_sources_modified=False, queue_observation_key_includes_process_identity=True,
          child_text_encoding_fixed_to_UTF8=True, prior_note_lease_not_used_for_research_time=True))
    write(base / 'runtime/TASK_QUEUE.json', [])
    print(json.dumps(dict(run_id=run, base=str(base), prior=str(oldbase)), ensure_ascii=False))


if __name__ == '__main__':
    main()
