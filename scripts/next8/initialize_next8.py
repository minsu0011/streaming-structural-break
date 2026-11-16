"""Initialize an isolated NEXT8 run by reusing the reviewed NEXT7 worker clock."""
from pathlib import Path
import datetime as dt
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import psutil
from src.next7.common import atomic, digest, utc_now


def main():
    prompt = Path(r'C:\Users\minsu\.codex\attachments\711b78f8-5c16-450b-97e6-45099267daf4\pasted-text.txt')
    run_id = 'run_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    base = ROOT / 'artifacts/next8' / run_id
    base.mkdir(parents=True, exist_ok=False)
    source_dir = ROOT / 'src/next8'
    source_dir.mkdir(parents=True, exist_ok=True)
    prompt_copy = base / 'USER_SPECIFICATION_NEXT8.txt'
    shutil.copyfile(prompt, prompt_copy)
    assert digest(prompt_copy) == digest(prompt)
    started = time.time()
    atomic(base / 'SESSION_NEXT8.json', {'started_utc': utc_now(), 'started_epoch': started,
        'status': 'START_RECEIPT', 'run_id': run_id,
        'research_name': 'NEXT8_CONDITIONAL_VARIABLE_FEATURE_DYNAMICS',
        'prompt_sha256': digest(prompt), 'research_minimum_seconds': 36000,
        'final_audit_target_seconds': 3600, 'no_idle_padding': True,
        'productive_accounting': 'Union of evidenced live work intervals, with separate admitted audit',
        'pre_initialization_work_credited_seconds': 0,
        'scope': 'Existing RP6400 and existing grouped outer folds only; Shadow/Seal/reduced/public/cloud forbidden'})
    affinity = psutil.Process().cpu_affinity()
    vm = psutil.virtual_memory()
    atomic(base / 'RESOURCE_POLICY_NEXT8.json', {'status': 'FROZEN', 'created_utc': utc_now(),
        'logical_cpus': psutil.cpu_count(), 'physical_cpus': psutil.cpu_count(logical=False),
        'inherited_allowed_cpu_affinity': affinity, 'memory_total_bytes': vm.total,
        'memory_available_at_start_bytes': vm.available,
        'maximum_workers': min(2, max(1, len(affinity) // 4)), 'math_threads_per_worker': 1,
        'reduce_to_one_above_RAM_percent': 85, 'GPU_used': False,
        'affinity_or_system_resource_restrictions_changed': False,
        'existing_environment_modified': False})
    for label, args in [('GIT_STATUS_BEFORE.bin', ['status', '--porcelain=v1', '-z']),
                        ('STAGED_PATCH_BEFORE.diff', ['diff', '--cached', '--binary']),
                        ('WORKTREE_PATCH_BEFORE.diff', ['diff', '--binary'])]:
        raw = subprocess.check_output(['git', *args], cwd=ROOT)
        (base / label).write_bytes(raw)
    pointer = ROOT / 'artifacts/next8/CURRENT_RUN.json'
    assert not pointer.exists(), 'Existing NEXT8 locator requires an explicit new-run setup; never overwrite'
    atomic(pointer, {'run_id': run_id, 'base': base.relative_to(ROOT).as_posix(),
                     'session_sha256': digest(base / 'SESSION_NEXT8.json')})
    init_text = '''"""NEXT8-only process environment; inherited CPU affinity is unchanged."""
import json
import os
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
_locator = json.loads((_root / 'artifacts/next8/CURRENT_RUN.json').read_text(encoding='utf-8'))
_run = os.environ.get('NEXT8_RUN_ID', _locator['run_id'])
if not _run.startswith('run_') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_' for c in _run):
    raise ValueError('Invalid NEXT8 run identifier')
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(_root / 'artifacts/next8' / _run / 'runtime/numba_cache')
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
'''
    with (source_dir / '__init__.py').open('x', encoding='utf-8', newline='') as stream:
        stream.write(init_text)
    sources = [
        ('src/next7/common.py', 'src/next8/common.py'),
        ('src/next7/queue.py', 'src/next8/queue.py'),
        ('src/next7/runtime_accounting.py', 'src/next8/runtime_accounting.py'),
        ('src/next7/worker_environment.py', 'src/next8/worker_environment.py'),
        ('scripts/next7/queue_runner.py', 'scripts/next8/queue_runner.py'),
        ('scripts/next7/run_next7_autonomous.py', 'scripts/next8/run_next8_autonomous.py'),
    ]
    records = []
    for original, target in sources:
        before = (ROOT / original).read_text(encoding='utf-8')
        after = before.replace('next7', 'next8').replace('NEXT7', 'NEXT8')
        if target == 'src/next8/common.py':
            old = "BASE = ROOT / 'artifacts/next8'"
            new = "_locator = json.loads((ROOT / 'artifacts/next8/CURRENT_RUN.json').read_text(encoding='utf-8'))\nRUN_ID = os.environ.get('NEXT8_RUN_ID', _locator['run_id'])\nBASE = ROOT / 'artifacts/next8' / RUN_ID"
            assert after.count(old) == 1
            after = after.replace(old, new)
        if target == 'scripts/next8/run_next8_autonomous.py':
            old = "        # Four single-thread jobs fit the measured eight-core host and RAM.\n        # The original high-memory admission reduction remains in force.\n        capacity = 1 if vm.percent > 85 else 4"
            new = "        resource = read_json(BASE / 'RESOURCE_POLICY_NEXT8.json')\n        capacity = 1 if vm.percent > resource['reduce_to_one_above_RAM_percent'] else resource['maximum_workers']"
            assert after.count(old) == 1
            after = after.replace(old, new)
            after = after.replace("    while True:\n        now = time.time()", "    while True:\n        if (BASE / 'runtime/STOP_SCHEDULER.json').exists():\n            break\n        now = time.time()")
            after = after.replace("'B1_SCALE_CONTRACTION'", "'B7_NORMALIZATION_AND_LINEAGE'")
        path = ROOT / target
        with path.open('x', encoding='utf-8', newline='') as stream:
            stream.write(after)
        diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                          fromfile=original, tofile=target))
        diff_path = base / 'infrastructure_reuse' / (Path(target).stem + '.diff')
        atomic(diff_path, diff)
        records.append({'original': original, 'original_sha256': digest(ROOT / original),
                        'derived': target, 'derived_sha256': digest(path), 'diff_sha256': digest(diff_path)})
    atomic(base / 'INFRASTRUCTURE_REUSE_NEXT8.json', {'status': 'FROZEN_REUSE',
        'created_utc': utc_now(), 'records': records,
        'intent': 'Reuse existing owned queue, one retry, activity union clock and checkpoints; isolate only NEXT8 namespace and measured resource cap',
        'previous_files_modified': False, 'scheduler_invents_research': False,
        'initializer_sha256': digest(Path(__file__))})
    atomic(base / 'runtime/TASK_QUEUE.json', [])
    atomic(base / 'runtime/CURRENT_RESEARCH_STATUS.json', {
        'best': 'FROZEN_PRIMARY_BL80_NO_NEXT8_SCORES',
        'bottleneck': ['B7_NORMALIZATION_AND_LINEAGE', 'B9_NESTED_SELECTION'],
        'next_task': 'Validate protected sources, RP cache and existing outer/inner reference provenance',
        'full_count': 0, 'screen_count': 0})
    from src.next8.common import note
    note('NEXT8 initialized; restoring prior authority', 'No prior source or model overwritten; consistent normalization and strictly nested feature selection first', hypothesis='RESTORE')
    print(json.dumps({'run_id': run_id, 'base': str(base), 'prompt_sha256': digest(prompt),
                      'maximum_workers': min(2, max(1, len(affinity)//4))}))


if __name__ == '__main__':
    main()
