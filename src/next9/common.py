"""Atomic NEXT9 receipts and evidence-backed activity accounting.

Productive time is a UNION of verified work intervals, never a sum of workers.
The scheduler's own polling and idle waiting do not count as productive work.
"""
from pathlib import Path
from contextlib import contextmanager
import csv
import datetime as dt
import hashlib
import json
import os
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
_locator = json.loads((ROOT / 'artifacts/next9/CURRENT_RUN.json').read_text(encoding='utf-8'))
RUN_ID = os.environ.get('NEXT9_RUN_ID', _locator['run_id'])
BASE = ROOT / 'artifacts/next9' / RUN_ID


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value, *, immutable=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        raise FileExistsError(str(path))
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('x', encoding='utf-8', newline='\n') as f:
            if isinstance(value, str):
                f.write(value)
            else:
                json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
                f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        if immutable:
            # Windows rename fails when a destination exists; no overwrite race.
            os.rename(tmp, path)
        else:
            os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()  # Only this function's unique, known temporary file.


def save(name, value, immutable=True):
    atomic(BASE / name, value, immutable=immutable)


def csv_write(name, rows, *, immutable=True):
    import io
    fields = list(dict.fromkeys(k for row in rows for k in row))
    f = io.StringIO(newline='')
    if fields:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    atomic(BASE / name, f.getvalue(), immutable=immutable)


def ids_hash(ids):
    return hashlib.sha256(','.join(map(str, sorted(map(int, ids)))).encode()).hexdigest()


def note(task, detail='', hypothesis='RESTORE', phase='PRODUCTIVE_RESEARCH'):
    """A short-lived controller work lease; refresh only while actually working."""
    record = {
        'timestamp': utc_now(), 'epoch': time.time(), 'lease_seconds': 150,
        'task': task, 'detail': detail, 'hypothesis': hypothesis, 'phase': phase,
    }
    save('runtime/controller_notes/' + str(time.time_ns()) + '.json', record)
    save('runtime/CONTROLLER_WORK.json', record, False)
    print(utc_now(), task, detail, flush=True)


def union_seconds(intervals):
    total = 0.0
    end = float('-inf')
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError('Reversed work interval')
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


@contextmanager
def activity(task, kind='RESEARCH', hypothesis='RESTORE'):
    token = uuid.uuid4().hex
    start = time.time()
    cpu = time.process_time()
    status = 'RUNNING'
    stop = threading.Event()
    path = 'runtime/jobs/' + token + '.json'
    base = {'id': token, 'task': task, 'kind': kind, 'hypothesis': hypothesis,
            'pid': os.getpid(), 'started_epoch': start, 'started_utc': utc_now()}

    def beat():
        while not stop.is_set():
            save(path, {**base, 'status': 'RUNNING', 'heartbeat_epoch': time.time(),
                        'cpu_seconds': time.process_time() - cpu}, False)
            stop.wait(30)  # Monitoring thread only; its idle time is excluded.

    worker = threading.Thread(target=beat, daemon=True)
    worker.start()
    print(utc_now(), 'START', task, flush=True)
    error = None
    try:
        yield token
        status = 'COMPLETE'
    except BaseException as exc:
        status = 'FAILED'
        error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        stop.set()
        worker.join()
        result = {**base, 'status': status, 'ended_epoch': time.time(), 'ended_utc': utc_now(),
                  'cpu_seconds': time.process_time() - cpu, 'error': error}
        save(path, result, False)
        save('ledger/' + token + '.json', result)
        print(utc_now(), status, task, flush=True)


def require_guard():
    receipt = read_json(BASE / 'PREFLIGHT_NEXT9.json')
    if receipt['status'] != 'PASS':
        raise RuntimeError('NEXT9 data guard gate failed')
    from src.next5.guard import ResearchGuard
    guard = ResearchGuard(ROOT)
    if guard.split.digest != receipt['split_sha256']:
        raise RuntimeError('Research split changed')
    guard.check_state()
    return guard
