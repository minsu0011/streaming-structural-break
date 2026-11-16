"""Atomic, durable reports and immutable registration for NEXT3."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import uuid

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/next3'


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, content, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    if immutable:
        try:
            # Publish the complete bytes exclusively; existing registrations
            # cannot be overwritten and partial JSON is never visible.
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        os.replace(temporary, path)


def progress(task, detail, *, phase='ACTIVE_RESEARCH', experiment=None):
    atomic_json(BASE/'runtime/ACTIVITY.json', {
        'timestamp': utc_now(), 'pid': os.getpid(), 'phase': phase,
        'task': task, 'detail': detail, 'experiment': experiment})
