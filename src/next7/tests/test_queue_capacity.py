"""Queue gates opening ahead of live tasks must not oversubscribe workers."""
from types import SimpleNamespace
import time
import pytest
from src.next7 import queue
from src.next7.common import atomic, digest


@pytest.mark.parametrize('capacity', [1, 2])
def test_newly_unblocked_earlier_tasks_respect_all_later_live_owners(tmp_path, monkeypatch, capacity):
    root, base = tmp_path / 'project', tmp_path / 'artifacts'
    script = root / 'scripts/next7/probe.py'
    script.parent.mkdir(parents=True)
    script.write_text('pass\n', encoding='utf-8')
    monkeypatch.setattr(queue, 'ROOT', root)
    monkeypatch.setattr(queue, 'BASE', base)
    monkeypatch.setattr(queue, 'save', lambda name, value: atomic(base / name, value))
    launched = []
    alive = {101, 102}
    monkeypatch.setattr(queue, 'identity_alive', lambda pid, created: pid in alive)
    monkeypatch.setattr(queue, 'progress', lambda record: (10., 100))
    def popen(*args, **kwargs):
        pid = 200 + len(launched)
        launched.append(pid)
        alive.add(pid)
        return SimpleNamespace(pid=pid)
    monkeypatch.setattr(queue.subprocess, 'Popen', popen)
    monkeypatch.setattr(queue.psutil, 'Process', lambda pid: SimpleNamespace(create_time=lambda: float(pid)))
    def item(name):
        return {'id': name, 'status': 'READY', 'script': 'scripts/next7/probe.py',
            'source_sha256': digest(script), 'arguments': [], 'gates': []}
    for index, name in enumerate(('running_a', 'running_b')):
        path = base / f'runtime/queue/{name}/attempt_1'
        atomic(path / 'START.json', {'id': name, 'attempt': 1, 'pid': 101 + index,
            'process_created': float(101 + index), 'log': f'runtime/queue/{name}/attempt_1/child.log'})
    items = [item('earlier_ready_a'), item('earlier_ready_b'), item('running_a'), item('running_b')]
    active, warnings = queue.dispatch(items, capacity, {})
    assert len(active) == 2 and not launched and not warnings
    atomic(base / 'runtime/queue/running_a/attempt_1/DONE.json', {'returncode': 0})
    active, warnings = queue.dispatch(items, capacity, {})
    assert len(launched) == (1 if capacity == 2 else 0)
    assert len(active) == capacity and not warnings


def test_source_hash_mismatch_is_quarantined_without_execution(tmp_path, monkeypatch):
    root, base = tmp_path / 'project', tmp_path / 'artifacts'
    script = root / 'scripts/next7/probe.py'
    script.parent.mkdir(parents=True)
    script.write_text('pass\n', encoding='utf-8')
    monkeypatch.setattr(queue, 'ROOT', root)
    monkeypatch.setattr(queue, 'BASE', base)
    monkeypatch.setattr(queue, 'save', lambda name, value: atomic(base / name, value))
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda *a, **k: pytest.fail('Mismatched source executed'))
    items = [{'id': 'changed_source', 'status': 'READY', 'script': 'scripts/next7/probe.py', 'source_sha256': 'wrong'}]
    active, warnings = queue.dispatch(items, 2, {})
    assert not active
    receipt = queue.read_json(base / 'runtime/queue/changed_source/TERMINAL.json')
    assert receipt['status'] == 'QUARANTINE'
