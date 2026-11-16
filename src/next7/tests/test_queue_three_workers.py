"""A third research slot must preserve reservation of every existing owner."""
from types import SimpleNamespace
import pytest
from src.next7 import queue
from src.next7.common import atomic, digest


@pytest.mark.parametrize('existing,capacity,new_expected', [(2,3,1),(3,3,0),(3,1,0)])
def test_ready_tasks_before_live_owners_never_exceed_available_slots(tmp_path, monkeypatch,
                                                                 existing, capacity, new_expected):
    root, base = tmp_path/'project', tmp_path/'artifacts'
    script = root/'scripts/next7/probe.py'
    script.parent.mkdir(parents=True)
    script.write_text('pass\n', encoding='utf-8')
    monkeypatch.setattr(queue, 'ROOT', root)
    monkeypatch.setattr(queue, 'BASE', base)
    monkeypatch.setattr(queue, 'save', lambda name, value: atomic(base/name, value))
    alive = set(range(101,101+existing))
    launched = []
    monkeypatch.setattr(queue, 'identity_alive', lambda pid, created: pid in alive)
    monkeypatch.setattr(queue, 'progress', lambda record: (10.,100))
    def popen(*args, **kwargs):
        pid = 201+len(launched)
        launched.append(pid)
        alive.add(pid)
        return SimpleNamespace(pid=pid)
    monkeypatch.setattr(queue.subprocess, 'Popen', popen)
    monkeypatch.setattr(queue.psutil, 'Process', lambda pid: SimpleNamespace(create_time=lambda:float(pid)))
    def item(name):
        return {'id':name,'status':'READY','script':'scripts/next7/probe.py',
                'source_sha256':digest(script),'arguments':[],'gates':[]}
    running = []
    for i in range(existing):
        name = f'running_{i}'
        atomic(base/f'runtime/queue/{name}/attempt_1/START.json',
               {'id':name,'attempt':1,'pid':101+i,'process_created':float(101+i),
                'log':f'runtime/queue/{name}/attempt_1/child.log'})
        running.append(item(name))
    items = [item(f'earlier_ready_{i}') for i in range(4)]+running
    active, warnings = queue.dispatch(items,capacity,{})
    assert len(launched)==new_expected and len(active)==existing+new_expected and not warnings
    active, warnings = queue.dispatch(items,capacity,{})
    assert len(launched)==new_expected and len(active)==existing+new_expected and not warnings
