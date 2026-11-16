"""Exercise artifact interruption boundaries and admission, without real jobs."""
import json
from pathlib import Path
import pytest
from src.next3 import artifacts, maintenance


def test_immutable_report_never_replaced(tmp_path):
    path = tmp_path/'POLICY.json'
    artifacts.atomic_json(path, {'version': 1}, immutable=True)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        artifacts.atomic_json(path, {'version': 2}, immutable=True)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob('*.tmp'))


def test_failed_atomic_publish_keeps_previous_report(tmp_path, monkeypatch):
    path = tmp_path/'HEARTBEAT.json'
    artifacts.atomic_json(path, {'complete': 1})
    def interrupted(*args):
        raise OSError('Simulated rename interruption')
    monkeypatch.setattr(artifacts.os, 'replace', interrupted)
    with pytest.raises(OSError):
        artifacts.atomic_json(path, {'complete': 2})
    assert json.loads(path.read_text()) == {'complete': 1}
    assert len(list(tmp_path.glob('*.tmp'))) == 1


def test_incomplete_or_nonfinite_registration_not_published(tmp_path):
    path = tmp_path/'RESULTS.json'
    with pytest.raises(ValueError):
        artifacts.atomic_json(path, {'score': float('nan')}, immutable=True)
    assert not path.exists()


def test_research_job_requires_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance, 'BASE', tmp_path)
    with pytest.raises(RuntimeError, match='blocked before preflight'):
        maintenance.register_job({'id':'R1','kind':'research','argv':['scripts/test.py'],
            'description':'Must not reach training','hypothesis':'missing.json'})


@pytest.mark.parametrize('identifier', ['../escape','a/b','a\\b','', 'a b'])
def test_queue_identifier_cannot_escape(tmp_path, monkeypatch, identifier):
    monkeypatch.setattr(maintenance, 'BASE', tmp_path)
    with pytest.raises(ValueError):
        maintenance.register_job({'id':identifier,'kind':'maintenance','argv':['scripts/test.py'],'description':'invalid'})


def test_registration_is_exclusive_and_no_shell(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance, 'BASE', tmp_path)
    job = {'id':'M1','kind':'maintenance','argv':['scripts/test.py'],'description':'safe example'}
    maintenance.register_job(job)
    with pytest.raises(FileExistsError):
        maintenance.register_job(job)
    with pytest.raises(ValueError):
        maintenance.register_job({**job, 'id':'M2', 'argv':'echo not a Python argument list'})
    with pytest.raises(ValueError):
        maintenance.register_job({**job, 'id':'M3', 'argv':['../outside.py']})


def test_completed_queue_job_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance, 'BASE', tmp_path)
    artifacts.atomic_json(tmp_path/'SESSION.json', {'started_utc':artifacts.utc_now()})
    maintenance.register_job({'id':'M1','kind':'maintenance','argv':['scripts/test.py'],'description':'done'})
    artifacts.atomic_json(tmp_path/'jobs/M1/STATE.json', {'status':'COMPLETE','attempt':1})
    supervisor = object.__new__(maintenance.Supervisor)
    assert supervisor.next_job() is None


def test_retry_limit_quarantines_and_moves_to_next_job(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance, 'BASE', tmp_path)
    artifacts.atomic_json(tmp_path/'SESSION.json', {'started_utc':artifacts.utc_now()})
    for name in ('M1','M2'):
        maintenance.register_job({'id':name,'kind':'maintenance','argv':['scripts/test.py'],
            'description':'retry control','retry_safe':True})
    artifacts.atomic_json(tmp_path/'jobs/M1/STATE.json', {'status':'FAILED','attempt':2})
    supervisor = object.__new__(maintenance.Supervisor)
    job, attempt = supervisor.next_job()
    assert job['id'] == 'M2' and attempt == 1
    assert artifacts.read_json(tmp_path/'jobs/M1/STATE.json')['status'] == 'QUARANTINED'


def test_one_retry_only_when_outputs_are_restart_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance, 'BASE', tmp_path)
    artifacts.atomic_json(tmp_path/'SESSION.json', {'started_utc':artifacts.utc_now()})
    maintenance.register_job({'id':'M1','kind':'maintenance','argv':['scripts/test.py'],
        'description':'safe retry','retry_safe':True})
    artifacts.atomic_json(tmp_path/'jobs/M1/STATE.json', {'status':'FAILED','attempt':1})
    supervisor = object.__new__(maintenance.Supervisor)
    job, attempt = supervisor.next_job()
    assert job['id'] == 'M1' and attempt == 2


def test_process_exit_during_resource_probe_is_not_catastrophic(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance,'ROOT',tmp_path)
    class Child:
        pid=123
        def poll(self):return None
    def vanished(pid):
        raise maintenance.psutil.NoSuchProcess(pid)
    monkeypatch.setattr(maintenance.psutil,'Process',vanished)
    supervisor=object.__new__(maintenance.Supervisor)
    supervisor.child=Child()
    assert supervisor.inspect_child() is False
    assert supervisor.child is not None


def test_invalid_module_argument_is_rejected_before_queue_publish(tmp_path,monkeypatch):
    monkeypatch.setattr(maintenance,'BASE',tmp_path)
    with pytest.raises(ValueError):
        maintenance.register_job({'id':'M1','kind':'maintenance','argv':['-m'],'description':'invalid'})
    assert not (tmp_path/'queue/M1.json').exists()
