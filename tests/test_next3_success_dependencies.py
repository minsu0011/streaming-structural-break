from pathlib import Path
import pytest
from src.next3 import maintenance
from src.next3.artifacts import atomic_json, read_json, utc_now
from src.next3.job_dependencies import validate_dependencies, dependency_observations, prerequisites_succeeded


def queued(base, name, **extra):
    job = {'id': name, 'kind': 'maintenance', 'argv': ['scripts/test.py'], 'description': 'isolated fixture', **extra}
    atomic_json(base/'queue'/f'{name}.json', job, immutable=True)
    return job


@pytest.mark.parametrize('status,code,allowed', [
    ('RUNNING', None, False), ('FAILED', 1, False), ('QUARANTINED', 0, False),
    ('COMPLETE', 0, True), ('COMPLETE', None, False), ('COMPLETE', 1, False), ('COMPLETE', False, False),
])
def test_success_requires_the_actual_zero_exit_terminal_state(tmp_path, status, code, allowed):
    queued(tmp_path, 'UNIT')
    job = {'id': 'COPY', 'depends_on_success': ['UNIT']}
    assert validate_dependencies(job, tmp_path) == ('UNIT',)
    atomic_json(tmp_path/'jobs/UNIT/STATE.json', {'id': 'UNIT', 'status': status, 'exit_code': code})
    assert prerequisites_succeeded(job, tmp_path) is allowed


@pytest.mark.parametrize('dependencies', ['UNIT', [None], ['../UNIT'], ['COPY'], ['UNIT', 'UNIT'], ['NOT_REGISTERED']])
def test_registration_rejects_ambiguous_or_unresolvable_dependencies(tmp_path, dependencies):
    queued(tmp_path, 'UNIT')
    with pytest.raises(ValueError): validate_dependencies({'id': 'COPY', 'depends_on_success': dependencies}, tmp_path)


def test_existing_artifact_cannot_replace_failed_test_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance, 'BASE', tmp_path); monkeypatch.setattr(maintenance, 'ROOT', tmp_path)
    atomic_json(tmp_path/'SESSION.json', {'status': 'ACTIVE_RESEARCH', 'started_utc': utc_now()})
    queued(tmp_path, 'A_UNIT')
    atomic_json(tmp_path/'jobs/A_UNIT/STATE.json', {'id': 'A_UNIT', 'status': 'QUARANTINED', 'attempt': 1, 'exit_code': 1})
    (tmp_path/'unit.xml').write_bytes(b'<testsuite failures="1"/>')
    queued(tmp_path, 'B_COPY', depends_on_success=['A_UNIT'], requires=['unit.xml'])
    queued(tmp_path, 'C_INDEPENDENT')
    supervisor = object.__new__(maintenance.Supervisor)
    selected, attempt = supervisor.next_job()
    assert selected['id'] == 'C_INDEPENDENT' and attempt == 1
    assert not (tmp_path/'jobs/B_COPY/STATE.json').exists()


def test_successful_dependency_and_required_artifact_both_must_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance, 'BASE', tmp_path); monkeypatch.setattr(maintenance, 'ROOT', tmp_path)
    atomic_json(tmp_path/'SESSION.json', {'status': 'ACTIVE_RESEARCH', 'started_utc': utc_now()})
    queued(tmp_path, 'A_UNIT')
    atomic_json(tmp_path/'jobs/A_UNIT/STATE.json', {'id': 'A_UNIT', 'status': 'COMPLETE', 'attempt': 1, 'exit_code': 0})
    queued(tmp_path, 'B_COPY', depends_on_success=['A_UNIT'], requires=['unit.xml'])
    supervisor = object.__new__(maintenance.Supervisor)
    assert supervisor.next_job() is None
    (tmp_path/'unit.xml').write_bytes(b'<testsuite failures="0"/>')
    assert supervisor.next_job()[0]['id'] == 'B_COPY'


def test_missing_state_is_reported_without_mutating_queue(tmp_path):
    job = queued(tmp_path, 'COPY', depends_on_success=['UNIT'])
    result = dependency_observations(job, tmp_path)
    assert result == [{'id': 'UNIT', 'status': 'PENDING', 'successful': False, 'reason': 'No completed worker attempt'}]
    assert read_json(tmp_path/'queue/COPY.json') == job


def test_registration_keeps_legacy_jobs_and_enforces_new_prerequisites(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance, 'BASE', tmp_path); monkeypatch.setattr(maintenance, 'ROOT', tmp_path)
    base = {'kind': 'maintenance', 'argv': ['scripts/test.py'], 'description': 'isolated fixture'}
    maintenance.register_job({'id': 'UNIT', **base})
    record = maintenance.register_job({'id': 'COPY', **base, 'depends_on_success': ['UNIT']})
    assert record['depends_on_success'] == ['UNIT']
    with pytest.raises(ValueError): maintenance.register_job({'id': 'INVALID', **base, 'depends_on_success': ['FUTURE']})
    assert not (tmp_path/'queue/INVALID.json').exists()
