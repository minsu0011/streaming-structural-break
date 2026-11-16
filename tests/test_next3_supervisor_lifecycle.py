"""Isolated real-child fault checks and conservative clock/resource boundaries."""
from pathlib import Path
import datetime
import os
import subprocess
import sys
import time
import pytest
from src.next3 import maintenance
from src.next3.artifacts import atomic_json, read_json, utc_now


def ready_child():
    child = subprocess.Popen([sys.executable, '-u', '-c',
        "import sys; print('READY', flush=True); sys.stdin.readline(); raise SystemExit(23)"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert child.stdout.readline().strip() == 'READY'
    return child, maintenance.psutil.Process(child.pid).create_time()


def close_child(child):
    if child.poll() is None:
        child.stdin.write('\n'); child.stdin.flush()
    child.communicate(timeout=10)


def test_resume_preserves_a_live_owned_worker_without_duplicate_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance, 'BASE', tmp_path)
    child, created = ready_child()
    try:
        path = tmp_path/'jobs/M1/STATE.json'
        record = {'id': 'M1', 'status': 'RUNNING', 'attempt': 1, 'pid': child.pid, 'process_create_time': created}
        atomic_json(path, record)
        supervisor = object.__new__(maintenance.Supervisor)
        with pytest.raises(RuntimeError, match='Owned child still running'): supervisor._recover_jobs()
        assert child.poll() is None and read_json(path) == record
    finally: close_child(child)


def test_real_crashed_worker_is_recovered_with_its_attempt_history(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance, 'BASE', tmp_path)
    child, created = ready_child(); close_child(child); assert child.returncode == 23
    path = tmp_path/'jobs/M1/STATE.json'
    atomic_json(path, {'id': 'M1', 'status': 'RUNNING', 'attempt': 1, 'pid': child.pid, 'process_create_time': created})
    supervisor = object.__new__(maintenance.Supervisor); supervisor._recover_jobs()
    recovered = read_json(path)
    assert recovered['status'] == 'FAILED' and recovered['attempt'] == 1 and recovered['pid'] == child.pid
    assert 'CRASH_RECOVERED' in (tmp_path/'runtime/HANG_RECOVERY_LOG.csv').read_text(encoding='utf-8')


def test_reused_pid_does_not_adopt_or_terminate_an_unrelated_live_process(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance, 'BASE', tmp_path)
    child, created = ready_child()
    try:
        path = tmp_path/'jobs/M1/STATE.json'
        atomic_json(path, {'id': 'M1', 'status': 'RUNNING', 'attempt': 1, 'pid': child.pid, 'process_create_time': created-20})
        supervisor = object.__new__(maintenance.Supervisor); supervisor._recover_jobs()
        assert read_json(path)['status'] == 'FAILED' and child.poll() is None
    finally: close_child(child)


def test_simulated_hang_terminates_only_the_owned_test_child(tmp_path, monkeypatch):
    monkeypatch.setattr(maintenance, 'ROOT', tmp_path); monkeypatch.setattr(maintenance, 'BASE', tmp_path/'state')
    child, _ = ready_child()
    attempt = tmp_path/'attempt_1'; attempt.mkdir(); (attempt/'RUN.log').write_text('', encoding='utf-8')
    supervisor = object.__new__(maintenance.Supervisor)
    supervisor.child = child; supervisor.job = {'id': 'OWNED_TEST', 'attempt': 1, 'attempt_directory': 'attempt_1'}
    supervisor.last_progress = supervisor.last_cpu_progress = time.monotonic()-1801
    supervisor.last_cpu = 1e9; supervisor.last_log_size = 0
    expected_owned = {child.pid, *(p.pid for p in maintenance.psutil.Process(child.pid).children(recursive=True))}
    try:
        supervisor.inspect_child(); child.wait(timeout=10)
        assert child.returncode is not None and maintenance.psutil.pid_exists(os.getpid())
        receipt = read_json(attempt/'HANG_SUSPECTED.json')
        assert receipt['event'] == 'HANG_SUSPECTED'
        assert {item['pid'] for item in receipt['owned_processes']} == expected_owned
        assert os.getpid() not in expected_owned
        assert receipt['no_progress_or_CPU_seconds'] >= 1800
    finally: close_child(child)


def tick_fixture(tmp_path, monkeypatch, *, interval=45., recent=False, busy=False, ram=86., disk=20*1024**3, complete=False):
    monkeypatch.setattr(maintenance, 'BASE', tmp_path)
    atomic_json(tmp_path/'SESSION.json', {'status': 'COMPLETE' if complete else 'ACTIVE_RESEARCH', 'started_utc': utc_now()})
    atomic_json(tmp_path/'runtime/ACTIVITY.json', {'timestamp': utc_now(), 'task': 'isolated test only'})
    monkeypatch.setattr(maintenance.time, 'monotonic', lambda: 1000.+interval)
    monkeypatch.setattr(maintenance, 'activity_age', lambda: 0 if recent else 1000)
    monkeypatch.setattr(maintenance, 'resource_snapshot', lambda: {'timestamp': utc_now(), 'RAM_percent': ram,
        'available_RAM_bytes': 10*1024**3, 'disk_free_bytes': disk, 'supervisor_RSS_bytes': 1, 'child_process_count': 0})
    supervisor = object.__new__(maintenance.Supervisor)
    supervisor.last_tick = 1000.; supervisor.productive = 10.; supervisor.audit_productive = 0.; supervisor.unobserved = 0.
    supervisor.last_resource = supervisor.last_checkpoint = supervisor.last_reprioritization = 1000.+interval
    supervisor.child = None; supervisor.job = None; supervisor.inspect_child = lambda: busy; supervisor.states = lambda: []
    launches = []; supervisor.next_job = lambda: ({'id': 'TEST'}, 1)
    supervisor.start = lambda *args: launches.append(args)
    return supervisor, launches


@pytest.mark.parametrize('interval,recent,busy,credited', [(45., False, False, 0.), (45., True, False, 45.),
    (45., False, True, 45.), (120., True, True, 0.)])
def test_productive_clock_does_not_credit_idle_or_suspend_gaps(tmp_path, monkeypatch, interval, recent, busy, credited):
    supervisor, launches = tick_fixture(tmp_path, monkeypatch, interval=interval, recent=recent, busy=busy)
    supervisor.tick(); receipt = read_json(tmp_path/'runtime/ACCOUNTING.json')
    assert receipt['last_interval_credited'] == credited
    assert receipt['productive_seconds'] == 10.+credited and receipt['unobserved_seconds'] == interval-credited
    assert not launches


@pytest.mark.parametrize('ram,disk,allowed', [(85., 20*1024**3, False), (84.9, 10*1024**3, False), (84.9, 10*1024**3+1, True)])
def test_pressure_boundaries_hold_new_worker_launches(tmp_path, monkeypatch, ram, disk, allowed):
    supervisor, launches = tick_fixture(tmp_path, monkeypatch, ram=ram, disk=disk)
    supervisor.tick(); assert bool(launches) is allowed


def test_completed_session_never_launches_a_remaining_queue_item(tmp_path, monkeypatch):
    supervisor, launches = tick_fixture(tmp_path, monkeypatch, ram=10., complete=True)
    assert supervisor.tick() is True
    assert not launches
