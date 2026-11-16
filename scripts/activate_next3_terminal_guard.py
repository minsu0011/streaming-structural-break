"""Activate the tested supervisor guard only at an observed idle boundary."""
from pathlib import Path
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now


def main():
    receipt_path = BASE / 'runtime/maintenance_updates/SUPERVISOR_V3_REPLACED.json'
    if receipt_path.exists():
        raise RuntimeError('This activation already has a receipt')
    suite_path = BASE / 'maintenance/FULL_TESTS_CHECKPOINT_5.xml'
    suites = ET.parse(suite_path).getroot().findall('testsuite')
    assert sum(int(s.attrib['tests']) for s in suites) == 685
    assert all(int(s.attrib.get('failures', 0)) == 0 and
               int(s.attrib.get('errors', 0)) == 0 for s in suites)
    prior = read_json(BASE / 'runtime/SUPERVISOR.json')
    process = psutil.Process(prior['pid'])
    assert abs(process.create_time() - prior['process_create_time']) < .1
    assert any('resume_next3.py' in arg for arg in process.cmdline())
    assert not process.children(recursive=True), 'Owned child is still alive'
    states = [read_json(p) for p in (BASE / 'jobs').glob('*/STATE.json')]
    assert all(s['status'] in ('COMPLETE', 'QUARANTINED') for s in states)
    terminal = {s['id'] for s in states}
    assert all(read_json(p)['id'] in terminal for p in (BASE / 'queue').glob('*.json'))
    session = read_json(BASE / 'SESSION.json')
    assert session['status'] != 'COMPLETE' and not session.get('audit_started_utc')
    accounting = read_json(BASE / 'runtime/ACCOUNTING.json')
    progress('SUPERVISOR_GUARD_ACTIVATION',
             '685 tests passed; verified no child or unprocessed queue entry before replacement')
    request = {'timestamp': utc_now(), 'event': 'PLANNED_SUPERVISOR_RESTART',
               'reason': 'Activate tested terminal-session launch guard; numerical models unchanged',
               'previous': prior, 'active_children': 0, 'accounting_before': accounting,
               'test_report_sha256': digest(suite_path),
               'new_source_sha256': digest(ROOT / 'src/next3/maintenance.py')}
    atomic_json(BASE / 'runtime/maintenance_updates/SUPERVISOR_V3_REPLACEMENT_REQUEST.json',
                request, immutable=True)
    # Identity and ownership are checked again immediately before termination.
    assert abs(process.create_time() - prior['process_create_time']) < .1
    assert not process.children(recursive=True)
    process.terminate()
    process.wait(timeout=10)
    log_path = BASE / 'runtime/maintenance_updates/SUPERVISOR_V3.log'
    with log_path.open('x', encoding='utf-8') as log:
        launcher = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scripts/resume_next3.py')],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            env=dict(os.environ, PYTHONUNBUFFERED='1'))
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        current = read_json(BASE / 'runtime/SUPERVISOR.json')
        if current['pid'] != prior['pid']:
            break
        if launcher.poll() is not None:
            raise RuntimeError('Replacement exited; preserved log requires review')
        time.sleep(.1)
    else:
        raise RuntimeError('Replacement receipt was not published')
    assert current['source_sha256'] == request['new_source_sha256']
    assert psutil.pid_exists(current['pid'])
    atomic_json(receipt_path, {**request, 'status': 'PASS', 'replacement': current,
                              'finished_utc': utc_now()}, immutable=True)
    repair_path = BASE / 'maintenance/supervisor_terminal_guard_fix/REPAIR.json'
    repair = read_json(repair_path)
    repair.update(status='TESTED_AND_ACTIVATED', live_supervisor_replaced=True,
                  receipt=receipt_path.relative_to(ROOT).as_posix(),
                  full_tests_passed=685, completed_utc=utc_now())
    atomic_json(repair_path, repair)
    print('NEXT3 SUPERVISOR GUARD ACTIVE', current['pid'], flush=True)


if __name__ == '__main__':
    main()
