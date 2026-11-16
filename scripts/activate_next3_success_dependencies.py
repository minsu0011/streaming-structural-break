"""Activate tested successful-job dependencies at a verified idle boundary."""
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
    directory = BASE/'runtime/maintenance_updates'
    receipt = directory/'SUPERVISOR_V4_REPLACED.json'
    if receipt.exists(): raise RuntimeError('Successful-dependency activation already recorded')
    tests = BASE/'maintenance/FULL_TESTS_CHECKPOINT_6.xml'
    suites = ET.parse(tests).getroot().findall('.//testsuite')
    assert suites and sum(int(s.attrib['tests']) for s in suites) == 753
    assert all(int(s.attrib.get('failures', 0)) == int(s.attrib.get('errors', 0)) ==
               int(s.attrib.get('skipped', 0)) == 0 for s in suites)
    prior = read_json(BASE/'runtime/SUPERVISOR.json')
    change = read_json(BASE/'maintenance/success_dependency_guard/CORRECTION.json')
    old_source = BASE/'maintenance/success_dependency_guard/original_maintenance.py'
    assert prior['source_sha256'] == digest(old_source)
    assert digest(ROOT/'src/next3/maintenance.py') == change['corrected_source_sha256']
    assert digest(ROOT/'src/next3/job_dependencies.py') == change['dependency_module_sha256']
    process = psutil.Process(prior['pid'])
    assert abs(process.create_time()-prior['process_create_time']) < .1
    assert any('resume_next3.py' in part for part in process.cmdline())
    assert not process.children(recursive=True)
    states = [read_json(p) for p in (BASE/'jobs').glob('*/STATE.json')]
    assert all(state['status'] in ('COMPLETE', 'QUARANTINED') for state in states)
    completed_ids = {state['id'] for state in states}
    assert all(read_json(p)['id'] in completed_ids for p in (BASE/'queue').glob('*.json'))
    session = read_json(BASE/'SESSION.json')
    assert session['status'] != 'COMPLETE' and not session.get('audit_started_utc')
    request = {'created_utc': utc_now(), 'event': 'PLANNED_SUPERVISOR_RESTART',
        'reason': 'Activate explicit successful-job prerequisites after full tests; preserve old queue records',
        'previous': prior, 'active_children': 0, 'accounting_before': read_json(BASE/'runtime/ACCOUNTING.json'),
        'full_test_report_sha256': digest(tests), 'passed_tests': 753,
        'new_source_sha256': change['corrected_source_sha256'], 'dependency_module_sha256': change['dependency_module_sha256']}
    progress('Successful-dependency guard activation', '753 tests passed; verified no child or unprocessed queue entry')
    atomic_json(directory/'SUPERVISOR_V4_REPLACEMENT_REQUEST.json', request, immutable=True)
    assert abs(process.create_time()-prior['process_create_time']) < .1 and not process.children(recursive=True)
    process.terminate(); process.wait(timeout=10)
    with (directory/'SUPERVISOR_V4.log').open('x', encoding='utf-8') as log:
        launcher = subprocess.Popen([sys.executable, '-u', str(ROOT/'scripts/resume_next3.py')], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            env=dict(os.environ, PYTHONUNBUFFERED='1'))
    deadline = time.monotonic()+15
    while time.monotonic() < deadline:
        current = read_json(BASE/'runtime/SUPERVISOR.json')
        if current['pid'] != prior['pid']: break
        if launcher.poll() is not None: raise RuntimeError('Replacement exited; inspect preserved log')
        time.sleep(.1)
    else: raise RuntimeError('Replacement did not publish its ownership receipt')
    assert current['source_sha256'] == request['new_source_sha256'] and psutil.pid_exists(current['pid'])
    atomic_json(receipt, {**request, 'status': 'PASS', 'finished_utc': utc_now(), 'replacement': current}, immutable=True)
    atomic_json(BASE/'maintenance/success_dependency_guard/ACTIVATION.json', {
        'status': 'PASS', 'created_utc': utc_now(), 'receipt_sha256': digest(receipt),
        'source_sha256': current['source_sha256'], 'passed_tests': 753,
        'old_queue_records_rewritten': False, 'accounting_rewritten': False}, immutable=True)
    print('NEXT3 SUCCESSFUL DEPENDENCIES ACTIVE', current['pid'], flush=True)


if __name__ == '__main__': main()
