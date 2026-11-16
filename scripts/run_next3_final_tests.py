"""Collect and run the complete final suite against one immutable source set."""
from pathlib import Path
import datetime
import re
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.final_source_contract import assert_source_snapshot
from src.next3.final_validation import tests_proof
from scripts.prepare_next3_independent_reproduction import time_gate


def main():
    timing = time_gate(); out = BASE/'final_audit'
    lock_path = out/'FINAL_EXECUTION_SOURCE_LOCK.json'; lock = read_json(lock_path)
    assert lock['status'] == 'FROZEN_FINAL_EXECUTION_SOURCE'
    assert_source_snapshot(ROOT, lock)
    for name in ('FULL_TESTS.xml', 'FULL_TESTS.log', 'TEST_COLLECTION.log', 'FULL_TESTS_RESULT.json'):
        if (out/name).exists(): raise RuntimeError('Preserve prior final test evidence: '+name)
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    collection_command = [sys.executable, '-m', 'pytest', '--collect-only', '-q', 'tests']
    with (out/'TEST_COLLECTION.log').open('x', encoding='utf-8') as stream:
        collection = subprocess.run(collection_command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
    if collection.returncode: raise RuntimeError('Final test collection failed; log preserved')
    text = (out/'TEST_COLLECTION.log').read_text(encoding='utf-8')
    counts = re.findall(r'(?m)^(\d+) tests? collected in ', text)
    if len(counts) != 1: raise RuntimeError('Unambiguous final collected test count required')
    count = int(counts[0])
    if count < 798: raise RuntimeError('Final collection lost required tests')
    assert_source_snapshot(ROOT, lock)
    command = [sys.executable, '-m', 'pytest', '-q', 'tests', '--junitxml='+str(out/'FULL_TESTS.xml')]
    atomic_json(out/'FULL_TESTS_START.json', {'created_utc': utc_now(), 'command': command,
        'source_lock_sha256': digest(lock_path), 'collected_tests': count,
        'collection_log_sha256': digest(out/'TEST_COLLECTION.log'), 'time_gate': timing}, immutable=True)
    progress('Final complete test suite', str(count)+' collected cases against frozen current sources', phase='FINAL_AUDIT')
    with (out/'FULL_TESTS.log').open('x', encoding='utf-8') as stream:
        completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
    if completed.returncode: raise RuntimeError('Final complete tests failed; original log and XML preserved')
    proof = tests_proof(out/'FULL_TESTS.xml', minimum_count=count,
        required_classnames={'tests.test_ts_auc', 'tests.test_scorer_official_checks',
            'tests.test_next3_final_validation', 'tests.test_next3_final_runtime_probe',
            'tests.test_next3_offline_environment', 'tests.test_next3_portable_report_contract',
            'tests.test_next3_final_source_contract'},
        not_before_timestamp=datetime.datetime.fromisoformat(timing['audit_started_utc']).timestamp())
    if proof['tests'] != count: raise RuntimeError('Collected and executed final test counts differ')
    assert_source_snapshot(ROOT, lock)
    atomic_json(out/'FULL_TESTS_RESULT.json', {'status': 'PASS', 'created_utc': utc_now(),
        'tests': proof, 'collected_tests': count, 'collected_and_executed_counts_equal': True,
        'source_set_unchanged_before_and_after': True, 'source_lock_sha256': digest(lock_path),
        'collection_log_sha256': digest(out/'TEST_COLLECTION.log'),
        'test_log_sha256': digest(out/'FULL_TESTS.log'), 'script_sha256': digest(Path(__file__))}, immutable=True)
    progress('Final complete tests passed', str(count)+' passed without failures, errors or skips; complete source set unchanged', phase='FINAL_AUDIT')
    print('NEXT3 FINAL FULL TESTS PASS', count, flush=True)


if __name__ == '__main__': main()
