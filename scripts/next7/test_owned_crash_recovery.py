"""Exercise real queue interruption, retained checkpoints and retry exhaustion."""
from pathlib import Path
import os
import sys
import time
import uuid
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import psutil
from src.next7.common import BASE, read_json, save, digest, utc_now, activity, note
from src.next7 import queue


def until(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.02)
    raise TimeoutError('Owned recovery probe failed to reach its required barrier')


def interrupt(record):
    assert queue.identity_alive(record['pid'], record['process_created'])
    parent = psutil.Process(record['pid'])
    assert any(Path(arg).resolve() == (ROOT / 'scripts/next7/queue_runner.py').resolve()
        for arg in parent.cmdline() if arg.endswith('.py'))
    processes = [parent, *parent.children(recursive=True)]
    identities = [{'pid': p.pid, 'created': p.create_time()} for p in processes]
    # Abrupt runner death first represents scheduler/runner interruption, not a
    # normally reported scientific child error. Only captured owned children.
    parent.kill()
    parent.wait(timeout=3)
    for identity in identities[1:]:
        if queue.identity_alive(identity['pid'], identity['created']):
            psutil.Process(identity['pid']).kill()
    psutil.wait_procs(processes[1:], timeout=3)
    assert all(not queue.identity_alive(i['pid'], i['created']) for i in identities)
    return identities


def probe(behavior):
    probe_id = 'recovery_probe_' + uuid.uuid4().hex
    script = 'scripts/next7/recovery_probe_worker.py'
    item = {'id': probe_id, 'status': 'READY', 'script': script,
        'source_sha256': digest(ROOT / script), 'arguments': ['--probe-id', probe_id, '--behavior', behavior],
        'gates': [], 'scope': 'Isolated harmless process recovery test; no scientific inputs'}
    observations = {}
    parent = BASE / f'runtime/queue/{probe_id}'
    output = BASE / f'audit/recovery_probes/{probe_id}'
    active, warning = queue.dispatch([item], 1, observations)
    assert len(active) == 1 and not warning
    first = active[0]
    identities = []
    try:
        until(lambda: (output / 'CHILD_READY_1.json').exists())
        sha = digest(output / 'CHECKPOINT.json')
        identities += interrupt(first)
        assert not (parent / 'attempt_1/DONE.json').exists()
        # New observation dictionary models monitor restart; disk receipts alone
        # must identify the dead owner and the allowed retry.
        observations = {}
        active, warning = queue.dispatch([item], 1, observations)
        assert len(active) == 1 and not warning and active[0]['attempt'] == 2
        recovery = read_json(parent / 'attempt_1/RECOVERY.json')
        assert recovery['status'] == 'CRASH_OR_INTERRUPTION'
        second = active[0]
        until(lambda: (output / 'CHILD_READY_2.json').exists())
        assert read_json(output / 'CHILD_READY_2.json')['resumed_checkpoint']
        assert digest(output / 'CHECKPOINT.json') == sha
        if behavior == 'resume_after_one':
            until(lambda: (parent / 'attempt_2/DONE.json').exists())
            queue.dispatch([item], 1, observations)
            assert read_json(parent / 'TERMINAL.json')['status'] == 'COMPLETE'
            assert read_json(output / 'RESULT.json')['total'] == sum(i * i for i in range(2000))
        else:
            identities += interrupt(second)
            assert not (parent / 'attempt_2/DONE.json').exists()
            active, warning = queue.dispatch([item], 1, {})
            assert not active and not warning
            assert read_json(parent / 'TERMINAL.json')['status'] == 'QUARANTINE'
            assert not (parent / 'attempt_3').exists()
        assert digest(output / 'CHECKPOINT.json') == sha
        return {'id': probe_id, 'behavior': behavior, 'status': 'PASS', 'attempts': 2,
            'checkpoint_preserved_sha256': sha, 'interruptions_of_owned_identities': identities,
            'terminal_status': read_json(parent / 'TERMINAL.json')['status'],
            'real_processes_used': True, 'simulated_wall_clock': False, 'part_reused_not_recomputed': True}
    finally:
        for p in sorted(parent.glob('attempt_*/START.json')):
            record = read_json(p)
            if queue.identity_alive(record['pid'], record['process_created']):
                queue.terminate_owned(record)


def main():
    assert not (BASE / 'audit/OWNED_CRASH_RECOVERY_TESTS_001.json').exists()
    with activity('Real owned queue crash and checkpoint recovery tests', 'ENGINEERING_TEST', 'AUTONOMOUS_MAINTENANCE'):
        results = [probe('resume_after_one'), probe('always_wait')]
    save('audit/OWNED_CRASH_RECOVERY_TESTS_001.json', {'status': 'PASS', 'created_utc': utc_now(),
        'checks': results, 'count': 2, 'real_abrupt_interruptions': 3, 'one_retry_exhaustion_verified': True,
        'checkpoint_bytes_preserved': True, 'scientific_jobs_interrupted': 0,
        'main_queue_modified': False, 'all_owned_probe_processes_stopped': True,
        'scope_limit': 'Abrupt dead-owner recovery and retry limit tested. Does not simulate a30-minute stall deadline.',
        'source_sha256': {n: digest(ROOT / n) for n in ('src/next7/queue.py', 'scripts/next7/queue_runner.py',
            'scripts/next7/recovery_probe_worker.py', 'scripts/next7/test_owned_crash_recovery.py')}})
    note('Owned crash recovery verified with real processes', 'Partial checkpoint reused after one interruption; second interruption quarantines without third attempt', 'AUTONOMOUS_MAINTENANCE')


if __name__ == '__main__':
    main()
