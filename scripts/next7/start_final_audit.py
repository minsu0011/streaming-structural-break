"""Admit the final audit only after the productive minimum and research closure."""
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, read_json, save, digest, utc_now, union_seconds, require_guard, note
from src.next6_extension.synthetic import CASES


def main():
    require_guard()
    session = read_json(BASE / 'SESSION_NEXT7.json')
    state = read_json(BASE / 'runtime/SCHEDULER_STATE.json')
    assert time.time() - state['last_epoch'] < 90, 'A fresh active monitor receipt is required'
    productive = union_seconds(state['intervals'])
    assert productive >= session['research_minimum_seconds'], 'Ten productive hours are not yet complete'
    assert not state['audit_intervals'], 'The formal audit cannot have been credited before admission'
    assert not (BASE / 'runtime/FINAL_AUDIT_ADMISSION.json').exists(), 'Audit admission is immutable'
    queue = read_json(BASE / 'runtime/TASK_QUEUE.json')
    queue_receipts = {}
    for item in queue:
        path = BASE / 'runtime/queue' / item['id'] / 'TERMINAL.json'
        terminal = read_json(path)
        if terminal['status'] != 'COMPLETE':
            resolution = path.parent / 'RESOLUTION.json'
            resolved = read_json(resolution)
            assert resolved['status'] in ('REPAIRED_AND_REPRODUCED', 'EXPECTED_FAILURE_TEST')
            assert resolved['terminal_sha256'] == digest(path)
            for relative, sha in resolved['evidence_sha256'].items():
                assert digest(BASE / relative) == sha
        queue_receipts[item['id']] = digest(path)
    required = [f'conditional_null/grid/{case}/H_{length}/DONE.json'
        for case in ('N0', 'N1', 'N2', 'N3') for length in (1000, 2048, 5000)]
    required += [f'classifier_stress/evaluation/{case}/DONE.json' for case in CASES]
    for relative in required:
        assert read_json(BASE / relative)['status'] == 'COMPLETE'
    frozen = BASE / 'FINAL_CANDIDATE_FREEZE_NEXT7.json'
    assert read_json(frozen)['status'] == 'FROZEN'
    ready_path = BASE / 'RESEARCH_COMPLETION_READY.json'
    ready = read_json(ready_path)
    assert ready['status'] == 'PASS' and ready['outstanding_scientific_tasks'] == 0
    assert ready['candidate_freeze_sha256'] == digest(frozen)
    assert ready['required_artifacts_sha256'], 'Concrete completed research artifacts required'
    for relative, sha in ready['required_artifacts_sha256'].items():
        path = (BASE / relative).resolve()
        assert path.is_relative_to(BASE.resolve()) and digest(path) == sha
    save('runtime/FINAL_AUDIT_ADMISSION.json', {'status': 'PASS', 'created_utc': utc_now(),
        'admitted_epoch': time.time(), 'session_sha256': digest(BASE / 'SESSION_NEXT7.json'),
        'productive_seconds_observed': productive, 'minimum_required': session['research_minimum_seconds'],
        'audit_target_seconds': session['final_audit_target_seconds'], 'queue_terminal_sha256': queue_receipts,
        'research_readiness_sha256': digest(ready_path), 'candidate_freeze_sha256': digest(frozen),
        'source_sha256': digest(__file__), 'idle_time_not_credited': True})
    note('Formal final audit admitted', 'Productive minimum and concrete research closure verified',
        'FINAL_AUDIT', phase='FINAL_AUDIT')


if __name__ == '__main__':
    main()
