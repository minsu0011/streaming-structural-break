"""The actual clock reader ignores completion notes and verifies work bytes."""
import json
import src.next8
import pytest
from src.next8 import strict_work_time as clock


def fixture_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(clock, 'BASE', tmp_path)
    (tmp_path / 'SESSION_NEXT8.json').write_text(json.dumps({'started_epoch': 0}), encoding='utf-8')
    (tmp_path / 'ledger').mkdir()
    (tmp_path / 'runtime/jobs').mkdir(parents=True)
    (tmp_path / 'runtime/controller_notes').mkdir()
    record = {'id': 'fit1', 'status': 'COMPLETE', 'kind': 'FIT',
              'started_epoch': 20., 'ended_epoch': 40., 'cpu_seconds': 1.}
    data = json.dumps(record).encode('utf-8')
    (tmp_path / 'ledger/fit1.json').write_bytes(data)
    (tmp_path / 'runtime/jobs/fit1.json').write_bytes(data)
    # This resembles the original defect: a finished worker's note outlives
    # the actual operation. No such lease may make a final minimum pass.
    (tmp_path / 'runtime/controller_notes/completed.json').write_text(json.dumps({
        'epoch': 40., 'lease_seconds': 36000., 'task': 'Complete graph state task',
        'phase': 'PRODUCTIVE_RESEARCH'}), encoding='utf-8')
    return record


def test_reader_cannot_replace_actual_work_with_worker_completion_lease(tmp_path, monkeypatch):
    fixture_receipts(tmp_path, monkeypatch)
    result = clock.completed_work_snapshot()
    assert result['productive_seconds'] == 20.
    assert result['controller_and_worker_completion_note_leases_credited'] is False
    assert set(result['ledger_sha256']) == {'ledger/fit1.json'}
    assert result['audit_seconds'] == 0.


def test_reader_rejects_changed_terminal_activity_receipt(tmp_path, monkeypatch):
    record = fixture_receipts(tmp_path, monkeypatch)
    record['ended_epoch'] = 36040.
    (tmp_path / 'runtime/jobs/fit1.json').write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(AssertionError, match='fit1.json'):
        clock.completed_work_snapshot()
