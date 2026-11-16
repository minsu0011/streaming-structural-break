import src.next8
import pytest
from src.next8.strict_work_time import aggregate_completed_work


def job(key, start, stop, *, kind='FIT', cpu=1., status='COMPLETE'):
    return {'id': key, 'started_epoch': start, 'ended_epoch': stop,
            'kind': kind, 'cpu_seconds': cpu, 'status': status}


def test_completed_work_union_does_not_credit_notes_gaps_or_idle_activity():
    records = [job('a', 10, 30), job('b', 20, 40), job('c', 70, 80),
               job('wait', 0, 100, kind='WAIT'), job('nocpu', 0, 100, cpu=0.),
               job('failed', 0, 100, status='FAILED'), job('running', 0, 100, status='RUNNING')]
    result = aggregate_completed_work(records, 0, 100)
    assert result['productive_seconds'] == 40
    assert result['intervals'] == [[10, 40], [70, 80]]
    assert len(result['counted_activities']) == 3
    assert result['audit_seconds'] == 0


def test_nested_final_tests_are_not_reclassified_as_more_research():
    records = [job('research', 0, 10), job('audit', 20, 60, kind='FINAL_AUDIT'),
               job('test_fit', 25, 40), job('nested_audit', 30, 50, kind='FINAL_AUDIT')]
    result = aggregate_completed_work(records, 0, 70, {'admitted_epoch': 20})
    assert result['productive_seconds'] == 10 and result['audit_seconds'] == 40
    assert result['excluded_activity_counts']['POST_ADMISSION_NON_AUDIT_HELPER'] == 1


def test_phase_crossing_duplicate_or_nonfinite_receipts_cannot_make_time_pass():
    for records, admission in [([job('a', 0, 30)], {'admitted_epoch': 20}),
                               ([job('a', 0, 10, kind='FINAL_AUDIT')], None),
                               ([job('a', 0, 10), job('a', 20, 30)], None),
                               ([job('a', 0, float('nan'))], None)]:
        with pytest.raises(AssertionError):
            aggregate_completed_work(records, 0, 100, admission)
