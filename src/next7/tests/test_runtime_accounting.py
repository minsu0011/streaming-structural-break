from src.next7.runtime_accounting import audit_admitted, allocate_interval
from src.next7.common import union_seconds


def admission(seconds=36000.):
    return {'status': 'PASS', 'session_sha256': 'same', 'productive_seconds_observed': seconds, 'admitted_epoch': 100.}


def test_audit_requires_actual_research_minimum_and_same_session():
    assert not audit_admitted(admission(), 35999.999, 'same')
    assert not audit_admitted(admission(35999.), 40000., 'same')
    assert not audit_admitted(admission(), 40000., 'different')
    assert not audit_admitted({}, 40000., 'same')
    assert audit_admitted(admission(), 36000., 'same')


def test_premature_audit_note_cannot_turn_research_into_audit():
    assert allocate_interval(1., 50., 'FINAL_AUDIT', admitted=False) is None
    assert allocate_interval(1., 50., 'RESEARCH', admitted=False) == ('intervals', [1., 50.])


def test_research_and_audit_never_share_time_after_admission():
    research = allocate_interval(80., 120., 'RESEARCH', admitted=True, admission_epoch=100.)
    audit = allocate_interval(80., 120., 'FINAL_AUDIT', admitted=True, admission_epoch=100.)
    assert research == ('intervals', [80., 100.])
    assert audit == ('audit_intervals', [100., 120.])
    assert union_seconds([research[1], audit[1]]) == 40.
    assert allocate_interval(101., 120., 'RESEARCH', admitted=True, admission_epoch=100.) is None


def test_parallel_activity_is_a_union_not_sum():
    assert union_seconds([[0., 10.], [5., 20.], [7., 8.], [21., 22.]]) == 21.
