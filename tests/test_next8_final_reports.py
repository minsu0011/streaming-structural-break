import pytest
from src.next8.final_reports import merge_intervals, time_contract


def test_final_time_contract_counts_union_and_separate_admitted_audit():
    state = {'intervals': [[0., 25000.], [20000., 36000.]],
             'audit_intervals': [[37000., 40000.], [39000., 40600.]], 'last_epoch': 40600.}
    admission = {'status': 'PASS', 'admitted_epoch': 37000., 'productive_seconds_observed': 36000.}
    r = time_contract(state, admission)
    assert r['productive_seconds_observed'] == 36000.
    assert r['separate_audit_seconds_observed'] == 3600.
    assert merge_intervals([[2., 4.], [1., 3.], [8., 9.]]) == [[1., 4.], [8., 9.]]


def test_padding_parallel_overlap_or_phase_reclassification_cannot_meet_contract():
    admission = {'status': 'PASS', 'admitted_epoch': 37000., 'productive_seconds_observed': 36000.}
    for state in [
        {'intervals': [[0., 20000.], [1000., 21000.]], 'audit_intervals': [[37000., 40600.]], 'last_epoch': 40600.},
        {'intervals': [[0., 36000.]], 'audit_intervals': [[37000., 39500.], [38000., 40000.]], 'last_epoch': 40600.},
        {'intervals': [[0., 38000.]], 'audit_intervals': [[38000., 42000.]], 'last_epoch': 42000.},
        {'intervals': [[0., 36000.]], 'audit_intervals': [[36000., 40600.]], 'last_epoch': 40600.},
    ]:
        with pytest.raises(AssertionError):
            time_contract(state, admission)
    with pytest.raises(AssertionError):
        merge_intervals([[0., float('inf')]])
