import numpy as np
import pytest
from src.next4.memory_diagnostics import local_recovery,aligned_outlier_excess,null_segments,paired_mean_interval


def test_recovery_matches_explicit_quiet_windows_and_censors_unobserved_future():
    values=np.array([[.2,.2,0,0,0,.3,0,0],[0,0,.2,0,0,.2,0,0],[-.2]*8,[.3]*8])
    time,event,limit=local_recovery(values,sustain=3)
    oracle=[]
    for row in values:
        starts=[j for j in range(6) if all(x<=.01 for x in row[j:j+3])]
        oracle.append(starts[0] if starts else 6)
    np.testing.assert_array_equal(time,oracle)
    np.testing.assert_array_equal(event,[True,False,True,False])
    assert limit==6


def test_alignment_excludes_the_entire_intervention_and_has_no_tail_truncation():
    clean=np.zeros((3,20));altered=np.broadcast_to(np.arange(20),(3,20)).copy()
    a,b,e=aligned_outlier_excess(np.vstack([clean,altered]),[2,4],burst_length=3,window=4)
    np.testing.assert_array_equal(e,[[5,6,7,8],[7,8,9,10],[5,6,7,8]])
    np.testing.assert_array_equal(a,0*b)
    with pytest.raises(ValueError):aligned_outlier_excess(np.vstack([clean,altered]),[18],burst_length=3,window=4)


def test_null_segment_hazard_risk_sets_are_not_eventual_or_recurrent_rates():
    p=np.array([[0,.8,0,.8,0,0],[0,0,0,.8,0,.8],[0,0,0,0,0,0.]])
    rows=null_segments(p,.5,segment_length=2)
    assert [r['at_risk_at_segment_start'] for r in rows]==[3,2,1]
    assert [r['new_first_alarms'] for r in rows]==[1,1,0]
    np.testing.assert_allclose([r['any_alarm_in_segment_fraction'] for r in rows],[1/3,2/3,1/3])
    np.testing.assert_allclose([r['conditional_first_alarm_fraction'] for r in rows],[1/3,1/2,0])
    np.testing.assert_allclose([r['eventual_alarm_fraction'] for r in rows],[1/3,2/3,2/3])


def test_paired_interval_matches_literal_seed_replication():
    a=np.array([.1,.2,.4]);b=np.array([.3,.1,.2]);counts=np.array([[1,1,1],[3,0,0],[0,1,2],[1,2,0]])
    result=paired_mean_interval(a,b,counts)
    oracle=np.array([np.repeat(a-b,row).mean() for row in counts])
    np.testing.assert_allclose(result['estimate_mean_delta'],oracle[0])
    np.testing.assert_allclose([result['ci_2_5'],result['ci_97_5']],np.quantile(oracle[1:],[.025,.975]))
    with pytest.raises(ValueError):paired_mean_interval(a,b,counts+1)
