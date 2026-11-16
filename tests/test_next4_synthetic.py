import numpy as np
import pytest
from src.next4.synthetic import threshold,alarm_summary,cohort_series,paired_seed_counts,background_mapping,scenario_background
from src.next2.counterexamples import policy


def test_calibration_strict_order_statistic_and_unattainable_alpha():
    maxima=np.arange(1,101)/101
    assert threshold(maxima,.05)==maxima[95]
    assert threshold(maxima,.01)==maxima[99]
    assert threshold(maxima[:3],.01)==1.
    with pytest.raises(ValueError):threshold([np.nan],.05)


def test_alarm_time_and_delay_exclude_prebreak_first_alarms():
    p=np.array([[.5,.5,.5,.5],[0,.8,0,.8],[0,0,.5,.8],[0,0,0,0]])
    result=alarm_summary(p,np.array([-1,2,2,-1]),.5)
    assert result['null_eventual_alarm_rate']==0
    assert result['prebreak_alarm_rate']==.5
    assert result['postbreak_first_detection_rate']==.5
    assert result['conditional_detection_delay_median']==1


def test_paired_cohorts_preserve_history_and_prechange_prefix():
    lock=policy()
    for scenario in ('V3','CB_N3_050','O1','OB16'):
        for index in range(4):
            h,a,t=cohort_series('counterexample',scenario,index,1024,True,lock)
            hb,b,tb=cohort_series('counterexample',scenario,index,1024,False,lock)
            np.testing.assert_array_equal(h,hb)
            stop=lock['tau_cycle'][index%len(lock['tau_cycle'])]
            np.testing.assert_array_equal(a[:stop],b[:stop])
            assert t==-1
            assert (tb==-1)==(scenario in ('O1','OB16'))
            hl,large,_=cohort_series('counterexample',scenario,index,16384,False,lock)
            np.testing.assert_array_equal(hb,hl);np.testing.assert_array_equal(b,large[:1024])


def test_fixed_horizon_null_prefix_and_calibration_partition():
    lock=policy()
    for scenario in ('N0','N1','N2','N3'):
        h,o,t=cohort_series('long_null',scenario,2,16384,False,lock)
        for horizon in (1024,4096):
            hh,oo,tt=cohort_series('long_null',scenario,2,horizon,False,lock)
            np.testing.assert_array_equal(h,hh);np.testing.assert_array_equal(o[:horizon],oo);assert t==tt==-1
        hc,_,_=cohort_series('calibration',scenario,2,1024,True,lock)
        assert not np.array_equal(h,hc)


def test_pair_seed_bootstrap_duplicates_controls_with_their_break():
    counts=paired_seed_counts(7,20,20260908)
    np.testing.assert_array_equal(counts.sum(axis=1),7)
    np.testing.assert_array_equal(counts,paired_seed_counts(7,20,20260908))
    code=np.repeat(np.tile(np.arange(7),2),5)
    assert np.array_equal(code[:35],code[35:])
    assert np.array_equal(counts[:,code[:35]],counts[:,code[35:]])


def test_background_calibration_uses_prechange_dgp():
    lock=policy();mapping=background_mapping(lock)
    assert mapping[scenario_background('V3',lock)]=='N2'
    assert mapping[scenario_background('CB_N3_070',lock)]=='N3'
    assert mapping[scenario_background('O1',lock)]=='M1'
    assert mapping[scenario_background('OB16',lock)]=='M1'
