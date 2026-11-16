"""Preserve original DGPs, prefix paths and independent null threshold rules."""
import numpy as np
import pytest
from src.next5.synthetic import cohort_series,threshold,alarm_summary,cohort_menu
from src.next5.artifacts import ROOT,BASE,read_json
from src.next4.synthetic import cohort_series as old_cohort


@pytest.mark.parametrize('scenario',['N0','N1','N2','N3','M1','V1','V4','A2','T2','D2','O1','X1','CB_N3_070','CB_N3_050','OB16'])
def test_exact_original_generator_and_longer_prefix(scenario):
    lock=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')
    for control in (False,True):
        for index in (0,1,2):
            h,o,tau=cohort_series('counterexample',scenario,index,2048,control,lock)
            old_h,old_o,old_tau=old_cohort('counterexample',scenario,index,2048,control,lock)
            np.testing.assert_array_equal(h,old_h);np.testing.assert_array_equal(o,old_o);assert tau==old_tau
            short_h,short_o,short_tau=cohort_series('counterexample',scenario,index,1024,control,lock)
            np.testing.assert_array_equal(h,short_h);np.testing.assert_array_equal(o[:1024],short_o);assert tau==short_tau


def test_exact_independent_maxima_order_statistic_and_strict_alarm():
    values=np.linspace(.1,.9,99)
    assert threshold(values,.05)==values[94]
    assert threshold(values,.01)==values[98]
    assert threshold(values[:10],.01)==1.
    p=np.array([[.2,.5,.5],[.2,.5,.6]]);tau=np.array([-1,2]);report=alarm_summary(p,tau,.5)
    assert report['null_eventual_alarm_rate']==0 and report['postbreak_first_detection_rate']==1 and report['conditional_detection_delay_median']==0


def test_menu_keeps_every_requested_impulse_lag_observable():
    menu=read_json(BASE/'synthetic/MENU_LOCK.json');lock=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json');cohorts=cohort_menu(menu,lock)
    for spec in cohorts:
        if spec['scenario'] in ('O1','OB16'):assert spec['horizon']>max(lock['tau_cycle'])+max(menu['impulse_lags'])
    assert {r['horizon'] for r in cohorts if r['kind']=='long_null'}=={16384}
