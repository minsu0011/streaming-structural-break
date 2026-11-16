"""Paired generating-seed resampling and conditional alarm denominators."""
import numpy as np
from src.next5.synthetic_metrics import score_inputs,score_draws,event_draws,paired_mean_difference
from src.next5.impulse_reports import recovery_times
from src.next5.synthetic import alarm_summary,cohort_menu
from src.next5.artifacts import BASE,ROOT,read_json
from src.scoring.ts_auc import ts_auc


def test_seed_bootstrap_literal_paired_curve_replication_including_near_ties():
    seed=np.repeat(np.arange(8),2);tau=np.tile([2,-1],8);horizon=6
    p=np.random.default_rng(20260908).integers(0,5,(16,horizon))/5.;p[0,3]=.4+1e-10
    counts=np.array([[1]*8,[2,0,1,1,0,2,1,1],[0,3,0,1,1,0,1,2]],dtype=np.int32)
    actual=score_draws(p,tau,seed,horizon,counts);target,times,code=score_inputs(tau,seed,horizon)
    for j,count in enumerate(counts):
        rows=np.repeat(np.arange(len(target)),count[code])
        assert abs(actual[j]-ts_auc(target[rows],p.ravel()[rows],times[rows]))<1e-14


def test_event_denominators_count_only_eligible_curves():
    p=np.array([[.1,.6,.1],[.1,.1,.1],[.1,.1,.8],[.1,.9,.1]])
    tau=np.array([2,-1,2,-1]);seed=np.array([0,0,1,1]);counts=np.array([[1,1],[2,0],[0,2]],dtype=np.int32)
    np.testing.assert_array_equal(event_draws(p,tau,seed,.5,'null_eventual',counts),[.5,0,1])
    np.testing.assert_array_equal(event_draws(p,tau,seed,.5,'prebreak',counts),[.5,1,0])
    np.testing.assert_array_equal(event_draws(p,tau,seed,.5,'postbreak_detection',counts),[.5,0,1])


def test_recovery_after_end_and_censoring_are_explicit():
    excess=np.full((2,32),.04);excess[0,5:]=0
    time,censored=recovery_times(excess,np.array([0,0]),1)
    np.testing.assert_array_equal(time,[4,31]);np.testing.assert_array_equal(censored,[False,True])


def test_precomputed_score_statistics_equal_direct_alarm_summary():
    p=np.array([[.1,.2,.5],[.2,.3,.8]]);tau=np.array([-1,2])
    stats={'score_q99':float(np.quantile(p,.99)),'mean_max_score':float(p.max(axis=1).mean())}
    assert alarm_summary(p,tau,.5)==alarm_summary(p,tau,.5,score_statistics=stats)


def test_actual_whole_research_artifacts_have_predeclared_larger_cohorts():
    menu=read_json(BASE/'synthetic/MENU_LOCK.json');generator=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')
    full=cohort_menu(menu,generator);whole=cohort_menu(menu,generator,whole_research=True)
    assert [r['scenario'] for r in full]==[r['scenario'] for r in whole]
    assert all(r['seeds']==1000 for r in whole if r['kind']=='calibration')
    assert all(r['seeds']==512 for r in whole if r['kind']!='calibration')
