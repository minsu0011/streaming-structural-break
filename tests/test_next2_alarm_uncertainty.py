import numpy as np
from src.next2.alarm_uncertainty import trajectory_maxima,event_matrix,bootstrap_thresholds,bootstrap_events,METRICS
from src.next2.synthetic_analysis import alarm_summary,threshold


def test_maxima_events_match_first_crossing_and_premature_exclusion():
    rng=np.random.default_rng(931)
    p=rng.random((20,150));onset=15
    p[10,3]=1.;p[10,20]=1.
    tau=np.r_[np.full(10,-1),np.full(10,onset)]
    maxima=trajectory_maxima(p,onset)
    for level in (.95,.999,1.,None):
        summary,_,_=alarm_summary(p,tau,level)
        np.testing.assert_array_equal(event_matrix(maxima,level).mean(axis=0),[summary[m] for m in METRICS])
    assert not event_matrix(maxima,.999)[0,2]


def test_bootstrap_matches_literal_whole_pair_and_order_statistic():
    rng=np.random.default_rng(43)
    calibration=rng.random(40);cal_idx=rng.integers(0,40,(9,40))
    levels=bootstrap_thresholds(calibration,cal_idx,.05)
    np.testing.assert_array_equal(levels,[threshold(calibration[row],.05) for row in cal_idx])
    p=rng.random((12,150));onset=12;maxima=trajectory_maxima(p,onset)
    pair_idx=rng.integers(0,6,(9,6));actual=bootstrap_events(maxima,levels,pair_idx)
    for k,idx in enumerate(pair_idx):
        joined=np.r_[p[idx],p[6+idx]];tau=np.r_[np.full(6,-1),np.full(6,onset)]
        summary,_,_=alarm_summary(joined,tau,levels[k])
        np.testing.assert_array_equal(actual[k],[summary[m] for m in METRICS])


def test_unattainable_finite_sample_threshold_has_no_alarms():
    levels=bootstrap_thresholds(np.arange(5.),np.tile(np.arange(5),(3,1)),.01)
    assert np.isnan(levels).all()
    assert not bootstrap_events(np.ones((2,6)),levels,np.zeros((3,2),dtype=int)).any()
