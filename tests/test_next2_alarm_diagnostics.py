import numpy as np
import pytest
from src.next2.synthetic_analysis import age_normalize,threshold
from src.next2.alarm_diagnostics import series_maxima64,first_alarm_records64,verify_within_time_ranks
from src.scoring.ts_auc import ts_auc


@pytest.mark.parametrize('beta',[0.,.5,1.])
def test_age_map_preserves_official_score_ties_and_extremes(beta):
    rng=np.random.default_rng(62219)
    p=np.r_[rng.uniform(size=4096),np.zeros(64),np.ones(64)].astype(np.float32)
    p[::17]=np.float32(.51)
    t=np.arange(len(p))%64
    y=rng.integers(0,2,len(p))
    q=age_normalize(p,t,beta)
    assert q.dtype==np.float64
    verify_within_time_ranks(p,q,t)
    assert ts_auc(y,p,t)==ts_auc(y,q,t)
    assert np.array_equal(q[p==0],p[p==0])
    assert np.array_equal(q[p==1],p[p==1])


def test_alarm_strict_boundary_preserves_float64_distinction():
    level=.5
    p=np.array([.5,np.nextafter(.5,1.),.1,.2])
    frame=first_alarm_records64([1,1,2,2],[0,1,0,1],p,{1:-1,2:1},level)
    assert frame.iloc[0].first_alarm_time==1
    assert frame.iloc[0].null_false_alarm
    assert frame.iloc[1].first_alarm_time==-1
    ids,maxima=series_maxima64([1,1,2,2],p)
    assert maxima[0]>level


def test_invalid_alarm_inputs_fail():
    with pytest.raises(ValueError):
        series_maxima64([1,2,1],[.1,.2,.3])
    with pytest.raises(ValueError):
        age_normalize([.2],[0],.7)
    assert threshold([.2,.3],.01) is None
    assert threshold([.1,.2,.3,.4],.25)==.4
