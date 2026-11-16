import numpy as np
from src.next4.support import scored_support


def test_future_break_membership_is_not_positive_slice_support():
    ids=np.repeat([11,22,33,44],3);times=np.tile(np.arange(3),4)
    tau=np.repeat([1,20,30,-1],3);y=((tau>=0)&(times>=tau)).astype(np.uint8)
    r=scored_support(ids,y,times,tau)
    assert r['break_series']==3 and r['positive_series']==1 and r['eligible_positive_series']==1
    assert r['null_series']==1 and r['negative_series']==4
    assert r['eligible_times']==2 and r['positive_negative_pair_mass']==6


def test_single_class_times_do_not_supply_scoring_pairs():
    ids=np.array([1,2,1,2]);times=np.array([0,0,1,1]);tau=np.array([0,0,0,0]);target=np.array([0,0,1,1])
    r=scored_support(ids,target,times,tau)
    assert r['positive_series']==2 and r['eligible_positive_series']==0
    assert r['positive_negative_pair_mass']==0 and r['eligible_times']==0
