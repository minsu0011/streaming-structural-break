"""Time/break-age pair partitions conserve official metric contributions."""
import numpy as np
from src.next5.fractures import masks,pair_mass
from src.scoring.ts_auc import ts_auc


def test_time_and_age_pair_partitions_conserve_scores():
    rng=np.random.default_rng(20260908);lengths=np.arange(70,110);ids=np.repeat(np.arange(40),lengths)
    times=np.concatenate([np.arange(n) for n in lengths]);series_tau=np.r_[np.full(20,-1),np.arange(20)*3]
    tau=series_tau[ids];target=((tau>=0)&(times>=tau)).astype(np.uint8)
    prediction=rng.random(len(ids));regime=ids%5;signature=np.where(tau<0,'NO_BREAK_CONTROL','SCALE_DOMINANT');direction=np.where(tau<0,'other','increase')
    scopes=masks(target,times,tau,regime,signature,direction);total=pair_mass(target,times);score=ts_auc(target,prediction,times)
    for prefix in ('time_','age_'):
        mass=0;wins=0.
        for name,mask in scopes.items():
            if not name.startswith(prefix):continue
            weight=pair_mass(target[mask],times[mask]);mass+=weight;wins+=weight*ts_auc(target[mask],prediction[mask],times[mask])
        assert mass==total and abs(wins/total-score)<1e-14


def test_mechanism_scopes_include_null_controls_but_not_other_changes():
    target=np.array([0,0,1,1]);time=np.zeros(4,dtype=int);tau=np.array([-1,-1,0,0]);regime=np.zeros(4,dtype=int)
    signature=np.array(['NO_BREAK_CONTROL','NO_BREAK_CONTROL','SCALE_DOMINANT','TAIL_DOMINANT']);direction=np.array(['other','other','decrease','other'])
    scope=masks(target,time,tau,regime,signature,direction)
    np.testing.assert_array_equal(scope['mechanism_SCALE_DOMINANT'],[True,True,True,False])
    np.testing.assert_array_equal(scope['mechanism_observed_scale_decrease'],[True,True,True,False])
