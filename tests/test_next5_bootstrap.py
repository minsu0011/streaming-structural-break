"""Weighted bootstrap agrees with literal series resampling, including near ties."""
import numpy as np
from src.next5.bootstrap import PreparedAUC
from src.scoring.ts_auc import ts_auc


def test_float64_near_ties_do_not_change_official_pair_order():
    y=np.array([0,1,0,1]);p=np.array([.5,.5+1e-10,.7,.7]);t=np.zeros(4,dtype=int);g=np.arange(4)
    counts=np.ones((1,4),dtype=np.int32)
    score=PreparedAUC.create(y,p,t,g,4).evaluate(counts)[0]
    assert score==ts_auc(y,p,t)==.625


def test_literal_whole_series_resampling_keeps_pair_weights_and_ties():
    rng=np.random.default_rng(20260908);lengths=np.arange(4,19);ids=np.repeat(np.arange(15),lengths)
    t=np.concatenate([np.arange(n) for n in lengths]);y=rng.integers(0,2,len(ids));p=rng.integers(0,5,len(ids))/5.
    p[0]=.4+1e-10
    counts=rng.multinomial(15,np.ones(15)/15,size=25).astype(np.int32)
    actual=PreparedAUC.create(y,p,t,ids,15).evaluate(counts)
    for k,multiplicity in enumerate(counts):
        rows=np.repeat(np.arange(len(ids)),multiplicity[ids])
        assert abs(actual[k]-ts_auc(y[rows],p[rows],t[rows]))<1e-14
