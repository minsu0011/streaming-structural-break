import src.next9
import numpy as np
from src.next9.inference import weights_for, rank_samples, interval


def test_whole_group_bootstrap_matches_literal_replicates():
    from src.scoring.ts_auc import ts_auc
    ids=np.repeat(np.arange(4),3)
    t=np.tile(np.arange(3),4)
    y=np.repeat([0,1,0,1],3)
    p=np.array([.1,.2,.3,.7,.8,.9,.3,.4,.5,.5,.6,.7])
    weights=np.array([[1,1,1,1],[2,1,0,1],[0,2,1,1]],dtype=np.uint16)
    point,boot=rank_samples(y,p,t,ids,np.arange(4),weights)
    for k,w in enumerate(weights):
        row=np.concatenate([np.tile(np.flatnonzero(ids==g),int(n)) for g,n in enumerate(w)])
        assert abs(boot[k]-ts_auc(y[row],p[row],t[row]))<1e-12
    assert point==boot[0]


def test_uncertainty_does_not_impute_unsupported_ranks():
    r=interval(np.nan,np.full(2000,np.nan))
    assert r['status']=='INSUFFICIENT_SUPPORT' and r['p_two_sided']==1
    a,ra=weights_for(np.arange(10),'toy',20)
    b,rb=weights_for(np.arange(10),'toy',20)
    np.testing.assert_array_equal(a,b)
    assert ra==rb
