import numpy as np
from src.next7.attribution import latch_pair_partition
from src.scoring.ts_auc import ts_auc


def test_all_tied_pair_cells_receive_half_credit():
    y=np.array([0,0,1,1])
    e=np.array([0,1,0,1])
    wins,pairs=latch_pair_partition(y,np.ones(4),np.zeros(4),e)
    np.testing.assert_array_equal(pairs,np.ones((2,2)))
    np.testing.assert_array_equal(wins,np.full((2,2),.5))


def test_partition_recovers_exact_official_pair_weighted_auc():
    rng=np.random.default_rng(20260913)
    for _ in range(100):
        n=200
        y=rng.integers(0,2,n)
        p=rng.integers(0,8,n)/7
        t=rng.integers(0,24,n)
        e=rng.integers(0,2,n)
        wins,pairs=latch_pair_partition(y,p,t,e)
        assert abs(wins.sum()/pairs.sum()-ts_auc(y,p,t))<1e-12
        for a in range(2):
            for b in range(2):
                mask=((y==1)&(e==a))|((y==0)&(e==b))
                if pairs[a,b]>0:
                    assert abs(wins[a,b]/pairs[a,b]-ts_auc(y[mask],p[mask],t[mask]))<1e-12
