import numpy as np
from src.next.bootstrap import PreparedAUC
from src.scoring.ts_auc import ts_auc
from scripts.bootstrap_next2_counterexamples import paired_group_codes


def test_seed_pair_counts_match_explicit_control_changed_resampling():
    n,h=3,5
    p=np.array([[.1,.1,.2,.2,.2],[.2,.2,.2,.3,.3],[.1,.1,.1,.1,.1],
                [.1,.1,.3,.6,.8],[.2,.2,.2,.6,.7],[.1,.1,.4,.4,.4]],dtype=np.float32)
    tau=np.array([-1,-1,-1,2,3,2])
    t=np.tile(np.arange(h),(2*n,1))
    y=((t>=tau[:,None])&(tau[:,None]>=0)).astype(np.uint8)
    counts=np.array([[1,1,1],[2,1,0],[0,0,3]],dtype=np.int32)
    prepared=PreparedAUC.create(y.ravel(),p.ravel(),t.ravel(),paired_group_codes(n,h),n)
    result=prepared.evaluate(counts)
    for j,row in enumerate(counts):
        selected=np.repeat(np.arange(n),row)
        selected=np.r_[selected,selected+n]
        assert result[j]==ts_auc(y[selected].ravel(),p[selected].ravel(),t[selected].ravel())
