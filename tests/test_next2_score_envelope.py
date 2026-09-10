from types import SimpleNamespace
import numpy as np
from src.next2.score_envelope import probability_upper_bound,no_alarm_after,mapped_upper
from src.next2.synthetic_analysis import score


def test_leafwise_envelope_dominates_every_leaf_combination():
    tree=SimpleNamespace(bias=.1,roots=np.array([0,3]),features=np.array([0,-1,-1,0,-1,-1]),
        values=np.array([0.,-1.,2.,0.,-3.,1.]))
    bound=probability_upper_bound(SimpleNamespace(compact=tree))['probability_upper_bound']
    for left in (-1.,2.):
        for right in (-3.,1.):
            assert np.float32(1/(1+np.exp(-(.1+left+right))))<=bound


def test_impossibility_boundary_with_strict_threshold():
    for beta in (.5,1.):
        index=no_alarm_after(.998,.62,beta)
        assert mapped_upper(.998,index,beta)<=.62
        assert mapped_upper(.998,index+100000,beta)<=.62
        assert mapped_upper(.998,index-1,beta)>.62
    assert no_alarm_after(1.,.5,1.) is None
    assert no_alarm_after(.9,.5,0.) is None
    assert no_alarm_after(.9,.9,1.)==0


def test_common_all_negative_prefix_can_be_removed_from_pair_weighted_auc():
    rng=np.random.default_rng(22);n=16;onset=37
    p=np.round(rng.random((2*n,128)),2)
    full_tau=np.r_[np.full(n,-1),np.full(n,onset)]
    cut_tau=np.r_[np.full(n,-1),np.zeros(n,dtype=int)]
    assert score(p,full_tau)==score(p[:,onset:],cut_tau)
