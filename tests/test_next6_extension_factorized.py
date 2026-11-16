import numpy as np
from src.next6_extension.factorized import ipf_null,fit_pair,Graph

def test_ipf_preserves_all_nuisance_marginals():
    rng=np.random.default_rng(121);counts=rng.lognormal(2,2,size=(2,3,3))+.5;null,n,error=ipf_null(counts)
    for axis in (0,1,2):np.testing.assert_allclose(null.sum(axis=axis),counts.sum(axis=axis),rtol=1e-10,atol=1e-10)
    # No phase-parent-current interaction means the parent-current odds ratio
    # is common to both phases even when every phase marginal differs.
    for a in range(2):
        for b in range(2):
            odds=null[:,a,b]*null[:,a+1,b+1]/(null[:,a,b+1]*null[:,a+1,b]);np.testing.assert_allclose(odds[0],odds[1],rtol=1e-9)

def test_ipf_recovers_known_no_interaction_distribution():
    rng=np.random.default_rng(432);a=rng.normal(size=(2,3,1));b=rng.normal(size=(2,1,3));c=rng.normal(size=(1,3,3));count=np.exp(a+b+c)
    null,_,_=ipf_null(count);np.testing.assert_allclose(null,count,rtol=1e-9)

def test_rare_context_backoff_prevents_spurious_zero_probabilities():
    current=np.ones((100,8),dtype=np.int8);previous=current.copy();phase=np.repeat([0,1],50);ids=np.repeat(np.arange(10),10)
    full,null,support,_=fit_pair(current,previous,phase,ids,0,1)
    assert (full>0).all() and (null>0).all();np.testing.assert_allclose(full.sum(axis=-1),1);np.testing.assert_array_equal(full,null)

def test_changed_marginals_without_interaction_are_removed():
    # Strong changes of target and parent frequencies, stable association.
    count=np.exp(np.array([[[1.],[0.],[-1.]],[[0.],[1.],[2.]]])+np.array([[[0.,1.,-1.]],[[2.,0.,1.]]])+np.array([[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]]))
    null,_,_=ipf_null(count);ratio=count[1]/count[0];nullratio=null[1]/null[0];np.testing.assert_allclose(np.log(ratio)-np.log(nullratio),0,atol=1e-9)
