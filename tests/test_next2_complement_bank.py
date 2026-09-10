import numpy as np
import pytest
from src.next2.complement_bank import ComplementState,FAMILIES,cdf_step


@pytest.mark.parametrize('family',FAMILIES)
def test_complement_streaming_replay_prefix(family):
    rng=np.random.default_rng(729)
    h=rng.standard_t(5,1024)
    o=rng.normal(size=801).astype(np.float32)
    o[400:]*=.5
    settings={'family':family}
    a=ComplementState(h,settings).replay(o)
    state=ComplementState(h,settings)
    b=np.stack([state.update(x).copy() for x in o])
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(a[:400],ComplementState(h,settings).replay(o[:400]))
    assert np.isfinite(a).all()


@pytest.mark.parametrize('bins',[8,16])
def test_incremental_histogram_matches_explicit_ewma(bins):
    p=np.full(bins,1/bins)
    state=np.tile(p,(3,1))
    expected=state.copy()
    output=np.empty(18,dtype=np.float32)
    for u in (.01,.99,.4,.7,.4,.01):
        actual=cdf_step(u,p,state,output)
        onehot=np.eye(bins)[min(int(u*bins),bins-1)]
        for j,h in enumerate((8.,32.,128.)):
            alpha=1-np.exp(-np.log(2)/h)
            expected[j]=(1-alpha)*expected[j]+alpha*onehot
            delta=expected[j]-p
            np.testing.assert_allclose(actual[j*6],np.sum(delta**2/p),rtol=1e-6)
        np.testing.assert_allclose(state,expected,rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(state.sum(axis=1),1.,atol=1e-14)
