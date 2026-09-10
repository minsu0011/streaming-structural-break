import numpy as np
import pytest
from src.next2.counterexamples import generate,policy,recurrence,noise


@pytest.mark.parametrize('scenario',[s['id'] for s in policy()['scenarios']])
def test_counterexample_prefix_and_h_are_horizon_invariant(scenario):
    h,a,t=generate(scenario,3,horizon=1024)
    h2,b,t2=generate(scenario,3,horizon=16384)
    np.testing.assert_array_equal(h,h2)
    np.testing.assert_array_equal(a,b[:1024])
    assert t==t2


def test_single_outlier_does_not_propagate_as_persistent_dgp_change():
    h,o,t=generate('O1',4)
    hc,c,tc=generate('O1',4,control=True)
    np.testing.assert_array_equal(h,hc)
    different=np.flatnonzero(o!=c)
    np.testing.assert_array_equal(different,[256])
    assert t==tc==-1
    assert abs(float(o[256]-c[256])-20)<1e-5


@pytest.mark.parametrize('factor',[1.5,2.,.7,.5])
def test_variance_multiplier_is_not_squared(factor):
    eps=noise([412],1000,None)
    before=recurrence(eps,0.,0.,0.,0.,1.,0.,0)
    after=recurrence(eps,0.,0.,0.,0.,factor,0.,0)
    np.testing.assert_allclose(after,before*np.sqrt(factor),rtol=1e-14,atol=1e-14)


def test_synthetic_streams_and_controls_have_independent_partitions():
    h,a,_=generate('V4',1,partition=1)
    hc,c,_=generate('V4',1,partition=1,control=True)
    other,_,_=generate('V4',1,partition=0)
    np.testing.assert_array_equal(h,hc)
    np.testing.assert_array_equal(a[:256],c[:256])
    assert not np.array_equal(h,other)
