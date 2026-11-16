import numpy as np
import pytest
from src.next4 import temporal_state as s


SETTINGS={'version':'coupled_linear_state_v1','kind':'two_pole_readout'}


def test_one_hundred_random_prefix_and_streaming_cases():
    rng=np.random.default_rng(20260908)
    for case in range(100):
        h=rng.standard_t(5,size=128).astype(np.float32);o=rng.normal(size=65).astype(np.float32)
        altered=o.copy();prefix=1+case%48;altered[prefix:]=rng.normal(0,20,len(o)-prefix)
        a=s.make_state(h,SETTINGS).replay(o);b=s.make_state(h,SETTINGS).replay(altered)
        np.testing.assert_array_equal(a[:prefix],b[:prefix])
        state=s.make_state(h,SETTINGS);live=np.array([state.update(v).copy() for v in o])
        np.testing.assert_array_equal(a,live)
        assert a.shape==(65,32) and np.isfinite(a).all() and np.all(np.abs(a)<=1)


def test_two_pole_state_and_variance_against_impulse_convolution():
    rng=np.random.default_rng(28);values=rng.normal(size=(65,4));states=np.zeros((4,3,2));variances=np.zeros((3,3))
    alphas=np.array([1-np.exp(-np.log(2)/h) for h in s.HALVES]);out=np.empty(32)
    for t,x in enumerate(values,1):
        s.filter_step(x,states,variances,alphas,out)
        k=np.arange(t)
        for j,a in enumerate(alphas):
            first=a*(1-a)**k;second=a*a*(k+1)*(1-a)**k
            np.testing.assert_allclose(states[:,j,0],first@values[:t][::-1],atol=1e-14,rtol=0)
            np.testing.assert_allclose(states[:,j,1],second@values[:t][::-1],atol=1e-14,rtol=0)
            np.testing.assert_allclose(variances[j],[first@first,second@second,first@second],atol=1e-14,rtol=0)


@pytest.mark.parametrize('length',[1024,4096,16384])
def test_degenerate_long_stream_fixed_state(length):
    state=s.make_state(np.ones(128),SETTINGS);arrays=[a for a in state.args if isinstance(a,np.ndarray)]
    shapes=[(id(a),a.shape,a.nbytes) for a in arrays];values=state.replay(np.zeros(length))
    assert np.isfinite(values).all() and np.all(np.abs(values)<=1)
    assert shapes==[(id(a),a.shape,a.nbytes) for a in arrays]


def test_forbidden_length_or_identity_configuration():
    with pytest.raises(ValueError):s.make_state(np.ones(128),{**SETTINGS,'online_length':1000})
