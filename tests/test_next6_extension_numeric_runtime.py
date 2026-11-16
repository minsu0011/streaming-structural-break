import warnings,pickle
import numpy as np
import pytest
from src.next6_extension.fresh_runtime import SafeState
from src.next6_extension.fresh_state import State

@pytest.mark.parametrize('value',[np.finfo(np.float32).max,-np.finfo(np.float32).max,1e30,-1e30,0.,1.])
def test_extreme_constant_H_remains_insufficient_without_overflow(value):
    with warnings.catch_warnings():
        warnings.simplefilter('error');state=SafeState(np.full(1024,value,dtype=np.float32),threshold=1.);out=state.replay(np.zeros(256,dtype=np.float32))
    assert np.all(out[:,0]==6);assert np.isfinite(out).all();assert not state.calibration['reference_valid']

def test_adapter_preserves_frozen_outputs_on_ordinary_inputs():
    for seed in range(32):
        rng=np.random.default_rng(seed);h=rng.normal(size=2048).astype(np.float32);o=rng.normal(size=1024).astype(np.float32)
        old=State(h,threshold=1.).replay(o);new=SafeState(h,threshold=1.).replay(o);np.testing.assert_array_equal(old,new)

def test_finite_extreme_current_observations_have_finite_bounded_evidence():
    rng=np.random.default_rng(32);h=rng.normal(size=2048);o=np.full(256,np.finfo(np.float32).max)
    with warnings.catch_warnings():
        warnings.simplefilter('error');out=SafeState(h,threshold=1.).replay(o)
    assert np.isfinite(out).all();assert np.max(abs(out[:,12:]))<=12
