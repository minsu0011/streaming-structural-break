import numpy as np
import pytest
from src.features.arch_input import arch_one,historical_variance_filter,ARCHCalibratedState
from src.features.config import CONFIG


def test_variance_forecast_uses_previous_observation_and_updates_after_output():
    parameters=np.array([1.,.5,10.,1e-8,2.]);state=np.array([4.])
    assert arch_one(3.,parameters,state)==np.float32(3./np.sqrt(3.))
    assert state[0]==9.
    assert arch_one(2.,parameters,state)==np.float32(2./np.sqrt(5.5))
    assert np.isnan(arch_one(np.nan,parameters,state)) and state[0]==2.


@pytest.mark.parametrize('historical',[[],[1.],[2.]*300,[np.nan,np.inf,0.,-1.]*100])
def test_arch_prefix_finite_and_fixed_memory(historical):
    config=CONFIG.variant(normalization='median_mad',scales=(5,20,160));a=ARCHCalibratedState(historical,config=config);b=ARCHCalibratedState(historical,config=config)
    before=a.state_array_bytes;points=np.array([0.,1.,np.nan,-2.,500.]*30,dtype=np.float32)
    actual=np.stack([a.update_and_get(p).copy() for p in points]);altered=np.r_[points[:25],np.full(13,-1000.)]
    changed=np.stack([b.update_and_get(p).copy() for p in altered]);np.testing.assert_array_equal(actual[:25],changed[:25])
    assert np.isfinite(actual).all() and a.state_array_bytes==before


def test_historical_normalization_preserves_causal_continuation():
    residual=np.array([.1,1.,-.5,2.,.2,1.1],dtype=np.float32)
    parameters,last,transformed=historical_variance_filter(residual)
    previous=np.array([min(float(residual[0])**2,parameters[2])])
    expected=np.array([arch_one(float(x),parameters,previous) for x in residual[1:]],dtype=np.float32)
    np.testing.assert_array_equal(expected,transformed);np.testing.assert_array_equal(previous,last)
    assert 0<=parameters[1]<=.95 and parameters[0]>0
