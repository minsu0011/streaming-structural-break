import numpy as np
import pytest
from src.features.magnitude_dependence import MagnitudeARState,MagnitudeFeatureState,update_magnitude,replay_magnitude,names
from src.features.ar_order_input import AROrderCalibratedState
from src.features.config import CONFIG


@pytest.mark.parametrize('historical',[[],[1.],[2.]*300,[np.nan,np.inf,0.,-1.]*100])
def test_magnitude_features_preserve_abcd_and_obey_prefix_and_memory_contract(historical):
    config=CONFIG.variant(normalization='median_mad',scales=(5,20,160));original=AROrderCalibratedState(historical,config=config,input_order=8)
    combined=MagnitudeARState(historical,config=config);only=MagnitudeARState(historical,config=config,include_abcd=False);before=combined.state_array_bytes
    points=np.array([0.,1.,np.nan,-2.,500.]*25,dtype=np.float32);rows=[]
    for point in points:
        reference=original.update_and_get(point).copy();actual=combined.update_and_get(point).copy();v=only.update_and_get(point).copy()
        np.testing.assert_array_equal(actual[:51],reference[:51]);np.testing.assert_array_equal(actual[51:],v);assert np.isfinite(actual).all();rows.append(actual)
    changed=MagnitudeARState(historical,config=config);alternate=np.array([changed.update_and_get(p).copy() for p in np.r_[points[:20],np.full(30,-300.)]])
    np.testing.assert_array_equal(np.array(rows)[:20],alternate[:20]);assert combined.state_array_bytes==before and len(names(config))==57


def test_historical_magnitude_replay_has_the_exact_live_state_continuation():
    h=np.random.default_rng(137).standard_t(3,600).astype(np.float32);state=MagnitudeFeatureState(h)
    first=tuple(x.copy() for x in state.args);second=tuple(x.copy() for x in state.args)
    expected=[]
    for point in h:update_magnitude(float(point),*first);expected.append(first[-1].copy())
    actual=replay_magnitude(h,second);np.testing.assert_array_equal(actual,np.array(expected))
    for left,right in zip(first,second):np.testing.assert_array_equal(left,right)
