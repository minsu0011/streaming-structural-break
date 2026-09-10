import numpy as np
import pytest
from src.features.historical_quantiles import HistoricalQuantileARState,historical_mapping,quantile_transform
from src.features.ar_order_input import AROrderCalibratedState
from src.features.streaming import FEATURE_FAMILIES
from src.features.config import CONFIG


@pytest.mark.parametrize('historical',[[],[1.],[2.]*300,[np.nan,np.inf,0.,-1.]*100])
def test_quantile_state_is_finite_causal_fixed_memory_and_preserves_other_families(historical):
    config=CONFIG.variant(normalization='median_mad',scales=(5,20,160));state=HistoricalQuantileARState(historical,families='C',config=config)
    original=AROrderCalibratedState(historical,config=config,input_order=8);before=state.state_array_bytes
    retained=np.array([family!='C' for family in FEATURE_FAMILIES]);points=np.array([0.,1.,np.nan,-2.,500.]*20,dtype=np.float32)
    actual=[]
    for point in points:
        left=state.update_and_get(point).copy();right=original.update_and_get(point).copy()
        np.testing.assert_array_equal(left[retained],right[retained]);assert np.isfinite(left).all();actual.append(left)
    alternate=HistoricalQuantileARState(historical,families='C',config=config)
    changed=np.array([alternate.update_and_get(p).copy() for p in np.r_[points[:20],np.full(7,-1000.)]])
    np.testing.assert_array_equal(np.array(actual)[:20],changed[:20]);assert state.state_array_bytes==before


def test_historical_knots_define_finite_monotone_bounded_mapping():
    rng=np.random.default_rng(745);mapping=historical_mapping(rng.standard_t(4,1200).astype(np.float32),families='ABCD')
    center,scale,mask,knots,scores,lengths=mapping;output=np.zeros(57,dtype=np.float32);values=[]
    for value in np.linspace(-100,100,301):
        quantile_transform(np.full(57,value,dtype=np.float32),*mapping,12.,output);values.append(output.copy())
    matrix=np.array(values);assert np.isfinite(matrix).all() and np.all(np.diff(matrix,axis=0)>=0)
    assert np.max(abs(matrix[:,lengths>=2]))<=3.1
    for j,count in enumerate(lengths):
        if count>=2:assert np.all(np.diff(knots[j,:count])>0) and np.all(np.diff(scores[j,:count])>=0)
