import numpy as np
import pytest
from src.next.legacy_extension import LegacyState,LegacyConfig,OLD_CONFIG,OLD_POLICY,extended_historical_filter,historical_arch2
from src.features.arch_input import ARCHCalibratedState
from src.features.ar_order_input import AROrderCalibratedState,historical_filter


@pytest.mark.parametrize('order',[4,8])
def test_legacy_extended_ar_filter_matches_frozen_historical_fit(order):
    rng = np.random.default_rng(534)
    h = rng.standard_t(5,1200).astype(np.float32)
    actual = extended_historical_filter(h,order)
    expected = historical_filter(h,OLD_CONFIG,order)
    for a,b in zip(actual,expected):
        np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('order,normalization',[(4,'none'),(8,'none'),(8,'arch1')])
def test_legacy_abcd_anchor_is_bitwise_identical_to_frozen_pipeline(order,normalization):
    rng = np.random.default_rng(648)
    h = rng.standard_t(4,1500).astype(np.float32)
    points = rng.normal(size=710).astype(np.float32)
    original = ARCHCalibratedState(h,config=OLD_CONFIG,policy=OLD_POLICY) if normalization=='arch1' else AROrderCalibratedState(h,config=OLD_CONFIG,policy=OLD_POLICY,input_order=order)
    expected = np.array([original.update_and_get(point).copy()[:51] for point in points])
    actual = LegacyState(h,LegacyConfig(order,normalization)).replay(points)
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('normalization',['none','arch1','arch2','gaussian'])
def test_ar12_extension_prefix_stream_and_fixed_memory(normalization):
    rng = np.random.default_rng(855)
    h = rng.normal(size=1400).astype(np.float32)
    points = rng.standard_t(5,1500).astype(np.float32)
    state = LegacyState(h,LegacyConfig(12,normalization))
    size = state.state_array_bytes
    prefix = np.array([state.update(point).copy() for point in points[:730]])
    batch = LegacyState(h,LegacyConfig(12,normalization)).replay(points)
    np.testing.assert_array_equal(prefix,batch[:730])
    assert size==state.state_array_bytes and np.isfinite(batch).all()


def test_arch2_coefficients_and_forecasts_obey_frozen_bounds():
    rng = np.random.default_rng(167)
    residual = rng.standard_t(4,1200).astype(np.float32)
    parameters,previous,transformed = historical_arch2(residual)
    assert parameters[0]>0 and parameters[1:3].min()>=0 and parameters[1:3].sum()<=.950000000000001
    assert np.isfinite(transformed).all() and np.isfinite(previous).all()
