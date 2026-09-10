import numpy as np
import pytest
from src.next.forward_variance_evidence import ForwardVarianceConfig,ForwardVarianceState
from src.next.whitening import fit_historical,innovation_update
from src.next.normalization import fit_normalization
from src.next.variance_evidence import variance_channel_update


@pytest.mark.parametrize('fraction',[.5,.75])
def test_forward_fit_calibration_matches_manual_causal_replay(fraction):
    rng = np.random.default_rng(348)
    h = rng.standard_t(7,1700).astype(np.float32)
    config = ForwardVarianceConfig(fit_fraction=fraction)
    state = ForwardVarianceState(h,config)
    cut = int(len(h)*fraction)
    fitted = fit_historical(h[:cut],8)
    profile = fit_normalization(fitted['historical_innovations'],'arch1')
    args = tuple(fitted[k] for k in ('reference','coefficients','ring','counter'))
    parameter_before = profile['parameters'].copy()
    channels = []
    work = np.empty(6)
    for point in h[cut:]:
        residual = innovation_update(float(point),*args)
        variance_channel_update(residual,profile['parameters'],profile['conditional'],profile['mode'],work)
        channels.append(work.copy())
    channels = np.asarray(channels)[32:]
    calibration = np.stack([channels.mean(axis=0),np.maximum(channels.std(axis=0),1e-6)])
    np.testing.assert_array_equal(state.calibration,calibration)
    np.testing.assert_array_equal(state.parameters,parameter_before)
    np.testing.assert_array_equal(state.conditional,profile['conditional'])
    for actual,expected in zip(state.filter_args,args):
        np.testing.assert_array_equal(actual,expected)
    points = rng.normal(size=600).astype(np.float32)
    full = state.replay(points)
    np.testing.assert_array_equal(ForwardVarianceState(h,config).replay(points[:315]),full[:315])
    np.testing.assert_array_equal(state.parameters,parameter_before)


def test_calibration_tail_cannot_change_fitted_parameters():
    rng = np.random.default_rng(128)
    h = rng.normal(size=1200).astype(np.float32)
    changed = h.copy()
    changed[900:] *= 10.
    left,right = ForwardVarianceState(h),ForwardVarianceState(changed)
    np.testing.assert_array_equal(left.parameters,right.parameters)
    np.testing.assert_array_equal(left.filter_args[0],right.filter_args[0])
    np.testing.assert_array_equal(left.filter_args[1],right.filter_args[1])
    assert not np.array_equal(left.calibration,right.calibration)


def test_nonfinite_calibration_tail_is_rejected():
    h = np.ones(100,dtype=np.float32)
    h[-1] = np.nan
    with pytest.raises(ValueError,match='finite'):
        ForwardVarianceState(h)
