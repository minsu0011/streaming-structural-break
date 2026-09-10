import numpy as np
import pytest
from scipy.special import logsumexp
from src.next.channel_evidence import make_channel_args, channel_step, CHANNEL_STATS
from src.next.mismatch_evidence import MismatchState, MismatchConfig, CHANNELS, mismatch_channels, feature_names
from src.next.whitening import fit_historical


@pytest.mark.parametrize('max_age',[128,512])
def test_channel_glr_and_integrated_mean_bayes_match_direct_suffixes(max_age):
    rng = np.random.default_rng(722)
    values = rng.normal(size=(1100,3))
    args = make_channel_args(3,max_age)
    for t, row in enumerate(values,1):
        output = channel_step(row,args).reshape(3,len(CHANNEL_STATS))
        if t not in (1,2,65,129,513,800,1100):
            continue
        ages = args[0][args[0]<=t]
        widths = args[1][:len(ages)]
        sums = np.array([np.clip(values[t-age:t],-12,12).sum(axis=0) for age in ages])
        glr = .5*sums**2/ages[:,None]
        np.testing.assert_allclose(output[:,6],glr.max(axis=0).astype(np.float32),rtol=1e-6,atol=1e-6)
        for prior,column in [(.25,9),(1.,10)]:
            logbf = -.5*np.log1p(prior*ages[:,None])+.5*prior*sums**2/(1+prior*ages[:,None])
            expected = logsumexp(logbf+np.log(widths[:,None]),axis=0)-np.log(widths.sum())
            np.testing.assert_allclose(output[:,column],expected.astype(np.float32),rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize('pair',[(4,8),(8,12)])
def test_mismatch_filters_match_direct_frozen_ar_predictions(pair):
    rng = np.random.default_rng(187)
    h = rng.normal(size=1600).astype(np.float32)
    points = rng.normal(size=600).astype(np.float32)
    config = MismatchConfig(*pair)
    state = MismatchState(h,config)
    fits = [fit_historical(h,p) for p in pair]
    all_points = np.r_[h,points].astype(float)
    ref = fits[0]['reference']
    z = np.clip((all_points-ref[0])/ref[1],-12,12)-ref[2]
    raw = np.empty(4)
    for j,point in enumerate(points):
        state.update(point)
        errors = [z[len(h)+j]-np.dot(fit['coefficients'],z[len(h)+j-order:len(h)+j][::-1]) for order,fit in zip(pair,fits)]
        mismatch_channels(*errors,state.calibration[0,0],raw)
        expected = (raw-state.calibration[1])/state.calibration[2]
        np.testing.assert_allclose(state.work,expected,atol=1e-11,rtol=1e-10)


def test_mismatch_future_invariance_and_fixed_state():
    rng = np.random.default_rng(836)
    h = rng.standard_t(6,1700).astype(np.float32)
    points = rng.normal(size=1500).astype(np.float32)
    state = MismatchState(h)
    size = state.state_array_bytes
    calibration = state.calibration.copy()
    stream = np.array([state.update(x).copy() for x in points[:850]])
    batch = MismatchState(h).replay(points)
    np.testing.assert_array_equal(stream,batch[:850])
    np.testing.assert_array_equal(calibration,state.calibration)
    assert state.state_array_bytes == size


def test_long_lag_change_reverses_high_order_prediction_advantage():
    rng = np.random.default_rng(246)
    values = np.zeros(8000,dtype=np.float32)
    for t in range(8,len(values)):
        phi = .6 if t<6500 else -.6
        values[t] = phi*values[t-8]+.8*rng.normal()
    result = MismatchState(values[:6000]).replay(values[6000:])
    column = feature_names({}).index('energy_advantage_ewma128')
    assert np.median(result[1100:,column]) < np.median(result[200:450,column])-5
