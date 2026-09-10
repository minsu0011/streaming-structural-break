import numpy as np
from src.next.channel_evidence import make_channel_args, channel_step
from src.next.serial_channel_evidence import sum_variances, serial_channel_step, historical_correlations
from src.next.variance_evidence import VarianceState
from src.next.serial_variance_evidence import SerialVarianceState, SerialVarianceConfig


def test_sum_variance_matches_full_ar1_covariance_and_stays_positive():
    ages = np.array([1, 2, 4, 8, 32, 128, 512])
    rho = np.array([-.95, -.4, 0., .6, .95])
    actual = sum_variances(ages, rho)
    for channel, correlation in enumerate(rho):
        for k, age in enumerate(ages):
            lag = np.abs(np.arange(age)[:, None]-np.arange(age))
            np.testing.assert_allclose(actual[channel, k], np.sum(correlation**lag), rtol=1e-12, atol=1e-10)
    assert np.all(actual > 0)


def test_zero_correlation_is_exact_original_channel_evidence():
    rng = np.random.default_rng(609)
    original, corrected = make_channel_args(2, 512), make_channel_args(2, 512)
    variances = sum_variances(corrected[0], [0., 0.])
    for values in rng.normal(size=(1100, 2)):
        np.testing.assert_array_equal(channel_step(values, original), serial_channel_step(values, corrected, variances))


def test_correction_preserves_local_features_and_future_prefix():
    rng = np.random.default_rng(610)
    historical = np.cumsum(rng.normal(size=1024)).astype(np.float32)
    points = rng.normal(size=600).astype(np.float32)
    ordinary, corrected = VarianceState(historical), SerialVarianceState(historical)
    np.testing.assert_array_equal(ordinary.parameters, corrected.parameters)
    np.testing.assert_array_equal(ordinary.calibration, corrected.calibration)
    np.testing.assert_array_equal(ordinary.conditional, corrected.conditional)
    local = [channel*13+stat for channel in range(6) for stat in range(6)]
    a, b = ordinary.replay(points), corrected.replay(points)
    np.testing.assert_array_equal(a[:, local], b[:, local])
    fresh = SerialVarianceState(historical)
    np.testing.assert_array_equal(fresh.replay(points[:311]), b[:311])
    assert np.all(np.abs(corrected.correlations) <= .95)


def test_h_parameters_stay_frozen_under_extreme_online_suffix():
    rng = np.random.default_rng(611)
    state = SerialVarianceState(rng.normal(size=1024), SerialVarianceConfig(max_age=512))
    rho, variance = state.correlations.copy(), state.age_variances.copy()
    size = state.state_array_bytes
    state.replay(rng.normal(size=1500).astype(np.float32)*20)
    np.testing.assert_array_equal(state.correlations, rho)
    np.testing.assert_array_equal(state.age_variances, variance)
    assert state.state_array_bytes == size
    np.testing.assert_array_equal(historical_correlations(np.ones((100, 2))), [0., 0.])
