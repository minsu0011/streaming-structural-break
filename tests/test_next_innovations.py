import numpy as np
import pytest
from scipy.special import ndtri
from src.next.whitening import fit_historical, innovation_update, _innovations, AR_ORDERS
from src.next.normalization import inverse_normal, ecdf_uniform, fit_normalization, normalization_update, MODES


@pytest.mark.parametrize('order', AR_ORDERS)
def test_ar_one_step_matches_direct_historical_plus_causal_prefix(order):
    rng = np.random.default_rng(415)
    h = rng.standard_t(4, 1300).astype(np.float32)
    o = rng.normal(size=101).astype(np.float32)
    f = fit_historical(h, order)
    reference = f['reference']
    z = np.clip((np.concatenate([h, o]).astype(float)-reference[0])/reference[1], -12, 12)-reference[2]
    expected = _innovations(z, f['coefficients'])[-len(o):]
    actual = np.array([innovation_update(x, reference, f['coefficients'], f['ring'], f['counter']) for x in o])
    np.testing.assert_array_equal(actual, expected)
    assert np.max(np.abs(np.roots(np.r_[1., -f['coefficients']]))) <= .995000001


def test_inverse_normal_accuracy_and_ecdf_ties():
    ps = np.r_[np.geomspace(1e-8, .02, 100), np.linspace(.025, .975, 100), 1-np.geomspace(1e-8, .02, 100)]
    np.testing.assert_allclose([inverse_normal(p) for p in ps], ndtri(ps), rtol=0, atol=8e-9)
    h = np.array([-1., 0., 0., 1.])
    assert ecdf_uniform(0., h) == .5
    assert ecdf_uniform(-10., h) == .1
    assert ecdf_uniform(10., h) == .9


@pytest.mark.parametrize('mode', MODES)
def test_normalization_prefix_independent_of_future_and_finite(mode):
    rng = np.random.default_rng(915)
    h = rng.standard_t(3, 1000)
    a, b = fit_normalization(h, mode), fit_normalization(h, mode)
    prefix = rng.normal(size=30)
    def replay(f, x):
        return np.array([normalization_update(v, f['parameters'], f['conditional'], f['sorted_h'], f['mode']) for v in x])
    one = replay(a, prefix)
    two = replay(b, np.r_[prefix, 10000., -10000.])
    np.testing.assert_array_equal(one, two[:len(prefix)])
    assert np.isfinite(two).all()
    assert np.all((two[:,1] > 0) & (two[:,1] < 1))


def test_arch_prediction_variance_uses_previous_values_only():
    f = fit_normalization(np.random.default_rng(51).normal(size=1000), 'arch2')
    previous = f['conditional'].copy()
    parameters = f['parameters']
    residual = .5
    variance = max(parameters[2]+parameters[3]*previous[0]+parameters[4]*previous[1], parameters[6])
    z, _ = normalization_update(residual, parameters, f['conditional'], f['sorted_h'], f['mode'])
    assert abs(z-(residual-parameters[0])/np.sqrt(variance)) < 1e-12
