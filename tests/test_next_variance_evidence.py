import numpy as np
import pytest
from src.next.variance_evidence import VarianceState,VarianceConfig,variance_channel_update,feature_names


def test_variance_scores_are_finite_difference_gaussian_likelihood_derivatives():
    parameters = np.array([.1,1.,.4,.3,.2,10.,1e-4,2/33])
    conditional = np.array([1.3,.8,1.1])
    residual = 1.9
    out = np.empty(6)
    variance_channel_update(residual,parameters,conditional.copy(),4,out)
    h_variance = .4/(1-.3-.2)
    def loglik(omega,alpha1,alpha2):
        v = omega+alpha1*conditional[0]+alpha2*conditional[1]
        return -.5*np.log(v)-.5*(residual-.1)**2/v
    delta = 1e-5
    args = [.4,.3,.2]
    for j,scale in [(0,h_variance),(1,1.),(2,1.)]:
        left,right = args.copy(),args.copy()
        left[j] -= delta*scale
        right[j] += delta*scale
        expected = (loglik(*right)-loglik(*left))/(2*delta)
        np.testing.assert_allclose(out[3+j],expected,rtol=1e-9,atol=1e-9)


@pytest.mark.parametrize('mode',['arch1','arch2','ewma'])
def test_conditional_forecast_excludes_current_observation_and_prefix_is_exact(mode):
    rng = np.random.default_rng(596)
    h = rng.standard_t(6,1800).astype(np.float32)
    points = rng.normal(size=1200).astype(np.float32)
    config = VarianceConfig(normalization=mode)
    low,high = VarianceState(h,config),VarianceState(h,config)
    low.update(-10.)
    high.update(10.)
    assert low.work[1] == high.work[1]
    state = VarianceState(h,config)
    size = state.state_array_bytes
    fixed = state.parameters.copy(),state.calibration.copy()
    stream = np.array([state.update(x).copy() for x in points[:731]])
    batch = VarianceState(h,config).replay(points)
    np.testing.assert_array_equal(stream,batch[:731])
    np.testing.assert_array_equal(state.parameters,fixed[0])
    np.testing.assert_array_equal(state.calibration,fixed[1])
    assert size == state.state_array_bytes and np.isfinite(batch).all()


def test_variance_scale_change_is_retained_even_when_forecast_adapts():
    rng = np.random.default_rng(673)
    h = rng.normal(size=5000).astype(np.float32)
    original = rng.normal(size=1200).astype(np.float32)
    changed = (2*rng.normal(size=1200)).astype(np.float32)
    config = VarianceConfig(normalization='ewma')
    reference = VarianceState(h,config).replay(original)
    candidate = VarianceState(h,config).replay(changed)
    column = feature_names({}).index('log_variance_forecast_ewma128')
    assert np.median(candidate[600:,column]) > np.median(reference[600:,column])+10
