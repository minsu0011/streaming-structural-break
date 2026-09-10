import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln
from src.next2.scale_likelihood import variance_log_bayes,directional_glr,ScaleState
from src.next.variance_evidence import VarianceState,VarianceConfig


@pytest.mark.parametrize('n,total',[(2,1.),(8,8.),(32,16.),(32,64.)])
@pytest.mark.parametrize('alpha',[2.,8.])
def test_variance_bayes_matches_independent_numeric_integral(n,total,alpha):
    beta=alpha-1.
    def integrand(log_variance):
        theta=np.exp(log_variance)
        prior=alpha*np.log(beta)-gammaln(alpha)-(alpha+1)*log_variance-beta/theta
        ratio=-.5*n*log_variance-.5*total/theta+.5*total
        return np.exp(prior+ratio+log_variance)
    value,error=quad(integrand,-25,25,epsabs=1e-10,epsrel=1e-10)
    np.testing.assert_allclose(variance_log_bayes(total,n,alpha),np.log(value),rtol=0,atol=1e-9)


@pytest.mark.parametrize('ratio',[.1,.5,1.,1.5,4.])
def test_directional_glr_is_exact_gaussian_log_likelihood_ratio(ratio):
    n=32; total=n*ratio
    lower,upper=directional_glr(total,n)
    direct=-.5*n*np.log(ratio)-.5*total/ratio+.5*total
    np.testing.assert_allclose(lower+upper,direct,atol=1e-12)
    assert (upper==0 if ratio<1 else lower==0)


@pytest.mark.parametrize('parent',['none','lower','ratios'])
def test_scale_evidence_is_prefix_causal_and_preserves_parent(parent):
    rng=np.random.default_rng(821)
    h=rng.standard_t(5,1024)
    o=rng.normal(size=1700).astype(np.float32)
    o[300:600]*=.4
    settings={'include_parent':parent}
    whole=ScaleState(h,settings).replay(o)
    prefix=ScaleState(h,settings).replay(o[:700])
    np.testing.assert_array_equal(whole[:700],prefix)
    assert np.isfinite(whole).all()
    if parent!='none':
        reference=VarianceState(h,VarianceConfig(max_age=512)).replay(o)[:,:26]
        np.testing.assert_array_equal(whole[:,:26],reference)


def test_scale_features_remain_finite_for_extreme_finite_observations():
    state=ScaleState(np.zeros(128),{})
    points=np.r_[np.zeros(20),np.float32(1e30),np.zeros(1200)].astype(np.float32)
    assert np.isfinite(state.replay(points)).all()
