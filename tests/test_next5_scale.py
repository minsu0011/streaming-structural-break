"""Independent density/window oracles and suffix-invariant scale evidence."""
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm,t
from src.next5 import scale as s,innovation


@pytest.mark.parametrize('student',[False,True])
def test_log_likelihood_matches_independent_distribution_density(student):
    for z in np.r_[np.linspace(-12,12,49),0.]:
        actual=s.likelihoods(z,student,np.empty(4))
        if student:
            expected=t.logpdf(z,df=5,scale=np.sqrt(3.*s.VARIANCES/5.))-t.logpdf(z,df=5,scale=np.sqrt(3./5.))
        else:expected=norm.logpdf(z,scale=np.sqrt(s.VARIANCES))-norm.logpdf(z)
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('kind',s.KINDS)
def test_hundred_prefixes_incremental_parity_and_final_length_invariance(kind):
    settings={'version':'next5_scale_unification_v1','kind':kind};rng=np.random.default_rng(20260908)
    for case in range(100):
        h=rng.standard_t(5,192).astype(np.float32);o=rng.normal(size=48).astype(np.float32)
        cutoff=case%40+1;changed=o.copy();changed[cutoff:]=rng.normal(0,30,len(o)-cutoff)
        full=s.make_state(h,settings).replay(o);other=s.make_state(h,settings).replay(changed)
        short=s.make_state(h,settings).replay(o[:cutoff]);state=s.make_state(h,settings)
        incremental=np.asarray([state.update(v).copy() for v in o])
        np.testing.assert_array_equal(full[:cutoff],other[:cutoff]);np.testing.assert_array_equal(full[:cutoff],short)
        np.testing.assert_array_equal(full,incremental)
        assert full.shape==(48,len(s.feature_names(settings))) and np.isfinite(full).all()


@pytest.mark.parametrize('kind',['directional','student'])
def test_rolling_likelihood_recomputed_from_current_prefix_across_ring_wrap(kind):
    rng=np.random.default_rng(20260908);h=rng.normal(size=192).astype(np.float32);o=rng.normal(size=300).astype(np.float32)
    args,_=innovation.prepare(h);z=np.asarray([innovation.step(v,*args).copy() for v in o])
    actual=s.make_state(h,{'version':'next5_scale_unification_v1','kind':kind}).replay(o)
    density=t if kind=='student' else norm
    for index in (0,7,8,31,32,127,128,129,255,299):
        for channel in range(2):
            for offset,window in enumerate(s.WINDOWS):
                values=z[max(0,index-window+1):index+1,channel,None]
                if kind=='student':lr=t.logpdf(values,df=5,scale=np.sqrt(3*s.VARIANCES/5))-t.logpdf(values,df=5,scale=np.sqrt(3/5))
                else:lr=norm.logpdf(values,scale=np.sqrt(s.VARIANCES))-norm.logpdf(values)
                sums=lr.sum(axis=0);lower=logsumexp(sums[:2])-np.log(2);upper=logsumexp(sums[2:])-np.log(2)
                expected=[lower,upper,upper-lower,logsumexp(sums)-np.log(4)]
                np.testing.assert_allclose(actual[index,channel*16+offset*4:channel*16+offset*4+4],expected,rtol=2e-6,atol=2e-5)


@pytest.mark.parametrize('kind',s.KINDS)
def test_constant_history_and_extreme_current_observation_are_finite(kind):
    state=s.make_state(np.ones(192,dtype=np.float32),{'version':'next5_scale_unification_v1','kind':kind})
    values=state.replay(np.r_[np.ones(32),np.zeros(32),np.full(8,1e8)].astype(np.float32))
    assert np.isfinite(values).all()


def test_no_tau_or_final_length_in_settings():
    with pytest.raises(ValueError):s.make_state(np.ones(192),{'version':'next5_scale_unification_v1','kind':'student','tau':4})
