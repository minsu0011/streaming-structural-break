"""Mixture priors, raw log densities, causality and long finite accumulation."""
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm
from src.next5 import mixture as m,innovation,scale


@pytest.mark.parametrize('kind',m.KINDS)
def test_hundred_current_prefix_and_length_cases(kind):
    rng=np.random.default_rng(20260908);settings={'version':'next5_fixed_mixture_v1','kind':kind}
    for case in range(100):
        h=rng.standard_t(5,192).astype(np.float32);o=rng.normal(size=80).astype(np.float32);cutoff=case%70+1
        changed=o.copy();changed[cutoff:]=rng.normal(0,20,len(o)-cutoff)
        full=m.make_state(h,settings).replay(o);other=m.make_state(h,settings).replay(changed)
        short=m.make_state(h,settings).replay(o[:cutoff]);state=m.make_state(h,settings)
        points=np.asarray([state.update(v).copy() for v in o])
        np.testing.assert_array_equal(full[:cutoff],other[:cutoff]);np.testing.assert_array_equal(full[:cutoff],short)
        np.testing.assert_array_equal(full,points);assert np.isfinite(full).all()


@pytest.mark.parametrize('kind',m.KINDS)
def test_mixture_agrees_with_direct_density_and_uniform_prior(kind):
    rng=np.random.default_rng(20260908);h=rng.normal(size=192).astype(np.float32);o=rng.normal(size=550).astype(np.float32)
    args,_=innovation.prepare(h);z=np.asarray([innovation.step(v,*args).copy() for v in o])
    actual=m.make_state(h,{'version':'next5_fixed_mixture_v1','kind':kind}).replay(o)
    for index in (0,1,7,31,127,255,256,257,549):
        for channel in range(2):
            evidence=[];ages=(index+1,) if kind=='cumulative' else m.AGES
            for age in ages:
                values=z[max(0,index-age+1):index+1,channel,None]
                evidence.append((norm.logpdf(values,scale=np.sqrt(scale.VARIANCES))-norm.logpdf(values)).sum(axis=0))
            evidence=np.asarray(evidence);lower=logsumexp(evidence[:,:2])-np.log(evidence[:,:2].size)
            upper=logsumexp(evidence[:,2:])-np.log(evidence[:,2:].size);total=logsumexp(evidence)-np.log(evidence.size)
            offset=channel*(6 if kind=='cumulative' else 9)
            np.testing.assert_allclose(actual[index,offset:offset+4],[lower,upper,total,upper-lower],rtol=2e-6,atol=2e-5)


@pytest.mark.parametrize('kind',m.KINDS)
def test_16k_null_and_constant_H_keep_finite_log_space(kind):
    rng=np.random.default_rng(20260908);state=m.make_state(np.ones(192,dtype=np.float32),{'version':'next5_fixed_mixture_v1','kind':kind})
    values=state.replay(rng.normal(size=16384).astype(np.float32));assert np.isfinite(values).all()
