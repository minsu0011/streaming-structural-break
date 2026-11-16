"""PIT ties/endpoints, adaptive density units and causal H-only calibration."""
import numpy as np
import pytest
from scipy.stats import t
from src.next5 import robust_null as r,scale


@pytest.mark.parametrize('kind',r.KINDS)
def test_hundred_prefix_length_and_incremental_cases(kind):
    rng=np.random.default_rng(20260908);settings={'version':'next5_robust_null_v1','kind':kind}
    for case in range(100):
        h=rng.standard_t(5,192).astype(np.float32);o=rng.normal(size=80).astype(np.float32);cutoff=case%70+1
        changed=o.copy();changed[cutoff:]=rng.normal(0,20,len(o)-cutoff)
        full=r.make_state(h,settings).replay(o);other=r.make_state(h,settings).replay(changed)
        short=r.make_state(h,settings).replay(o[:cutoff]);state=r.make_state(h,settings)
        incremental=np.asarray([state.update(v).copy() for v in o])
        np.testing.assert_array_equal(full[:cutoff],other[:cutoff]);np.testing.assert_array_equal(full[:cutoff],short)
        np.testing.assert_array_equal(full,incremental);assert np.isfinite(full).all()


def test_PIT_midrank_ties_and_open_interval_endpoints():
    history=np.array([0.,0.,1.,2.,5.])
    for value in (-1.,0.,.5,1.,2.,5.,10.):
        expected=(np.sum(history<value)+np.sum(history<=value)+1)/(2*(len(history)+1))
        assert r.energy_pit(value,history)==expected and 0<expected<1


def test_adaptive_t_variance_ratio_matches_density():
    for nu in (4.,5.,10.,30.):
        for z in (-12.,-.1,0.,.5,4.,12.):
            actual=r.t_likelihoods(z,nu,np.empty(4))
            expected=t.logpdf(z,df=nu,scale=np.sqrt((nu-2)*scale.VARIANCES/nu))-t.logpdf(z,df=nu,scale=np.sqrt((nu-2)/nu))
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)


def test_adaptive_null_parameters_depend_on_H_only():
    h=np.random.default_rng(20260908).standard_t(5,512).astype(np.float32)
    state=r.make_state(h,{'version':'next5_robust_null_v1','kind':'adaptive_t'});before=state.args[1].copy()
    state.replay(np.full(200,100.,dtype=np.float32));np.testing.assert_array_equal(before,state.args[1]);assert np.all((before>=4)&(before<=30))


@pytest.mark.parametrize('kind',r.KINDS)
def test_degenerate_H_and_extreme_point_finite(kind):
    state=r.make_state(np.ones(192,dtype=np.float32),{'version':'next5_robust_null_v1','kind':kind})
    assert np.isfinite(state.replay(np.r_[np.zeros(128),1e8,np.ones(128)].astype(np.float32))).all()
