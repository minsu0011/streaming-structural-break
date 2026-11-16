"""Independent robust location density, sign-run state and prefix falsification."""
import numpy as np
import pytest
from scipy.stats import t
from src.next5 import location as m,innovation


def test_robust_location_density_units():
    for z in np.linspace(-12,12,49):
        for shift in m.SHIFTS:
            expected=t.logpdf(z,df=5,loc=shift,scale=np.sqrt(3/5))-t.logpdf(z,df=5,scale=np.sqrt(3/5))
            assert abs(m.location_lr(z,shift)-expected)<1e-12


@pytest.mark.parametrize('kind',m.KINDS)
def test_hundred_suffix_final_length_and_reset_cases(kind):
    rng=np.random.default_rng(20260908);settings={'version':'next5_location_v1','kind':kind}
    for case in range(100):
        h=rng.standard_t(5,192).astype(np.float32);o=rng.normal(size=80).astype(np.float32);cutoff=case%70+1
        changed=o.copy();changed[cutoff:]=rng.normal(0,20,len(o)-cutoff)
        full=m.make_state(h,settings).replay(o);other=m.make_state(h,settings).replay(changed)
        short=m.make_state(h,settings).replay(o[:cutoff]);state=m.make_state(h,settings)
        incremental=np.asarray([state.update(v).copy() for v in o])
        np.testing.assert_array_equal(full[:cutoff],other[:cutoff]);np.testing.assert_array_equal(full[:cutoff],short)
        np.testing.assert_array_equal(full,incremental);assert np.isfinite(full).all()


def test_sign_run_matches_direct_history_of_signs():
    rng=np.random.default_rng(20260908);h=rng.normal(size=192).astype(np.float32);o=rng.normal(size=150).astype(np.float32)
    state=m.make_state(h,{'version':'next5_location_v1','kind':'sign_run'});parameters=state.args[2];args,_=innovation.prepare(h)
    z=np.asarray([innovation.step(v,*args).copy() for v in o]);actual=state.replay(o)
    for channel in range(2):
        signs=np.sign(z[:,channel]-parameters[channel,0]);run=0;last=0
        for j,sign in enumerate(signs):
            run=run+1 if sign!=0 and sign==last else 1 if sign!=0 else 0;last=sign
            assert abs(actual[j,channel*6+4]-sign*np.log1p(run))<1e-6
            assert abs(actual[j,channel*6+5]-sign*(1-parameters[channel,3]**run))<1e-6


@pytest.mark.parametrize('kind',m.KINDS)
def test_constant_history_finite(kind):
    state=m.make_state(np.ones(192,dtype=np.float32),{'version':'next5_location_v1','kind':kind})
    assert np.isfinite(state.replay(np.r_[np.zeros(128),np.ones(128)].astype(np.float32))).all()
