"""Causality, direct occupancy oracle, and analytic shock-state recovery."""
import numpy as np
import pytest
from src.next5 import persistence as p,innovation
from src.next4.scale_asymmetry import branch_values


@pytest.mark.parametrize('kind',p.KINDS)
def test_hundred_prefix_cases(kind):
    rng=np.random.default_rng(20260908);settings={'version':'next5_persistence_v1','kind':kind}
    for case in range(100):
        h=rng.standard_t(5,192).astype(np.float32);o=rng.normal(size=80).astype(np.float32);cutoff=case%70+1
        changed=o.copy();changed[cutoff:]=rng.normal(0,20,len(o)-cutoff)
        full=p.make_state(h,settings).replay(o);other=p.make_state(h,settings).replay(changed)
        short=p.make_state(h,settings).replay(o[:cutoff]);state=p.make_state(h,settings)
        incremental=np.asarray([state.update(v).copy() for v in o])
        np.testing.assert_array_equal(full[:cutoff],other[:cutoff]);np.testing.assert_array_equal(full[:cutoff],short)
        np.testing.assert_array_equal(full,incremental);assert np.isfinite(full).all() and full.shape==(80,16)


def test_exact_recent_confirmation_count_with_H_only_thresholds():
    rng=np.random.default_rng(20260908);h=rng.normal(size=192).astype(np.float32);o=rng.normal(size=150).astype(np.float32)
    settings={'version':'next5_persistence_v1','kind':'soft_count'};state=p.make_state(h,settings)
    threshold=state.args[2][:,2].copy();reference=state.args[1];args,_=innovation.prepare(h);indicators=[]
    for point in o:
        z=innovation.step(point,*args);raw=np.zeros(6);raw[0]=z[0]**2-1;raw[2]=z[1]**2-1
        indicators.append(branch_values(raw,reference,1,np.empty(4))>threshold)
    indicators=np.asarray(indicators);values=state.replay(o)
    np.testing.assert_array_equal(values[:,::4],indicators)
    for index in range(len(o)):
        np.testing.assert_allclose(values[index,1::4],indicators[max(0,index-31):index+1].mean(axis=0),rtol=1e-7,atol=1e-8)


def test_shock_decay_state_matches_direct_peak_envelope():
    rng=np.random.default_rng(20260908);h=rng.normal(size=192).astype(np.float32);o=np.r_[rng.normal(size=20),30.,np.zeros(100)].astype(np.float32)
    state=p.make_state(h,{'version':'next5_persistence_v1','kind':'shock_recovery'})
    reference=state.args[1];peak=state.args[2][:,0].copy();args,_=innovation.prepare(h);expected=[]
    for point in o:
        z=innovation.step(point,*args);raw=np.zeros(6);raw[0]=z[0]**2-1.;raw[2]=z[1]**2-1.
        value=branch_values(raw,reference,1,np.empty(4));peak=np.maximum(value,np.exp(-np.log(2.)/4.)*peak)
        expected.append(np.log1p(peak))
    values=state.replay(o);np.testing.assert_allclose(values[:,::4],expected,rtol=1e-7,atol=1e-7)


@pytest.mark.parametrize('kind',p.KINDS)
def test_degenerate_H_finite(kind):
    state=p.make_state(np.ones(192,dtype=np.float32),{'version':'next5_persistence_v1','kind':kind})
    assert np.isfinite(state.replay(np.r_[np.zeros(64),np.ones(64)].astype(np.float32))).all()
