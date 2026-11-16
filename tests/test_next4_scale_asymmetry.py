"""Causal append contract and independent original-feature parity."""
import numpy as np
import pytest
from src.next4 import scale_asymmetry as s


def settings(kind):return {'version':'separate_energy_branches_v1','kind':kind}


@pytest.mark.parametrize('kind',s.KINDS)
def test_hundred_prefix_cases_and_exact_original_77_columns(kind):
    rng=np.random.default_rng(20260908)
    for case in range(100):
        h=rng.standard_t(5,size=192).astype(np.float32);a=rng.normal(size=48).astype(np.float32)
        b=a.copy();prefix=case%40+1;b[prefix:]=rng.normal(0,20,len(b)-prefix)
        left=s.make_state(h,settings(kind)).replay(a);right=s.make_state(h,settings(kind)).replay(b)
        np.testing.assert_array_equal(left[:prefix],right[:prefix])
        old=s.ARCHCalibratedState(h,config=s.OLD_CONFIG)
        serial=s.SerialVarianceState(h,s.SerialVarianceConfig(order=8,normalization='arch1',max_age=512))
        original=np.asarray([np.r_[old.update_and_get(point)[s.OLD_COLUMNS],serial.update(point)[:26]] for point in a],dtype=np.float32)
        np.testing.assert_array_equal(left[:,:77],original)
        state=s.make_state(h,settings(kind));online=np.asarray([state.update(point).copy() for point in a])
        np.testing.assert_array_equal(left,online)
        assert left.shape==(48,93) and np.isfinite(left).all()


@pytest.mark.parametrize('kind',s.KINDS)
def test_degenerate_reference_is_finite(kind):
    state=s.make_state(np.ones(192,dtype=np.float32),settings(kind))
    assert np.isfinite(state.replay(np.r_[np.ones(64),np.zeros(64)].astype(np.float32))).all()


def test_separate_branches_not_globally_recentered():
    channels=np.zeros((100,6));channels[:,0]=np.linspace(-.99,10,100);channels[:,2]=channels[:,0]
    parameters,branches=s.branch_reference(channels,'split_energy')
    assert np.all(branches>=0) and np.all(branches[:,0]*branches[:,1]==0)
    assert np.all(branches[:,2]*branches[:,3]==0)
    raw=channels[20].copy();actual=s.branch_values(raw,parameters,0,np.empty(4))
    np.testing.assert_array_equal(actual,branches[20])


@pytest.mark.parametrize('bad',[{}, {'version':'separate_energy_branches_v1','kind':'split_energy','tau':1}])
def test_unregistered_or_target_input_rejected(bad):
    with pytest.raises(ValueError):s.make_state(np.ones(192),bad)
