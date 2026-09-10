import numpy as np
import pytest
from src.next2.variance_bank import BankState,AGE_GRIDS,ENERGIES,MEMORIES
from src.next.variance_evidence import VarianceState,VarianceConfig


def test_variance_bridge_is_exact_parent():
    rng=np.random.default_rng(719)
    h=rng.standard_t(5,2048)
    o=rng.normal(size=1700).astype(np.float32)
    expected=VarianceState(h,VarianceConfig(max_age=512)).replay(o)[:,:26]
    actual=BankState(h,{}).replay(o)
    np.testing.assert_array_equal(expected,actual)


@pytest.mark.parametrize('ages',list(AGE_GRIDS))
def test_sparse_age_has_finite_neutral_prior_and_fixed_ring(ages):
    rng=np.random.default_rng(712)
    state=BankState(rng.normal(size=512),{'ages':ages})
    value=state.replay(rng.normal(size=2000))
    assert np.isfinite(value).all()
    assert state.args[10][5].shape==(max(AGE_GRIDS[ages])+1,2)
    warmup=min(AGE_GRIDS[ages])-1
    if warmup:
        np.testing.assert_array_equal(value[:warmup,[6,7,8,9,10,11,12]],0.)


@pytest.mark.parametrize('energy',ENERGIES)
def test_energy_coordinates_are_causal_and_finite(energy):
    rng=np.random.default_rng(91)
    h=rng.standard_t(3,512)
    o=rng.normal(size=750).astype(np.float32)
    o[220]=1e6
    state=BankState(h,{'energy':energy})
    prefix=state.replay(o[:350])
    state=BankState(h,{'energy':energy})
    full=state.replay(o)
    assert np.isfinite(full).all()
    np.testing.assert_array_equal(prefix,full[:350])


@pytest.mark.parametrize('memory',MEMORIES)
def test_memory_modes_do_not_change_nonmemory_features(memory):
    rng=np.random.default_rng(191)
    h,o=rng.normal(size=512),rng.normal(size=701)
    a=BankState(h,{}).replay(o)
    b=BankState(h,{'memory':memory}).replay(o)
    columns=[j for j in range(26) if j%13<11]
    np.testing.assert_array_equal(a[:,columns],b[:,columns])
