"""Reference adapters must preserve original values and current-prefix causality."""
import numpy as np
import pytest
from src.next5 import reference_features as r


@pytest.mark.parametrize('kind',r.KINDS)
def test_reference_original_parity_and_hundred_prefix_cases(kind):
    settings={'version':'next5_reference_features_v1','kind':kind}
    rng=np.random.default_rng(20260908)
    for case in range(100):
        h=rng.standard_t(5,192).astype(np.float32);online=rng.normal(size=48).astype(np.float32)
        cutoff=case%40+1;changed=online.copy();changed[cutoff:]=rng.normal(0,20,len(online)-cutoff)
        first=r.make_state(h,settings).replay(online);second=r.make_state(h,settings).replay(changed)
        np.testing.assert_array_equal(first[:cutoff],second[:cutoff])
        state=r.make_state(h,settings);incremental=np.asarray([state.update(v).copy() for v in online])
        np.testing.assert_array_equal(first,incremental)
        if kind=='J102':
            old=r.scale_asymmetry.ARCHCalibratedState(h,config=r.scale_asymmetry.OLD_CONFIG)
            serial=r.scale_asymmetry.SerialVarianceState(h,r.scale_asymmetry.SerialVarianceConfig(order=8,normalization='arch1',max_age=512))
            expected=np.asarray([np.r_[old.update_and_get(v)[r.scale_asymmetry.OLD_COLUMNS],serial.update(v)[:26]] for v in online],dtype=np.float32)
        elif kind=='C205':
            original=r.CompactDependenceState(h);expected=np.asarray([original.update(v).copy() for v in online])
        else:
            original=r.sequential.make_state(h,r.D2) if kind=='D2_expert' else r.scale_asymmetry.make_state(h,r.B2)
            expected=original.replay(online)
        np.testing.assert_array_equal(first,expected)
        assert first.shape==(48,len(r.feature_names(settings))) and np.isfinite(first).all()
    assert len(r.implementation_hash(settings))==64


@pytest.mark.parametrize('kind',r.KINDS)
def test_reference_constant_historical_finite(kind):
    settings={'version':'next5_reference_features_v1','kind':kind}
    values=r.make_state(np.ones(192,dtype=np.float32),settings).replay(np.r_[np.ones(32),np.zeros(32)].astype(np.float32))
    assert np.isfinite(values).all()


def test_reference_cannot_accept_outcome_settings():
    with pytest.raises(ValueError):r.make_state(np.ones(192),{'version':'next5_reference_features_v1','kind':'J102','tau':4})
