import numpy as np
import pytest
from src.next4 import historical_context as c
from src.features.arch_input import ARCHCalibratedState
from src.next.serial_variance_evidence import SerialVarianceState,SerialVarianceConfig


SETTINGS={'version':'historical_context_v1','kind':'base'}


def test_one_hundred_prefix_cases_preserve_all_old_J_features():
    rng=np.random.default_rng(20260908)
    for case in range(100):
        h=rng.standard_t(5,size=192).astype(np.float32);o=rng.normal(size=48).astype(np.float32)
        altered=o.copy();prefix=1+case%32;altered[prefix:]=rng.normal(0,20,len(o)-prefix)
        result=c.make_state(h,SETTINGS).replay(o)
        np.testing.assert_array_equal(result[:prefix],c.make_state(h,SETTINGS).replay(altered)[:prefix])
        state=c.make_state(h,SETTINGS);stream=np.array([state.update(v).copy() for v in o])
        np.testing.assert_array_equal(result,stream)
        old=ARCHCalibratedState(h,config=c.OLD_CONFIG);serial=SerialVarianceState(h,SerialVarianceConfig(order=8,normalization='arch1',max_age=512))
        reference=np.array([np.r_[old.update_and_get(v)[c.OLD_COLUMNS],serial.update(v)[:26]] for v in o],dtype=np.float32)
        np.testing.assert_array_equal(result[:,:77],reference)
        np.testing.assert_array_equal(result[:,77:],np.tile(c.context(h),(len(o),1)))


def test_context_translation_invariance_and_selected_formula_oracles():
    rng=np.random.default_rng(88);h=rng.normal(size=256)
    original=c.context(h);shifted=c.context(h+8)
    np.testing.assert_allclose(original,shifted,atol=2e-6,rtol=0)
    assert original[0]==pytest.approx(np.log(h.std())/6,abs=1e-7)
    assert original[2]==pytest.approx(np.corrcoef(h[1:],h[:-1])[0,1],abs=1e-7)
    assert original[8]==pytest.approx(np.log(len(h))/10,abs=1e-7)


def test_constant_context_and_stream_finite_immutable():
    h=np.ones(128);state=c.make_state(h,SETTINGS);reference=state.args[2].copy()
    result=state.replay(np.zeros(4096))
    assert np.isfinite(result).all();np.testing.assert_array_equal(reference,state.args[2])
    np.testing.assert_array_equal(result[:,77:],np.tile(reference,(4096,1)))


@pytest.mark.parametrize('bad',[np.ones(4),np.full(128,np.nan),np.ones((16,8))])
def test_context_rejects_invalid_history(bad):
    with pytest.raises(ValueError):c.context(bad)


def test_no_identity_or_online_length_in_configuration():
    with pytest.raises(ValueError):c.make_state(np.ones(128),{**SETTINGS,'dataset_id':2})
