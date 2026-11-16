import numpy as np
import pytest
from src.next3.scale_calibration import KINDS,fit_transform,transform_point,make_state
from src.next.serial_variance_evidence import SerialVarianceState,SerialVarianceConfig


@pytest.mark.parametrize('kind',KINDS)
def test_h_only_calibration_is_centered_and_live_mapping_exact(kind):
    rng=np.random.default_rng(299)
    raw=np.column_stack([rng.standard_t(5,512)**2-1.,rng.normal(size=512)])
    parameters,x=fit_transform(raw,kind)
    np.testing.assert_allclose(x.mean(axis=0),0.,rtol=0,atol=1e-14)
    np.testing.assert_allclose(x.std(axis=0),1.,rtol=0,atol=1e-14)
    actual=np.asarray([transform_point(row,parameters,KINDS.index(kind),np.empty(2)) for row in raw])
    np.testing.assert_array_equal(actual,x)
    fixed=parameters.copy();transform_point(np.array([1e20,-1e20]),parameters,KINDS.index(kind),np.empty(2))
    np.testing.assert_array_equal(fixed,parameters)


@pytest.mark.parametrize('kind',KINDS)
def test_original_variance_inputs_exact_and_future_suffix_excluded(kind):
    rng=np.random.default_rng(355);h=rng.normal(size=512).astype(np.float32);o=rng.normal(size=300).astype(np.float32)
    settings={'kind':kind,'version':'scale_calibration_append_v1'}
    original=SerialVarianceState(h,SerialVarianceConfig(max_age=512)).replay(o)[:,:26]
    actual=make_state(h,settings).replay(o)
    np.testing.assert_array_equal(actual[:,:26],original)
    state=make_state(h,settings)
    from src.next.fused_runtime import unique_array_bytes
    before=unique_array_bytes(state.args)
    live=np.asarray([state.update(v).copy() for v in o]);np.testing.assert_array_equal(live,actual)
    assert unique_array_bytes(state.args)==before
    changed=o.copy();changed[127:]+=100.
    np.testing.assert_array_equal(make_state(h,settings).replay(changed)[:127],actual[:127])


@pytest.mark.parametrize('kind',KINDS)
def test_constant_h_channel_has_no_spurious_calibrated_drift(kind):
    p,x=fit_transform(np.zeros((64,2)),kind)
    np.testing.assert_array_equal(x,np.zeros_like(x))
    assert np.isfinite(transform_point(np.zeros(2),p,KINDS.index(kind),np.empty(2))).all()
