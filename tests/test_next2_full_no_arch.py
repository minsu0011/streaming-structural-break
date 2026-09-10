import numpy as np
from src.next2.full_no_arch import FullNoARCHState
from src.features.ar_order_input import historical_filter
from src.features.ar_residual_input import innovation_one
from src.features.historical_calibration import HistoricalCalibratedState,CalibrationPolicy
from src.next.fast_variance_initialization import OLD_CONFIG


def test_legacy_branch_is_exact_ar_only_calibrated_features():
    rng=np.random.default_rng(31291)
    h=rng.standard_t(4,640).astype(np.float32)
    o=rng.normal(size=181).astype(np.float32)
    state=FullNoARCHState(h,{'version':'full_pipeline_no_arch_v1'})
    reference,coefficients,ring,residual=historical_filter(h,OLD_CONFIG,8)
    legacy=HistoricalCalibratedState(residual,config=OLD_CONFIG,policy=CalibrationPolicy(method='historical_median_mad'))
    counter=np.zeros(1,dtype=np.int64)
    for p in o:
        value=innovation_one(float(p),reference,coefficients,ring,counter)
        expected=legacy.update_and_get(value)[:51]
        actual=state.update(p)
        np.testing.assert_array_equal(expected,actual[:51])
        assert np.isfinite(actual).all()
    np.testing.assert_array_equal(state.args[3][1][3:5],[0.,0.])


def test_full_arch_off_prefix_exact():
    rng=np.random.default_rng(9411)
    h=rng.normal(size=512).astype(np.float32)
    o=rng.normal(size=151).astype(np.float32)
    settings={'version':'full_pipeline_no_arch_v1'}
    a=FullNoARCHState(h,settings).replay(o)
    b=FullNoARCHState(h,settings).replay(o[:39])
    np.testing.assert_array_equal(a[:39],b)
