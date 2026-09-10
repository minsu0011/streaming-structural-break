import numpy as np
from src.next.engine import EngineConfig,FEATURE_NAMES
from src.next.calibrated_engine import CalibratedSequentialState


def test_h_only_calibration_stream_batch_exact_and_prefix_safe():
    rng=np.random.default_rng(942)
    h=rng.standard_t(4,1500).astype(np.float32)
    points=rng.normal(size=250).astype(np.float32)
    a=CalibratedSequentialState(h)
    before=a.location.copy(),a.scale.copy()
    stream=np.array([a.update(point).copy() for point in points[:120]])
    batch=CalibratedSequentialState(h).replay(points)
    np.testing.assert_array_equal(stream,batch[:120])
    np.testing.assert_array_equal(before[0],a.location)
    np.testing.assert_array_equal(before[1],a.scale)
    assert np.isfinite(batch).all()
    assert np.max(np.abs(batch[:,a.apply_mask]))<=12


def test_nonstationary_time_and_memory_are_not_h_null_calibrated():
    rng=np.random.default_rng(316)
    h=rng.normal(size=1000).astype(np.float32)
    state=CalibratedSequentialState(h,EngineConfig(normalization='gaussian'))
    for name in ['elapsed_log1p','glr_mean_argmax_log_age','evidence_cumulative_max']:
        assert not state.apply_mask[FEATURE_NAMES.index(name)]
