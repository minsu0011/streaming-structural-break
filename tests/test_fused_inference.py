import numpy as np
import pytest
from src.models.learners import fit_candidate
from src.models.calibrated import CalibratedBundle
from src.features.streaming import replay,StreamingFeatureState
from src.features.historical_calibration import transform_cached_features,CalibrationPolicy
from src.streaming.fused import PreparedFusedPredictor

@pytest.mark.parametrize('model',['M3_histgb','M4_lightgbm','M5_xgboost'])
@pytest.mark.parametrize('calibrated',[None,'historical_median_mad','historical_median_mad_startup'])
def test_fused_prefix_and_float32_prediction_identity(model,calibrated):
    rng=np.random.default_rng(95);h=rng.normal(size=600).astype(np.float32);o=rng.normal(size=180).astype(np.float32);o[90:]*=2
    raw=np.array(list(replay(h,o)));policy=CalibrationPolicy(method=calibrated or 'historical_median_mad')
    if calibrated and calibrated.endswith('_startup'):
        from src.features.startup_calibration import transform_cached_features as transform
        features=transform(raw,h,policy=policy)
    else:features=transform_cached_features(raw,h,policy=policy) if calibrated else raw
    bundle=fit_candidate(model,features,(np.arange(len(o))>=90).astype('uint8'))
    if calibrated:bundle=CalibratedBundle(bundle,policy)
    prepared=PreparedFusedPredictor(bundle);detector=prepared.new_state(h)
    actual=np.array([detector.predict_one(x) for x in o],dtype=np.float32)
    expected=bundle.predict(features).astype(np.float32)
    np.testing.assert_array_equal(actual,expected)
    before=detector.state_array_bytes
    for _ in range(100):detector.predict_one(0.)
    assert detector.state_array_bytes==before
    changed=np.r_[o[:40],np.full(7,100,dtype=np.float32)];second=prepared.new_state(h)
    alternative=np.array([second.predict_one(x) for x in changed],dtype=np.float32)
    np.testing.assert_array_equal(actual[:40],alternative[:40])
