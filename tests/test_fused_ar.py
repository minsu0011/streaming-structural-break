import numpy as np
import pytest
from src.features.config import CONFIG
from src.features.historical_calibration import CalibrationPolicy
from src.features.ar_residual_input import ARResidualCalibratedState
from src.models.ar_calibrated import ARCalibratedBundle
from src.models.learners import fit_candidate
from src.streaming.fused_ar import PreparedFusedARPredictor


@pytest.mark.parametrize('model',['M3_histgb','M4_lightgbm','M5_xgboost'])
@pytest.mark.parametrize('method',['historical_mean_std','historical_median_mad'])
@pytest.mark.parametrize('fast',[False,True])
def test_fused_ar_matches_cached_features_and_preserves_prefix(model,method,fast):
    rng=np.random.default_rng(901);h=rng.standard_t(4,800).astype(np.float32);o=rng.normal(size=160).astype(np.float32);o[80:]*=2
    for t in range(4,len(h)):h[t]+=.3*h[t-1]-.2*h[t-4]
    config=CONFIG.variant(normalization='median_mad',scales=(5,20,160));policy=CalibrationPolicy(method=method)
    state=ARResidualCalibratedState(h,config=config,policy=policy);features=np.array([state.update_and_get(p).copy() for p in o])
    bundle=ARCalibratedBundle(fit_candidate(model,features,np.arange(len(o))>=80,config=config),policy,config)
    prepared=PreparedFusedARPredictor(bundle,fast_initialization=fast);detector=prepared.new_state(h);before=detector.state_array_bytes
    actual=np.array([detector.predict_one(p) for p in o],dtype=np.float32);np.testing.assert_array_equal(actual,bundle.predict(features).astype(np.float32))
    assert detector.state_array_bytes==before
    second=prepared.new_state(h);changed=np.r_[o[:40],np.full(7,1e5,dtype=np.float32)]
    alternate=np.array([second.predict_one(p) for p in changed],dtype=np.float32);np.testing.assert_array_equal(actual[:40],alternate[:40])
