import inspect
import json
import numpy as np
from src.features.historical_calibration import HistoricalCalibratedState,transform_cached_features,implementation_hash,CalibrationPolicy
from src.features.streaming import replay

def test_historical_calibration_has_no_suffix_access_and_matches_cache():
    rng=np.random.default_rng(20);h=rng.normal(size=600).astype(np.float32);o=rng.normal(size=80).astype(np.float32)
    raw=np.array(list(replay(h,o)))
    expected=transform_cached_features(raw,h)
    first=HistoricalCalibratedState(h);second=HistoricalCalibratedState(h)
    actual=np.array([first.update_and_get(x).copy() for x in o])
    changed=np.r_[o[:30],np.full(4,100,dtype=np.float32)]
    other=np.array([second.update_and_get(x).copy() for x in changed])
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(actual[:30],other[:30])
    before=first.state_array_bytes
    for _ in range(1000):first.update_and_get(0.)
    assert first.state_array_bytes==before
    assert tuple(inspect.signature(HistoricalCalibratedState.update_and_get).parameters)==('self','point')
    assert implementation_hash()!=implementation_hash(CalibrationPolicy(method='historical_median_mad'))

def test_calibrated_model_live_serialization_and_prefix(tmp_path):
    from src.models.learners import fit_candidate
    from src.models.calibrated import CalibratedBundle
    from src.streaming.inference import infer
    from src.validation.protocol import protocol_predictions
    rng=np.random.default_rng(80);h=rng.normal(size=500).astype(np.float32);o=rng.normal(size=160).astype(np.float32)
    o[80:]*=2
    features=transform_cached_features(np.array(list(replay(h,o))),h)
    target=(np.arange(len(o))>=80).astype('uint8')
    bundle=CalibratedBundle(fit_candidate('M4_lightgbm',features,target));bundle.save(tmp_path/'model.joblib')
    (tmp_path/'model.json').write_text(json.dumps({'kind':'historical_calibrated','name':bundle.name,'feature_hash':bundle.feature_hash}),encoding='utf-8')
    direct=np.asarray(list(infer(iter([(h,iter(o))]),tmp_path))[1:],dtype=np.float32)
    wire=protocol_predictions(infer,[(h,o)],tmp_path)
    np.testing.assert_array_equal(direct,wire)
    np.testing.assert_array_equal(direct,bundle.predict(features).astype(np.float32))
    changed=np.r_[o[:90],np.full(30,100,dtype=np.float32)]
    prefix=np.asarray(list(infer(iter([(h,iter(changed))]),tmp_path))[1:],dtype=np.float32)
    np.testing.assert_array_equal(direct[:90],prefix[:90])
