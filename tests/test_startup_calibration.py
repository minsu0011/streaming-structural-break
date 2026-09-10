import numpy as np
from src.features.config import CONFIG
from src.features.streaming import replay
from src.features.historical_calibration import CalibrationPolicy
from src.features.startup_calibration import StartupCalibratedState,transform_cached_features,correct_startup

def test_startup_correction_exact_prefix_and_live_cache_identity():
    rng=np.random.default_rng(101);h=rng.normal(size=800).astype(np.float32);o=rng.normal(size=140).astype(np.float32)
    policy=CalibrationPolicy(method='historical_median_mad_startup')
    raw=np.array(list(replay(h,o)))
    expected=transform_cached_features(raw,h,policy=policy)
    state=StartupCalibratedState(h,config=CONFIG,policy=policy)
    actual=np.array([state.update_and_get(x).copy() for x in o])
    np.testing.assert_array_equal(actual,expected)
    alternate=np.r_[o[:40],np.full(17,20,dtype=np.float32)]
    second=StartupCalibratedState(h,config=CONFIG,policy=policy)
    other=np.array([second.update_and_get(x).copy() for x in alternate])
    np.testing.assert_array_equal(actual[:40],other[:40])
    np.testing.assert_array_equal(expected[:40],transform_cached_features(raw[:40],h,policy=policy))
    late=np.zeros(57,dtype=np.float32);correct_startup(raw[-1],np.asarray(CONFIG.alphas),100000.,late)
    np.testing.assert_array_equal(late,raw[-1])
