import numpy as np
import pytest
from src.features.streaming import StreamingFeatureState, replay
from src.features.config import CONFIG
from src.features.historical_calibration import CalibrationPolicy, historical_reference, transform_cached_features
from src.models.learners import fit_candidate
from src.models.calibrated import CalibratedBundle
from src.streaming.fast_initialization import shared_mad_reference, PreparedFastInitFusedPredictor
from src.streaming.fused import PreparedFusedPredictor


@pytest.mark.parametrize('historical', [[], [1.], [2.] * 320, [np.nan, np.inf, 0., -1.] * 100])
def test_shared_reference_degenerate_exact_and_state_unchanged(historical):
    policy = CalibrationPolicy(method='historical_median_mad'); state = StreamingFeatureState(historical)
    arrays = {k: v.copy() for k, v in vars(state).items() if isinstance(v, np.ndarray)}
    expected = historical_reference(historical, policy=policy); actual = shared_mad_reference(historical, state, policy)
    for a, b in zip(actual, expected): np.testing.assert_array_equal(a, b)
    for name, original in arrays.items(): np.testing.assert_array_equal(getattr(state, name), original)


@pytest.mark.parametrize('startup', [False, True])
@pytest.mark.parametrize('families',['ABCD','B','EF'])
def test_shared_initialization_realistic_prediction_identity(startup,families):
    rng = np.random.default_rng(359); h = rng.standard_t(3, 1200).astype(np.float32); o = rng.normal(size=150).astype(np.float32); o[70:] += .5
    config = CONFIG.variant(normalization='median_mad', scales=(5, 20, 160))
    policy = CalibrationPolicy(method='historical_median_mad' + ('_startup' if startup else ''))
    features = np.array(list(replay(h, o, config=config)))
    if startup:
        from src.features.startup_calibration import transform_cached_features as transform
    else: transform = transform_cached_features
    features = transform(features, h, config=config, policy=policy)
    bundle = CalibratedBundle(fit_candidate('M3_histgb', features, np.arange(len(o)) >= 70, config=config, families=families), policy, config)
    original = PreparedFusedPredictor(bundle).new_state(h); optimized = PreparedFastInitFusedPredictor(bundle).new_state(h)
    for a, b in zip(original.state_args, optimized.state_args): np.testing.assert_array_equal(a, b)
    needed=np.zeros(len(features[0]),dtype=bool);needed[bundle.base_model.columns]=True
    active=optimized.calibration_args[2]
    assert np.all(active<=original.calibration_args[2]) and np.all(active<=needed)
    if families=='EF':assert not active.any()
    for a,b in zip(original.calibration_args[:2],optimized.calibration_args[:2]):np.testing.assert_array_equal(a[active],b[active])
    for a,b in zip(original.calibration_args[3:],optimized.calibration_args[3:]):np.testing.assert_array_equal(a,b)
    expected = np.array([original.predict_one(p) for p in o], dtype=np.float32)
    actual = np.array([optimized.predict_one(p) for p in o], dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)
