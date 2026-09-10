import numpy as np
from src.features.config import CONFIG
from src.features.rank_gaussian import historical_mapping, map_one, map_array, RankGaussianCalibratedState
from src.features.historical_calibration import CalibrationPolicy, HistoricalCalibratedState
from src.models.rank_calibrated import RankCalibratedBundle
from src.models.learners import fit_candidate


def test_mapping_monotonicity_ties_and_nonfinite_policy():
    for h in (np.ones(100), np.tile([0., 0., 1., 2.], 100), np.array([np.nan, np.inf])):
        knots, scores, transformed = historical_mapping(h)
        assert np.all(np.diff(knots) > 0) and np.all(np.diff(scores) >= 0)
        points = np.r_[np.linspace(-5, 5, 100), np.nan].astype(np.float32)
        batch = map_array(points, knots, scores)
        pointwise = np.array([map_one(float(p), knots, scores) for p in points], dtype=np.float32)
        np.testing.assert_array_equal(batch, pointwise)
        assert np.all(np.diff(batch[:-1]) >= 0) and np.isnan(batch[-1])


def test_rank_streaming_equals_independent_transformed_replay_and_prefix(tmp_path):
    rng = np.random.default_rng(570); h = rng.standard_t(3, 700).astype(np.float32); o = rng.standard_t(3, 150).astype(np.float32); o[80:] *= 2
    config = CONFIG.variant(normalization='median_mad', scales=(5, 20, 160)); policy = CalibrationPolicy(method='historical_median_mad')
    knots, scores, transformed_h = historical_mapping(h); transformed_o = map_array(o, knots, scores)
    reference = HistoricalCalibratedState(transformed_h, config=config, policy=policy)
    stream = RankGaussianCalibratedState(h, config=config, policy=policy); before = stream.state_array_bytes
    expected = np.array([reference.update_and_get(p).copy() for p in transformed_o]); actual = np.array([stream.update_and_get(p).copy() for p in o])
    np.testing.assert_array_equal(actual, expected); assert stream.state_array_bytes == before
    changed = np.r_[o[:40], np.full(10, 500, dtype=np.float32)]; second = RankGaussianCalibratedState(h, config=config, policy=policy)
    alternate = np.array([second.update_and_get(p).copy() for p in changed]); np.testing.assert_array_equal(actual[:40], alternate[:40])
    bundle = RankCalibratedBundle(fit_candidate('M4_lightgbm', actual, np.arange(150) >= 80, config=config), policy, config)
    bundle.save(tmp_path / 'model.joblib'); loaded = RankCalibratedBundle.load(tmp_path / 'model.joblib')
    stream = loaded.make_state(h); live = np.array([loaded.predict_one(stream.update_and_get(p)) for p in o], dtype=np.float32)
    np.testing.assert_array_equal(live, loaded.predict(actual).astype(np.float32))
