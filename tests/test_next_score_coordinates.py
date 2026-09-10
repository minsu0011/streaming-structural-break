import numpy as np
from src.next.score_coordinates import ScoreCoordinateState, feature_names
from src.next.score_evidence import ARScoreState, SCORE_NAMES


def test_original_aggregate_features_are_bitwise_unchanged():
    rng = np.random.default_rng(8657)
    h = rng.normal(size=1300).astype(np.float32)
    points = rng.standard_t(5, 700).astype(np.float32)
    original = ARScoreState(h).replay(points)
    extended = ScoreCoordinateState(h).replay(points)
    np.testing.assert_array_equal(extended[:, :len(SCORE_NAMES)], original)
    coords = extended[:, len(SCORE_NAMES):].reshape(len(points), 3, 8)
    for k, half in enumerate((8,32,128)):
        np.testing.assert_allclose(np.mean(coords[:, k].astype(float)**2, axis=1), original[:, SCORE_NAMES.index(f'score_ewma_q_{half}')], rtol=2e-7, atol=1e-6)


def test_score_coordinate_prefix_and_constant_state_size():
    rng = np.random.default_rng(934)
    h = rng.normal(size=1400).astype(np.float32)
    points = rng.normal(size=1500).astype(np.float32)
    state = ScoreCoordinateState(h)
    size = state.state_array_bytes
    first = np.array([state.update(x).copy() for x in points[:810]])
    all_points = ScoreCoordinateState(h).replay(points)
    np.testing.assert_array_equal(first, all_points[:810])
    assert size == state.state_array_bytes
    assert len(feature_names({})) == 43
