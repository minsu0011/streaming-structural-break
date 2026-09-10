import numpy as np
import pytest
from pathlib import Path
from src.next2.shared_blend_runtime import SharedChannels
from src.next2.pruned_runtime import CompactPairState
from src.next2.dependence_runtime import CompactDependenceState
from src.next2.counterexamples import generate
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor, SharedParts
from src.next2.canonical_model import load


@pytest.mark.parametrize('historical_length', [128, 2048])
def test_shared_recurrence_is_bitwise_exact_through_age_boundaries_and_extremes(historical_length):
    rng = np.random.default_rng(72031+historical_length)
    historical = rng.standard_t(5, historical_length).astype(np.float32)
    points = rng.standard_t(3, 2048).astype(np.float32)
    points[[0, 127, 511, 512, 1023]] = [20., -20., 100., -100., 0.]
    left = CompactPairState(historical, {'order': 8, 'normalization': 'arch1', 'max_age': 512}, 'serial')
    right = CompactDependenceState(historical)
    shared = SharedChannels(historical)
    np.testing.assert_array_equal(left.parameters, right.args[1])
    np.testing.assert_array_equal(left.conditional, right.args[2])
    for point in points:
        a, b = shared.update(point)
        np.testing.assert_array_equal(a.view(np.uint32), left.update(point).view(np.uint32))
        np.testing.assert_array_equal(b.view(np.uint32), right.update(point).view(np.uint32))
        np.testing.assert_array_equal(shared.call_args[2], left.conditional)
        np.testing.assert_array_equal(shared.call_args[2], right.args[2])


@pytest.mark.parametrize('background', ['N0', 'N1', 'N2', 'N3'])
def test_shared_channels_match_over_locked_long_null_backgrounds(background):
    h, online, tau = generate(background, 7, partition=2, horizon=16384)
    assert tau == -1
    left = CompactPairState(h, {'order': 8, 'normalization': 'arch1', 'max_age': 512}, 'serial')
    right = CompactDependenceState(h)
    shared = SharedChannels(h)
    for point in online:
        a, b = shared.update(point)
        np.testing.assert_array_equal(a.view(np.uint32), left.update(point).view(np.uint32))
        np.testing.assert_array_equal(b.view(np.uint32), right.update(point).view(np.uint32))


def test_shared_fused_and_separate_feature_tree_boundaries_match_saved_model():
    root = Path(__file__).resolve().parents[1]
    model = load(root/'artifacts/next2/canonical_deployment/PERFORMANCE_compact_v1/model.sbnext2')
    h, online, _ = generate('N3', 9, partition=2, horizon=1024)
    prepared = PreparedSharedBlendPredictor(model)
    fused = prepared.new_state(h)
    parts = SharedParts(prepared.new_state(h))
    for point in online:
        expected = fused.predict_one(point)
        parts.feature_update(point)
        assert parts.model_predict() == expected
        assert parts.model_predict() == expected  # prediction itself does not mutate state
