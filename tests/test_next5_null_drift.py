"""Nontrivial arithmetic and actual frozen-state checks for null diagnostics."""
import numpy as np
import pytest
from src.next5 import mixture, scale
from src.next5.null_drift import energy_boundary, cumulative_from_energy, evaluate


def test_variance_boundary_changes_lr_sign_in_the_correct_direction():
    boundary = energy_boundary(scale.VARIANCES)
    for j, variance in enumerate(scale.VARIANCES):
        at_boundary = cumulative_from_energy(np.repeat(boundary[j] * 1024, 2), 1024, scale.VARIANCES)
        assert abs(at_boundary[0, j]) < 1e-12
        below = cumulative_from_energy(np.repeat(boundary[j] * .9 * 1024, 2), 1024, scale.VARIANCES)[0, j]
        above = cumulative_from_energy(np.repeat(boundary[j] * 1.1 * 1024, 2), 1024, scale.VARIANCES)[0, j]
        assert (below > 0) == (variance < 1)
        assert (above > 0) == (variance > 1)
    with pytest.raises(ValueError):
        energy_boundary([1.])


def test_snapshot_features_match_original_replay_across_ring_wraps():
    rng = np.random.default_rng(20260908)
    historical = rng.standard_t(5, 256).astype(np.float32)
    online = rng.normal(size=1024).astype(np.float32)
    horizons = [1, 256, 257, 512, 1024]
    values, error = evaluate(historical, online, horizons)
    for kind, actual in [('cumulative', values[2]), ('sparse_starts', values[3])]:
        replay = mixture.make_state(historical, {'version': 'next5_fixed_mixture_v1', 'kind': kind}).replay(online)
        np.testing.assert_array_equal(actual, replay[np.asarray(horizons) - 1])
    assert error < 1e-7
    with pytest.raises(ValueError):
        evaluate(historical, online, [1024, 512])
