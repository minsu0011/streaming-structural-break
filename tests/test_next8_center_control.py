"""Mean-returning independent noise must not establish history information."""
import src.next8
import numpy as np
from src.next8.center_control import fixed_center_delta, loss_comparisons


def test_iid_center_predictor_beats_persistence_without_memory_gain():
    rng = np.random.default_rng(7712)
    current = rng.normal(size=50000)
    future = rng.normal(size=50000)
    delta = future - current
    losses = loss_comparisons(current, delta, fixed_center_delta(current))
    assert losses['PERSISTENCE_TO_H_CENTER_MSE'].mean() > .9
    np.testing.assert_array_equal(losses['H_CENTER_TO_D1_MSE'], np.zeros(len(current)))
    np.testing.assert_array_equal(losses['H_CENTER_TO_D1_MAE'], np.zeros(len(current)))


def test_actual_autoregressive_signal_can_exceed_center_baseline():
    rng = np.random.default_rng(9812)
    current = rng.normal(size=50000)
    future = .8 * current + rng.normal(size=50000) * .3
    losses = loss_comparisons(current, future - current, -.2 * current)
    assert losses['H_CENTER_TO_D1_MSE'].mean() > .6
    assert losses['H_CENTER_TO_D1_MAE'].mean() > .3
