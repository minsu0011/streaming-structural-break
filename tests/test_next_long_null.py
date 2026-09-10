import numpy as np
import pytest
from scripts.stress_next_long_null import predict_complete


class Prepared:
    def __init__(self, grow=False, invalid=False):
        self.grow, self.invalid = grow, invalid
    def new_state(self, historical):
        prepared = self
        class State:
            state_array_bytes = 16
            def predict_one(self, point):
                if prepared.grow:
                    self.state_array_bytes += 8
                return np.nan if prepared.invalid else min(max(float(point), 0.), 1.)
        return State()


def test_complete_scalar_stream_is_finite_and_constant_state():
    values, size = predict_complete(Prepared(), np.ones(10), iter([.1, .2, 1.5]))
    np.testing.assert_array_equal(values, np.array([.1, .2, 1.], dtype=np.float32))
    assert size == 16


def test_growing_or_invalid_detector_is_not_hidden_in_aggregation():
    with pytest.raises(RuntimeError, match='growing'):
        predict_complete(Prepared(grow=True), np.ones(10), [.2])
    with pytest.raises(RuntimeError, match='invalid'):
        predict_complete(Prepared(invalid=True), np.ones(10), [.2])
