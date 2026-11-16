import copy
import pickle
import numpy as np
import pytest
from src.next9.states import MovementState, describe_history, DOWN, FLAT, UP, UNKNOWN
from src.next9.boundary import target_free_npz
from src.next9.work_time import aggregate


def test_state_uses_only_t_minus_1_or_earlier():
    seed = np.arange(16, dtype=np.float32)
    state = MovementState(seed, .02)
    assert state.peek()['S1_DIRECTION'] == UP
    state.observe(-99)
    assert state.peek()['S1_DIRECTION'] == DOWN


def test_current_F_not_used_for_pre_state():
    a = MovementState(np.arange(16), .02)
    b = a.snapshot()
    pre_a, pre_b = a.peek(), b.peek()
    a.observe(100)
    b.observe(-100)
    for key in pre_a:
        np.testing.assert_array_equal(pre_a[key], pre_b[key])
    assert a.peek()['S1_DIRECTION'] != b.peek()['S1_DIRECTION']


def test_future_suffix_invariance():
    rng = np.random.default_rng(9101604)
    values = rng.normal(size=100)
    changed = values.copy(); changed[37:] *= -7
    a = MovementState(np.zeros(16), .02).replay(values)
    b = MovementState(np.zeros(16), .02).replay(changed)
    for x, y in zip(a[:38], b[:38]):
        for key in x:
            np.testing.assert_array_equal(x[key], y[key])


def test_prefix_invariance():
    values = np.sin(np.arange(100))
    a = MovementState(np.zeros(16), .02).replay(values)
    b = MovementState(np.zeros(16), .02).replay(values[:41])
    for x, y in zip(a, b):
        for key in x:
            np.testing.assert_array_equal(x[key], y[key])


def test_save_restore_state():
    state = MovementState(np.arange(16), .02)
    state.replay(np.arange(13))
    restored = MovementState.restore(pickle.loads(pickle.dumps(state)))
    for x, y in zip(state.replay(np.arange(25)), restored.replay(np.arange(25))):
        for key in x:
            np.testing.assert_array_equal(x[key], y[key])


def test_reset_between_series():
    seed = np.zeros(16)
    first = MovementState(seed, .02)
    first.replay(np.arange(100))
    second = MovementState(seed, .02)
    assert second.peek()['S1_DIRECTION'] == FLAT
    np.testing.assert_array_equal(second.history, seed)


def test_reverse_order():
    inputs = [np.arange(20), -np.arange(20)]
    forward = [MovementState(np.zeros(16), .02).replay(v)[-1] for v in inputs]
    reverse = [MovementState(np.zeros(16), .02).replay(v)[-1] for v in inputs[::-1]][::-1]
    for a, b in zip(forward, reverse):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_run_length_causal():
    h = np.array([0] * 12 + [1, 2, 3, 4], dtype=np.float32)
    s = MovementState(h, .02).peek()
    assert s['direction_run_capped15'] == 4
    assert s['S2_DIRECTION_RUN'] == UP * 3 + 2


def test_reversal_age_causal():
    h = np.arange(16, dtype=np.float32)
    state = MovementState(h, .02)
    assert state.peek()['S4_REVERSAL'] == 0
    state.observe(14)
    assert state.peek()['S4_REVERSAL'] == 2
    assert state.peek()['reversal_age_capped15'] == 0
    state.observe(13)
    assert state.peek()['reversal_age_capped15'] == 1


def test_H_flat_threshold_training_only():
    state = MovementState(np.linspace(0, .015, 16), .02)
    assert state.peek()['S1_DIRECTION'] == FLAT
    state.replay([100, -100])
    assert state.epsilon_H == .02
    assert MovementState(np.zeros(16), .02, False).peek()['S1_DIRECTION'] == UNKNOWN


def test_serialized_float32_H_floor_preserves_original_threshold():
    epsilon = float(np.float32(.02))
    assert epsilon < .02
    state = MovementState(np.zeros(16), epsilon)
    state.observe(np.float32(.02))
    assert state.peek()['S1_DIRECTION'] == FLAT
    assert state.epsilon_H == epsilon
    with pytest.raises(ValueError):
        MovementState(np.zeros(16), .019)


def test_stream_offline_parity():
    rng = np.random.default_rng(9)
    seed = rng.normal(size=16).astype(np.float32)
    values = rng.normal(size=60).astype(np.float32)
    joined = np.r_[seed, values]
    histories = np.stack([joined[t:t+16] for t in range(len(values))])
    batch = describe_history(histories, .02)
    stream = MovementState(seed, .02).replay(values)
    for t, one in enumerate(stream):
        for key in one:
            np.testing.assert_array_equal(batch[key][t], one[key])


def test_candidate_movement_screen_no_break_label(tmp_path):
    path = tmp_path / 'toy.npz'
    np.savez(path, dataset_id=[1], target=[1], tau=[3])
    with target_free_npz() as read:
        with np.load(path) as z:
            np.testing.assert_array_equal(z['dataset_id'], [1])
            for key in ('target', 'tau'):
                with pytest.raises(PermissionError):
                    z[key]
        assert read == ['dataset_id']


def test_time_clock_excludes_wait_failed_and_double_count():
    def row(i, a, b, kind='RESEARCH', status='COMPLETE', cpu=1):
        return dict(id=i, started_epoch=a, ended_epoch=b, kind=kind, status=status, cpu_seconds=cpu)
    result = aggregate([row('a', 0, 10), row('b', 5, 15), row('c', 0, 100, kind='WAIT'),
                        row('d', 0, 100, status='FAILED'), row('e', 15, 100, cpu=0)])
    assert result['productive_seconds'] == 15
    assert result['audit_seconds'] == 0
