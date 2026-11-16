import copy
import pickle
import numpy as np
import pytest
from src.next7.null_growth import (GRID, COLUMNS, joint_ratio, bootstrap_maxima,
    historical_admission, new_admission_args, admission_step, admission_replay,
    QualifiedState, qualified_features, shared_uniforms)
from src.next7.event import CombinedState
from src.next7.features import from_state
from src.next7.scale import confirmation


@pytest.fixture(scope='module')
def sample():
    rng = np.random.default_rng(12020)
    return rng.normal(size=2048), np.r_[rng.normal(size=96), rng.normal(scale=.1, size=320), rng.normal(size=320)]


def args(cuts=None, valid=True):
    return new_admission_args({'admission_cuts': [1.] * 9 if cuts is None else cuts,
        'reference_valid': valid, 'unsupported_cut': 8.})


def fake_raw(t, ratio, confirmed=True):
    raw = np.zeros(30)
    raw[2] = ratio
    raw[15] = 1. if (t + 1) % 16 == 0 else 0.
    raw[16] = (t + 1) // 16
    raw[18] = 1. if confirmed and raw[15] and raw[16] >= 4 else 0.
    raw[8] = -1.
    raw[22:25] = [8, raw[16] - 4, 3]
    return raw


def test_score_matches_independent_rules():
    rng = np.random.default_rng(433)
    for _ in range(200):
        z = rng.uniform(-4, 4, (4, 9))
        if _ % 3 == 0:
            z[:, :3] = np.abs(z[:, :3])
        core = max([0.] + [min(np.abs(z[:, j])) for j in range(8)
            if np.all(z[:, j] > 0) or np.all(z[:, j] < 0)])
        scale = confirmation(np.column_stack((z[:, 8], z[:, 8])), 4, np.zeros(18))[3]
        assert joint_ratio(z, 2., 1.7) == pytest.approx(max(core / 2., scale / 1.7), abs=1e-15)


def test_bootstrap_matches_explicit_sampled_paths():
    rng = np.random.default_rng(44)
    z = rng.uniform(-4, 4, (14, 9))
    u = rng.random((3, 4))
    grid = np.array([4, 8, 16])
    actual = bootstrap_maxima(z, u, grid, 2., 1.7)
    for r in range(3):
        starts = (u[r] * (len(z) - 3)).astype(int)
        sampled = np.concatenate([z[s:s + 4] for s in starts])
        scores = [joint_ratio(sampled[t - 3:t + 1], 2., 1.7) for t in range(3, 16)]
        expected = [max(scores[:b - 3]) for b in grid]
        np.testing.assert_allclose(actual[r], expected, rtol=0, atol=1e-15)


def test_shared_draws_and_H_determinism(sample):
    h, _ = sample
    np.testing.assert_array_equal(shared_uniforms(), shared_uniforms())
    a = historical_admission(h, core_cut=2., scale_cut=2.)
    b = historical_admission(h.copy(), core_cut=2., scale_cut=2.)
    assert a == b
    assert a['reference_valid']
    assert np.all(np.diff(a['admission_cuts']) >= 0)
    assert min(a['admission_cuts']) >= 1.


def test_short_and_constant_H_abstain():
    rng = np.random.default_rng(4)
    for h in (rng.normal(size=512), np.ones(2048)):
        state = QualifiedState(h, core_cut=2., scale_cut=2.)
        out = state.replay(rng.normal(scale=.01, size=256))
        assert not state.calibration['reference_valid']
        assert np.all(out[:, 30] == 0.)
        assert np.all(out[:, 43] == 0.)
        assert np.isfinite(out).all()


def test_scalar_batch_chunk_pickle_and_current_parity(sample):
    h, o = sample
    batch = QualifiedState(h, core_cut=2., scale_cut=2.)
    mutable_bytes = batch.bounded_mutable_state_bytes
    expected = batch.replay(o)
    scalar = QualifiedState(h, core_cut=2., scale_cut=2.)
    actual = np.array([scalar.predict_one(x) for x in o])
    np.testing.assert_array_equal(actual, expected)
    raw = CombinedState(h, core_cut=2., scale_cut=2.).replay(o)
    np.testing.assert_array_equal(expected[:, :30], raw)
    chunked = QualifiedState(h, core_cut=2., scale_cut=2.)
    prefix = chunked.replay(o[:149])
    restored = pickle.loads(pickle.dumps(chunked))
    np.testing.assert_array_equal(np.vstack((prefix, restored.replay(o[149:]))), expected)
    assert batch.bounded_mutable_state_bytes == mutable_bytes
    assert np.all(np.diff(expected[:, 30]) >= 0)
    assert np.all(expected[expected[:, 40] > 0, 18] > 0)


def test_online_prefix_future_suffix_and_length_cannot_change_output(sample):
    h, o = sample
    state = QualifiedState(h, core_cut=2., scale_cut=2.)
    a, b = copy.deepcopy(state), copy.deepcopy(state)
    x = a.replay(o)
    y = b.replay(np.r_[o[:301], np.full(500, 20.)])
    np.testing.assert_array_equal(x[:301], y[:301])
    np.testing.assert_array_equal(state.replay(o[:301]), x[:301])


def test_shock_or_raw_confirmation_alone_cannot_qualify_and_strict_cut():
    state = args([2.] * 9)
    for t in range(128):
        raw = fake_raw(t, 2.)
        raw[14] = 1.
        out = admission_step(raw, state)
        assert out[30] == 0.
    out = admission_step(fake_raw(128, 3.), state)
    assert out[30] == 0.  # Incomplete block.
    for t in range(129, 144):
        out = admission_step(fake_raw(t, 3., confirmed=False), state)
    assert out[30] == 0.


def test_grid_routes_observed_block_count_and_stops_new_late_events():
    state = args(np.arange(1., 10.).tolist())
    for b in [0, 1, 4, 5, 8, 9, 1024, 1025, 100000]:
        raw = fake_raw(15, 20.)
        raw[16] = b
        out = admission_step(raw, state).copy()
        expected = np.arange(1., 10.)[np.searchsorted(GRID, b)] if b <= 1024 else 8.
        assert out[42] == expected
        assert out[43] == (b <= 1024)
        if b > 1024:
            assert out[40] == 0.
    fresh = args()
    raw[16] = 1025
    assert admission_step(raw, fresh)[30] == 0.


def test_qualified_episodes_age_and_retention_after_current_recovery():
    state = args()
    rows = []
    for t in range(208):
        ratio = 2. if 63 <= t <= 95 or 159 <= t <= 175 else .2
        raw = fake_raw(t, ratio, confirmed=ratio > 1.)
        rows.append(admission_step(raw, state).copy())
    out = np.array(rows)
    assert out[62, 30] == 0.
    assert out[-1, 31] == 2.
    assert out[-1, 35] == 63.
    assert out[-1, 36] == 175.
    assert out[-1, 32] == 207 - 63
    assert out[-1, 33] == 207 - 175
    assert out[-1, 30] == out[63, 30] > 0.
    assert out[-1, 41] == 0.
    assert out[-1, 34] == -1.


def test_features_only_replace_qualified_memory(sample):
    h, o = sample
    v = QualifiedState(h, core_cut=2., scale_cut=2.).replay(o)
    raw = from_state(v[:, :30])
    new = qualified_features(v)
    np.testing.assert_array_equal(new[:, [0, 5, 7, 8, 9]], raw[:, [0, 5, 7, 8, 9]])
    np.testing.assert_array_equal(new[:, 1], v[:, 30])
    np.testing.assert_array_equal(qualified_features(v[120]), new[120])


def test_invalid_online_input_does_not_mutate_state(sample):
    h, _ = sample
    state = QualifiedState(h, core_cut=2., scale_cut=2.)
    before = pickle.dumps(state)
    for values in ([1., np.nan], [1., np.inf], [1., 1e100]):
        with pytest.raises(ValueError):
            state.replay(values)
        assert pickle.dumps(state) == before
    with pytest.raises(ValueError):
        state.predict_one(float('nan'))
    assert pickle.dumps(state) == before
