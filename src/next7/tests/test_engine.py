import copy
import inspect
import pickle
import numpy as np
import pytest
from src.next7.scale import ScaleState, confirmation, log_block_mad
from src.next7.event import CombinedState, new_event_args, event_replay


def historical(n=2048):
    return np.random.default_rng(20260913).normal(size=n).astype(np.float32)


def state(h=None, **kwargs):
    return CombinedState(historical() if h is None else h, core_cut=1.9975371345973232,
        scale_cut=kwargs.pop('scale_cut', 1.5), **kwargs)


def test_fresh_block_nonoverlap():
    h = historical()
    stream = np.arange(96, dtype=np.float32) * .1
    altered = stream.copy()
    altered[7] = 100000.
    a = ScaleState(h).replay(stream)[15::16, 2:4]
    b = ScaleState(h).replay(altered)[15::16, 2:4]
    assert np.any(a[0] != b[0])
    np.testing.assert_array_equal(a[1:], b[1:])


def test_single_shock_not_reused_as_distinct_blocks():
    memory = np.zeros((4, 2))
    memory[0] = 4.
    out = np.zeros(18)
    confirmation(memory, 4, out)
    assert np.all(out[::3] == 0)
    # Repeated evaluation of the same memory is not another fresh block.
    for _ in range(20):
        confirmation(memory, 4, out)
        assert np.all(out[::3] == 0)


def event_fixture(confirm=False):
    core = np.zeros((128, 20))
    scale = np.zeros((128, 22))
    for t in range(128):
        core[t, 0] = 1
        core[t, 3] = t == 20
        scale[t, 0] = (t+1) % 16 == 0
        scale[t, 1] = (t+1) // 16
        if confirm and t == 63:
            scale[t, 7:10] = [2., -1., 3.]
    return core, scale


def test_event_memory_only_after_confirm():
    core, scale = event_fixture(False)
    out = event_replay(core, scale, new_event_args(2., 1., 0, 1, True, True, True))
    assert out[20, 14] == 1
    assert np.all(out[:, [3, 4, 18, 26, 27]] == 0)
    assert np.all(out[:, 21] == -1)


def test_recovery_does_not_clear_event():
    core, scale = event_fixture(True)
    out = event_replay(core, scale, new_event_args(2., 1., 0, 1, True, True, True))
    assert np.all(out[:63, 3] == 0)
    assert out[63, 3] > 0
    assert out[63, 0] == 4
    assert out[79, 0] == 5
    assert out[-1, 0] == 1
    assert np.all(out[63:, 3] == out[63, 3])
    assert out[-1, 21] == 63
    assert out[-1, 4] == 1


def test_reset_between_series():
    stream = historical(1024) * .4
    first = state().replay(stream)
    state().replay(historical(1024) * 5.)
    np.testing.assert_array_equal(first, state().replay(stream))


@pytest.mark.parametrize('prefix', [1, 15, 16, 47, 63, 64, 129, 257])
def test_prefix_invariance(prefix):
    stream = historical(1024)[::-1].copy()
    whole = state().replay(stream)
    np.testing.assert_array_equal(whole[:prefix], state().replay(stream[:prefix]))


def test_future_suffix_invariance():
    stream = historical(1024)
    changed = stream.copy()
    changed[199:] = np.finfo(np.float32).max
    np.testing.assert_array_equal(state().replay(stream)[:199], state().replay(changed)[:199])


def test_future_length_absent():
    assert 'length' not in inspect.signature(CombinedState.__init__).parameters
    s = state()
    with pytest.raises(TypeError):
        s.predict_one(1., future_total_length=10000)


def test_tau_absent():
    for function in (CombinedState.__init__, CombinedState.predict_one, CombinedState.replay,
                     ScaleState.__init__, ScaleState.predict_one):
        assert 'tau' not in inspect.signature(function).parameters
    with pytest.raises(TypeError):
        state().predict_one(1., tau=20)


def test_save_load():
    stream = historical(1024)[::-1].copy()
    s = state()
    s.replay(stream[:301])
    restored = pickle.loads(pickle.dumps(s))
    np.testing.assert_array_equal(s.replay(stream[301:]), restored.replay(stream[301:]))


def test_determinism_scalar_batch_chunk_and_interleave():
    stream = historical(1024)[::-1].copy()
    expected = state().replay(stream)
    scalar = state()
    interloper = state(historical(1500))
    rows = []
    for x in stream:
        rows.append(scalar.predict_one(x))
        interloper.predict_one(-x * 3)
    np.testing.assert_array_equal(expected, np.array(rows))
    chunked = state()
    parts = [chunked.replay(stream[a:b]) for a,b in ((0,3),(3,34),(34,515),(515,1024))]
    np.testing.assert_array_equal(expected, np.concatenate(parts))


def test_scale_sign_symmetry():
    s = ScaleState(historical())
    center, spread, eps, _ = s.args[2]
    pattern = np.arange(16, dtype=np.float64) - 7.5
    pattern *= np.exp(center) / np.median(np.abs(pattern))
    z = []
    for log_change in (-.4, .4):
        output = ScaleState(historical()).replay((pattern * np.exp(log_change)).astype(np.float32))
        z.append(output[-1,2])
    assert z[0] < 0 < z[1]
    assert abs(z[0]+z[1]) < 2e-6
    assert abs(z[1] - .4/spread) < 2e-6


def test_translation_invariance_of_scale():
    block = np.arange(16, dtype=np.float64)
    assert log_block_mad(block,1e-20) == log_block_mad(block+100000,1e-20)


def test_H_only_calibration():
    s = ScaleState(historical())
    reference = [a.copy() for a in s.args[:3]]
    cal = copy.deepcopy(s.calibration)
    s.replay(historical(1024)*100.)
    for before, after in zip(reference,s.args[:3]):
        np.testing.assert_array_equal(before,after)
    assert cal == s.calibration


def test_short_H_fallback():
    s = ScaleState(historical(512))
    assert s.calibration['deterministic_fit_only_fallback']
    assert s.calibration['reference_valid']
    np.testing.assert_array_equal(s.args[0],s.args[1])
    assert np.isfinite(s.replay(historical(512))).all()
    with pytest.raises(ValueError):
        ScaleState(historical(511))


def test_long_null_state_bounded():
    s = state()
    before = s.bounded_mutable_state_bytes
    out = s.replay(historical(16384))
    assert s.bounded_mutable_state_bytes == before
    assert out.shape == (16384,30)
    assert np.isfinite(out).all()
    assert np.all((out[:,3] >= 0) & (out[:,3] <= 1))
    assert np.all(np.diff(out[:,3]) >= 0)
    assert out[-1,4] <= 7


def test_invalid_input_does_not_mutate():
    s = state()
    before = pickle.dumps(s)
    for value in (np.nan,np.inf,1e100):
        with pytest.raises(ValueError):
            s.predict_one(value)
        assert pickle.dumps(s) == before
    with pytest.raises(ValueError):
        s.replay([1.,2.,np.nan])
    assert pickle.dumps(s) == before


def test_invalid_reference_no_latch():
    s = state(np.ones(1024,dtype=np.float32))
    out = s.replay(historical(1024)*100.)
    assert np.all(out[:,0] == 6)
    assert np.all(out[:,[1,2,3,4,18,26,27]] == 0)


def test_current_finite_influence_separate_event_history():
    stream = historical(2048)[::-1].copy()
    altered = stream.copy()
    altered[127:159] *= 20.
    first = state()
    second = state()
    a = first.replay(stream)
    b = second.replay(altered)
    current_columns = [0,1,2,7,9,10,11,12,13,14,15,16,17,18,25,28,29]
    np.testing.assert_array_equal(a[159+128:,current_columns], b[159+128:,current_columns])
    for left,right in zip(first.scale.args,second.scale.args):
        np.testing.assert_array_equal(left,right)
    for left,right in zip(first.core.args,second.core.args):
        np.testing.assert_array_equal(left,right)


def test_incomplete_blocks_cannot_confirm():
    s = state(scale_cut=.01)
    result = s.replay(np.zeros(63,dtype=np.float32))
    assert not result[:,18].any()
    assert not result[:,3].any()
