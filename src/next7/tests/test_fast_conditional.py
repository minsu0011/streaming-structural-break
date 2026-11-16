import copy
import pickle
import numpy as np
import pytest
from src.next7.common import ROOT, read_json
from src.next7.calibration import load_calibration, cutoff
from src.next7.null_growth import QualifiedState, GRID
from src.next7.fast_state import FastQualifiedState, SUMMARY_COLUMNS, HORIZONS
from src.next7.conditional_null import source_components, conditional_points, conditional_summary
from src.next6_extension.synthetic import generated


@pytest.mark.parametrize('case', ['N0', 'N1', 'N2', 'N3'])
@pytest.mark.parametrize('seed', [0, 17, 511])
def test_conditional_diagonal_raw_path_exactly_matches_original_generator(case, seed):
    lock = copy.deepcopy(read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json'))
    lock['historical_length'] = 5000
    h, eps, terminal, spec = source_components(case, seed, lock)
    actual = conditional_points(eps, float(spec['phi_before']), float(spec['arch_before']), *terminal)
    expected_h, expected, tau, _ = generated(case, seed, partition=33, horizon=16384, control=False, lock=lock)
    np.testing.assert_array_equal(h, expected_h)
    np.testing.assert_array_equal(actual, expected)
    assert tau == -1


@pytest.mark.parametrize('case', ['N2', 'V4', 'RETURN_NORMAL', 'OB16'])
def test_fused_all_coordinates_exact_reference_parity(case):
    lock = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    h, o, _, _ = generated(case, 1, partition=31, horizon=4096, control=False, lock=lock)
    cal = load_calibration()
    args = {'core_cut': cal['core_cut'], 'scale_cut': cutoff(cal, len(h))}
    fast, original = FastQualifiedState(h, **args), QualifiedState(h, **args)
    before = fast.bounded_mutable_state_bytes
    np.testing.assert_array_equal(fast.replay(o), original.replay(o))
    assert fast.bounded_mutable_state_bytes == before


def test_fused_scalar_clone_pickle_prefix_and_invalid_input():
    lock = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    h, o, _, _ = generated('V4', 1, partition=31, horizon=1024, control=False, lock=lock)
    cal = load_calibration()
    state = FastQualifiedState(h, core_cut=cal['core_cut'], scale_cut=cutoff(cal, len(h)))
    scalar, batch, chunked = state.clone(), state.clone(), state.clone()
    expected = batch.replay(o)
    np.testing.assert_array_equal(np.array([scalar.predict_one(x) for x in o]), expected)
    prefix = chunked.replay(o[:317])
    restored = pickle.loads(pickle.dumps(chunked))
    np.testing.assert_array_equal(np.vstack((prefix, restored.replay(o[317:]))), expected)
    before = pickle.dumps(state)
    with pytest.raises(ValueError):
        state.replay([1., np.nan])
    assert pickle.dumps(state) == before
    changed = np.r_[o[:317], np.full(1200, 20., dtype=np.float32)]
    np.testing.assert_array_equal(state.clone().replay(changed)[:317], expected[:317])


def test_bounded_summary_matches_every_reference_prefix_and_boundary_statistic():
    lock = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    h, eps, terminal, spec = source_components('N2', 17, lock)
    cal = load_calibration()
    state = FastQualifiedState(h[-2048:], core_cut=cal['core_cut'], scale_cut=cutoff(cal, 2048))
    points = conditional_points(eps, float(spec['phi_before']), float(spec['arch_before']), *terminal)
    full = state.clone().replay(points)
    summary, maxima = state.clone().summarize(points)
    joint, jmax = conditional_summary(eps, float(spec['phi_before']), float(spec['arch_before']),
        terminal[0], terminal[1], state.clone().args)
    np.testing.assert_array_equal(summary, joint)
    np.testing.assert_array_equal(maxima, jmax)
    assert summary.shape == (3, len(SUMMARY_COLUMNS)) and maxima.shape == (9, 2)
    for j, horizon in enumerate(HORIZONS):
        out = full[:horizon]
        assert summary[j, 0] == np.max(out[:, 2])
        assert summary[j, 1] == np.max(out[:, 2] / out[:, 42])
        for column, source in [(5, 26), (6, 3), (7, 30)]:
            assert summary[j, column] == (out[-1, source] > 0)
        assert summary[j, 12] == (out[:, 2].max() > out[-1, 42])
    for j, blocks in enumerate(GRID):
        assert maxima[j, 0] == full[:blocks * 16, 2].max()


def test_conditional_future_innovation_suffix_cannot_change_prefix():
    lock = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    _, eps, terminal, spec = source_components('N3', 511, lock)
    a = conditional_points(eps, float(spec['phi_before']), float(spec['arch_before']), *terminal)
    b = conditional_points(eps[:777], float(spec['phi_before']), float(spec['arch_before']), *terminal)
    np.testing.assert_array_equal(a[:777], b)


def test_fused_insufficient_H_preserves_reference_abstention():
    lock = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    h, o, _, _ = generated('N0', 0, partition=31, horizon=1024, control=False, lock=lock)
    cal = load_calibration()
    args = {'core_cut': cal['core_cut'], 'scale_cut': cutoff(cal, 512)}
    a = FastQualifiedState(h[-512:], **args).replay(o)
    b = QualifiedState(h[-512:], **args).replay(o)
    np.testing.assert_array_equal(a, b)
    assert not a[:, 30].any() and not a[:, 43].any()
