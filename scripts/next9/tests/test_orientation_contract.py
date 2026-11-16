import src.next9
import inspect
import numpy as np
import pytest
from src.next9.data import FitScope
from src.next9.orientation_contract import fit_common_range, fit_run_fallback, PAIR_MENU, orientation_design


def fixture():
    groups = np.repeat(np.arange(128), 20)
    states = np.tile(np.repeat([0, 2], 10), 128)
    f = np.tile(np.linspace(-2, 2, 10), 256)
    y = groups % 2
    scope = FitScope(frozenset(range(128)), frozenset(range(128, 256)), frozenset(range(256, 384)), 'toy')
    return f, states, groups, y, scope


def test_up_down_common_range_training_only():
    f, states, groups, y, scope = fixture()
    fitted = fit_common_range(f, states, groups, y, scope, PAIR_MENU[0])
    assert fitted.status == 'SUPPORTED'
    original = fitted.record()
    fitted.mask(f * 100, states)
    assert fitted.record() == original
    with pytest.raises(ValueError, match='outside'):
        fit_common_range(f, states, groups + 128, y, scope, PAIR_MENU[0])


def test_common_range_does_not_adapt_to_labels():
    f, s, g, y, scope = fixture()
    a = fit_common_range(f, s, g, y, scope, PAIR_MENU[0])
    b = fit_common_range(f, s, g, 1-y, scope, PAIR_MENU[0])
    assert (a.low, a.high) == (b.low, b.high)
    c = fit_common_range(f + (s == 2)*20, s, g, y, scope, PAIR_MENU[0])
    assert c.status == 'INSUFFICIENT_COMMON_RANGE'


def test_sparse_run_mapping_is_training_fixed():
    f, s, g, y, scope = fixture()
    run = np.where(s == 2, 6, 0)
    mapping = fit_run_fallback(run, g, scope)
    assert mapping[0] == 0 and mapping[6] == 6
    assert mapping[1] == 10 and mapping[7] == 12
    assert mapping[9] == 9


def test_same_input_tabular_exact_matrix():
    f, s, g, y, scope = fixture()
    args = (f, s, np.where(s == 2, 6, 0), np.where(s == 2, 0, 1), s)
    np.testing.assert_array_equal(orientation_design(*args, 'O5_MOVE_REV'), orientation_design(*args, 'O6_TABULAR'))
    np.testing.assert_array_equal(orientation_design(*args, 'O1'), orientation_design(*args, 'F_ONLY'))


def test_tau_absent():
    assert 'tau' not in inspect.signature(orientation_design).parameters


def test_break_age_absent():
    assert 'break_age' not in inspect.signature(orientation_design).parameters


def test_future_length_absent():
    assert 'length' not in inspect.signature(orientation_design).parameters
