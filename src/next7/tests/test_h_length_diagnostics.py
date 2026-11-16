import numpy as np
import pytest
from src.next7.h_length_diagnostics import discordant_latches


def test_shared_seed_multiplicity_matches_materialized_crossed_grid():
    candidate = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=np.uint8)
    reference = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 1]], dtype=np.uint8)
    counts = np.array([[2, 0, 1], [0, 3, 0], [1, 1, 1]])
    rates, draws, delta = discordant_latches(candidate, reference, counts)
    for draw, multiplicities in enumerate(counts):
        selected = np.repeat(np.arange(3), multiplicities)
        cand = candidate[np.ix_(selected, selected)].astype(bool)
        ref = reference[np.ix_(selected, selected)].astype(bool)
        assert draws['ADDED_FALSE_MEMORY'][draw] == np.mean(cand & ~ref)
        assert draws['REMOVED_FALSE_MEMORY'][draw] == np.mean(ref & ~cand)
        assert delta[draw] == (cand.astype(np.int8) - ref.astype(np.int8)).mean()
    assert np.isclose(sum(rates.values()), 1.)


def test_unsigned_input_cannot_wrap_negative_differences():
    candidate = np.zeros((2, 2), dtype=np.uint8)
    reference = np.ones((2, 2), dtype=np.uint8)
    rates, draws, delta = discordant_latches(candidate, reference, np.array([[1, 1], [2, 0]]))
    assert rates['REMOVED_FALSE_MEMORY'] == 1
    np.testing.assert_array_equal(delta, [-1., -1.])
    np.testing.assert_array_equal(draws['ADDED_FALSE_MEMORY'], [0., 0.])


def test_identical_paths_have_zero_discordance_even_with_duplicate_seeds():
    values = np.eye(3, dtype=bool)
    rates, draws, delta = discordant_latches(values, values, np.array([[0, 2, 1]]))
    assert rates['ADDED_FALSE_MEMORY'] == rates['REMOVED_FALSE_MEMORY'] == 0
    np.testing.assert_array_equal(delta, [0.])
    assert draws['BOTH_FALSE_MEMORY'][0] == 5 / 9


def test_missing_or_nonbinary_decisions_are_rejected():
    counts = np.ones((1, 2), dtype=int)
    for invalid in (np.array([[0, np.nan], [1, 0]]), np.array([[0, 2], [1, 0]])):
        with pytest.raises(ValueError):
            discordant_latches(invalid, np.zeros((2, 2)), counts)
    with pytest.raises(ValueError):
        discordant_latches(np.zeros((2, 2)), np.zeros((3, 3)), counts)
