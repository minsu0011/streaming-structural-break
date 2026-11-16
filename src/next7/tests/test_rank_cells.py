import numpy as np
import pytest
from src.next7.rank_cells import rank_cells
from src.next7.attribution import latch_pair_partition


def test_three_groups_match_literal_pairs_with_ties_and_zero_support():
    y = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1])
    p = np.array([.2, .2, .8, .7, .2, .6, .8, .3, .3])
    t = np.array([0, 0, 0, 0, 1, 1, 1, 2, 3])
    g = np.array([0, 1, 2, 2, 2, 0, 1, 2, 2])
    wins, mass = rank_cells(y, p, t, g, 3)
    expected_w, expected_m = np.zeros((3, 3)), np.zeros((3, 3))
    for i in np.flatnonzero(y):
        for j in np.flatnonzero(y == 0):
            if t[i] == t[j]:
                expected_m[g[i], g[j]] += 1
                expected_w[g[i], g[j]] += (1. if p[i] > p[j] else .5 if p[i] == p[j] else 0.)
    np.testing.assert_array_equal(wins, expected_w)
    np.testing.assert_array_equal(mass, expected_m)
    binary = latch_pair_partition(y, p, t, g > 0)
    for source, result in zip((wins, mass), binary):
        collapsed = np.array([[source[0, 0], source[0, 1:].sum()], [source[1:, 0].sum(), source[1:, 1:].sum()]])
        np.testing.assert_array_equal(collapsed, result)


@pytest.mark.parametrize('groups', [[0, 3], [0, -.1], [0, 1.5]])
def test_invalid_categories_do_not_silently_cast(groups):
    with pytest.raises(ValueError):
        rank_cells([0, 1], [.2, .4], [0, 0], groups, 3)
