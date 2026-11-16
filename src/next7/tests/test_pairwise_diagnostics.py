import numpy as np
import pytest
from src.next7.pairwise import objective
from src.next7.pairwise_diagnostics import exact_stratum, sampled_stratum


@pytest.mark.parametrize('block', [1, 2, 64])
def test_exhaustive_matches_literal_pairs(block):
    p = np.array([[1., 2.], [-1., 3.], [0., 0.]])
    n = np.array([[2., -2.], [1., 2.]])
    po, no = np.array([2., -1., .5]), np.array([-.4, 1.2])
    beta = np.array([.3, -.2])
    x = (p[:, None] - n[None, :]).reshape(-1, 2)
    offset = (po[:, None] - no[None, :]).ravel()
    expected = objective(beta, x, offset, np.full(6, 1 / 6), penalty=0.)
    actual = exact_stratum(beta, p, n, po, no, block)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-14, atol=1e-14)
    sampled = sampled_stratum(beta, x, offset)
    np.testing.assert_allclose(sampled[:2][1], actual[1], atol=1e-14)
    assert abs(sampled[0] - actual[0]) < 1e-14


def test_constant_within_time_features_cannot_change_ranking():
    p, n = np.ones((3, 2)), np.ones((4, 2))
    po, no = np.array([1., 2., 3.]), np.array([-1., 0., 1., 2.])
    before = exact_stratum(np.zeros(2), p, n, po, no)
    after = exact_stratum(np.array([100., -70.]), p, n, po, no)
    np.testing.assert_allclose(before[0], after[0], atol=1e-14)
    np.testing.assert_allclose(after[1], 0., atol=1e-14)


def test_reject_no_eligible_pair():
    with pytest.raises(ValueError):
        exact_stratum(np.zeros(2), np.empty((0, 2)), np.ones((1, 2)), np.empty(0), np.zeros(1))
