from itertools import product
import numpy as np
import pytest
from src.next7.crossed_moments import exact_product_moments


def enumerate_moments(matrix, size):
    sequences = list(product(range(len(matrix)), repeat=size))
    same = np.array([matrix[np.ix_(a, a)].mean() for a in sequences])
    independent = np.array([matrix[np.ix_(a, b)].mean() for a in sequences for b in sequences])
    return same.mean(), same.var(), independent.mean(), independent.var()


@pytest.mark.parametrize('size', [1, 2, 3, 4])
def test_exact_moments_match_exhaustive_sequences_without_monte_carlo(size):
    matrix = np.array([[1., -.2, .3], [.7, .1, 1.8], [-.4, .8, -.3]])
    expected = enumerate_moments(matrix, size)
    actual = exact_product_moments(matrix, size)
    np.testing.assert_allclose([actual[k] for k in ('same_axis_exact_mean', 'same_axis_exact_variance',
        'independent_axis_exact_mean', 'independent_axis_exact_variance')], expected, rtol=2e-13, atol=1e-14)


def test_antisymmetric_grid_has_zero_same_axis_variance_but_independent_variation():
    matrix = np.array([[0., 1., 2.], [-1., 0., 3.], [-2., -3., 0.]])
    result = exact_product_moments(matrix)
    assert result['same_axis_exact_mean'] == result['same_axis_exact_variance'] == 0.
    assert result['independent_axis_exact_variance'] > 0.


def test_constant_and_row_only_have_the_correct_degenerate_moments():
    constant = exact_product_moments(np.full((3, 3), .25))
    assert constant['same_axis_exact_mean'] == .25
    assert constant['same_axis_exact_variance'] == constant['independent_axis_exact_variance'] == 0.
    row = np.broadcast_to(np.array([0., .5, 1.])[:, None], (3, 3))
    result = exact_product_moments(row)
    assert abs(result['same_axis_exact_variance'] - np.var(row[:, 0]) / 3) < 1e-15
    assert abs(result['same_to_independent_SD_ratio'] - 1) < 1e-15


def test_large_constant_shift_preserves_resampling_variance():
    matrix = np.array([[1., 2.], [3., 4.]])
    a, b = exact_product_moments(matrix), exact_product_moments(matrix + 1e9)
    assert a['same_axis_exact_variance'] == b['same_axis_exact_variance']
    assert a['independent_axis_exact_variance'] == b['independent_axis_exact_variance']


def test_invalid_or_nonfinite_matrices_and_sizes_are_rejected():
    for matrix, size in [(np.zeros((2, 3)), None), (np.empty((0, 0)), None),
                         (np.array([[np.nan]]), None), (np.eye(2), 0),
                         (np.eye(2), 1.5), (np.eye(2), True)]:
        with pytest.raises(ValueError):
            exact_product_moments(matrix, size)
