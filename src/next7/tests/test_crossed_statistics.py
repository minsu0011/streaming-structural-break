import numpy as np
import pytest
from src.next7.crossed_statistics import seed_pair_product_draws, conditional_H_draws, complete_grid_decomposition


def test_product_weights_equal_literal_seed_pair_resampling_including_duplicate_diagonal():
    matrix = np.array([[1., 0., .2], [.3, 1., .4], [0., .8, 1.]])
    counts = np.array([[2, 0, 1], [0, 1, 2], [1, 1, 1]])
    expected = []
    for row in counts:
        indices = np.repeat(np.arange(3), row)
        expected.append(matrix[indices][:, indices].mean())
    np.testing.assert_allclose(seed_pair_product_draws(matrix, counts), expected, atol=1e-15)


def test_conditional_bootstrap_resamples_innovation_axis_only():
    matrix = np.array([[0., 1., 0.], [1., 0., 1.], [.2, .3, .4]])
    counts = np.array([[2, 0, 1], [0, 1, 2]])
    expected = np.array([matrix[:, np.repeat(np.arange(3), c)].mean(axis=1) for c in counts])
    np.testing.assert_allclose(conditional_H_draws(matrix, counts), expected, atol=1e-15)


def test_anova_descriptive_decomposition_has_exact_row_and_column_additivity():
    matrix = np.array([1., 2., 4.])[:, None] + np.array([0., -2., 3.])[None, :]
    result = complete_grid_decomposition(matrix)
    assert result['interaction_mean_square'] < 1e-28
    assert abs(result['H_main_effect_mean_square'] - np.var([1., 2., 4.])) < 1e-14


def test_fractional_multiplicities_are_rejected():
    with pytest.raises(ValueError):
        seed_pair_product_draws(np.zeros((2, 2)), [[.5, 1.5]])
