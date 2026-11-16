import numpy as np
import pytest
from src.next7.diagnostic_statistics import support_partition, fixed_fold_draws, fixed_fold_pair


def test_support_partition_includes_strict_boundary_and_retained_crossing():
    result = support_partition([4., 3., 2., 2.5, 1.], 2., 3., [1, 1, 1, 1, 0])
    np.testing.assert_array_equal(result, [[1, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0]])


def test_support_partition_refuses_incompatible_cut_or_missing_latch():
    with pytest.raises(ValueError):
        support_partition([4.], 3., 2., [1])
    with pytest.raises(ValueError):
        support_partition([4.], 2., 3., [0])


def test_fixed_fold_equal_weight_is_not_pooled_available_cells():
    value = np.array([[0., np.nan, np.nan], [1., 2., 3.]])
    estimate, draws, means, support = fixed_fold_draws(value, [[1, 1, 1], [3, 0, 0], [0, 2, 1]])
    assert estimate == 1. and estimate != np.nanmean(value)
    np.testing.assert_array_equal(draws[:2], [1., .5])
    assert np.isnan(draws[2])
    np.testing.assert_array_equal(support, [1, 3])


def test_paired_delay_comparison_uses_same_observable_seeds_per_fold():
    a = np.array([[1., np.nan, 100.], [3., 2., 1.]])
    b = np.array([[0., 1000., np.nan], [1., np.nan, 1.]])
    row, draws = fixed_fold_pair(a, b, [[1, 1, 1], [1, 0, 2], [0, 3, 0]])
    assert row['eligible_seed_counts_by_fixed_fold'] == [1, 2]
    assert row['mean_delta'] == 1.
    np.testing.assert_allclose(draws[:2], [1., 5 / 6])
    assert np.isnan(draws[2]) and row['usable_draws'] == 2


def test_shared_seed_draws_do_not_treat_duplicated_folds_as_new_samples():
    v = np.array([[0., 1., 1.]])
    counts = np.array([[3, 0, 0], [1, 1, 1], [0, 0, 3]])
    one = fixed_fold_draws(v, counts)
    five = fixed_fold_draws(np.repeat(v, 5, axis=0), counts)
    assert one[0] == five[0]
    np.testing.assert_allclose(one[1], five[1], atol=0, rtol=1e-15)


def test_entire_missing_fold_does_not_silently_change_fixed_model_pool():
    estimate, draws, _, _ = fixed_fold_draws([[1., 2.], [np.nan, np.nan]], [[1, 1], [2, 0]])
    assert estimate is None and np.isnan(draws).all()
