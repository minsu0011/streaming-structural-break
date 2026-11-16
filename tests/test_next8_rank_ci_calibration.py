import numpy as np
import pytest
from src.next8.rank_ci_calibration import sample, cells, evaluate_sample, wilson


def test_known_truth_control_uses_whole_series_and_same_simulated_labels():
    a = sample(16, 0., 0., 1)
    b = sample(16, .95, .36, 1)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[2], b[2])
    np.testing.assert_array_equal(a[3], b[3])
    assert a[4] == .5 and b[4] > .6
    assert np.unique(a[3], return_counts=True)[1].tolist() == [512] * 16
    assert np.all(np.diff(a[0].reshape(16, 512).astype(int), axis=1) >= 0)
    assert len(cells()) == len(set(cells())) == 12
    record = evaluate_sample(16, .95, .36, 1, draws=40)
    assert record['resampling']['group_count'] == 16
    assert record['bootstrap_valid_draws'] == 40
    assert set(record['methods']) == {'CLUSTER_NORMAL', 'EXACT_GROUP_PERCENTILE'}
    assert record['real_RP_values_or_predictions_used'] is False


def test_monte_carlo_zero_and_full_coverage_remain_uncertain():
    low, high = wilson(0, 100)
    assert low == 0 and 0 < high < .05
    low, high = wilson(100, 100)
    assert .95 < low < 1 and high == 1
    with pytest.raises(ValueError):
        wilson(0, 0)
