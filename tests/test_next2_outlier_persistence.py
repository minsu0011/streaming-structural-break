import numpy as np
import pytest
from src.next2.outlier_persistence import aligned_pair_predictions, paired_mean_interval


def test_alignment_preserves_each_scheduled_injection_and_prepoint():
    control = np.arange(24, dtype=np.float32).reshape(3, 8)
    changed = control.copy()
    onsets = np.array([1, 2, 4])
    for i, onset in enumerate(onsets):
        changed[i, onset:] += 100
    before, after = aligned_pair_predictions(np.r_[control, changed], onsets, 4)
    np.testing.assert_array_equal(after[:, 0], before[:, 0])
    np.testing.assert_array_equal(after[:, 1:]-before[:, 1:], np.full((3, 4), 100))
    changed[0, 0] += 1
    with pytest.raises(AssertionError):
        aligned_pair_predictions(np.r_[control, changed], onsets, 4)


def test_interval_keeps_signed_whole_pair_difference():
    counts = np.array([[2, 0], [1, 1], [0, 2]])
    result = paired_mean_interval(np.array([-1., 3.]), counts)
    assert result['estimate'] == 1.
    assert result['ci_low'] == pytest.approx(-.9)
    assert result['ci_high'] == pytest.approx(2.9)
    with pytest.raises(ValueError):
        paired_mean_interval(np.array([-1., 3.]), np.array([[1, 0]]))
