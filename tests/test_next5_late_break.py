import numpy as np
import pytest
from src.next5.late_break import onset_mask, stratum_event_draws


def test_late_stratum_retains_both_clean_and_changed_seed_partners():
    ids = np.repeat(np.arange(8), 2)
    mask = onset_mask(ids, [128, 256, 512], 512)
    np.testing.assert_array_equal(ids[mask], [2, 2, 5, 5])
    with pytest.raises(ValueError):
        onset_mask(ids, [128, 256, 512], 1024)


def test_eligible_denominator_uses_original_seed_columns_not_subset_maximum():
    ids = np.array([2, 2, 5, 5])
    tau = np.array([-1, 1, -1, 1])
    p = np.array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.], [1., 0., 0.]])
    counts = np.ones((2, 8), dtype=np.int16)
    counts[1, 2], counts[1, 5] = 3, 1
    result = stratum_event_draws(p, tau, ids, .5, 'postbreak_detection', counts)
    np.testing.assert_array_equal(result, [.5, .75])
    result = stratum_event_draws(p, tau, ids, .5, 'prebreak', counts)
    np.testing.assert_array_equal(result, [.5, .25])
    result = stratum_event_draws(p, tau, ids, .5, 'null_eventual', counts)
    np.testing.assert_array_equal(result, [.5, .75])
