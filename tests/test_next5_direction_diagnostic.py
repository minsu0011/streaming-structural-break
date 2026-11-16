import numpy as np
import pytest
from src.next5.direction_diagnostic import project, summaries, COORDINATES


def test_onset_coordinates_and_right_censoring_are_explicit():
    values = np.arange(1024, dtype=float)
    actual = project(values, 512)[:, 0]
    assert actual[COORDINATES.index('lag_0')] == 512
    assert np.isnan(actual[COORDINATES.index('lag_512')])
    assert actual[COORDINATES.index('age_0_4')] == 514
    assert actual[COORDINATES.index('age_50_plus')] == np.mean(values[562:])


def test_paired_support_and_seed_mean_not_pooled_time_mean():
    changed = np.ones((256, 16, 1))
    clean = np.zeros_like(changed)
    changed[:85, 10] = clean[:85, 10] = np.nan
    rows = summaries(changed, clean, ['response'])
    assert rows[10]['paired_generating_seeds'] == 171
    assert rows[10]['paired_response_mean'] == 1
    clean[0, 0] = np.nan
    with pytest.raises(ValueError, match='support'):
        summaries(changed, clean, ['response'])
