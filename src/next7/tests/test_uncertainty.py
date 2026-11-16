import numpy as np
import pytest
from src.next5.bootstrap import PreparedAUC
from src.next7.uncertainty import prepared_totals, paired_summary
from src.next5.fractures import masks


def explicit_totals(y, p, t):
    wins = pairs = 0.
    for time in np.unique(t):
        a, b = p[(t == time) & (y == 1)], p[(t == time) & (y == 0)]
        pairs += len(a) * len(b)
        wins += sum(float(x > z) + .5 * float(x == z) for x in a for z in b)
    return wins, pairs


def test_weighted_totals_match_literal_series_resampling_with_ties_and_empty_support():
    y = np.array([0, 1, 1, 0, 0, 1, 0, 1])
    p = np.array([.2, .2, .3, .4, .8, .9, .7, .7])
    t = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    g = np.array([0, 1, 0, 1, 2, 3, 2, 3])
    counts = np.array([[1, 1, 1, 1], [2, 0, 3, 1], [0, 0, 0, 0], [0, 0, 0, 4]])
    prepared = PreparedAUC.create(y, p, t, g, 4)
    totals = prepared_totals(prepared, counts)
    for i, count in enumerate(counts):
        indices = np.repeat(np.arange(len(y)), count[g])
        np.testing.assert_array_equal(totals[i], explicit_totals(y[indices], p[indices], t[indices]))
    legacy = prepared.evaluate(counts)
    good = totals[:, 1] > 0
    np.testing.assert_array_equal(totals[good, 0] / totals[good, 1], legacy[good])
    assert np.all(totals[~good, 1] == 0)


def test_time_and_positive_age_pair_totals_partition_all_pairs_for_every_draw():
    rng = np.random.default_rng(32)
    groups, length = 12, 80
    g = np.repeat(np.arange(groups), length)
    t = np.tile(np.arange(length), groups)
    tau = np.repeat(np.r_[np.full(6, -1), [0, 5, 11, 20, 40, 70]], length)
    y = ((tau >= 0) & (t >= tau)).astype(np.uint8)
    p = rng.integers(0, 6, size=len(y)).astype(float)
    counts = rng.multinomial(groups, np.full(groups, 1 / groups), size=20)
    all_totals = prepared_totals(PreparedAUC.create(y, p, t, g, groups), counts)
    scopes = masks(y, t, tau, np.zeros(len(y)), np.full(len(y), 'WEAK_OR_AMBIGUOUS'), np.full(len(y), 'increase'))
    for prefix in ('time_', 'age_'):
        total = np.zeros_like(all_totals)
        for name, select in scopes.items():
            if name.startswith(prefix) and np.any(select):
                total += prepared_totals(PreparedAUC.create(y[select], p[select], t[select], g[select], groups), counts)
        np.testing.assert_array_equal(total, all_totals)


def test_median_of_paired_fold_differences_is_distinct_from_difference_of_medians():
    ref = np.tile([.1, .5, .9], (4, 1))
    cand = np.tile([.8, .6, .4], (4, 1))
    out = paired_summary(ref, cand)
    assert out['estimate_median_matched_fold_delta'] == pytest.approx(.1)
    assert out['estimate_difference_of_fold_score_medians'] == pytest.approx(.1)
    cand[:] = [.8, .4, .6]
    out = paired_summary(ref, cand)
    assert out['estimate_median_matched_fold_delta'] == pytest.approx(-.1)
    assert out['estimate_difference_of_fold_score_medians'] == pytest.approx(.1)


def test_fractional_or_negative_counts_rejected():
    prepared = PreparedAUC.create([0, 1], [.1, .2], [0, 0], [0, 1], 2)
    for counts in ([[1., .5]], [[-1, 2]], [[np.inf, 0]]):
        with pytest.raises(ValueError):
            prepared_totals(prepared, counts)
