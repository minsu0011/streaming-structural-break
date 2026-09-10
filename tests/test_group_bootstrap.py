import numpy as np
from src.scoring.ts_auc import ts_auc
from src.validation.group_bootstrap import draw_group_multiplicities, weighted_ts_auc_replicates


def test_group_weighted_auc_matches_explicit_timeline_resampling():
    rng = np.random.default_rng(847)
    group = np.repeat(np.arange(12), 17)
    time = np.tile(np.arange(17), 12)
    y = rng.integers(0, 2, len(group))
    p = rng.integers(0, 5, len(group)).astype(np.float32) / 4
    code, counts = draw_group_multiplicities(group, replicates=16)
    actual = weighted_ts_auc_replicates(y, p, time, code, counts)
    for k, count in enumerate(counts):
        ix = np.repeat(np.arange(len(group)), count[code])
        np.testing.assert_allclose(actual[k], ts_auc(y[ix], p[ix], time[ix]), atol=1e-15, rtol=0)
    # Resampling twice as many copies leaves a pair-weighted ranking unchanged.
    np.testing.assert_array_equal(actual, weighted_ts_auc_replicates(y, p, time, code, 2 * counts))


def test_bootstrap_no_pairs_ties_and_deterministic_groups():
    group = np.array([10, 10, 20, 20, 30, 30])
    code, counts = draw_group_multiplicities(group, replicates=4)
    np.testing.assert_array_equal(counts, draw_group_multiplicities(group, replicates=4)[1])
    np.testing.assert_array_equal(counts.sum(axis=1), np.full(5, 3))
    for target in (np.zeros(6), np.array([0, 0, 1, 1, 0, 1])):
        np.testing.assert_array_equal(weighted_ts_auc_replicates(target, np.ones(6), np.tile([0, 1], 3), code, counts), np.full(5, .5))
