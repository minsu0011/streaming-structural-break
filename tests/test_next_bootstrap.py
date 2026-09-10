import numpy as np
import pytest
from src.next.bootstrap import PreparedAUC, paired_summary
from src.scoring.ts_auc import ts_auc
from src.validation.group_bootstrap import draw_group_multiplicities, weighted_ts_auc_replicates


def test_cached_exact_bootstrap_matches_explicit_series_resampling():
    rng = np.random.default_rng(9831)
    lengths = rng.integers(3, 40, size=25)
    ids = np.repeat(np.arange(25), lengths)
    times = np.concatenate([np.arange(n) for n in lengths])
    target = rng.integers(0, 2, len(ids), dtype=np.uint8)
    pred = rng.integers(0, 9, len(ids)).astype(np.float32)/8
    code, counts = draw_group_multiplicities(ids, replicates=40, seed=441)
    cached = PreparedAUC.create(target, pred, times, code, 25)
    result = cached.evaluate(counts)
    np.testing.assert_array_equal(result, weighted_ts_auc_replicates(target, pred, times, code, counts))
    for b in range(len(counts)):
        rows = np.repeat(np.arange(len(ids)), counts[b, code])
        assert abs(result[b] - ts_auc(target[rows], pred[rows], times[rows])) < 1e-15


def test_cached_bootstrap_all_ties_one_class_and_zero_counts():
    code = np.repeat(np.arange(4), 5)
    times = np.tile(np.arange(5), 4)
    for y in [np.zeros(20, dtype=np.uint8), (code % 2).astype(np.uint8)]:
        state = PreparedAUC.create(y, np.ones(20), times, code, 4)
        np.testing.assert_array_equal(state.evaluate(np.array([[1]*4, [0]*4])), [.5, .5])


def test_pairing_identity_and_sign():
    a = np.arange(55).reshape(11, 5)/100
    assert paired_summary(a, a)['probability_delta_positive'] == 0
    result = paired_summary(a, a+.01)
    assert result['probability_delta_positive'] == 1
    assert abs(result['estimate_mean_delta']-.01) < 1e-15
