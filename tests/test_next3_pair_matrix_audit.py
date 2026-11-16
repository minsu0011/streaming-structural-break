"""An independent explicit-pair oracle tests the numerical audit's contract."""
import numpy as np
import pytest
from numba import set_num_threads
from src.next3.pair_matrix_audit import dense_series, enumerate_pair_matrices, evaluate_pair_matrices


def fixture():
    probabilities = [[.1, .3, .5, .8], [.2, .3, .6], [.3, .7], [.1, .3, .2, .8, .4]]
    changes = [1, 3, 0, 3]
    rows = [(i, t, int(t >= changes[i]), p) for i, series in enumerate(probabilities) for t, p in enumerate(series)]
    return (np.array([r[2] for r in rows], dtype=np.uint8),
            np.array([r[3] for r in rows], dtype=np.float32),
            np.array([r[1] for r in rows], dtype=np.int32),
            np.array([r[0] for r in rows], dtype=np.int32))


def literal_pair_oracle(y, p, t, g, weights):
    scores = []
    for w in weights:
        # Literal replicated-row enumeration deliberately shares no rank or matrix logic.
        indices = [i for i in range(len(y)) for _ in range(int(w[g[i]]))]
        wins = 0.; pairs = 0
        for i in indices:
            for j in indices:
                if y[i] == 1 and y[j] == 0 and t[i] == t[j]:
                    pairs += 1
                    wins += 1. if p[i] > p[j] else .5 if p[i] == p[j] else 0.
        scores.append(wins/pairs if pairs else .5)
    return np.asarray(scores)


@pytest.mark.parametrize('permuted', [False, True])
@pytest.mark.parametrize('chunk_size', [1, 3, 64])
def test_pair_matrix_matches_literal_duplicate_rows_and_ties(permuted, chunk_size):
    set_num_threads(2)
    y, p, t, g = fixture()
    if permuted:
        order = np.random.default_rng(4).permutation(len(y))
        y, p, t, g = (a[order] for a in (y, p, t, g))
    weights = np.asarray([[1, 1, 1, 1], [0, 0, 0, 0], [2, 0, 1, 3], [0, 4, 0, 0], [0, 1, 1, 2]], dtype=np.int32)
    n, d = enumerate_pair_matrices(*dense_series(y, p, t, g))
    assert np.diag(n).sum() == np.diag(d).sum() == 0
    np.testing.assert_array_equal(evaluate_pair_matrices(n, d, weights, chunk_size=chunk_size),
                                  literal_pair_oracle(y, p, t, g, weights))


@pytest.mark.parametrize('label', [0, 1])
def test_single_class_returns_frozen_half_score(label):
    y, p, t, g = fixture(); y[:] = label
    n, d = enumerate_pair_matrices(*dense_series(y, p, t, g))
    assert not n.any() and not d.any()
    np.testing.assert_array_equal(evaluate_pair_matrices(n, d, np.ones((3, 4), dtype=np.int32)), [.5]*3)


@pytest.mark.parametrize('failure', ['time_gap', 'duplicate_time', 'reverse_label', 'nonfinite', 'wrong_precision', 'group_gap'])
def test_invalid_original_timeline_is_rejected(failure):
    y, p, t, g = fixture()
    if failure == 'time_gap': t[1] += 10
    if failure == 'duplicate_time': t[1] = 0
    if failure == 'reverse_label': y[2] = 0
    if failure == 'nonfinite': p[0] = np.nan
    if failure == 'wrong_precision': p = p.astype(np.float64)
    if failure == 'group_gap': g[g == 3] = 6
    with pytest.raises(ValueError): dense_series(y, p, t, g)


def test_quadratic_counts_reject_inexact_or_invalid_inputs():
    n = np.ones((2, 2), dtype=np.int32); d = n.copy()
    for weights in (np.array([[-1, 2]], dtype=np.int32), np.array([[1., 2.]]),
                    np.array([[2**30, 2**30]], dtype=np.int64)):
        with pytest.raises(ValueError): evaluate_pair_matrices(n, d, weights)
    with pytest.raises(ValueError): evaluate_pair_matrices(n*3, d, np.ones((1, 2), dtype=np.int32))
