"""Independent conditional TS-AUC audit by explicit ordered series pairs."""
import numpy as np
from numba import njit, prange


def dense_series(target, prediction, online_time, group):
    y = np.asarray(target)
    p = np.asarray(prediction)
    t = np.asarray(online_time)
    g = np.asarray(group)
    if y.ndim != 1 or not len(y) or any(a.shape != y.shape for a in (p, t, g)):
        raise ValueError('Nonempty equal-length row vectors required')
    if p.dtype != np.float32 or not np.isfinite(p).all() or not np.isin(y, (0, 1)).all():
        raise ValueError('Finite original float32 predictions and binary targets required')
    if t.dtype.kind not in 'iu' or g.dtype.kind not in 'iu' or np.any(t < 0):
        raise ValueError('Nonnegative integer time and integer group codes required')
    unique = np.unique(g)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError('Contiguous zero-based whole-series codes required')
    order = np.lexsort((t, g)); yy, pp, tt, gg = y[order], p[order], t[order], g[order]
    starts = np.r_[0, np.flatnonzero(gg[1:] != gg[:-1])+1, len(gg)]
    lengths = np.diff(starts).astype(np.int32)
    matrix = np.full((len(unique), int(lengths.max())), np.nan, dtype=np.float32)
    first_positive = np.empty(len(unique), dtype=np.int32)
    for code, (start, stop) in enumerate(zip(starts[:-1], starts[1:])):
        local_y, local_t = yy[start:stop], tt[start:stop]
        if not np.array_equal(local_t, np.arange(stop-start)):
            raise ValueError('Each original timeline must contain exactly times 0 through length minus one')
        positive = np.flatnonzero(local_y)
        first = int(positive[0]) if len(positive) else stop-start
        if not np.array_equal(local_y, np.arange(stop-start) >= first):
            raise ValueError('Frozen structural-break binary target must be a zero prefix and one suffix')
        first_positive[code] = first
        matrix[code, :stop-start] = pp[start:stop]
    return matrix, lengths, first_positive


@njit(cache=False, parallel=True)
def enumerate_pair_matrices(predictions, lengths, first_positive):
    """N_ij=2*wins+ties; D_ij=eligible common times with i positive and j negative."""
    n = len(lengths)
    numerator_twice = np.zeros((n, n), dtype=np.int32)
    denominator = np.zeros((n, n), dtype=np.int32)
    for left in prange(n):
        start = first_positive[left]
        for right in range(n):
            end = min(lengths[left], lengths[right], first_positive[right])
            if start >= end:
                continue
            wins = 0
            for time in range(start, end):
                a, b = predictions[left, time], predictions[right, time]
                if a > b:
                    wins += 2
                elif a == b:
                    wins += 1
            numerator_twice[left, right] = wins
            denominator[left, right] = end-start
    return numerator_twice, denominator


def evaluate_pair_matrices(numerator_twice, denominator, multiplicities, *, chunk_size=64):
    n = np.asarray(numerator_twice); d = np.asarray(denominator); w = np.asarray(multiplicities)
    if n.ndim != 2 or n.shape[0] != n.shape[1] or d.shape != n.shape:
        raise ValueError('Square pair-count matrices required')
    if any(a.dtype.kind not in 'iu' for a in (n, d, w)) or np.any(n < 0) or np.any(d < 0):
        raise ValueError('Nonnegative integer pair counts required')
    if np.any(n > 2*d) or w.ndim != 2 or w.shape[1] != n.shape[0] or np.any(w < 0):
        raise ValueError('Invalid win counts or whole-series multiplicities')
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('Positive bounded chunk size required')
    max_weight_sum = int(w.sum(axis=1, dtype=np.int64).max(initial=0))
    upper_bound = 2*int(d.max(initial=0))*max_weight_sum*max_weight_sum
    if upper_bound >= 2**53:
        raise ValueError('Integer quadratic sums could exceed exact float64 representation')
    nn, dd = n.astype(np.float64), d.astype(np.float64)
    result = np.full(len(w), .5, dtype=np.float64)
    for start in range(0, len(w), chunk_size):
        stop = min(start+chunk_size, len(w)); weights = w[start:stop].astype(np.float64)
        numerator = np.sum((weights@nn)*weights, axis=1)
        pairs = np.sum((weights@dd)*weights, axis=1)
        if not np.array_equal(numerator, np.floor(numerator)) or not np.array_equal(pairs, np.floor(pairs)):
            raise AssertionError('Nonintegral result in exact integer-count quadratic sum')
        np.divide(numerator, 2*pairs, out=result[start:stop], where=pairs > 0)
    return result
