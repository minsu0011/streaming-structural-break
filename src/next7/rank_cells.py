"""Exact same-time rank-pair decomposition by finite diagnostic categories."""
import numpy as np
from numba import njit


@njit(cache=True)
def sorted_cells(y, p, t, group, size):
    wins, pairs = np.zeros((size, size)), np.zeros((size, size))
    cursor = 0
    while cursor < len(y):
        at = t[cursor]
        positive, negative = np.zeros(size), np.zeros(size)
        while cursor < len(y) and t[cursor] == at:
            score = p[cursor]
            tp, tn = np.zeros(size), np.zeros(size)
            while cursor < len(y) and t[cursor] == at and p[cursor] == score:
                g = group[cursor]
                if y[cursor]:
                    tp[g] += 1
                else:
                    tn[g] += 1
                cursor += 1
            for i in range(size):
                for j in range(size):
                    wins[i, j] += tp[i] * (negative[j] + .5 * tn[j])
            positive += tp
            negative += tn
        for i in range(size):
            for j in range(size):
                pairs[i, j] += positive[i] * negative[j]
    return wins, pairs


def rank_cells(target, prediction, time, group, size):
    arrays = tuple(np.asarray(a) for a in (target, prediction, time, group))
    y, p, t, g = arrays
    if not all(a.ndim == 1 and a.shape == y.shape for a in arrays):
        raise ValueError('Aligned vectors required')
    if not isinstance(size, int) or not 1 <= size <= 16:
        raise ValueError('Bounded category count required')
    if not all(np.isfinite(a).all() for a in arrays) or not np.isin(y, [0, 1]).all():
        raise ValueError('Finite arrays and binary target required')
    if np.any(t < 0) or np.any(t != np.floor(t)) or np.any(g < 0) or np.any(g >= size) or np.any(g != np.floor(g)):
        raise ValueError('Nonnegative integer times and declared categories required')
    order = np.lexsort((p, t))
    return sorted_cells(y[order].astype(np.uint8), p[order].astype(np.float64),
        t[order].astype(np.int64), g[order].astype(np.int64), size)
