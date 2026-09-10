"""Conditional uncertainty of fixed OOF predictions under whole-group resampling.

These intervals do not account for fitting uncertainty or adaptive model selection.
Every bootstrap sample preserves each series' timeline and its frozen fold.
"""
import numpy as np
from numba import njit


@njit(cache=False)
def _weighted_scores(target, prediction, online_time, group, multiplicities):
    result = np.empty(len(multiplicities), dtype=np.float64)
    for replicate in range(len(multiplicities)):
        wins, pairs = 0.0, 0.0
        positive, negative = 0.0, 0.0
        i = 0
        while i < len(target):
            time = online_time[i]
            positive, negative = 0.0, 0.0
            while i < len(target) and online_time[i] == time:
                score = prediction[i]
                tie_positive, tie_negative = 0.0, 0.0
                while i < len(target) and online_time[i] == time and prediction[i] == score:
                    weight = multiplicities[replicate, group[i]]
                    if target[i] == 1:
                        tie_positive += weight
                    else:
                        tie_negative += weight
                    i += 1
                wins += tie_positive * (negative + 0.5 * tie_negative)
                positive += tie_positive
                negative += tie_negative
            pairs += positive * negative
        result[replicate] = wins / pairs if pairs > 0 else 0.5
    return result


def weighted_ts_auc_replicates(target, prediction, online_time, group, multiplicities):
    y = np.asarray(target, dtype=np.uint8)
    p = np.asarray(prediction, dtype=np.float32)
    t = np.asarray(online_time, dtype=np.int64)
    g = np.asarray(group, dtype=np.int32)
    w = np.asarray(multiplicities, dtype=np.int32)
    if y.ndim != 1 or p.shape != y.shape or t.shape != y.shape or g.shape != y.shape:
        raise ValueError('Equal-length row vectors required')
    if w.ndim != 2 or np.any(w < 0) or not np.isfinite(p).all() or np.any(y > 1):
        raise ValueError('Invalid predictions, binary labels or multiplicities')
    if len(g) and (g.min() < 0 or g.max() >= w.shape[1]):
        raise ValueError('Group multiplicity table does not cover the rows')
    order = np.lexsort((p, t))
    return _weighted_scores(y[order], p[order], t[order], g[order], w)


def draw_group_multiplicities(groups, replicates=200, seed=20260908):
    """The first row is the original sample; remaining rows sample whole groups."""
    unique, code = np.unique(groups, return_inverse=True)
    if not len(unique) or replicates < 1:
        raise ValueError('At least one group and bootstrap replicate required')
    rng = np.random.default_rng(seed)
    counts = np.empty((replicates + 1, len(unique)), dtype=np.int32)
    counts[0] = 1
    for row in counts[1:]:
        row[:] = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
    return code.astype(np.int32), counts
