"""Evaluation-only attribution of official pair mass; never router inputs."""
import numpy as np


def positive_pair_credit(target, prediction, online_time):
    y = np.asarray(target)
    p = np.asarray(prediction, dtype=float)
    t = np.asarray(online_time)
    if y.ndim != 1 or p.shape != y.shape or t.shape != y.shape:
        raise ValueError('Aligned vectors required')
    if not np.isfinite(p).all() or not np.isfinite(t).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Finite predictions/time and binary targets required')
    if (t < 0).any() or (t != np.floor(t)).any():
        raise ValueError('Nonnegative integral time required')
    order = np.argsort(t, kind='stable')
    cuts = np.r_[0, np.flatnonzero(np.diff(t[order]))+1, len(t)]
    credit = np.zeros(len(y), dtype=np.float64)
    pairs = np.zeros(len(y), dtype=np.int64)
    for begin, end in zip(cuts[:-1], cuts[1:]):
        rows = order[begin:end]
        positive = rows[y[rows] == 1]
        negative = np.sort(p[rows[y[rows] == 0]])
        if not len(positive) or not len(negative):
            continue
        low = np.searchsorted(negative, p[positive], side='left')
        high = np.searchsorted(negative, p[positive], side='right')
        credit[positive] = .5*(low+high)
        pairs[positive] = len(negative)
    return credit, pairs
