"""Exact time-stratified AUC with cached ordering and paired series weights.

Parallelism is over independent bootstrap replicates; each weighted rank sum
uses float64 and exactly the same tie/pair definition as explicit resampling.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def _replicates(y, group, time_ends, tie_ends, multiplicities):
    result = np.empty(len(multiplicities), dtype=np.float64)
    for b in prange(len(multiplicities)):
        wins = 0.0
        pairs = 0.0
        i = 0
        tie = 0
        for time_end in time_ends:
            positive = 0.0
            negative = 0.0
            while i < time_end:
                tie_positive = 0.0
                tie_negative = 0.0
                end = tie_ends[tie]
                while i < end:
                    weight = multiplicities[b, group[i]]
                    if y[i]:
                        tie_positive += weight
                    else:
                        tie_negative += weight
                    i += 1
                wins += tie_positive * (negative + 0.5 * tie_negative)
                positive += tie_positive
                negative += tie_negative
                tie += 1
            pairs += positive * negative
        result[b] = wins / pairs if pairs > 0 else 0.5
    return result


@dataclass
class PreparedAUC:
    y: np.ndarray
    group: np.ndarray
    time_ends: np.ndarray
    tie_ends: np.ndarray
    n_groups: int

    @classmethod
    def create(cls, target, prediction, time_online, series_code, n_groups):
        y = np.asarray(target, dtype=np.uint8)
        p = np.asarray(prediction, dtype=np.float32)
        t = np.asarray(time_online, dtype=np.int32)
        g = np.asarray(series_code, dtype=np.int32)
        if any(a.ndim != 1 or a.shape != y.shape for a in [p, t, g]):
            raise ValueError('Equal-length vectors required')
        if not len(y) or not np.isfinite(p).all() or np.any(y > 1) or np.any(t < 0):
            raise ValueError('Invalid binary targets, times or predictions')
        if g.min() < 0 or g.max() >= n_groups:
            raise ValueError('Group code outside weight matrix')
        order = np.lexsort((p, t))
        sy, sp, st, sg = y[order], p[order], t[order], g[order]
        time_change = st[1:] != st[:-1]
        time_ends = np.r_[np.flatnonzero(time_change)+1, len(y)].astype(np.int32)
        tie_ends = np.r_[np.flatnonzero(time_change | (sp[1:] != sp[:-1]))+1, len(y)].astype(np.int32)
        return cls(sy, sg, time_ends, tie_ends, n_groups)

    def evaluate(self, multiplicities):
        counts = np.asarray(multiplicities, dtype=np.int32)
        if counts.ndim != 2 or counts.shape[1] != self.n_groups or np.any(counts < 0):
            raise ValueError('Nonnegative whole-series counts required')
        return _replicates(self.y, self.group, self.time_ends, self.tie_ends, counts)


def paired_summary(reference, candidate):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError('Paired replicate-by-fold matrices required')
    mean_delta = candidate.mean(axis=1) - reference.mean(axis=1)
    median_delta = np.median(candidate, axis=1) - np.median(reference, axis=1)
    return {'estimate_mean_delta': float(mean_delta[0]),
            'bootstrap_mean_delta': float(mean_delta[1:].mean()),
            'bootstrap_median_delta': float(np.median(mean_delta[1:])),
            'ci_2_5': float(np.quantile(mean_delta[1:], .025)),
            'ci_97_5': float(np.quantile(mean_delta[1:], .975)),
            'probability_delta_positive': float(np.mean(mean_delta[1:] > 0)),
            'estimate_median_of_folds_delta': float(median_delta[0]),
            'median_of_folds_delta_ci_2_5': float(np.quantile(median_delta[1:], .025)),
            'median_of_folds_delta_ci_97_5': float(np.quantile(median_delta[1:], .975)),
            'replicates': int(len(mean_delta)-1)}
