"""Whole-pair alarm events and independent calibration/evaluation bootstrap."""
import math
import numpy as np

METRICS = ('null_eventual_alarm_rate', 'prebreak_alarm_rate',
           'postbreak_first_detection_rate', 'clean_detection_within_10',
           'clean_detection_within_50', 'clean_detection_within_128')


def trajectory_maxima(prediction, onset):
    p = np.asarray(prediction)
    if p.ndim != 2 or len(p) % 2 or not 0 < onset < p.shape[1] or not np.isfinite(p).all():
        raise ValueError('Finite matched controls followed by interventions required')
    n = len(p) // 2
    changed = p[n:]
    return np.column_stack([p[:n].max(axis=1), changed[:, :onset].max(axis=1),
        changed[:, onset:].max(axis=1),
        *[changed[:, onset:onset+w].max(axis=1) for w in (10, 50, 128)]])


def event_matrix(maxima, level):
    m = np.asarray(maxima)
    if m.ndim != 2 or m.shape[1] != 6 or not np.isfinite(m).all():
        raise ValueError('Six finite trajectory maxima required')
    if level is None:
        return np.zeros(m.shape, dtype=bool)
    if not np.isfinite(level):
        raise ValueError('Finite threshold or None required')
    pre = m[:, 1] > level
    result = m > level
    result[:, 2:] &= ~pre[:, None]
    return result


def bootstrap_thresholds(calibration_maxima, indices, alpha):
    values = np.asarray(calibration_maxima, dtype=np.float64)
    idx = np.asarray(indices)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not 0 < alpha < 1:
        raise ValueError('Finite calibration maxima and valid alpha required')
    if idx.ndim != 2 or idx.shape[1] != len(values) or idx.dtype.kind not in 'iu' or np.any((idx < 0) | (idx >= len(values))):
        raise ValueError('Whole calibration-series index vectors required')
    rank = math.ceil((len(values)+1)*(1-alpha))
    if rank > len(values):
        return np.full(len(idx), np.nan)
    samples = values[idx]
    return np.partition(samples, rank-1, axis=1)[:, rank-1]


def bootstrap_events(maxima, thresholds, pair_indices):
    m = np.asarray(maxima)
    levels = np.asarray(thresholds, dtype=np.float64)
    idx = np.asarray(pair_indices)
    if idx.shape != (len(levels), len(m)) or idx.dtype.kind not in 'iu' or np.any((idx < 0) | (idx >= len(m))):
        raise ValueError('One matched-pair resample per threshold required')
    output = np.empty((len(levels), len(METRICS)), dtype=np.float64)
    for k, level in enumerate(levels):
        output[k] = event_matrix(m[idx[k]], None if np.isnan(level) else level).mean(axis=0)
    return output
