"""Paired summaries of one-observation contamination, without model selection."""
import numpy as np


def aligned_pair_predictions(prediction, onsets, horizon=512):
    prediction = np.asarray(prediction)
    onsets = np.asarray(onsets, dtype=np.int64)
    n = len(onsets)
    if prediction.ndim != 2 or prediction.shape[0] != 2*n:
        raise ValueError('Expected complete control and outlier halves')
    if np.any(onsets < 1) or np.any(onsets+horizon > prediction.shape[1]):
        raise ValueError('Common complete post-injection horizon required')
    for i, onset in enumerate(onsets):
        if not np.array_equal(prediction[i, :onset], prediction[n+i, :onset]):
            raise AssertionError('Predictions differ before the sole intervention')
    time = onsets[:, None] + np.arange(-1, horizon)[None, :]
    index = np.arange(n)[:, None]
    return prediction[index, time], prediction[n+index, time]


def paired_mean_interval(values, counts):
    values = np.asarray(values, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if values.ndim != 1 or counts.ndim != 2 or counts.shape[1] != len(values):
        raise ValueError('One scalar per whole matched pair required')
    if not np.isfinite(values).all() or np.any(counts < 0) or not np.all(counts.sum(axis=1) == len(values)):
        raise ValueError('Finite values and complete bootstrap multiplicities required')
    boot = counts @ values / len(values)
    return {'estimate': float(values.mean()),
            'ci_low': float(np.quantile(boot, .025)),
            'ci_high': float(np.quantile(boot, .975))}
