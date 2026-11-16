"""Post-hoc fixed synthetic response coordinates, never inference features."""
import numpy as np

LAGS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
AGES = ((0, 4), (5, 9), (10, 19), (20, 49), (50, None))
COORDINATES = tuple('lag_' + str(v) for v in LAGS) + tuple(
    'age_' + str(a) + '_' + ('plus' if b is None else str(b)) for a, b in AGES)


def project(values, onset):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or not np.isfinite(array).all() or not 0 <= onset < len(array):
        raise ValueError('Finite time-by-coordinate values and valid diagnostic onset required')
    result = np.full((len(COORDINATES), array.shape[1]), np.nan)
    for i, lag in enumerate(LAGS):
        if onset + lag < len(array):
            result[i] = array[onset + lag]
    for i, (start, end) in enumerate(AGES, len(LAGS)):
        stop = len(array) if end is None else min(len(array), onset + end + 1)
        if onset + start < stop:
            result[i] = array[onset + start:stop].mean(axis=0)
    return result


def summaries(changed, clean, names):
    changed, clean = np.asarray(changed), np.asarray(clean)
    if changed.shape != clean.shape or changed.shape != (256, len(COORDINATES), len(names)):
        raise ValueError('Exactly the original 256 paired seed coordinates are required')
    if not np.array_equal(np.isnan(changed), np.isnan(clean)):
        raise ValueError('Paired missing-coordinate support differs')
    rows = []
    for coordinate_index, coordinate in enumerate(COORDINATES):
        for feature_index, name in enumerate(names):
            a = changed[:, coordinate_index, feature_index]
            b = clean[:, coordinate_index, feature_index]
            keep = np.isfinite(a) & np.isfinite(b)
            delta = a[keep] - b[keep]
            rows.append({'coordinate': coordinate, 'measurement': name,
                'paired_generating_seeds': int(keep.sum()),
                'changed_mean': float(a[keep].mean()), 'clean_mean': float(b[keep].mean()),
                'paired_response_mean': float(delta.mean()), 'paired_response_median': float(np.median(delta)),
                'positive_response_fraction': float((delta > 0).mean()),
                'negative_response_fraction': float((delta < 0).mean()),
                'interpretation': 'Descriptive response at the fixed generator; not held-out discrimination or causal feature importance'})
    return rows
