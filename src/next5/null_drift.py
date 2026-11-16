"""Descriptive energy identity for the unchanged working Gaussian evidence."""
import numpy as np
from numba import njit
from src.next5 import mixture, scale


def energy_boundary(variances):
    values = np.asarray(variances, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values <= 0) or np.any(values == 1):
        raise ValueError('Positive non-null variance alternatives required')
    return np.log(values) / (1 - 1 / values)


def cumulative_from_energy(energy_sum, count, variances):
    energy = np.asarray(energy_sum, dtype=np.float64)
    alternatives = np.asarray(variances, dtype=np.float64)
    energy_boundary(alternatives)
    if energy.shape != (2,) or not np.isfinite(energy).all() or np.any(energy < 0) or type(count) is not int or count <= 0:
        raise ValueError('Two finite nonnegative accumulated energies and an actual positive count required')
    return -.5 * count * np.log(alternatives)[None, :] + .5 * energy[:, None] * (1 - 1 / alternatives)[None, :]


@njit(cache=False)
def snapshots(points, cumulative_args, sparse_args, horizons):
    energy = np.zeros(2)
    energies = np.empty((len(horizons), 2))
    likelihood = np.empty((len(horizons), 2, 4))
    cumulative_features = np.empty((len(horizons), 12), dtype=np.float32)
    sparse_features = np.empty((len(horizons), 18), dtype=np.float32)
    maximum_channel_difference = 0.
    cursor = 0
    for index in range(len(points)):
        a = mixture.mixture_step(float(points[index]), *cumulative_args)
        b = mixture.mixture_step(float(points[index]), *sparse_args)
        za, zb = cumulative_args[0][-1], sparse_args[0][-1]
        maximum_channel_difference = max(maximum_channel_difference, np.max(np.abs(za - zb)))
        energy += za * za
        if cursor < len(horizons) and index + 1 == horizons[cursor]:
            energies[cursor] = energy
            likelihood[cursor] = cumulative_args[2]
            cumulative_features[cursor] = a
            sparse_features[cursor] = b
            cursor += 1
    if cursor != len(horizons):
        raise ValueError('Requested horizon not observed')
    return energies, likelihood, cumulative_features, sparse_features, maximum_channel_difference


def evaluate(historical, online, horizons):
    points = np.asarray(online, dtype=np.float32)
    stops = np.asarray(horizons)
    if points.ndim != 1 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Finite one-dimensional synthetic observations required')
    if stops.ndim != 1 or stops.dtype.kind not in 'iu' or not len(stops) or stops[0] <= 0 or stops[-1] > len(points) or np.any(np.diff(stops) <= 0):
        raise ValueError('Strictly increasing actual observed horizons required')
    cumulative = mixture.make_state(historical, {'version': 'next5_fixed_mixture_v1', 'kind': 'cumulative'})
    sparse = mixture.make_state(historical, {'version': 'next5_fixed_mixture_v1', 'kind': 'sparse_starts'})
    values = snapshots(points, cumulative.args, sparse.args, stops.astype(np.int64))
    assert values[-1] == 0., 'The two frozen feature banks must use identical innovation channels'
    errors = []
    for j, horizon in enumerate(stops):
        expected = cumulative_from_energy(values[0][j], int(horizon), scale.VARIANCES)
        error = float(np.max(np.abs(expected - values[1][j])))
        # The identity differs only in floating summation order. This bound is
        # arithmetic verification, not an estimated statistical interval.
        np.testing.assert_allclose(values[1][j], expected, rtol=1e-10, atol=1e-7)
        errors.append(error)
    return values[:4], max(errors)


def describe(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite descriptive seed values required')
    return {'generating_seeds': len(x), 'mean': float(x.mean()), 'median': float(np.median(x)),
        'q05': float(np.quantile(x, .05)), 'q95': float(np.quantile(x, .95)),
        'fraction_strictly_positive': float(np.mean(x > 0))}
