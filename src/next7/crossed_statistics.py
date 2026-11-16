"""Fixed seed-pair product reweighting for a complete crossed diagnostic grid."""
import numpy as np
from threadpoolctl import threadpool_limits


def validate(matrix, counts):
    m, c = np.asarray(matrix, dtype=np.float64), np.asarray(counts, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or c.ndim != 2 or c.shape[1] != len(m):
        raise ValueError('Square crossed matrix and aligned seed count matrix required')
    if not np.isfinite(m).all() or not np.isfinite(c).all() or np.any(c < 0) or np.any(c != np.floor(c)):
        raise ValueError('Finite matrix and nonnegative integer multiplicities required')
    total = c.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError('Every resample needs a positive seed count')
    return m, c, total


def seed_pair_product_draws(matrix, counts):
    m, c, n = validate(matrix, counts)
    with threadpool_limits(limits=1):
        return np.sum((c @ m) * c, axis=1) / (n * n)


def conditional_H_draws(matrix, counts):
    m, c, n = validate(matrix, counts)
    with threadpool_limits(limits=1):
        return (c @ m.T) / n[:, None]


def complete_grid_decomposition(matrix):
    m = np.asarray(matrix, dtype=np.float64)
    grand = float(m.mean())
    row, column = m.mean(axis=1) - grand, m.mean(axis=0) - grand
    residual = m - grand - row[:, None] - column[None, :]
    pieces = {'H_main_effect_mean_square': float(np.mean(row * row)),
        'innovation_main_effect_mean_square': float(np.mean(column * column)),
        'interaction_mean_square': float(np.mean(residual * residual))}
    total = float(np.mean((m - grand) ** 2))
    assert abs(sum(pieces.values()) - total) < 1e-12
    return {'grand_mean': grand, 'total_cell_variance_descriptive': total, **pieces,
        **{key.replace('mean_square', 'fraction_total_variance'): value / total if total > 0 else None
            for key, value in pieces.items()}}
