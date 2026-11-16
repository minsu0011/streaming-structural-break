"""Exact finite multinomial moments of a fixed crossed-grid functional.

This is an audit of resampling arithmetic, not a coverage theorem or a new
uncertainty interval. No random variates are generated here.
"""
import numpy as np


def exact_product_moments(matrix, resample_size=None):
    """Moments of C' M C / N**2, C ~ Multinomial(N, uniform(n)).

    The same count vector is used on both axes. The independent-axis variance
    is an algebraic comparator only; it does not replace the frozen method.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size:
        raise ValueError('A nonempty square matrix is required')
    if not np.isfinite(matrix).all():
        raise ValueError('Finite grid values are required')
    size = len(matrix) if resample_size is None else resample_size
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError('A positive integer resample size is required')
    size = int(size)
    # Center before taking moments to avoid cancellation for nearly constant M.
    grand = float(matrix.mean())
    centered = matrix - grand
    symmetric = (centered + centered.T) / 2
    projection = symmetric.mean(axis=1)
    diagonal = np.diag(centered)
    diagonal_mean = float(diagonal.mean())
    zeta1 = float(np.mean(projection ** 2))
    zeta2 = float(np.mean(symmetric ** 2))
    diagonal_variance = float(np.var(diagonal))
    projection_diagonal_covariance = float(np.mean(projection * (diagonal - diagonal_mean)))
    numerator = (4 * (size - 1) * (size - 2) * zeta1
                 + 2 * (size - 1) * zeta2 + diagonal_variance
                 + 4 * (size - 1) * projection_diagonal_covariance)
    same_variance = numerator / size ** 3
    if same_variance < -1e-13 * max(1., float(np.mean(centered ** 2))):
        raise ArithmeticError('Negative exact variance beyond roundoff')
    same_variance = max(0., same_variance)
    row, column = centered.mean(axis=1), centered.mean(axis=0)
    residual = centered - row[:, None] - column[None, :]
    independent_variance = (float(np.mean(row ** 2)) + float(np.mean(column ** 2))) / size
    independent_variance += float(np.mean(residual ** 2)) / size ** 2
    return {'grid_mean': grand, 'diagonal_mean': grand + diagonal_mean,
            'same_axis_exact_mean': grand + diagonal_mean / size,
            'same_axis_exact_mean_minus_grid': diagonal_mean / size,
            'same_axis_exact_variance': same_variance,
            'same_axis_exact_SD': float(np.sqrt(same_variance)),
            'independent_axis_exact_mean': grand,
            'independent_axis_exact_variance': independent_variance,
            'independent_axis_exact_SD': float(np.sqrt(independent_variance)),
            'same_to_independent_SD_ratio': float(np.sqrt(same_variance / independent_variance))
                if independent_variance > 0 else None,
            'paired_row_column_covariance': float(np.mean(row * column)),
            'symmetrized_projection_variance': zeta1,
            'symmetrized_kernel_variance': zeta2,
            'diagonal_variance': diagonal_variance,
            'projection_diagonal_covariance': projection_diagonal_covariance}
