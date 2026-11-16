"""Read-only comparison of the two H-fitted references already used by G2."""
import numpy as np
from threadpoolctl import threadpool_limits
from src.next6_extension.fresh_runtime import SafeState
from src.next6_extension.fresh_state import NAMES
from src.next7.scale import ScaleState, valid_points
from src.next7.crossed_statistics import validate, conditional_H_draws

CHANNELS = (*NAMES, 'signed_log_MAD')
COVARIATES = tuple('center_shift_full_units__' + name for name in CHANNELS) + tuple(
    'log_prefix_spread_over_full__' + name for name in CHANNELS) + (
    'max_rank_reference_gap_on_H', 'RMS_rank_reference_gap_on_H',
    'prefix_minus_full_rank_second_moment', 'prefix_core_floor_nodes', 'full_core_floor_nodes')


def historical_parameters(historical, core_cut):
    h = valid_points(historical, historical=True)
    prefix_length = int(.6 * len(h))
    if prefix_length < 512:
        raise ValueError('The original qualified calibration needs a supported H prefix')
    prefix_core, full_core = SafeState(h[:prefix_length], threshold=core_cut), SafeState(h, threshold=core_cut)
    prefix_scale, full_scale = ScaleState(h[:prefix_length]), ScaleState(h)
    centers = np.stack((np.r_[prefix_core.args[1], prefix_scale.args[2][0]],
                        np.r_[full_core.args[1], full_scale.args[2][0]]))
    spreads = np.stack((np.r_[prefix_core.args[2], prefix_scale.args[2][1]],
                        np.r_[full_core.args[2], full_scale.args[2][1]]))
    assert centers.shape == spreads.shape == (2, 9) and np.all(spreads > 0)
    center_shift = (centers[0] - centers[1]) / spreads[1]
    log_spread_ratio = np.log(spreads[0] / spreads[1])
    def ranks(reference):
        left = np.searchsorted(reference, h, side='left')
        right = np.searchsorted(reference, h, side='right')
        return (left + right + 1.) / (len(reference) + 1.) - 1.
    rank_difference = ranks(prefix_core.args[0]) - ranks(full_core.args[0])
    context = [float(np.max(np.abs(rank_difference))), float(np.sqrt(np.mean(rank_difference ** 2))),
        float(prefix_core.args[9][2] - full_core.args[9][2]),
        len(prefix_core.calibration['spread_floor_nodes']), len(full_core.calibration['spread_floor_nodes'])]
    covariates = np.r_[center_shift, log_spread_ratio, context]
    assert covariates.shape == (len(COVARIATES),) and np.isfinite(covariates).all()
    return {'centers': centers, 'spreads': spreads, 'covariates': covariates,
        'prefix_length': prefix_length,
        'prefix_core_calibration': prefix_core.calibration, 'full_core_calibration': full_core.calibration,
        'prefix_scale_calibration': prefix_scale.calibration, 'full_scale_calibration': full_scale.calibration}


def paired_risk_correlations(covariates, decisions, counts):
    """Weighted Pearson with H and innovations resampled as the same seed unit."""
    matrix, multiplicity, size = validate(decisions, counts)
    x = np.asarray(covariates, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(matrix) or not np.isfinite(x).all():
        raise ValueError('Finite aligned H covariates required')
    x = x - x.mean(axis=0)  # Correlation invariant; limits cancellation for nearly constant H diagnostics.
    risk = conditional_H_draws(matrix, multiplicity)
    with threadpool_limits(limits=1):
        mean_x = (multiplicity @ x) / size[:, None]
        variance_x = (multiplicity @ (x * x)) / size[:, None] - mean_x * mean_x
        mean_risk = np.sum(multiplicity * risk, axis=1) / size
        variance_risk = np.sum(multiplicity * risk * risk, axis=1) / size - mean_risk * mean_risk
        covariance = ((multiplicity * risk) @ x) / size[:, None] - mean_risk[:, None] * mean_x
    denominator = np.sqrt(np.maximum(variance_x, 0) * np.maximum(variance_risk[:, None], 0))
    result = np.divide(covariance, denominator, out=np.full_like(covariance, np.nan), where=denominator > 0)
    # Roundoff may place a correlation a few ulps beyond its mathematical range.
    if np.isfinite(result).any():
        assert np.nanmax(np.abs(result)) <= 1 + 1e-9
    return np.clip(result, -1., 1.)
