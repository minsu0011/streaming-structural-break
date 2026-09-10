"""H-fitted innovation normalization and binary-search empirical CDF mapping."""
import numpy as np
from numba import njit

MODES = ('raw', 'sd', 'mad', 'arch1', 'arch2', 'ewma', 'uniform', 'gaussian')


@njit(cache=True)
def inverse_normal(p):
    """Acklam rational approximation; verified against scipy.special.ndtri."""
    if p < .02425:
        q = np.sqrt(-2.*np.log(p))
        return (((((-7.784894002430293e-3*q-.3223964580411365)*q-2.400758277161838)*q-2.549732539343734)*q+4.374664141464968)*q+2.938163982698783)/((((7.784695709041462e-3*q+.3224671290700398)*q+2.445134137142996)*q+3.754408661907416)*q+1.)
    if p > 1.-.02425:
        return -inverse_normal(1.-p)
    q = p-.5
    r = q*q
    return (((((-39.69683028665376*r+220.9460984245205)*r-275.9285104469687)*r+138.3577518672690)*r-30.66479806614716)*r+2.506628277459239)*q/(((((-54.47609879822406*r+161.5858368580409)*r-155.6989798598866)*r+66.80131188771972)*r-13.28068155288572)*r+1.)


@njit(cache=True)
def ecdf_uniform(value, sorted_h):
    left = np.searchsorted(sorted_h, value, side='left')
    right = np.searchsorted(sorted_h, value, side='right')
    # Mid-rank smoothing handles repeated values and both outside-H extremes.
    return (left+right+1.)/(2.*(len(sorted_h)+1.))


@njit(cache=True)
def normalization_update(residual, parameters, conditional, sorted_h, mode):
    u = ecdf_uniform(residual, sorted_h)
    centered = residual-parameters[0]
    if mode == 0:
        z = residual
    elif mode == 1 or mode == 2:
        z = centered/parameters[1]
    elif mode == 3 or mode == 4:
        variance = max(parameters[2]+parameters[3]*conditional[0]+parameters[4]*conditional[1], parameters[6])
        z = centered/np.sqrt(variance)
    elif mode == 5:
        z = centered/np.sqrt(max(conditional[2], parameters[6]))
    elif mode == 6:
        z = np.sqrt(12.)*(u-.5)
    else:
        z = inverse_normal(u)
    squared = min(centered*centered, parameters[5])
    conditional[1] = conditional[0]
    conditional[0] = squared
    conditional[2] = (1.-parameters[7])*conditional[2]+parameters[7]*squared
    return min(max(z, -12.), 12.), u


@njit(cache=True)
def replay_normalization(residuals, parameters, conditional, sorted_h, mode):
    z = np.empty(len(residuals), dtype=np.float64)
    u = np.empty(len(residuals), dtype=np.float64)
    for j in range(len(residuals)):
        z[j], u[j] = normalization_update(residuals[j], parameters, conditional, sorted_h, mode)
    return z, u


def fit_normalization(historical_innovations, mode='sd'):
    if mode not in MODES:
        raise ValueError('Innovation normalization outside preregistered family')
    r = np.asarray(historical_innovations, dtype=np.float64)
    mean, median = float(r.mean()), float(np.median(r))
    sd = max(float(r.std()), 1e-8)
    mad = max(float(1.4826*np.median(np.abs(r-median))), sd*.01, 1e-8)
    location, scale = (median, mad) if mode == 'mad' else (mean, sd)
    sq = (r-location)**2
    cap = max(float(np.quantile(sq, .99)), 1e-12)
    capped = np.minimum(sq, cap)
    variance = max(float(capped.mean()), 1e-12)
    slopes = np.zeros(2)
    if mode in ('arch1', 'arch2'):
        order = 1 if mode == 'arch1' else 2
        design = np.column_stack([capped[order-1-j:len(r)-1-j] for j in range(order)])
        target = capped[order:]
        center = design-design.mean(axis=0)
        gram = center.T@center
        ridge = max(float(np.trace(gram))/order*.001, 1e-12)
        slopes[:order] = np.maximum(np.linalg.solve(gram+ridge*np.eye(order), center.T@(target-target.mean())), 0.)
        if slopes.sum() > .95:
            slopes *= .95/slopes.sum()
    parameters = np.array([location, scale, variance*(1.-slopes.sum()), slopes[0], slopes[1], cap,
                           max(variance*.0001, 1e-12), 2./33.], dtype=np.float64)
    conditional = np.array([variance, variance, variance], dtype=np.float64)
    sorted_h = np.sort(r)
    transformed, uniforms = replay_normalization(r, parameters, conditional, sorted_h, MODES.index(mode))
    return {'parameters': parameters, 'conditional': conditional, 'sorted_h': sorted_h,
            'mode': MODES.index(mode), 'historical_z': transformed, 'historical_uniform': uniforms}
