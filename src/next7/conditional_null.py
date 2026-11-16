"""Known DGP states are used only to generate null continuations, never features."""
import math
import numpy as np
from numba import njit
from src.next2.counterexamples import noise
from src.next7.fast_state import fast_step, update_summary, SUMMARY_COLUMNS


@njit(cache=True)
def historical_path(eps, phi, alpha):
    values = np.empty(len(eps), dtype=np.float64)
    y = 0.
    q = 1. - phi * phi
    for t in range(len(eps)):
        target = 1. - phi * phi
        variance = (1. - alpha) * target + alpha * q
        innovation = np.sqrt(max(variance, 1e-12)) * eps[t]
        y = phi * y + innovation + (1. - phi) * 0.
        values[t] = y
        q = innovation * innovation
    return values, y, q


@njit(cache=True)
def conditional_points(eps, phi, alpha, y, q):
    points = np.empty(len(eps), dtype=np.float32)
    for t in range(len(eps)):
        target = 1. - phi * phi
        variance = (1. - alpha) * target + alpha * q
        innovation = np.sqrt(max(variance, 1e-12)) * eps[t]
        y = phi * y + innovation + (1. - phi) * 0.
        q = innovation * innovation
        points[t] = y
    return points


@njit(cache=True)
def conditional_summary(eps, phi, alpha, y, q, detector_args):
    summary = np.zeros((3, len(SUMMARY_COLUMNS)))
    maxima = np.zeros((9, 2))
    running = np.array([0., 0., -1., -1., -1.])
    for t in range(len(eps)):
        target = 1. - phi * phi
        variance = (1. - alpha) * target + alpha * q
        innovation = np.sqrt(max(variance, 1e-12)) * eps[t]
        y = phi * y + innovation + (1. - phi) * 0.
        q = innovation * innovation
        # The detector sees only the same canonical current float32 observation.
        out = fast_step(float(np.float32(y)), detector_args)
        update_summary(t, out, summary, maxima, running)
    return summary, maxima


def source_components(case, index, lock, *, H_length=5000, horizon=16384):
    if case not in ('N0', 'N1', 'N2', 'N3') or not 0 <= index < 512:
        raise ValueError('Fixed canonical null and fixed index0..511 required')
    ordinal = next(j for j, spec in enumerate(lock['scenarios']) if spec['id'] == case)
    spec = lock['scenarios'][ordinal]
    assert spec['phi_before'] == spec['phi_after'] and spec['arch_before'] == spec['arch_after']
    assert spec['variance_factor'] == 1. and spec['mean_shift_stationary_sd'] == 0.
    assert spec['t_df_before'] == spec['t_df_after']
    seed = [lock['seed'], 33, ordinal, index]
    hist = noise(seed + [0], lock['burn_in'] + H_length, spec['t_df_before'])
    values, y, q = historical_path(hist, float(spec['phi_before']), float(spec['arch_before']))
    scheduled = int(lock['tau_cycle'][index % len(lock['tau_cycle'])])
    pre = noise(seed + [1], min(scheduled, horizon), spec['t_df_before'])
    post = noise(seed + [2], max(horizon - scheduled, 0), spec['t_df_after'])
    return values[lock['burn_in']:].astype(np.float32), np.r_[pre, post], np.array([y, q]), spec
