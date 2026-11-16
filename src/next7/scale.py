"""G1 signed, translation-invariant scale on disjoint 16-observation blocks.

All distribution fitting uses H. Current input never updates those references.
Outputs are bounded evidence coordinates, not calibrated probabilities.
"""
import math
import numpy as np
from numba import njit

FLOAT32_MAX = float(np.finfo(np.float32).max)
SCALES = ('S1_BLOCK_LOG_MAD', 'S2_BLOCK_SCALE_PIT')
RULES = ('C0_HARD4', 'C1_SOFT_3OF4', 'C2_ACCUMULATED_4BLOCK')
COLUMNS = ('completed', 'block_count', 'S1_signed', 'S2_signed') + tuple(
    f'{s}_{r}_{quantity}' for s in SCALES for r in RULES for quantity in ('strength', 'direction', 'same_direction_blocks'))


def valid_points(points, *, historical=False):
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any(np.abs(p) > FLOAT32_MAX):
        raise ValueError('Finite float32 one-dimensional observations required')
    if historical and len(p) < 512:
        raise ValueError('H requires at least512 observations')
    return p.astype(np.float32)


@njit(cache=True)
def log_block_mad(block, eps):
    center = np.median(block)
    return math.log(max(np.median(np.abs(block - center)), eps))


@njit(cache=True)
def historical_blocks(values, eps):
    n = len(values) // 16
    result = np.empty(n, dtype=np.float64)
    for j in range(n):
        result[j] = log_block_mad(values[16*j:16*j+16].astype(np.float64), eps)
    return result


@njit(cache=True)
def midcount(sorted_values, value):
    low = np.searchsorted(sorted_values, value, side='left')
    high = np.searchsorted(sorted_values, value, side='right')
    return (low + high) * .5


@njit(cache=True)
def signed_evidence(value, fit_sorted, calibration_sorted, params):
    center, dispersion, eps, valid = params
    if valid < .5:
        return np.zeros(2)
    z = min(max((value - center) / dispersion, -4.), 4.)
    prior = (midcount(fit_sorted, value) + .5) / (len(fit_sorted) + 1.)
    u = (midcount(calibration_sorted, value) + 32.*prior + .5) / (len(calibration_sorted) + 33.)
    pit = min(max(math.log(u / (1. - u)) / (math.pi / math.sqrt(3.)), -4.), 4.)
    return np.array([z, pit])


@njit(cache=True)
def confirmation(memory, available, output):
    """Six independent finite-rule diagnostics; thresholding happens elsewhere."""
    output[:] = 0.
    if available < 4:
        return output
    for channel in range(2):
        for rule in range(3):
            best = 0.
            direction = 0.
            support = 0
            for d in (-1., 1.):
                count = 0
                total = 0.
                smallest = 4.
                for k in range(4):
                    evidence = min(max(d * memory[k, channel], -4.), 4.)
                    count += int(evidence > 0.)
                    total += evidence
                    smallest = min(smallest, evidence)
                strength = 0.
                if rule == 0:
                    if count == 4:
                        strength = smallest
                elif count >= (3 if rule == 1 else 2):
                    strength = max(0., total / 4.)
                if strength > best:
                    best = strength
                    direction = d
                    support = count
            at = (channel * 3 + rule) * 3
            output[at] = best
            output[at + 1] = direction
            output[at + 2] = support
    return output


@njit(cache=True)
def step(point, args):
    fit_sorted, calibration_sorted, params, block, memory, counters, output = args
    t = counters[0]
    block[t % 16] = point
    output[0] = 0.
    if (t + 1) % 16 == 0:
        b = counters[1]
        value = log_block_mad(block, params[2])
        z = signed_evidence(value, fit_sorted, calibration_sorted, params)
        memory[b % 4] = z
        counters[1] += 1
        output[0] = 1.
        output[1] = counters[1]
        output[2:4] = z
        confirmation(memory, counters[1], output[4:])
    counters[0] += 1
    return output


@njit(cache=True)
def replay_kernel(points, args):
    out = np.empty((len(points), len(COLUMNS)))
    for i in range(len(points)):
        out[i] = step(float(points[i]), args)
    return out


class ScaleState:
    def __init__(self, historical):
        h = valid_points(historical, historical=True)
        fit = int(.6 * len(h))
        fit_h = h[:fit].astype(np.float64)
        mad = float(np.median(np.abs(fit_h - np.median(fit_h))))
        eps = max(1e-12 * mad, 1e-30)
        fit_blocks = historical_blocks(h[:fit], eps)
        calibration_blocks = historical_blocks(h[fit:], eps)
        valid = float(np.std(fit_h) > 1e-8 and len(fit_blocks) >= 16 and len(calibration_blocks) >= 8)
        n = len(calibration_blocks)
        fallback = n < 16
        used = fit_blocks if fallback else calibration_blocks
        fit_center = float(np.median(fit_blocks))
        cal_center = float(np.median(used))
        fit_spread = max(float(np.diff(np.quantile(fit_blocks, [.25, .75]))[0]) / 1.349, .15)
        cal_spread = max(float(np.diff(np.quantile(used, [.25, .75]))[0]) / 1.349, .15)
        center = (n * cal_center + 32. * fit_center) / (n + 32.)
        dispersion = math.sqrt((n * cal_spread**2 + 32. * fit_spread**2) / (n + 32.))
        self.args = (np.sort(fit_blocks), np.sort(used), np.array([center, dispersion, eps, valid]),
            np.zeros(16), np.zeros((4, 2)), np.zeros(2, dtype=np.int64), np.zeros(len(COLUMNS)))
        self.calibration = {'H_length': len(h), 'fit_length': fit, 'fit_blocks': len(fit_blocks),
            'calibration_blocks': n, 'used_calibration_blocks': len(used),
            'deterministic_fit_only_fallback': fallback, 'reference_valid': bool(valid),
            'center': center, 'dispersion': dispersion, 'epsilon': eps,
            'source_scope': 'Historical H only; no labels or online observations'}

    def predict_one(self, point):
        x = float(point)
        if not np.isfinite(x) or abs(x) > FLOAT32_MAX:
            raise ValueError('Finite float32 current scalar required')
        return step(float(np.float32(x)), self.args).copy()

    def replay(self, points):
        return replay_kernel(valid_points(points), self.args)

    @property
    def bounded_mutable_state_bytes(self):
        return sum(self.args[j].nbytes for j in (3, 4, 5, 6))
