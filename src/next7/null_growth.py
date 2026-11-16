"""G2 H-only prefix-maximum admission. No anytime probability guarantee.

The first30 output columns are the frozen G1 state, copied without alteration.
Columns30:45 describe qualified memory separately. Only five memory inputs are
substituted when producing the existing ten correction features.
"""
import math
import numpy as np
from numba import njit
from src.next7.event import CombinedState, COLUMNS as RAW_COLUMNS
from src.next7.scale import ScaleState, valid_points, FLOAT32_MAX
from src.next6_extension.fresh_runtime import SafeState

GRID = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024], dtype=np.int64)
DRAW_SEED = 20260913
REPLICATES = 512
ORDER_INDEX = 487
QUALIFIED_COLUMNS = ('qualified_ever', 'qualified_episodes', 'qualified_first_age',
    'qualified_last_age', 'qualified_direction', 'first_qualified_time',
    'last_qualified_time', 'qualified_channel', 'qualified_oldest_block',
    'qualified_same_direction_blocks', 'qualified_fresh', 'qualified_block_status',
    'admission_cut', 'admission_supported', 'qualified_strength')
COLUMNS = RAW_COLUMNS + QUALIFIED_COLUMNS


def shared_uniforms():
    return np.random.default_rng(DRAW_SEED).random((REPLICATES, 256))


@njit(cache=True)
def joint_ratio(memory, core_cut, scale_cut):
    best_core = 0.
    for channel in range(8):
        sign = 1. if memory[0, channel] > 0. else -1.
        weakest = 12.
        for k in range(4):
            weakest = min(weakest, sign * memory[k, channel])
        best_core = max(best_core, weakest)
    total = 0.
    positive = 0
    negative = 0
    for k in range(4):
        z = min(max(memory[k, 8], -4.), 4.)
        total += z
        positive += int(z > 0.)
        negative += int(z < 0.)
    best_scale = 0.
    if positive >= 3:
        best_scale = max(best_scale, total / 4.)
    if negative >= 3:
        best_scale = max(best_scale, -total / 4.)
    return max(best_core / core_cut, best_scale / scale_cut)


@njit(cache=True)
def bootstrap_maxima(vectors, uniforms, grid, core_cut, scale_cut):
    maxima = np.empty((len(uniforms), len(grid)), dtype=np.float64)
    for replicate in range(len(uniforms)):
        memory = np.zeros((4, 9))
        maximum = 0.
        grid_index = 0
        b = 0
        for segment in range(uniforms.shape[1]):
            start = int(uniforms[replicate, segment] * (len(vectors) - 3))
            for k in range(4):
                memory[b % 4] = vectors[start + k]
                b += 1
                if b >= 4:
                    maximum = max(maximum, joint_ratio(memory, core_cut, scale_cut))
                if grid_index < len(grid) and b == grid[grid_index]:
                    maxima[replicate, grid_index] = maximum
                    grid_index += 1
    return maxima


def historical_admission(historical, *, core_cut, scale_cut):
    h = valid_points(historical, historical=True)
    if not np.isfinite(core_cut) or core_cut <= 0 or not np.isfinite(scale_cut) or scale_cut <= 0:
        raise ValueError('Finite positive frozen cutoffs required')
    prefix_length = int(.6 * len(h))
    unavailable_cut = max(12. / core_cut, 4. / scale_cut) + 1e-9
    result = {'H_length': len(h), 'reference_prefix_length': prefix_length,
        'pseudo_blocks': (len(h) - prefix_length) // 16, 'reference_valid': False,
        'grid_blocks': GRID.tolist(), 'admission_cuts': [unavailable_cut] * len(GRID),
        'unsupported_cut': unavailable_cut, 'bootstrap_replicates': REPLICATES,
        'bootstrap_seed': DRAW_SEED, 'bootstrap_order_1based': ORDER_INDEX + 1,
        'source_scope': 'H only; shared draws; no series ID, online length, labels or tau',
        'probability_guarantee': None, 'support_reason': 'prefix_shorter_than512'}
    if prefix_length < 512:
        return result
    core = SafeState(h[:prefix_length], threshold=core_cut)
    scale = ScaleState(h[:prefix_length])
    result['core_reference_valid'] = bool(core.calibration['reference_valid'])
    result['scale_reference_valid'] = bool(scale.calibration['reference_valid'])
    if not core.calibration['reference_valid'] or not scale.calibration['reference_valid']:
        result['support_reason'] = 'invalid_historical_reference'
        return result
    c = core.replay(h[prefix_length:])
    s = scale.replay(h[prefix_length:])
    completed = s[:, 0] > .5
    vectors = np.column_stack((c[completed, 12:20], s[completed, 2]))
    if len(vectors) < 8:
        result['support_reason'] = 'fewer_than8_pseudo_blocks'
        return result
    maxima = bootstrap_maxima(vectors, shared_uniforms(), GRID, core_cut, scale_cut)
    cuts = np.maximum(1., np.sort(maxima, axis=0)[ORDER_INDEX])
    if not np.isfinite(cuts).all() or np.any(np.diff(cuts) < 0.):
        raise RuntimeError('Invalid H-null maximum table')
    result.update(reference_valid=True, admission_cuts=cuts.tolist(),
        support_reason='supported_through1024_blocks',
        pseudo_vector_min=np.min(vectors, axis=0).tolist(),
        pseudo_vector_max=np.max(vectors, axis=0).tolist(),
        bootstrap_maximum_range=[float(np.min(maxima)), float(np.max(maxima))])
    return result


def new_admission_args(calibration):
    cuts = np.asarray(calibration['admission_cuts'], dtype=np.float64)
    if cuts.shape != (9,) or not np.isfinite(cuts).all() or np.any(cuts < 1.) or np.any(np.diff(cuts) < 0.):
        raise ValueError('Nine finite monotone admission cuts >=1 required')
    params = np.array([calibration['reference_valid'], calibration['unsupported_cut']])
    # t, episodes, previous qualification, first, last, direction, channel, oldest, support.
    clock = np.array([0, 0, 0, -1, -1, 0, -1, -1, 0], dtype=np.int64)
    output = np.zeros(len(COLUMNS))
    output[[32, 33, 35, 36, 37, 38]] = -1.
    return cuts.copy(), params, clock, np.zeros(1), output


@njit(cache=True)
def admission_step(raw, args):
    cuts, params, clock, memory, output = args
    output[:30] = raw
    t = clock[0]
    b = int(raw[16])
    supported = params[0] > .5 and b <= 1024
    cut = params[1]
    if supported:
        j = 0
        while j < len(GRID) - 1 and b > GRID[j]:
            j += 1
        cut = cuts[j]
    output[40] = 0.
    output[44] = 0.
    if raw[15] > .5:
        qualified = supported and raw[18] > .5 and raw[2] > cut
        if qualified:
            if clock[2] == 0:
                clock[1] = min(clock[1] + 1, 7)
            if clock[3] < 0:
                clock[3] = t
            clock[4] = t
            clock[5] = int(raw[8])
            clock[6] = int(raw[22])
            clock[7] = int(raw[23])
            clock[8] = int(raw[24])
            strength = -math.expm1(-raw[2] / cut)
            memory[0] = max(memory[0], strength)
            output[40] = 1.
            output[44] = strength
        clock[2] = int(qualified)
    output[30] = memory[0]
    output[31] = clock[1]
    output[32] = t - clock[3] if clock[3] >= 0 else -1.
    output[33] = t - clock[4] if clock[4] >= 0 else -1.
    output[34] = clock[5]
    output[35] = clock[3]
    output[36] = clock[4]
    output[37] = clock[6]
    output[38] = clock[7]
    output[39] = clock[8]
    output[41] = clock[2]
    output[42] = cut
    output[43] = 1. if supported else 0.
    clock[0] += 1
    return output


@njit(cache=True)
def admission_replay(raw, args):
    output = np.empty((len(raw), len(COLUMNS)))
    for t in range(len(raw)):
        output[t] = admission_step(raw[t], args)
    return output


def qualified_features(values):
    from src.next7.features import from_state
    v = np.asarray(values, dtype=np.float64)
    if v.ndim not in (1, 2) or v.shape[-1] != len(COLUMNS):
        raise ValueError('G2 coordinate vector or matrix required')
    raw = v[..., :30].copy()
    raw[..., [3, 4, 5, 6, 8]] = v[..., [30, 31, 32, 33, 34]]
    return from_state(raw)


class QualifiedState:
    def __init__(self, historical, *, core_cut, scale_cut):
        self.raw = CombinedState(historical, core_cut=core_cut, scale_cut=scale_cut)
        self.calibration = historical_admission(historical, core_cut=core_cut, scale_cut=scale_cut)
        self.args = new_admission_args(self.calibration)

    def predict_one(self, point):
        value = float(point)
        if not np.isfinite(value) or abs(value) > FLOAT32_MAX:
            raise ValueError('Finite float32 current scalar required')
        return admission_step(self.raw.predict_one(value), self.args).copy()

    def replay(self, points):
        return admission_replay(self.raw.replay(valid_points(points)), self.args)

    @property
    def bounded_mutable_state_bytes(self):
        return self.raw.bounded_mutable_state_bytes + sum(a.nbytes for a in self.args[2:])
