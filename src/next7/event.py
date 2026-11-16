"""Confirmation-only event memory, independent of recoverable current state."""
import math
import numpy as np
from numba import njit
from src.next7.scale import ScaleState, valid_points, FLOAT32_MAX
from src.next6_extension.fresh_runtime import SafeState

COLUMNS = ('current_state', 'current_score', 'confirmation_ratio', 'ever_score',
    'episodes_capped', 'time_since_first', 'time_since_last', 'recovery', 'confirmed_direction',
    'signed_scale', 'scale_current_score', 'scale_confirmation_ratio', 'core_state',
    'core_confirmation_ratio', 'raw_shock', 'block_complete', 'completed_blocks',
    'confirmed_block_status', 'fresh_confirmation', 'first_shock_time', 'first_confirmation_time',
    'first_latch_time', 'confirmation_channel', 'oldest_confirmation_block', 'same_direction_blocks',
    'reference_valid', 'core_ever_score', 'scale_ever_score', 'core_fresh_confirmation',
    'scale_fresh_confirmation')


@njit(cache=True)
def event_step(core, scale, args):
    # params: core cut, scale cut, scale index, rule index, include core, include scale, valid.
    # clock: t, current state, recovery blocks left, episodes, previous confirmation,
    # first shock, first confirm, last confirm, first latch, direction, winning channel.
    params, clock, memories, output = args
    t = clock[0]
    complete = scale[0] > .5
    output[18] = output[28] = output[29] = 0.
    output[14] = core[3]
    if core[3] > .5 and clock[5] < 0:
        clock[5] = t
    if complete:
        channel = int(params[2])
        rule = int(params[3])
        at = 4 + (channel * 3 + rule) * 3
        core_ratio = core[1] / params[0] if params[4] > .5 else 0.
        scale_ratio = scale[at] / params[1] if params[5] > .5 else 0.
        if params[6] < .5:
            core_ratio = scale_ratio = 0.
        strength = max(core_ratio, scale_ratio)
        core_confirm = core_ratio > 1. and core[11] >= 4.
        scale_confirm = scale_ratio > 1. and scale[1] >= 4.
        confirmed = core_confirm or scale_confirm
        output[1] = -math.expm1(-strength)
        output[2] = strength
        output[9] = scale[2 + channel]
        output[10] = -math.expm1(-abs(scale[2 + channel]))
        output[11] = scale_ratio
        output[13] = core_ratio
        output[17] = 1. if confirmed else 0.
        output[18] = 1. if confirmed else 0.
        output[28] = 1. if core_confirm else 0.
        output[29] = 1. if scale_confirm else 0.
        if core_confirm:
            memories[1] = max(memories[1], -math.expm1(-core_ratio))
        if scale_confirm:
            memories[2] = max(memories[2], -math.expm1(-scale_ratio))
        if confirmed:
            if clock[4] == 0:
                clock[3] = min(clock[3] + 1, 7)
            if clock[6] < 0:
                clock[6] = t
                clock[8] = t
            clock[7] = t
            memories[0] = max(memories[0], -math.expm1(-strength))
            if scale_confirm and scale_ratio >= core_ratio:
                clock[9] = int(scale[at + 1])
                clock[10] = 8 + channel
                output[24] = scale[at + 2]
            else:
                j = int(core[6])
                clock[9] = 1 if core[12 + j] > 0 else -1
                clock[10] = j
                output[24] = 4.
            output[23] = scale[1] - 4.
            clock[1] = 4
            clock[2] = 0
        elif params[6] < .5:
            clock[1] = 6
        elif clock[1] == 4:
            clock[1] = 5
            clock[2] = 2
        elif clock[1] == 5 and clock[2] > 1:
            clock[2] -= 1
        else:
            clock[2] = 0
            clock[1] = 2 if core[0] == 2 else 3 if max(core[2], abs(scale[2 + channel])) >= 2. else 1
        clock[4] = int(confirmed)
    elif params[6] < .5:
        clock[1] = 6
    elif clock[1] not in (0, 4, 5):
        if core[0] == 2:
            clock[1] = 2
        elif clock[1] == 2:
            clock[1] = 3 if max(core[2], abs(output[9])) >= 2. else 1
    output[0] = clock[1]
    output[3] = memories[0]
    output[4] = clock[3]
    output[5] = t - clock[6] if clock[6] >= 0 else -1.
    output[6] = t - clock[7] if clock[7] >= 0 else -1.
    output[7] = 1. if clock[1] == 5 else 0.
    output[8] = clock[9]
    output[12] = core[0]
    output[15] = scale[0]
    output[16] = scale[1]
    output[19] = clock[5]
    output[20] = clock[6]
    output[21] = clock[8]
    output[22] = clock[10]
    output[25] = params[6]
    output[26] = memories[1]
    output[27] = memories[2]
    clock[0] += 1
    return output


@njit(cache=True)
def event_replay(core, scale, args):
    output = np.empty((len(core), len(COLUMNS)))
    for t in range(len(core)):
        output[t] = event_step(core[t], scale[t], args)
    return output


def new_event_args(core_cut, scale_cut, scale_index, rule, include_core, include_scale, valid):
    if not np.isfinite(core_cut) or core_cut <= 0 or not np.isfinite(scale_cut) or scale_cut <= 0:
        raise ValueError('Finite positive frozen cutoffs required')
    if scale_index not in (0, 1) or rule not in (0, 1, 2):
        raise ValueError('Preregistered scale/rule index required')
    params = np.array([core_cut, scale_cut, scale_index, rule, include_core, include_scale, valid], dtype=np.float64)
    clock = np.array([0, 0 if valid else 6, 0, 0, 0, -1, -1, -1, -1, 0, -1], dtype=np.int64)
    output = np.zeros(len(COLUMNS))
    output[[5, 6, 19, 20, 21, 22, 23]] = -1.
    output[0] = clock[1]
    output[25] = valid
    return params, clock, np.zeros(3), output


class CombinedState:
    def __init__(self, historical, *, core_cut, scale_cut, scale_index=0, rule=1,
                 include_core=True, include_scale=True):
        self.scale = ScaleState(historical)
        self.core = SafeState(historical, threshold=core_cut)
        valid = self.scale.calibration['reference_valid'] and self.core.calibration['reference_valid']
        self.args = new_event_args(core_cut, scale_cut, scale_index, rule, include_core, include_scale, valid)
        self.calibration = {'scale': self.scale.calibration, 'core': self.core.calibration,
            'core_cut': core_cut, 'scale_cut': scale_cut, 'scale_index': scale_index,
            'rule': rule, 'include_core': include_core, 'include_scale': include_scale}

    def predict_one(self, point):
        value = float(point)
        if not np.isfinite(value) or abs(value) > FLOAT32_MAX:
            raise ValueError('Finite float32 current scalar required')
        c = self.core.predict_one(value)
        s = self.scale.predict_one(value)
        return event_step(c, s, self.args).copy()

    def replay(self, points):
        p = valid_points(points)
        c = self.core.replay(p)
        s = self.scale.replay(p)
        return event_replay(c, s, self.args)

    @property
    def bounded_mutable_state_bytes(self):
        return (self.core.bounded_mutable_state_bytes + self.scale.bounded_mutable_state_bytes
                + sum(a.nbytes for a in self.args[1:]))
