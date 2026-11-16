"""Fused execution of unchanged G1/G2 kernels, with bounded summary export."""
import copy
import numpy as np
from numba import njit
from src.next6_extension.fresh_state import step as core_step
from src.next7.scale import step as scale_step, valid_points, FLOAT32_MAX
from src.next7.event import event_step
from src.next7.null_growth import QualifiedState, admission_step, GRID

HORIZONS = np.array([1024, 4096, 16384], dtype=np.int64)
SUMMARY_COLUMNS = ('max_raw_ratio', 'max_routed_ratio', 'first_core_time', 'first_raw_time',
    'first_qualified_time', 'core_latched', 'raw_latched', 'qualified_latched', 'core_ever',
    'raw_ever', 'qualified_ever', 'terminal_admission_cut', 'max_raw_exceeds_terminal_cut', 'current_state')


@njit(cache=True)
def fast_step(point, args):
    c = core_step(point, args[0])
    s = scale_step(point, args[1])
    raw = event_step(c, s, args[2])
    return admission_step(raw, args[3])


@njit(cache=True)
def fast_replay(points, args):
    output = np.empty((len(points), 45))
    for t in range(len(points)):
        output[t] = fast_step(float(points[t]), args)
    return output


@njit(cache=True)
def update_summary(t, out, summary, grid_maxima, running):
    running[0] = max(running[0], out[2])
    running[1] = max(running[1], out[2] / out[42])
    if running[2] < 0 and out[26] > 0:
        running[2] = t
    if running[3] < 0 and out[3] > 0:
        running[3] = t
    if running[4] < 0 and out[30] > 0:
        running[4] = t
    if out[15] > .5:
        b = int(out[16])
        for j in range(len(GRID)):
            if b == GRID[j]:
                grid_maxima[j, 0] = running[0]
                grid_maxima[j, 1] = running[1]
    for j in range(len(HORIZONS)):
        if t + 1 == HORIZONS[j]:
            summary[j, :5] = running
            summary[j, 5] = 1. if out[26] > 0 else 0.
            summary[j, 6] = 1. if out[3] > 0 else 0.
            summary[j, 7] = 1. if out[30] > 0 else 0.
            summary[j, 8:11] = np.array([out[26], out[3], out[30]])
            summary[j, 11] = out[42]
            summary[j, 12] = 1. if running[0] > out[42] else 0.
            summary[j, 13] = out[0]


@njit(cache=True)
def summarize_points(points, args):
    summary = np.zeros((3, len(SUMMARY_COLUMNS)))
    maxima = np.zeros((9, 2))
    running = np.array([0., 0., -1., -1., -1.])
    for t in range(len(points)):
        out = fast_step(float(points[t]), args)
        update_summary(t, out, summary, maxima, running)
    return summary, maxima


class FastQualifiedState:
    def __init__(self, historical, *, core_cut, scale_cut):
        self.reference = QualifiedState(historical, core_cut=core_cut, scale_cut=scale_cut)
        self.args = (self.reference.raw.core.args, self.reference.raw.scale.args,
            self.reference.raw.args, self.reference.args)
        self.calibration = self.reference.calibration

    def predict_one(self, point):
        value = float(point)
        if not np.isfinite(value) or abs(value) > FLOAT32_MAX:
            raise ValueError('Finite float32 current scalar required')
        return fast_step(float(np.float32(value)), self.args).copy()

    def replay(self, points):
        return fast_replay(valid_points(points), self.args)

    def summarize(self, points):
        p = valid_points(points)
        if len(p) != 16384 or self.args[3][2][0] != 0:
            raise ValueError('Summary requires a fresh state and exactly16384 points')
        return summarize_points(p, self.args)

    def clone(self):
        return copy.deepcopy(self)

    @property
    def bounded_mutable_state_bytes(self):
        return self.reference.bounded_mutable_state_bytes
