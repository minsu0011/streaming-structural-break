"""Streaming execution adapter for frozen NEXT7 corrections, without fitting."""
import copy
from functools import lru_cache
import numpy as np
from numba import njit, typeof
from src.next5.runtime import SharedResearchBaseline, baseline_step, resident_class
from src.next.fused_runtime import unique_array_bytes
from src.next6_extension.fresh_state import step as core_step
from src.next7.scale import step as scale_step, valid_points, FLOAT32_MAX
from src.next7.event import CombinedState, event_step
from src.next7.null_growth import QualifiedState, admission_step, new_admission_args
from src.next7.calibration import cutoff
from src.models.compact_trees import tree_margin


@njit(cache=True)
def coordinates(raw, admission, qualified, out):
    out[0] = raw[1]
    out[1] = admission[30] if qualified else raw[3]
    out[2] = (admission[31] if qualified else raw[4]) / 7.
    first = admission[32] if qualified else raw[5]
    last = admission[33] if qualified else raw[6]
    out[3] = np.log1p(min(max(first, 0.), 16384.))
    out[4] = np.log1p(min(max(last, 0.), 16384.))
    out[5] = raw[7]
    out[6] = admission[34] if qualified else raw[8]
    out[7] = raw[9]
    out[8] = raw[10]
    out[9] = raw[11]
    return out


@njit(cache=True)
def correction_value(baseline, features, args):
    kind, columns, center, spread, beta, tree, work = args
    if kind == 0:  # Frozen M_EVENT / G2_EVENT static recipe.
        return .9 * baseline + .1 * features[1]
    if kind == 1:
        b = min(max(baseline, 1e-6), 1. - 1e-6)
        adjustment = 0.
        for i in range(len(columns)):
            z = min(max((features[columns[i]] - center[i]) / spread[i], -8.), 8.)
            adjustment += z * beta[i]
        eta = np.log(b) - np.log1p(-b) + adjustment
        if eta >= 0.:
            return 1. / (1. + np.exp(-eta))
        e = np.exp(eta)
        return e / (1. + e)
    for i in range(len(columns)):
        work[i] = features[columns[i]]
    residual = tree_margin(work, *tree)
    return min(max(baseline + residual, 1e-6), 1. - 1e-6)


@njit(cache=False)
def kernel(point, args):
    baseline_args, state_args, head_args, qualified, features, output = args
    baseline = baseline_step(point, baseline_args)
    c = core_step(point, state_args[0])
    s = scale_step(point, state_args[1])
    raw = event_step(c, s, state_args[2])
    admission = state_args[3][-1]
    if qualified:
        admission = admission_step(raw, state_args[3])
    coordinates(raw, admission, qualified, features)
    output[0] = baseline
    output[1] = correction_value(baseline, features, head_args)
    return output[1]


@njit(cache=False)
def replay_kernel(points, args):
    output = np.empty((len(points), 2))
    for t in range(len(points)):
        kernel(float(points[t]), args)
        output[t] = args[-1]
    return output


def dummy_tree():
    # The static/logistic paths do not evaluate this type-compatible leaf.
    return (np.array([0], dtype=np.int32), np.array([-1], dtype=np.int32), np.zeros(1),
        np.array([-1], dtype=np.int32), np.array([-1], dtype=np.int32), np.zeros(1), np.zeros(1, dtype=bool), 0.)


def head_arguments(model):
    if model is None:
        return (0, np.array([1], dtype=np.int64), np.zeros(1), np.ones(1), np.zeros(1), dummy_tree(), np.empty(1, dtype=np.float32))
    model.validate()
    columns = np.asarray(model.columns, dtype=np.int64)
    if model.tree is None:
        return (1, columns, model.center, model.spread, model.beta, dummy_tree(), np.empty(len(columns), dtype=np.float32))
    return (2, columns, model.center, model.spread, np.zeros(len(columns)), model.tree.args(), np.empty(len(columns), dtype=np.float32))


class Prepared:
    def __init__(self, j102, c205, correction, calibration, *, qualified=False, resident=True):
        self.baseline = SharedResearchBaseline(j102, c205)
        self.correction = correction
        self.calibration = copy.deepcopy(calibration)
        self.qualified = bool(qualified)
        self.resident = bool(resident)
        self.head = head_arguments(correction)

    def new_state(self, historical):
        return IntegratedState(self, historical)


class IntegratedState:
    def __init__(self, prepared, historical):
        h = valid_points(historical, historical=True)
        self.prepared = prepared
        cut = cutoff(prepared.calibration, len(h))
        if prepared.qualified:
            state = QualifiedState(h, core_cut=prepared.calibration['core_cut'], scale_cut=cut)
            raw, admission = state.raw, state.args
            self.admission_calibration = state.calibration
        else:
            raw = CombinedState(h, core_cut=prepared.calibration['core_cut'], scale_cut=cut)
            admission = new_admission_args({'admission_cuts': [1.] * 9, 'reference_valid': False, 'unsupported_cut': 10.})
            self.admission_calibration = None
        state_args = (raw.core.args, raw.scale.args, raw.args, admission)
        head = list(prepared.head)
        head[-1] = np.empty_like(head[-1])  # Per-state mutable tree input, never shared across streams.
        self.args = (prepared.baseline.arguments(h), state_args, tuple(head), prepared.qualified, np.empty(10), np.empty(2))
        self.added_bounded_mutable_bytes = raw.bounded_mutable_state_bytes + (sum(a.nbytes for a in admission[2:]) if prepared.qualified else 0)
        self.added_bounded_mutable_bytes += self.args[-2].nbytes + self.args[-1].nbytes + head[-1].nbytes
        self.resident = resident_class(kernel, typeof(self.args))(self.args) if prepared.resident else None

    def predict_one(self, point):
        value = float(point)
        if not np.isfinite(value) or abs(value) > FLOAT32_MAX:
            raise ValueError('Finite float32 current scalar required')
        value = float(np.float32(value))
        prediction = float(self.resident.predict_one(value) if self.resident is not None else kernel(value, self.args))
        if not np.isfinite(prediction) or not 0. <= prediction <= 1.:
            raise RuntimeError('Finite bounded detector score required')
        return prediction

    def replay(self, points):
        return replay_kernel(valid_points(points), self.args)

    def snapshot(self):
        return self.args[1][3][-1].copy() if self.prepared.qualified else self.args[1][2][-1].copy()

    @property
    def instance_array_bytes_excluding_models(self):
        return unique_array_bytes((self.args[0][:-1], self.args[1], self.args[-2:], self.args[2][-1]))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('resident', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.resident = resident_class(kernel, typeof(self.args))(self.args) if self.prepared.resident else None

    def clone(self):
        return copy.deepcopy(self)
