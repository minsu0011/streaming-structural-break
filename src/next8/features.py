"""H-prefix fitted reference bank, with one unchanged calibration/online engine.

These are NEW scientific coordinates, not replacements for the frozen BL80
features. No target, dataset id, tau or eventual online length enters this API.
"""
from dataclasses import dataclass
import copy
import hashlib
from pathlib import Path
import numpy as np
from numba import njit
from src.next5.reference_features import ReferenceState, feature_names
from src.next2.dependence_runtime import step as dependence_step

KINDS = ('B2_augmented_J', 'C205', 'D2_expert')
SETTINGS = {k: {'version': 'next5_reference_features_v1', 'kind': k} for k in KINDS}
NAMES = tuple(n for k in KINDS for n in feature_names(SETTINGS[k]))
N_FEATURES = len(NAMES)
HORIZONS = (1, 4, 16)
LAGS = (1, 4, 16)
UNKNOWN = 3
NORMALIZATION = 'H60_fit_same_engine_H40_cal_continue_online_IQR1349_floor005_clip12_v1'
# Fixed from formulas and prior history BEFORE any NEXT8 label association.
DETAILED = (0, 6, 9, 11, 15, 21, 51, 64, 77, 81, 93, 105)
CONTEXT_NAMES = ('H_log_length', 'H_log_MAD', 'H_lag1', 'H_lag4', 'H_tail3', 'H_abs_skew')


@njit(cache=False)
def _replay_c(points, args):
    out = np.empty((len(points), 24), dtype=np.float32)
    for t in range(len(points)):
        out[t] = dependence_step(float(points[t]), *args)
    return out


def family(j):
    n = NAMES[int(j)]
    if j >= 117:
        return 'conditional_scale_likelihood'
    if j >= 93:
        return 'conditional_dependence_' + n.split('dependence_')[1].split('_')[0]
    if j >= 77:
        return 'conditional_energy' if 'conditional_energy' in n else 'raw_energy'
    if j >= 64:
        return 'variance_forecast'
    if j >= 51:
        return 'raw_energy'
    if any(s in n for s in ('lag', 'ar_')):
        return 'innovation_dependence'
    if any(s in n for s in ('tail', 'sign', 'quantile')):
        return 'innovation_distribution'
    if any(s in n for s in ('second', 'variance', 'moment', 'abs_')):
        return 'innovation_scale'
    return 'innovation_location'


def mechanical_group(j):
    """Conservative shared primitive grouping, not a causal independence claim."""
    if j >= 117:
        return 'conditional_z_squared_windows'
    if j >= 93:
        return family(j)
    if j >= 85:
        return 'conditional_z_squared_windows'
    if 51 <= j < 64 or 77 <= j < 85:
        return 'raw_residual_energy'
    if 64 <= j < 77:
        return 'ARCH_variance_forecast'
    return family(j)


def relation_type(source, destination):
    if source == destination:
        return 'SELF_HISTORY'
    if mechanical_group(source) == mechanical_group(destination):
        return 'MECHANICAL_DERIVED'
    return 'CROSS_PREDICTIVE_CANDIDATE_SHARED_RAW'


def mechanical_controls(j):
    group = mechanical_group(j)
    return tuple(sorted({j, 0, 2, 51, 64} | {
        k for k in range(N_FEATURES) if mechanical_group(k) == group}))


def value_sign(value):
    return np.sign(np.asarray(value)).astype(np.int8)


def movement_direction(change, epsilon):
    change, epsilon = np.asarray(change), np.asarray(epsilon)
    return np.where(change > epsilon, 2, np.where(change < -epsilon, 0, 1)).astype(np.int8)


def categorize(value, low, high, valid=True):
    value = np.asarray(value)
    return np.where(~np.isfinite(value) | ~np.asarray(valid, dtype=bool), UNKNOWN,
                    np.where(value < low, 0, np.where(value > high, 2, 1))).astype(np.int8)


def h_context(historical):
    h = np.asarray(historical, dtype=np.float64)
    if not len(h):
        return np.zeros(len(CONTEXT_NAMES), dtype=np.float32)
    center = np.median(h)
    scale = max(float(1.4826 * np.median(np.abs(h-center))), 1e-6)
    z = np.clip((h-center)/scale, -12, 12)
    corr = []
    for lag in (1, 4):
        a, b = z[lag:], z[:-lag]
        corr.append(float(np.mean((a-a.mean())*(b-b.mean())) /
                          max(float(a.std()*b.std()), 1e-6)) if len(a) else 0.)
    return np.asarray([np.log1p(len(h)), np.log(scale), *corr,
                       np.mean(np.abs(z)>3), abs(np.mean(z**3))], dtype=np.float32)


class FeatureState:
    """Streaming bank, four observable states, and H-only change thresholds.

    Short H (<160) abstains deterministically: zero coordinates, UNKNOWN state.
    All buffers (including hidden AR/ARCH/EWMA/prefix memory) survive restore.
    State labels alone are therefore explicitly NOT a complete Markov state.
    """
    def __init__(self, historical):
        h = np.asarray(historical, dtype=np.float32)
        if h.ndim != 1 or not np.isfinite(h).all():
            raise ValueError('Finite one-dimensional H required')
        self.context = h_context(h)
        self.fit_length = int(.6*len(h))
        self.valid = np.zeros(N_FEATURES, dtype=bool)
        self.center = np.zeros(N_FEATURES)
        self.spread = np.ones(N_FEATURES)
        self.low = np.full(N_FEATURES, -1.)
        self.high = np.full(N_FEATURES, 1.)
        self.epsilon = np.full((3, N_FEATURES), .02)
        self.history = np.zeros((16, N_FEATURES), dtype=np.float32)
        self.counter = 0
        self.engines = []
        self.calibration_clip_count = np.zeros(N_FEATURES, dtype=np.int64)
        self.last_raw = np.zeros(N_FEATURES, dtype=np.float32)
        self.last_unclipped = np.zeros(N_FEATURES)
        if len(h) < 160:
            self.fit_length = len(h)
            self.normalizer_fit_sha256 = hashlib.sha256(h.tobytes()).hexdigest()
            self.calibration_reference_sha256 = self.normalizer_fit_sha256
            self.online_reference_sha256 = self.normalizer_fit_sha256
            return
        reference = h[:self.fit_length]
        self.normalizer_fit_sha256 = hashlib.sha256(reference.tobytes()).hexdigest()
        self.calibration_reference_sha256 = self.normalizer_fit_sha256
        self.engines = [ReferenceState(reference, SETTINGS[k]) for k in KINDS]
        # Exactly the SAME engine instances continue after this replay.
        raw = self._raw_replay(h[self.fit_length:])
        burn = min(32, len(raw)//4)
        calibration = raw[burn:].astype(np.float64)
        self.center = np.median(calibration, axis=0)
        q25, q75 = np.quantile(calibration, [.25, .75], axis=0)
        self.spread = np.maximum((q75-q25)/1.349, .05)
        normalized = (raw.astype(np.float64)-self.center)/self.spread
        self.calibration_clip_count = (np.abs(normalized[burn:])>=12).sum(axis=0)
        normalized = np.clip(normalized, -12, 12).astype(np.float32)
        self.low, self.high = np.quantile(normalized[burn:], [.2, .8], axis=0)
        self.valid = (self.high-self.low) > 1e-6
        for k, horizon in enumerate(HORIZONS):
            differences = np.abs(normalized[burn+horizon:]-normalized[burn:-horizon])
            self.epsilon[k] = np.maximum(.1*np.median(differences, axis=0), .02)
        self.history[:] = normalized[-16:]
        self.online_reference_sha256 = self.normalizer_fit_sha256

    def _raw_replay(self, points):
        pieces = [self.engines[0].state.replay(points),
                  _replay_c(points, self.engines[1].state.args),
                  self.engines[2].state.replay(points)]
        return np.concatenate(pieces, axis=1)

    def lag(self, lag):
        if lag not in LAGS:
            raise ValueError('Only frozen lags 1,4,16')
        return self.history[(self.counter-lag) % 16].copy()

    def update(self, point):
        if not np.isfinite(point):
            raise ValueError('Finite scalar online observation required')
        before = self.lag(1)
        if self.engines:
            self.last_raw = np.concatenate([e.update(float(point)).copy() for e in self.engines])
            self.last_unclipped = (self.last_raw.astype(np.float64)-self.center)/self.spread
            value = np.clip(self.last_unclipped, -12, 12).astype(np.float32)
        else:
            value = np.zeros(N_FEATURES, dtype=np.float32)
        # pre_state is known BEFORE current F is consumed.
        pre_state = categorize(before, self.low, self.high, self.valid)
        state = categorize(value, self.low, self.high, self.valid)
        movement = movement_direction(value-before, self.epsilon[0])
        self.history[self.counter % 16] = value
        self.counter += 1
        return value, state, pre_state, movement

    def replay(self, points):
        p = np.asarray(points, dtype=np.float32)
        if p.ndim != 1 or not np.isfinite(p).all():
            raise ValueError('Finite one-dimensional observed prefix required')
        # Batched raw kernels and scalar output policy have identical order.
        if not self.engines:
            values = np.zeros((len(p), N_FEATURES), dtype=np.float32)
        else:
            raw = self._raw_replay(p)
            values = np.clip((raw.astype(np.float64)-self.center)/self.spread,
                             -12, 12).astype(np.float32)
            if len(p):
                self.last_raw = raw[-1].copy()
                self.last_unclipped = (self.last_raw.astype(np.float64)-self.center)/self.spread
        states = categorize(values, self.low, self.high, self.valid)
        for row in values:
            self.history[self.counter % 16] = row
            self.counter += 1
        return values, states

    def snapshot(self):
        return copy.deepcopy(self)

    @classmethod
    def restore(cls, snapshot):
        if not isinstance(snapshot, cls):
            raise TypeError('Exact FeatureState snapshot required')
        return copy.deepcopy(snapshot)


@dataclass(frozen=True)
class FutureTargets:
    """Auxiliary supervision only. Never accepted by inference feature APIs."""
    horizon: int
    value: np.ndarray
    delta: np.ndarray
    direction: np.ndarray
    state: np.ndarray
    reversal: np.ndarray
    available: np.ndarray


def future_targets(values, ids, times, epsilon, low, high, valid, horizon, cutoff=None):
    if horizon not in HORIZONS:
        raise ValueError('Unregistered forecast horizon')
    n = len(values)
    available = np.zeros(n, dtype=bool)
    available[:-horizon] = ((ids[:-horizon] == ids[horizon:]) &
                           (times[horizon:] == times[:-horizon]+horizon))
    if cutoff is not None:
        available &= times+horizon <= cutoff
    following = np.full_like(values, np.nan, dtype=np.float32)
    rows = np.flatnonzero(available)
    following[rows] = values[rows+horizon]
    delta = following-values
    direction = movement_direction(delta, epsilon)
    state = categorize(following, low, high, valid)
    reversal = (value_sign(following)*value_sign(values) < 0).astype(np.uint8)
    return FutureTargets(horizon, following, delta, direction, state, reversal, available)


def source_hash():
    from src.next5.reference_features import implementation_hash
    return hashlib.sha256(Path(__file__).read_bytes() + ''.join(
        implementation_hash(SETTINGS[k]) for k in KINDS).encode()).hexdigest()
