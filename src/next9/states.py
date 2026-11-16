"""Observable movement states emitted before consuming the current feature."""
from dataclasses import dataclass
import copy
import numpy as np

DOWN, FLAT, UP, UNKNOWN = 0, 1, 2, 3
MEMORY = 16
LAGS = (1, 2, 4, 8, 16)


def directions(delta, epsilon):
    return np.where(delta > epsilon, UP, np.where(delta < -epsilon, DOWN, FLAT)).astype(np.int8)


def describe_history(history, epsilon, available=True):
    """history[...,16] is F[t-16:t], never F[t] or any suffix."""
    h = np.asarray(history, dtype=np.float64)
    if h.ndim < 1 or h.shape[-1] != MEMORY or not np.isfinite(h).all():
        raise ValueError('Exactly sixteen finite past/H-seed feature values required')
    e = np.broadcast_to(np.asarray(epsilon, dtype=float), h.shape[:-1])
    if not np.isfinite(e).all() or np.any(e < float(np.float32(.02))):
        raise ValueError('Frozen H-only epsilon >= .02 required')
    valid = np.broadcast_to(np.asarray(available, dtype=bool), h.shape[:-1])
    movement = directions(np.diff(h, axis=-1), e[..., None])
    last = movement[..., -1]
    run = np.ones(h.shape[:-1], dtype=np.int16)
    same = np.ones(h.shape[:-1], dtype=bool)
    for k in range(2, MEMORY):
        same &= movement[..., -k] == last
        run += same
    # A reversal is an opposite non-FLAT movement, bridging intervening FLATs.
    # Only this observed finite history is used; >=15 is explicitly censored.
    prior_nonflat = np.full(h.shape[:-1], UNKNOWN, dtype=np.int8)
    reversal_age = np.full(h.shape[:-1], MEMORY - 1, dtype=np.int16)
    last_reversal_to = np.full(h.shape[:-1], UNKNOWN, dtype=np.int8)
    transitions = np.zeros(h.shape[:-1], dtype=np.int16)
    for k in range(MEMORY - 1):
        current = movement[..., k]
        nonflat = current != FLAT
        reversal = nonflat & (prior_nonflat != UNKNOWN) & (prior_nonflat != current)
        reversal_age = np.where(reversal, MEMORY - 2 - k, reversal_age)
        last_reversal_to = np.where(reversal, current, last_reversal_to)
        prior_nonflat = np.where(nonflat, current, prior_nonflat)
        if k:
            transitions += current != movement[..., k - 1]
    slope = np.sum(h[..., -8:] * np.arange(-3.5, 4., 1.), axis=-1) / 42.
    strength = (np.abs(slope) > e).astype(np.int8)
    bucket = np.where(run == 1, 0, np.where(run <= 3, 1, 2))
    s1 = np.where(valid, last, UNKNOWN).astype(np.int8)
    s2 = np.where(valid, last * 3 + bucket, 9).astype(np.int8)
    s3 = np.where(valid, last * 2 + strength, 6).astype(np.int8)
    recent = (reversal_age <= 3) & (last_reversal_to == last)
    # 0 continue UP, 1 continue DOWN, 2 UP->DOWN, 3 DOWN->UP, 4 FLAT/UNKNOWN.
    s4 = np.where(last == UP, np.where(recent, 3, 0),
                  np.where(last == DOWN, np.where(recent, 2, 1), 4))
    s4 = np.where(valid, s4, 4).astype(np.int8)
    return dict(S1_DIRECTION=s1, S2_DIRECTION_RUN=s2, S3_DIRECTION_SLOPE=s3,
                S4_REVERSAL=s4, direction_run_capped15=run, slope8=slope,
                delta_prev=h[..., -1] - h[..., -2], reversal_age_capped15=reversal_age,
                recent_transition_count15=transitions,
                lags=np.stack([h[..., -lag] for lag in LAGS], axis=-1),
                history_available=valid)


@dataclass
class MovementState:
    history: np.ndarray
    epsilon_H: float
    available: bool = True

    def __post_init__(self):
        self.history = np.asarray(self.history, dtype=np.float32).copy()
        if self.history.shape != (MEMORY,):
            raise ValueError('Sixteen H seed values required')
        self.peek()

    def peek(self):
        return describe_history(self.history, self.epsilon_H, self.available)

    def observe(self, current_F):
        if not np.isfinite(current_F):
            raise ValueError('Finite current scalar feature required')
        self.history[:-1] = self.history[1:]
        self.history[-1] = np.float32(current_F)

    def snapshot(self):
        return copy.deepcopy(self)

    @classmethod
    def restore(cls, snapshot):
        if not isinstance(snapshot, cls):
            raise TypeError('Exact movement state snapshot required')
        return copy.deepcopy(snapshot)

    def replay(self, values):
        records = []
        for value in values:
            records.append(self.peek())
            self.observe(value)
        return records
