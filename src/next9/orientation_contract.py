"""TRAIN-only common-current-F support and finite movement pair contract."""
from dataclasses import dataclass, asdict
import numpy as np

PAIR_MENU = (
    ('S1_DIRECTION', 2, 0),
    ('S2_DIRECTION_RUN', 6, 0),
    ('S2_DIRECTION_RUN', 7, 1),
    ('S2_DIRECTION_RUN', 8, 2),
    ('S3_DIRECTION_SLOPE', 4, 0),
    ('S3_DIRECTION_SLOPE', 5, 1),
    ('S4_REVERSAL', 0, 1),
    ('S4_REVERSAL', 3, 2),
)


def support(groups, labels, mask):
    groups, labels = np.asarray(groups), np.asarray(labels)
    return dict(groups=int(len(np.unique(groups[mask]))),
                positive_groups=int(len(np.unique(groups[mask & (labels == 1)]))),
                negative_groups=int(len(np.unique(groups[mask & (labels == 0)]))))


def sufficient(s):
    return s['groups'] >= 64 and s['positive_groups'] >= 32 and s['negative_groups'] >= 32


@dataclass(frozen=True)
class CommonRange:
    family: str
    first: int
    second: int
    low: float
    high: float
    status: str
    training_support: dict
    provenance: dict

    def mask(self, values, states):
        values, states = np.asarray(values), np.asarray(states)
        return (np.isfinite(values) & (values >= self.low) & (values <= self.high)
                & np.isin(states, [self.first, self.second]))

    def record(self):
        return asdict(self)


def fit_common_range(values, states, groups, labels, scope, pair):
    """Labels assess support only; neither quantile nor pair is label selected."""
    if pair not in PAIR_MENU:
        raise ValueError('Unregistered state pair')
    values, states, groups, labels = map(np.asarray, (values, states, groups, labels))
    if not (values.ndim == states.ndim == groups.ndim == labels.ndim == 1
            and len(values) == len(states) == len(groups) == len(labels)):
        raise ValueError('Aligned one-dimensional observations required')
    scope.admit_training(groups)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Binary labels required')
    family, a, b = pair
    ranges = []
    for state in (a, b):
        v = values[(states == state) & np.isfinite(values)]
        ranges.append(np.quantile(v, [.1, .9]) if len(v) else [0., 0.])
    low, high = float(max(v[0] for v in ranges)), float(min(v[1] for v in ranges))
    width_ok = high - low > 1e-6
    stats = {}
    for state in (a, b):
        all_state = (states == state) & np.isfinite(values)
        mask = all_state & (values >= low) & (values <= high)
        s = support(groups, labels, mask)
        s['retained_fraction'] = float(mask.sum() / max(1, all_state.sum()))
        stats[str(state)] = s
    ok = width_ok and all(sufficient(s) and s['retained_fraction'] >= .1 for s in stats.values())
    return CommonRange(family, a, b, low, high, 'SUPPORTED' if ok else 'INSUFFICIENT_COMMON_RANGE', stats, scope.provenance())


def fit_run_fallback(states, groups, scope):
    """Freeze sparse RUN levels as distinct parent codes10..12; UNKNOWN9."""
    states, groups = np.asarray(states), np.asarray(groups)
    scope.admit_training(groups)
    mapping = np.arange(10, dtype=np.int16)
    for s in range(9):
        if len(np.unique(groups[states == s])) < 64:
            mapping[s] = 10 + s // 3
    return mapping


def fixed_nonlinear(values):
    f = np.asarray(values, dtype=float)
    if f.ndim != 1 or not np.isfinite(f).all():
        raise ValueError('Finite current F required')
    return np.column_stack([f, np.abs(f), np.maximum(f, 0), np.minimum(f, 0), f*f])


def one_hot(states, count):
    s = np.asarray(states)
    if s.ndim != 1 or not np.isin(s, np.arange(count)).all():
        raise ValueError('State outside frozen vocabulary')
    return np.eye(count, dtype=float)[s.astype(int)]


def orientation_design(values, movement, run, reversal, level, model):
    """All interactions use fixed basis and pre-current state, never labels."""
    f = np.asarray(values, dtype=float)
    nl = fixed_nonlinear(f)
    if model == 'O0':
        return f[:, None]
    if model in ('O1', 'F_ONLY'):
        return nl
    move = one_hot(movement, 4)[:, :3]
    rev = one_hot(reversal, 5)[:, :4]
    if model == 'STATE_ONLY':
        return np.column_stack([move, rev])
    if model == 'O2_LEVEL':
        state = one_hot(level, 4)[:, :3]
    elif model == 'O3_MOVE':
        state = move
    elif model == 'O4_MOVE_RUN':
        state = one_hot(run, 13)[:, [*range(9), 10, 11, 12]]
    elif model in ('O5_MOVE_REV', 'O6_TABULAR'):
        state = np.column_stack([move, rev])
    else:
        raise ValueError('Unregistered comparator')
    # O6 receives exactly the same numeric matrix, including interaction columns.
    return np.column_stack([nl, state, state * f[:, None]])
