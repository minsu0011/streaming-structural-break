"""Causal H-calibrated categorical transition evidence with fixed memory.

Every lag is a separate conditional multinomial model. Both null and changed
models integrate transition probabilities; their prior means are the same
historical posterior mean. No claim of an anytime-valid e-process is made.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit


@dataclass(frozen=True)
class MarkovConfig:
    lags: tuple = (1, 2, 4, 8)
    prior_strength: float = 16.
    max_age: int = 512


STATS = ('surprise', 'glr_max', 'glr_logmean', 'glr_logage', 'bf_max',
         'bf_logmixture', 'bf_logage', 'bf_memory', 'bf_decay', 'stay_ewma32')


def _config(settings):
    settings = dict(settings)
    if 'lags' in settings:
        settings['lags'] = tuple(settings['lags'])
    config = MarkovConfig(**settings)
    if config.lags not in ((1,), (1, 2, 4, 8)):
        raise ValueError('Unregistered transition lags')
    if config.prior_strength not in (4., 16.) or config.max_age not in (128, 512):
        raise ValueError('Unregistered transition prior or maximum age')
    return config


@njit(cache=True)
def _logadd(left, right):
    maximum = max(left, right)
    return maximum + np.log(np.exp(left - maximum) + np.exp(right - maximum))


@njit(cache=True)
def transition_step(point, args):
    (edges, lags, ages, widths, logprob, cell_lookup, row_lookup, nlogn,
     ring, cells, totals, glr, bayes, memory, counter, output) = args
    current = np.searchsorted(edges, point, side='right')
    t = counter[0]
    ring[t % len(ring)] = current
    alpha = 1. - np.exp(-np.log(2.) / 32.)
    for li in range(len(lags)):
        previous = ring[(t - lags[li]) % len(ring)]
        maximum_g, age_g = -1e300, 1
        maximum_b, age_b = -1e300, 1
        mixture_g, mixture_b = -1e300, -1e300
        active, width = 0, 0.
        for ai in range(len(ages)):
            age = ages[ai]
            if t >= age:
                old_current = ring[(t - age) % len(ring)]
                old_previous = ring[(t - age - lags[li]) % len(ring)]
                cells[li, ai, old_previous, old_current] -= 1
                totals[li, ai, old_previous] -= 1
                n = cells[li, ai, old_previous, old_current]
                total = totals[li, ai, old_previous]
                bayes[li, ai] -= cell_lookup[li, old_previous, old_current, n] - row_lookup[li, old_previous, total]
                glr[li, ai] -= nlogn[n + 1] - nlogn[n] - nlogn[total + 1] + nlogn[total] - logprob[li, old_previous, old_current]
            n = cells[li, ai, previous, current]
            total = totals[li, ai, previous]
            bayes[li, ai] += cell_lookup[li, previous, current, n] - row_lookup[li, previous, total]
            glr[li, ai] += nlogn[n + 1] - nlogn[n] - nlogn[total + 1] + nlogn[total] - logprob[li, previous, current]
            cells[li, ai, previous, current] += 1
            totals[li, ai, previous] += 1
            if age <= t + 1:
                active += 1
                width += widths[ai]
                g, b = max(glr[li, ai], 0.), bayes[li, ai]
                if g > maximum_g:
                    maximum_g, age_g = g, age
                if b > maximum_b:
                    maximum_b, age_b = b, age
                mixture_g = _logadd(mixture_g, g)
                mixture_b = _logadd(mixture_b, b + np.log(widths[ai]))
        memory[li, 0] = max(memory[li, 0], maximum_b)
        memory[li, 1] = max((1. - alpha) * memory[li, 1], maximum_b)
        p_stay = np.exp(logprob[li, previous, previous])
        centered_stay = ((1. if previous == current else 0.) - p_stay) / np.sqrt(p_stay * (1. - p_stay) + .01)
        memory[li, 2] = (1. - alpha) * memory[li, 2] + alpha * centered_stay
        cursor = li * len(STATS)
        output[cursor] = -logprob[li, previous, current]
        output[cursor + 1] = maximum_g
        output[cursor + 2] = mixture_g - np.log(active)
        output[cursor + 3] = np.log2(age_g)
        output[cursor + 4] = maximum_b
        output[cursor + 5] = mixture_b - np.log(width)
        output[cursor + 6] = np.log2(age_b)
        output[cursor + 7] = memory[li, 0]
        output[cursor + 8] = memory[li, 1]
        output[cursor + 9] = memory[li, 2]
    counter[0] += 1
    return output


@njit(cache=True)
def replay_transitions(points, args):
    output = np.empty((len(points), len(args[-1])), dtype=np.float32)
    for i in range(len(points)):
        output[i] = transition_step(points[i], args)
    return output


class MarkovState:
    def __init__(self, historical, config=MarkovConfig()):
        config = _config(asdict(config))
        self.config = config
        h = np.asarray(historical, dtype=np.float32)
        if h.ndim != 1 or len(h) < 32 or not np.isfinite(h).all():
            raise ValueError('Historical input must contain at least 32 finite values')
        edges = np.quantile(h.astype(float), [.25, .5, .75])
        bins = np.searchsorted(edges, h, side='right')
        lags = np.asarray(config.lags, dtype=np.int64)
        ages = np.asarray([2**k for k in range(config.max_age.bit_length())], dtype=np.int64)
        widths = np.diff(np.r_[0, ages]).astype(float)
        counts = np.empty((len(lags), 4, 4), dtype=float)
        for li, lag in enumerate(lags):
            counts[li] = np.bincount(4 * bins[:-lag] + bins[lag:], minlength=16).reshape(4, 4)
        beta = counts + .5
        beta_rows = beta.sum(axis=2)
        probabilities = beta / beta_rows[:, :, None]
        prior = config.prior_strength * probabilities
        n = np.arange(config.max_age + 1, dtype=float)
        cell_lookup = np.log(prior[:, :, :, None] + n) - np.log(beta[:, :, :, None] + n)
        row_lookup = np.log(config.prior_strength + n)[None, None, :] - np.log(beta_rows[:, :, None] + n)
        nlogn = np.zeros(len(n))
        nlogn[1:] = n[1:] * np.log(n[1:])
        ring = np.zeros(config.max_age + max(config.lags) + 1, dtype=np.int64)
        for j in range(1, max(config.lags) + 1):
            ring[-j] = bins[-j]
        self.historical_counts = counts
        self.args = (edges, lags, ages, widths, np.log(probabilities), cell_lookup, row_lookup, nlogn,
            ring, np.zeros((len(lags), len(ages), 4, 4), dtype=np.int64),
            np.zeros((len(lags), len(ages), 4), dtype=np.int64),
            np.zeros((len(lags), len(ages))), np.zeros((len(lags), len(ages))),
            np.zeros((len(lags), 3)), np.zeros(1, dtype=np.int64), np.empty(len(lags) * len(STATS), dtype=np.float32))

    def update(self, point):
        return transition_step(float(point), self.args)

    def replay(self, points):
        return replay_transitions(np.asarray(points, dtype=np.float32), self.args)

    @property
    def state_array_bytes(self):
        return sum(a.nbytes for a in self.args) + self.historical_counts.nbytes


def feature_names(settings):
    config = _config(settings)
    return tuple(f'transition_lag{lag}_{stat}' for lag in config.lags for stat in STATS)


def feature_groups(settings):
    return tuple('BAYES' if stat.startswith('bf_') else 'GLR' if stat.startswith('glr_') else 'TRANSITION'
                 for lag in _config(settings).lags for stat in STATS)


def make_state(historical, settings):
    return MarkovState(historical, _config(settings))


def implementation_hash(settings):
    return hashlib.sha256(Path(__file__).read_bytes() + json.dumps(asdict(_config(settings)), sort_keys=True).encode()).hexdigest()
