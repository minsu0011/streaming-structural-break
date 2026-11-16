"""Existing late-onset seed stratum; preserve original global seed weights."""
import numpy as np
from src.next5.synthetic_metrics import event_by_seed


def onset_mask(seed_index, tau_cycle, onset):
    if list(tau_cycle).count(onset) != 1:
        raise ValueError('An existing unique frozen onset is required')
    ids = np.asarray(seed_index)
    if ids.ndim != 1 or np.any(ids < 0) or np.any(ids != np.floor(ids)):
        raise ValueError('Original nonnegative generating-seed indices required')
    return ids % len(tau_cycle) == list(tau_cycle).index(onset)


def stratum_event_draws(prediction, tau, seed_index, cutoff, event, counts):
    numerator, denominator = event_by_seed(prediction, tau, seed_index, cutoff, event)
    if len(numerator) > counts.shape[1]:
        raise ValueError('Seed index lies beyond the original frozen count columns')
    top = np.zeros(counts.shape[1])
    bottom = np.zeros(counts.shape[1])
    top[:len(numerator)] = numerator
    bottom[:len(denominator)] = denominator
    divisor = counts @ bottom
    if not np.all(divisor > 0):
        raise ValueError('This original bootstrap contains an empty eligible stratum')
    return (counts @ top) / divisor
