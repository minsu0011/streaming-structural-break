"""Matched crossed-grid diagnostics for already frozen historical lengths."""
import numpy as np
from src.next7.crossed_statistics import seed_pair_product_draws


def discordant_latches(candidate, reference, counts):
    """Partition changed decisions; preserve the original seed on both axes."""
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise ValueError('Aligned two-dimensional decision matrices required')
    if not np.isin(candidate, (0, 1)).all() or not np.isin(reference, (0, 1)).all():
        raise ValueError('Only observed binary decisions are allowed')
    candidate, reference = candidate.astype(bool), reference.astype(bool)
    added = candidate & ~reference
    removed = reference & ~candidate
    both = candidate & reference
    neither = ~(candidate | reference)
    pieces = {'ADDED_FALSE_MEMORY': added, 'REMOVED_FALSE_MEMORY': removed,
              'BOTH_FALSE_MEMORY': both, 'NEITHER_FALSE_MEMORY': neither}
    draws = {name: seed_pair_product_draws(matrix, counts) for name, matrix in pieces.items()}
    delta = candidate.astype(np.int8) - reference.astype(np.int8)
    delta_draws = seed_pair_product_draws(delta, counts)
    np.testing.assert_allclose(draws['ADDED_FALSE_MEMORY'] - draws['REMOVED_FALSE_MEMORY'],
                               delta_draws, rtol=0, atol=5e-16)
    np.testing.assert_allclose(sum(draws.values()), 1., rtol=0, atol=5e-16)
    return {name: float(matrix.mean()) for name, matrix in pieces.items()}, draws, delta_draws
