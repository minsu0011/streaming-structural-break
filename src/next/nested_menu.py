"""Fixed-menu nested diagnostics using unchanged original folds only."""
from itertools import combinations
import numpy as np


def unique_training_tasks():
    """One three-fold fit serves both directions of a held-out fold pair."""
    return [{'held_out_folds': [left, right], 'training_folds': [k for k in range(5) if k not in (left, right)]}
        for left, right in combinations(range(5), 2)]


def select_inner_candidate(inner_scores):
    """No outer score argument exists; every candidate must have four scores."""
    if not inner_scores:
        raise ValueError('A nonempty frozen candidate menu is required')
    ranking = []
    for candidate, scores in inner_scores.items():
        values = np.asarray(scores, dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError('Exactly four valid inner held-out scores are required')
        ranking.append((float(values.mean()), float(values.min()), str(candidate)))
    return sorted(ranking, key=lambda row: (-row[0], -row[1], row[2]))[0][2]
