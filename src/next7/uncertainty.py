"""Paired uncertainty with explicit pair support and unambiguous fold medians."""
import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def rank_totals(y, group, time_ends, tie_ends, multiplicities):
    result = np.empty((len(multiplicities), 2), dtype=np.float64)
    for replicate in prange(len(multiplicities)):
        wins = 0.
        pairs = 0.
        cursor = 0
        tie = 0
        for time_end in time_ends:
            positive = 0.
            negative = 0.
            while cursor < time_end:
                tie_positive = 0.
                tie_negative = 0.
                end = tie_ends[tie]
                while cursor < end:
                    weight = multiplicities[replicate, group[cursor]]
                    if y[cursor]:
                        tie_positive += weight
                    else:
                        tie_negative += weight
                    cursor += 1
                wins += tie_positive * (negative + .5 * tie_negative)
                positive += tie_positive
                negative += tie_negative
                tie += 1
            pairs += positive * negative
        result[replicate, 0] = wins
        result[replicate, 1] = pairs
    return result


def prepared_totals(prepared, multiplicities):
    raw = np.asarray(multiplicities)
    if raw.ndim != 2 or raw.shape[1] != prepared.n_groups or not np.isfinite(raw).all():
        raise ValueError('Finite group multiplicity matrix required')
    if np.any(raw < 0) or np.any(raw != np.floor(raw)) or np.max(raw, initial=0) > np.iinfo(np.int32).max:
        raise ValueError('Nonnegative integer multiplicities required')
    return rank_totals(prepared.y, prepared.group, prepared.time_ends, prepared.tie_ends, raw.astype(np.int32))


def paired_summary(reference, candidate):
    ref, cand = np.asarray(reference), np.asarray(candidate)
    if ref.ndim != 2 or ref.shape != cand.shape or len(ref) < 2 or not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ValueError('Finite paired replicate-by-fold arrays including identity row required')
    paired = cand - ref
    mean = paired.mean(axis=1)
    median = np.median(paired, axis=1)
    difference_of_medians = np.median(cand, axis=1) - np.median(ref, axis=1)
    return {'estimate_mean_matched_delta': float(mean[0]),
        'mean_delta_ci_2_5': float(np.quantile(mean[1:], .025)),
        'mean_delta_ci_97_5': float(np.quantile(mean[1:], .975)),
        'probability_mean_delta_positive': float(np.mean(mean[1:] > 0)),
        'estimate_median_matched_fold_delta': float(median[0]),
        'median_matched_fold_delta_ci_2_5': float(np.quantile(median[1:], .025)),
        'median_matched_fold_delta_ci_97_5': float(np.quantile(median[1:], .975)),
        'estimate_difference_of_fold_score_medians': float(difference_of_medians[0]),
        'difference_of_medians_ci_2_5': float(np.quantile(difference_of_medians[1:], .025)),
        'difference_of_medians_ci_97_5': float(np.quantile(difference_of_medians[1:], .975)),
        'identity_positive_matched_folds': int(np.sum(paired[0] > 0)),
        'identity_worst_matched_fold_delta': float(np.min(paired[0])),
        'replicates': len(ref) - 1, 'folds': ref.shape[1]}
