"""Fixed-support diagnostic summaries; no detector fitting or new randomness."""
import numpy as np
from threadpoolctl import threadpool_limits


def support_partition(maximum, terminal_cut, envelope, qualified):
    maximum, terminal_cut, envelope, qualified = np.broadcast_arrays(
        np.asarray(maximum, float), np.asarray(terminal_cut, float),
        np.asarray(envelope, float), np.asarray(qualified, float))
    if not all(np.isfinite(x).all() for x in (maximum, terminal_cut, envelope, qualified)):
        raise ValueError('Finite inputs required')
    if np.any(terminal_cut > envelope) or np.any((qualified != 0) & (qualified != 1)):
        raise ValueError('Original cut must lie within support; qualification must be binary')
    beyond = maximum > envelope
    within = (maximum > terminal_cut) & ~beyond
    early = (qualified > .5) & (maximum <= terminal_cut)
    if np.any((beyond | within) & (qualified < .5)):
        raise ValueError('Terminal exceedance must imply qualification in this frozen horizon')
    pieces = np.stack((beyond, within, early)).astype(np.uint8)
    np.testing.assert_array_equal(pieces.sum(axis=0), qualified)
    return pieces


def fixed_fold_draws(values, counts, *, common_mask=None):
    """Equal-weight fixed-fold conditional means; resample generating seeds only.

    values is (fitted fold, generating seed). Missing support is never zero.
    Every draw needs observations in every fixed fold; otherwise it stays NaN.
    A paired comparison passes the same intersection mask to both arguments.
    """
    v, c = np.asarray(values, float), np.asarray(counts, float)
    if v.ndim != 2 or c.ndim != 2 or v.shape[1] != c.shape[1]:
        raise ValueError('Expected aligned fold-by-seed values and draw-by-seed counts')
    if np.isinf(v).any() or not np.isfinite(c).all() or np.any(c < 0) or np.any(c != np.floor(c)):
        raise ValueError('Values may be missing but not infinite; counts must be nonnegative integers')
    if np.any(c.sum(axis=1) <= 0):
        raise ValueError('Every resample requires positive mass')
    mask = np.isfinite(v)
    if common_mask is not None:
        common_mask = np.asarray(common_mask, bool)
        if common_mask.shape != v.shape:
            raise ValueError('Paired support shape mismatch')
        mask &= common_mask
    support = mask.sum(axis=1)
    numerator = np.where(mask, v, 0.)
    with threadpool_limits(limits=1):
        mass = c @ mask.T
        sums = c @ numerator.T
    per_fold = np.full(sums.shape, np.nan)
    np.divide(sums, mass, out=per_fold, where=mass > 0)
    usable = np.all(mass > 0, axis=1)
    draws = np.full(len(c), np.nan)
    draws[usable] = per_fold[usable].mean(axis=1)
    fold_estimate = np.full(v.shape[0], np.nan)
    np.divide(numerator.sum(axis=1), support, out=fold_estimate, where=support > 0)
    estimate = float(fold_estimate.mean()) if np.all(support > 0) else None
    return estimate, draws, fold_estimate, support


def interval(draws):
    a = np.asarray(draws, float)
    a = a[np.isfinite(a)]
    return {'CI_low': float(np.quantile(a, .025)) if len(a) else None,
        'CI_high': float(np.quantile(a, .975)) if len(a) else None,
        'usable_draws': len(a), 'requested_draws': len(draws)}


def fixed_fold_pair(candidate, reference, counts):
    a, b = np.asarray(candidate, float), np.asarray(reference, float)
    if a.shape != b.shape:
        raise ValueError('Matched fold and seed shapes required')
    common = np.isfinite(a) & np.isfinite(b)
    ae, ad, af, support = fixed_fold_draws(a, counts, common_mask=common)
    be, bd, bf, support_b = fixed_fold_draws(b, counts, common_mask=common)
    np.testing.assert_array_equal(support, support_b)
    delta = ad - bd
    valid = delta[np.isfinite(delta)]
    return {'candidate_mean_on_common_support': ae, 'reference_mean_on_common_support': be,
        'mean_delta': ae - be if ae is not None and be is not None else None,
        **interval(delta), 'probability_delta_positive': float(np.mean(valid > 0)) if len(valid) else None,
        'probability_delta_negative': float(np.mean(valid < 0)) if len(valid) else None,
        'eligible_seed_counts_by_fixed_fold': support.tolist(),
        'candidate_fold_means_common_support': af.tolist(),
        'reference_fold_means_common_support': bf.tolist()}, delta
