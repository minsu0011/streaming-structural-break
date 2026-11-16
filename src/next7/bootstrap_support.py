"""Exact support envelope of G2's four-block empirical resampling mechanism."""
import numpy as np
from numba import njit
from src.next7.null_growth import joint_ratio


@njit(cache=True)
def support_envelope(vectors, core_cut, scale_cut):
    """Every four-block window spans one sampled chunk or two adjacent chunks."""
    count = len(vectors) - 3
    internal = 0.
    stitched = 0.
    memory = np.empty((4, 9))
    for a in range(count):
        internal = max(internal, joint_ratio(vectors[a:a + 4], core_cut, scale_cut))
        for b in range(count):
            for left_count in range(1, 4):
                # Match the physical ring in bootstrap_maxima exactly: the new
                # chunk head occupies the first slots, then the old chunk tail.
                # This matters for the last rounding bits of the four-term sum.
                memory[:4 - left_count] = vectors[b:b + 4 - left_count]
                memory[4 - left_count:] = vectors[a + 4 - left_count:a + 4]
                stitched = max(stitched, joint_ratio(memory, core_cut, scale_cut))
    return internal, stitched, max(1., internal, stitched)


def exact_support(vectors, core_cut, scale_cut):
    v = np.asarray(vectors, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 9 or len(v) < 4 or not np.isfinite(v).all():
        raise ValueError('At least four finite nine-channel pseudo blocks required')
    if not np.isfinite(core_cut) or not np.isfinite(scale_cut) or min(core_cut, scale_cut) <= 0:
        raise ValueError('Positive finite cutoffs required')
    return support_envelope(v, float(core_cut), float(scale_cut))
