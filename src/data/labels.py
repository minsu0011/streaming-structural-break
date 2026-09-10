import numbers
import numpy as np

def normalize_tau(tau):
    if tau is None or tau == -1:
        return None
    if isinstance(tau, (bool, np.bool_)) or not isinstance(tau, numbers.Real) or not np.isfinite(tau) or int(tau) != tau or tau < 0:
        raise ValueError("tau must be None, -1, or a nonnegative integer")
    return int(tau)

def labels_from_tau(length, tau):
    """Official documented convention: online t >= zero-based tau is positive.

    Empirical verification against downloaded official labels is tracked separately.
    This function is never imported by the feature engine.
    """
    tau = normalize_tau(tau)
    if length < 0 or int(length) != length:
        raise ValueError("Invalid length")
    if tau is not None and tau >= length:
        raise ValueError("Break index outside online segment")
    return np.zeros(length, dtype=np.uint8) if tau is None else (np.arange(length) >= tau).astype(np.uint8)
