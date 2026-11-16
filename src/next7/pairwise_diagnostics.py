"""Bounded-memory exhaustive pair diagnostics, without fitting or random draws."""
import numpy as np
from scipy.special import expit


def exact_stratum(beta, positive_features, negative_features, positive_offset,
                  negative_offset, block_size=64):
    beta = np.asarray(beta, dtype=np.float64)
    p, n = np.asarray(positive_features), np.asarray(negative_features)
    po, no = np.asarray(positive_offset), np.asarray(negative_offset)
    if p.ndim != 2 or n.ndim != 2 or p.shape[1] != len(beta) or n.shape[1] != len(beta):
        raise ValueError('Aligned feature matrices required')
    if po.shape != (len(p),) or no.shape != (len(n),) or not len(p) or not len(n):
        raise ValueError('Two nonempty classes and aligned offsets required')
    if block_size <= 0 or not all(np.isfinite(v).all() for v in (beta, p, n, po, no)):
        raise ValueError('Finite inputs and positive block size required')
    ne = no + n @ beta
    loss = 0.
    gradient = np.zeros(len(beta))
    for a in range(0, len(p), block_size):
        z = p[a:a + block_size]
        margin = (po[a:a + block_size] + z @ beta)[:, None] - ne[None, :]
        loss += float(np.logaddexp(0., -margin).sum())
        derivative = -expit(-margin)
        gradient += z.T @ derivative.sum(axis=1) - n.T @ derivative.sum(axis=0)
    mass = len(p) * len(n)
    return loss / mass, gradient / mass


def sampled_stratum(beta, differences, offset_differences):
    x, offset = np.asarray(differences), np.asarray(offset_differences)
    margin = offset + x @ beta
    losses = np.logaddexp(0., -margin)
    gradient = x.T @ (-expit(-margin)) / len(x)
    return float(losses.mean()), gradient, float(losses.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.
