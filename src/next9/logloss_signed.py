"""Binary loss explicitly uses signed floating labels and probabilities."""
import numpy as np


def binary_loss(y,p):
    target=np.asarray(y,dtype=np.float64)
    probability=np.asarray(p,dtype=np.float64)
    if target.shape!=probability.shape or not np.isin(target,[0.,1.]).all():
        raise ValueError('Aligned binary labels required')
    if not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)):
        raise ValueError('Finite probabilities required')
    probability=np.clip(probability,1e-8,1-1e-8)
    loss=-target*np.log(probability)-(1-target)*np.log1p(-probability)
    assert np.isfinite(loss).all() and np.all(loss>=0)
    return loss
