"""Explicit unavailable auxiliary targets; no NaN-to-integer conversion.

The H-only feature engine and its completed cache remain byte-unchanged.
This replaces only the initial unused auxiliary-target helper whose unavailable
suffix generated NumPy cast warnings. Both helpers agree on available targets.
"""
import numpy as np
from src.next8.features import FutureTargets,HORIZONS,UNKNOWN,categorize,movement_direction


def future_targets(values,ids,times,epsilon,low,high,valid,horizon,cutoff=None):
    values=np.asarray(values,dtype=np.float32)
    ids=np.asarray(ids);times=np.asarray(times)
    if horizon not in HORIZONS:raise ValueError('Unregistered forecast horizon')
    if values.ndim!=1 or len(ids)!=len(values) or len(times)!=len(values) or not np.isfinite(values).all():
        raise ValueError('Finite single-coordinate observed sequence and aligned membership/time required')
    n=len(values);available=np.zeros(n,dtype=bool)
    if n>horizon:
        available[:-horizon]=(ids[:-horizon]==ids[horizon:]) & (times[horizon:]==times[:-horizon]+horizon)
    if cutoff is not None:available &= times+horizon<=cutoff
    rows=np.flatnonzero(available)
    following=np.full(n,np.nan,dtype=np.float32);delta=np.full(n,np.nan,dtype=np.float32)
    direction=np.full(n,UNKNOWN,dtype=np.int8);state=direction.copy();reversal=direction.copy()
    following[rows]=values[rows+horizon];delta[rows]=following[rows]-values[rows]
    def at(value):return np.broadcast_to(np.asarray(value),(n,))[rows]
    direction[rows]=movement_direction(delta[rows],at(epsilon))
    state[rows]=categorize(following[rows],at(low),at(high),at(valid))
    reversal[rows]=(np.sign(following[rows])*np.sign(values[rows])<0).astype(np.int8)
    return FutureTargets(horizon,following,delta,direction,state,reversal,available)
