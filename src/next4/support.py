"""Distinguish eventual-break membership from positive observations in a slice."""
import numpy as np


def scored_support(ids,target,times,tau):
    ids=np.asarray(ids);y=np.asarray(target);t=np.asarray(times);tau=np.asarray(tau)
    if any(a.ndim!=1 or a.shape!=y.shape for a in (ids,t,tau)) or not np.isin(y,[0,1]).all():
        raise ValueError('Aligned diagnostic coordinates required')
    if not np.isfinite(t).all() or (t<0).any() or (t!=np.floor(t)).any():raise ValueError('Integral nonnegative online times required')
    if not len(y):return {'rows':0,'break_series':0,'null_series':0,'positive_series':0,'negative_series':0,
        'eligible_positive_series':0,'eligible_negative_series':0,'eligible_times':0,'positive_negative_pair_mass':0}
    size=int(t.max())+1;total=np.bincount(t.astype(np.int64),minlength=size)
    positive=np.bincount(t.astype(np.int64),weights=y,minlength=size).astype(np.int64)
    negative=total-positive;pairs=positive*negative;eligible=pairs[t.astype(np.int64)]>0
    return {'rows':len(y),'break_series':len(np.unique(ids[tau>=0])),'null_series':len(np.unique(ids[tau<0])),
        'positive_series':len(np.unique(ids[y==1])),'negative_series':len(np.unique(ids[y==0])),
        'eligible_positive_series':len(np.unique(ids[(y==1)&eligible])),
        'eligible_negative_series':len(np.unique(ids[(y==0)&eligible])),
        'eligible_times':int(np.count_nonzero(pairs)),'positive_negative_pair_mass':int(pairs.sum())}
