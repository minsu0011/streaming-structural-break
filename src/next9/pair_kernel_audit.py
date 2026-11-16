"""Independent TS-AUC from explicit positive-negative pairs, without score sorting.

Weights multiply original groups. Integer pair kernels preserve exact tie halves;
the statistic pools comparable pairs over time, not per-time AUC means.
"""
import numpy as np


def prepare_pairs(y,p,t,groups):
    y,p,t,groups=map(np.asarray,(y,p,t,groups))
    if not (y.ndim==p.ndim==t.ndim==groups.ndim==1 and len(y)==len(p)==len(t)==len(groups)):
        raise ValueError('One aligned vector per input required')
    if not np.isin(y,[0,1]).all() or not np.isfinite(p).all():
        raise ValueError('Binary labels and finite scores required')
    unique,index=np.unique(groups,return_inverse=True);n=len(unique)
    count=np.zeros((n,n),dtype=np.int64);win2=np.zeros_like(count)
    order=np.argsort(t,kind='stable');tt=t[order]
    bounds=np.r_[0,np.flatnonzero(tt[1:]!=tt[:-1])+1,len(order)] if len(order) else np.array([0])
    for start,stop in zip(bounds[:-1],bounds[1:]):
        rows=order[start:stop]
        if len(np.unique(index[rows]))!=len(rows):
            raise ValueError('This independent kernel requires one observation per original group and time')
        pos=rows[y[rows]==1];neg=rows[y[rows]==0]
        if not len(pos) or not len(neg):continue
        where=np.ix_(index[pos],index[neg]);left=p[pos,None];right=p[None,neg]
        count[where]+=1
        win2[where]+=2*(left>right).astype(np.int64)+(left==right)
    return unique,count,win2


def weighted_pairs(count,win2,weights):
    weights=np.asarray(weights)
    if weights.ndim!=2 or weights.shape[1]!=len(count) or not np.isfinite(weights).all():
        raise ValueError('Finite draw by group weights required')
    if np.any(weights<0) or not np.array_equal(weights,np.floor(weights)):
        raise ValueError('Nonnegative integer multiplicities required')
    if count.shape!=win2.shape or count.shape!=(weights.shape[1],)*2:
        raise ValueError('Square aligned pair kernels required')
    # A conservative bound on every intermediate positive sum; below 2**53 all
    # binary64 products and additions are exact integers regardless of BLAS order.
    limit=int(max(int(count.max(initial=0)),int(win2.max(initial=0))))
    maxsum=int(weights.sum(axis=1).max(initial=0))
    if limit*maxsum*maxsum>=2**53:raise OverflowError('Exact integer range exceeded')
    w=weights.astype(np.float64)
    denominator=np.sum((w@count.astype(np.float64))*w,axis=1)
    twice_wins=np.sum((w@win2.astype(np.float64))*w,axis=1)
    out=np.full(len(w),np.nan)
    np.divide(twice_wins,2*denominator,out=out,where=denominator>0)
    return out,denominator
