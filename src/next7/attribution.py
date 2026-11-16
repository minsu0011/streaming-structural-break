"""Exact AUC rank-pair partition by positive/negative event-latch status."""
import numpy as np
from numba import njit


@njit(cache=True)
def partition_sorted(y,p,t,latch):
    wins=np.zeros((2,2))
    pairs=np.zeros((2,2))
    cursor=0
    while cursor<len(y):
        time=t[cursor]
        positive=np.zeros(2)
        negative=np.zeros(2)
        while cursor<len(y) and t[cursor]==time:
            score=p[cursor]
            tie_pos=np.zeros(2)
            tie_neg=np.zeros(2)
            while cursor<len(y) and t[cursor]==time and p[cursor]==score:
                group=int(latch[cursor])
                if y[cursor]:
                    tie_pos[group]+=1
                else:
                    tie_neg[group]+=1
                cursor+=1
            for ip in range(2):
                for jn in range(2):
                    wins[ip,jn]+=tie_pos[ip]*(negative[jn]+.5*tie_neg[jn])
            positive+=tie_pos
            negative+=tie_neg
        for ip in range(2):
            for jn in range(2):
                pairs[ip,jn]+=positive[ip]*negative[jn]
    return wins,pairs


def latch_pair_partition(target,prediction,time,latch):
    y=np.asarray(target,dtype=np.uint8)
    p=np.asarray(prediction,dtype=np.float64)
    t=np.asarray(time,dtype=np.int64)
    e=np.asarray(latch,dtype=np.uint8)
    if not (y.ndim==1 and p.shape==y.shape and t.shape==y.shape and e.shape==y.shape):
        raise ValueError('Aligned point arrays required')
    if not np.isfinite(p).all() or np.any(y>1) or np.any(e>1):
        raise ValueError('Finite score and binary target/latch required')
    order=np.lexsort((p,t))
    return partition_sorted(y[order],p[order],t[order],e[order])
