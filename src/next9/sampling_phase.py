"""Group-weighted phase support diagnostics; never used for model inputs."""
import numpy as np
from numba import njit

FAMILIES=('S1_DIRECTION','S2_DIRECTION_RUN','S3_DIRECTION_SLOPE','S4_REVERSAL')
WIDTHS=(4,10,7,5)
OFFSETS=(0,4,14,21)


@njit(cache=False)
def phase_counts(states,times):
    n,width,_=states.shape
    counts=np.zeros((8,9,width,26),dtype=np.int64)
    totals=np.zeros((8,9),dtype=np.int64)
    for i in range(n):
        t=times[i]
        domain=1 if t<=9 else 2 if t<=24 else 3 if t<=49 else 4 if t<=99 else 5 if t<=199 else 6 if t<=399 else 7
        phase=t%8
        for d in (0,domain):
            totals[d,phase]+=1;totals[d,8]+=1
            for j in range(width):
                for k in range(4):
                    code=int(states[i,j,k])+OFFSETS[k]
                    counts[d,phase,j,code]+=1;counts[d,8,j,code]+=1
    return counts,totals


def eligible_group_frequencies(counts,totals):
    # The same series must supply every phase before comparing them. Without
    # this rule a short tail could change the cohort, not only sample timing.
    eligible=np.all(totals[:,:8]>0,axis=1)
    frequencies=np.zeros_like(counts,dtype=np.float64)
    np.divide(counts,totals[:,:,None,None],out=frequencies,
              where=eligible[:,None,None,None] & (totals[:,:,None,None]>0))
    return frequencies,eligible
