"""Full observed-prefix diagnostic; never substitutes for the frozen16 window."""
import numpy as np
from numba import njit


@njit(cache=True)
def observed_prefix_reference(seed,values,epsilon):
    n,width=values.shape
    runs=np.empty((n,width),dtype=np.int32);ages=np.empty((n,width),dtype=np.int32);s4=np.empty((n,width),dtype=np.int8)
    for j in range(width):
        last=-1;run=0;age=15;nonflat=-1;reversal_to=-1
        previous=np.float64(seed[0,j]);e=np.float64(epsilon[j])
        for k in range(1,16):
            value=np.float64(seed[k,j]);delta=value-previous;direction=2 if delta>e else 0 if delta<-e else 1
            run=run+1 if direction==last else 1;age+=1
            if direction!=1:
                if nonflat>=0 and nonflat!=direction:age=0;reversal_to=direction
                nonflat=direction
            last=direction;previous=value
        # No seed reversal means only15 observed differences, not a known age30.
        if reversal_to<0:age=15
        for t in range(n):
            runs[t,j]=run;ages[t,j]=age
            recent=age<=3 and reversal_to==last
            s4[t,j]=(3 if recent else 0) if last==2 else (2 if recent else 1) if last==0 else 4
            value=np.float64(values[t,j]);delta=value-previous;direction=2 if delta>e else 0 if delta<-e else 1
            run=run+1 if direction==last else 1;age+=1
            if direction!=1:
                if nonflat>=0 and nonflat!=direction:age=0;reversal_to=direction
                nonflat=direction
            last=direction;previous=value
    return runs,ages,s4


def finite_window_states(seed,values,epsilon,available=True,stride=8):
    from src.next9.states import describe_history
    rows=np.arange(0,len(values),stride,dtype=int)
    joined=np.vstack([seed,values])
    history=joined[rows[:,None]+np.arange(16)[None,:]].transpose(0,2,1)
    return rows,describe_history(history,np.asarray(epsilon)[None,:],available)
