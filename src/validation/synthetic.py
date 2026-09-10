"""Small deterministic engineering fixtures, NEVER competition performance data."""
import numpy as np

KINDS=('no_break','mean_up','mean_down','variance_up','variance_down','heavy_tail','ar_up','ar_reverse','single_outlier')

def ar_process(noise,rho,previous=0.0):
    output=np.empty(len(noise))
    for i,value in enumerate(noise):
        previous=rho*previous+np.sqrt(1-rho*rho)*value
        output[i]=previous
    return output

def fixture(kind,seed=20260908,historical_length=2048,online_length=320,break_at=100):
    rng=np.random.default_rng(seed)
    h=rng.normal(size=historical_length)
    o=rng.normal(size=online_length)
    tau=break_at
    if kind=='no_break': tau=None
    elif kind=='mean_up': o[tau:]+=2
    elif kind=='mean_down': o[tau:]-=2
    elif kind=='variance_up': o[tau:]*=2.5
    elif kind=='variance_down': o[tau:]*=.25
    elif kind=='heavy_tail': o[tau:]=rng.standard_t(3,size=online_length-tau)/np.sqrt(3)
    elif kind=='ar_up':
        o[tau:]=ar_process(o[tau:],.85,o[tau-1] if tau else h[-1])
    elif kind=='ar_reverse':
        h=ar_process(h,.75)
        o[:tau]=ar_process(o[:tau],.75,h[-1])
        o[tau:]=ar_process(o[tau:],-.75,o[tau-1] if tau else h[-1])
    elif kind=='single_outlier':
        o[tau]=15; tau=None
    else: raise ValueError(kind)
    return h.astype(np.float32),o.astype(np.float32),tau
