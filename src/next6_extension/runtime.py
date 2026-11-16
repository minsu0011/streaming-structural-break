"""Fused numerical equivalent of the readable observable-network adapter."""
import numpy as np
from numba import njit
from src.next6_extension.fresh_runtime import SafeState
from src.next6_extension.fresh_state import step as feature_step,FLOAT32_MAX

@njit(cache=True)
def step(point,args):
    features,probability,support,previous,out=args
    base=feature_step(point,features);out[:20]=base;current=int(base[0]);old=int(previous[0]);out[20:]=0.;out[20]=out[21]=1.;out[26]=old
    if old>=0:
        p0=probability[0,old,current];p1=probability[1,old,current];out[20]=p0;out[21]=p1;out[22]=np.log(p1/p0);out[23]=-np.log(p0);out[24]=1.;out[25]=1. if support[0,old,current]>=8 and support[1,old,current]>=8 else 0.
    previous[0]=current;return out

@njit(cache=True)
def replay_kernel(points,args):
    out=np.empty((len(points),27))
    for t in range(len(points)):out[t]=step(float(points[t]),args)
    return out

class Prepared:
    def __init__(self,model):model.validate();self.model=model
    def new_state(self,historical):return StreamingState(self.model,historical)

class StreamingState:
    def __init__(self,model,historical):
        self.model=model;self.features=SafeState(historical,threshold=model.threshold,graph=model.graph)
        self.args=(self.features.args,model.macro_probabilities,model.macro_support,np.array([-1],dtype=np.int64),np.zeros(27))
    def predict_one(self,point):
        value=float(point)
        if not np.isfinite(value) or abs(value)>FLOAT32_MAX:raise ValueError('Finite float32 current scalar required')
        return step(float(np.float32(value)),self.args).copy()
    def replay(self,points):
        p=np.asarray(points,dtype=np.float64)
        if p.ndim!=1 or not np.isfinite(p).all() or np.any(np.abs(p)>FLOAT32_MAX):raise ValueError('Finite float32 scalar stream required')
        return replay_kernel(p.astype(np.float32),self.args)
    @property
    def bounded_mutable_state_bytes(self):return self.features.bounded_mutable_state_bytes+self.args[3].nbytes+self.args[4].nbytes
