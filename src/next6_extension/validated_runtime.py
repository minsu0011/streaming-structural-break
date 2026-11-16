"""Neutral outputs when H cannot establish a reference; unchanged valid-H kernel.

An insufficient reference cannot supply evidence, a confirmed block count, or a
learned transition probability. Neutral probabilities are 1 with availability 0.
This is an input-validity repair, not a new fitted detector or recalibration.
"""
import numpy as np
from src.next6_extension.runtime import Prepared as FusedPrepared
from src.next6_extension.fresh_state import FLOAT32_MAX

class Prepared(FusedPrepared):
    def new_state(self,historical):
        state=super().new_state(historical)
        if state.features.calibration['reference_valid']:
            state.reference_valid=True
            return state
        return InsufficientReferenceState(state.features.calibration)

class InsufficientReferenceState:
    reference_valid=False
    def __init__(self,calibration):
        self.calibration=calibration.copy();self.counter=np.zeros(1,dtype=np.int64)
    def _emit(self,n):
        t=np.arange(int(self.counter[0]),int(self.counter[0])+n,dtype=np.int64);out=np.zeros((n,27));out[:,0]=6;out[:,4]=(t+1)%16==0;out[:,5]=(t+1)//16;out[:,20:22]=1;out[:,26]=np.where(t==0,-1,6);self.counter[0]+=n;return out
    def predict_one(self,point):
        value=float(point)
        if not np.isfinite(value) or abs(value)>FLOAT32_MAX:raise ValueError('Finite float32 current scalar required')
        return self._emit(1)[0]
    def replay(self,points):
        p=np.asarray(points,dtype=np.float64)
        if p.ndim!=1 or not np.isfinite(p).all() or np.any(np.abs(p)>FLOAT32_MAX):raise ValueError('Finite float32 scalar stream required')
        return self._emit(len(p))
    @property
    def bounded_mutable_state_bytes(self):return self.counter.nbytes
