"""Input-complete nongraph control, frozen before outer score release."""
from dataclasses import dataclass
import numpy as np,joblib
from numba import njit
from src.next6.common import ROOT,BASE,digest,read_json
from src.next6.nodes import NodeState
from src.next5.reference_features import ReferenceState
from src.next6.model import EXTRA_COLUMNS

@njit(cache=False)
def extras(states,ring,ewma,duration,counter):
    out=np.empty((len(states),145),dtype=np.float32)
    for t in range(len(states)):
        now=int(counter[0]);out[t,:16]=states[t]
        for k,lag in enumerate((1,2,4,8,16)):out[t,16+16*k:32+16*k]=ring[(now-lag)%16]
        for j in range(16):
            value=float(states[t,j]-1) if states[t,j]<3 else 0.
            for k,half in enumerate((8.,32.)):
                alpha=1-np.exp(-np.log(2.)/half);ewma[k,j]=(1-alpha)*ewma[k,j]+alpha*value;out[t,96+16*k+j]=ewma[k,j]
            if states[t,j]<3 and states[t,j]==ring[(now-1)%16,j]:duration[j]=min(duration[j]+1,4096)
            else:duration[j]=0
            out[t,128+j]=np.log1p(duration[j])
        out[t,144]=np.log1p(now)/8.;ring[now%16]=states[t];counter[0]+=1
    return out

@dataclass
class StrictModel:
    columns:np.ndarray
    compact:object
    training:dict
    source_sha256:dict
    method:str='M1_STRICT'
    def validate(self):
        for name,sha in self.source_sha256.items():
            if digest(ROOT/name)!=sha:raise RuntimeError('Strict baseline source mismatch '+name)
    def predict(self,x):return self.compact.predict(np.asarray(x,dtype=np.float32)[:,self.columns])
    def new_state(self,h):self.validate();return StrictState(self,h)

class StrictState:
    def __init__(self,model,h):
        self.model=model;self.j=ReferenceState(h,{'version':'next5_reference_features_v1','kind':'J102'});self.c=ReferenceState(h,{'version':'next5_reference_features_v1','kind':'C205'});self.n=NodeState(h)
        self.ring=self.n.pseudo_s[-16:].copy();self.ewma=np.zeros((2,16));self.duration=np.zeros(16);self.counter=np.zeros(1)
    def replay(self,o):
        j=self.j.replay(o);c=self.c.replay(o);_,s,t=self.n.replay(o);e=extras(s,self.ring,self.ewma,self.duration,self.counter)
        return self.model.predict(np.column_stack((j,c,t,e)))
    def predict_one(self,point):return float(self.replay(np.array([point],dtype=np.float32))[0])
