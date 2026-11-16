"""Actual streaming observable-state network with both learned macro models.

The output contains descriptive transition probabilities, not the probability
that a structural break has ever occurred. No label, phase or final length API.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np,joblib
from src.next6_extension.common import ROOT,digest
from src.next6_extension.fresh_runtime import SafeState
from src.next6_extension.fresh_state import COLUMNS as STATE_COLUMNS

COLUMNS=STATE_COLUMNS+('normal_transition_probability','post_transition_probability','post_minus_normal_transition_logscore','normal_transition_surprise','transition_available','both_conditions_supported','previous_observable_state')

@dataclass
class ObservableModel:
    threshold:float
    graph:object
    macro_probabilities:np.ndarray
    macro_support:np.ndarray
    provenance:dict
    source_sha256:dict
    def validate(self):
        self.graph.validate()
        p=self.macro_probabilities
        if p.shape!=(2,7,7) or not np.isfinite(p).all() or np.any(p<=0) or not np.allclose(p.sum(axis=-1),1,atol=1e-12):raise ValueError('Invalid learned macro transition probabilities')
        if self.macro_support.shape!=(2,7,7) or np.any(self.macro_support<0):raise ValueError('Invalid macro support')
        if not np.isfinite(self.threshold) or self.threshold<0:raise ValueError('Fixed finite threshold required')
        for name,sha in self.source_sha256.items():
            if digest(ROOT/name)!=sha:raise ValueError('Observable model source changed: '+name)
    def new_state(self,historical):self.validate();return ObservableState(self,historical)

class ObservableState:
    def __init__(self,model,historical):
        self.model=model;self.features=SafeState(historical,threshold=model.threshold,graph=model.graph);self.previous=-1
    def _append(self,base):
        base=np.asarray(base);out=np.empty((len(base),27));out[:,:20]=base
        current=base[:,0].astype(np.int64)
        if not len(current):return out
        previous=np.r_[self.previous,current[:-1]];good=previous>=0;out[:,20:]=0;out[:,20:22]=1;out[:,26]=previous
        out[good,20]=self.model.macro_probabilities[0,previous[good],current[good]];out[good,21]=self.model.macro_probabilities[1,previous[good],current[good]]
        out[good,22]=np.log(out[good,21]/out[good,20]);out[good,23]=-np.log(out[good,20]);out[good,24]=1
        out[good,25]=(self.model.macro_support[0,previous[good],current[good]]>=8)&(self.model.macro_support[1,previous[good],current[good]]>=8)
        self.previous=int(current[-1]);return out
    def predict_one(self,point):return self._append(self.features.predict_one(point)[None,:])[0]
    def replay(self,points):return self._append(self.features.replay(points))
    @property
    def bounded_mutable_state_bytes(self):return self.features.bounded_mutable_state_bytes+8

def load(path):
    model=joblib.load(path)
    if not isinstance(model,ObservableModel):raise TypeError('ObservableModel required')
    model.validate();return model
