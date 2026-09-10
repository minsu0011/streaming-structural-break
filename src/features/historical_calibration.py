"""Experimental historical-null standardization, independent of outcomes.

This module does not change the original 57-feature engine. It supplies an
alternative representation using only the historical reference and each prefix.
"""
from dataclasses import dataclass,asdict
import hashlib,json
from pathlib import Path
import numpy as np
from numba import njit
from src.features.streaming import StreamingFeatureState,_update,feature_version,FEATURE_NAMES,FEATURE_FAMILIES
from src.features.config import CONFIG

@dataclass(frozen=True)
class CalibrationPolicy:
    burn_in: int = 160
    standard_deviation_floor: float = 0.05
    clip: float = 12.0
    signed_families: str = 'ABCD'
    method: str = 'historical_mean_std'

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()

POLICY=CalibrationPolicy()

def implementation_hash(policy=POLICY,config=CONFIG):
    return hashlib.sha256(Path(__file__).read_bytes()+policy.digest().encode()+feature_version(config).encode()).hexdigest()

@njit(cache=False)
def _historical_replay(points,reference,baseline,thresholds,ring,ew,counters,output,parameters,alphas,lags):
    matrix=np.empty((len(points),len(output)),dtype=np.float32)
    for i in range(len(points)):
        _update(float(points[i]),reference,baseline,thresholds,ring,ew,counters,output,parameters,alphas,lags)
        matrix[i,:]=output
    return matrix

def historical_reference(historical,*,config=CONFIG,policy=POLICY):
    h=np.asarray(historical,dtype=np.float32)
    state=StreamingFeatureState(h,config=config)
    reference=_historical_replay(h,state.reference,state.baseline,state.thresholds,state.ring,state.ew,state.counters,state.output,state.parameters,state.alphas,state.lags)
    if not len(reference):reference=np.zeros((1,len(FEATURE_NAMES)),dtype=np.float32)
    burn=min(policy.burn_in,max(0,len(reference)//2))
    reference=reference[burn:].astype(np.float64)
    if policy.method=='historical_mean_std':center=reference.mean(axis=0);scale=reference.std(axis=0)
    elif policy.method=='historical_median_mad':
        center=np.median(reference,axis=0);scale=np.median(abs(reference-center),axis=0)*1.4826
    else:raise ValueError('Unknown calibration method')
    scale=np.maximum(scale,policy.standard_deviation_floor)
    mask=np.array([family in policy.signed_families for family in FEATURE_FAMILIES])
    # Prefix CUSUM and cumulative summaries have nonstationary null drift. Keep
    # those original values instead of calibrating them from a longer history.
    mask[[3,4,5]]=False
    return center,scale,mask

@njit(cache=False)
def _transform(source,center,scale,mask,clip,output):
    for j in range(len(source)):
        output[j]=min(max((source[j]-center[j])/scale[j],-clip),clip) if mask[j] else source[j]

class HistoricalCalibratedState:
    def __init__(self,historical,*,config=CONFIG,policy=POLICY):
        self.state=StreamingFeatureState(historical,config=config)
        self.center,self.scale,self.mask=historical_reference(historical,config=config,policy=policy)
        self.policy=policy;self.output=np.zeros(len(FEATURE_NAMES),dtype=np.float32)

    def update_and_get(self,point):
        source=self.state.update_and_get(point)
        _transform(source,self.center,self.scale,self.mask,self.policy.clip,self.output)
        return self.output

    @property
    def state_array_bytes(self):
        return self.state.state_array_bytes+self.center.nbytes+self.scale.nbytes+self.mask.nbytes+self.output.nbytes

def transform_cached_features(matrix,historical,*,config=CONFIG,policy=POLICY):
    center,scale,mask=historical_reference(historical,config=config,policy=policy)
    result=np.asarray(matrix,dtype=np.float32).copy()
    result[:,mask]=np.clip((result[:,mask].astype(np.float64)-center[mask])/scale[mask],-policy.clip,policy.clip)
    return result
