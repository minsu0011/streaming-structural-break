"""Causal startup-variance correction before historical-null standardization."""
from dataclasses import replace
import hashlib,json
from pathlib import Path
import numpy as np
from numba import njit
from src.features.config import CONFIG
from src.features.streaming import StreamingFeatureState,feature_version,FEATURE_NAMES
from src.features.historical_calibration import historical_reference,_transform,implementation_hash as null_hash

def implementation_hash(policy,config=CONFIG):
    return hashlib.sha256(Path(__file__).read_bytes()+null_hash(replace(policy,method=policy.method.removesuffix('_startup')),config).encode()+json.dumps(policy.__dict__,sort_keys=True).encode()).hexdigest()

@njit(cache=False)
def correct_startup(source,alphas,t,output):
    output[:]=source
    for scale in range(3):
        decay=(1.-alphas[scale])**t
        unscaled_ratio=np.sqrt(max(1.-decay*decay,1e-12))
        rootn_ratio=np.sqrt(max((1.-decay)*(1.-decay*decay),1e-12))
        for j in range(15):
            ratio=unscaled_ratio if j in (2,3,14) else rootn_ratio
            index=6+15*scale+j
            output[index]=source[index]/ratio

class StartupCalibratedState:
    def __init__(self,historical,*,config,policy):
        self.state=StreamingFeatureState(historical,config=config)
        reference_policy=replace(policy,method=policy.method.removesuffix('_startup'))
        self.center,self.scale,self.mask=historical_reference(historical,config=config,policy=reference_policy)
        self.policy=policy;self.corrected=np.zeros(len(FEATURE_NAMES),dtype=np.float32);self.output=np.zeros(len(FEATURE_NAMES),dtype=np.float32)

    def update_and_get(self,point):
        raw=self.state.update_and_get(point)
        correct_startup(raw,self.state.alphas,self.state.counters[0],self.corrected)
        _transform(self.corrected,self.center,self.scale,self.mask,self.policy.clip,self.output)
        return self.output

    @property
    def state_array_bytes(self):
        return self.state.state_array_bytes+self.center.nbytes+self.scale.nbytes+self.mask.nbytes+self.corrected.nbytes+self.output.nbytes

@njit(cache=False)
def transform_matrix(matrix,alphas,center,scale,mask,clip):
    result=np.empty_like(matrix);corrected=np.empty(matrix.shape[1],dtype=np.float32)
    for i in range(len(matrix)):
        correct_startup(matrix[i],alphas,float(i+1),corrected)
        _transform(corrected,center,scale,mask,clip,result[i])
    return result

def transform_cached_features(matrix,historical,*,config=CONFIG,policy):
    reference_policy=replace(policy,method=policy.method.removesuffix('_startup'))
    center,scale,mask=historical_reference(historical,config=config,policy=reference_policy)
    return transform_matrix(np.asarray(matrix,dtype=np.float32),np.asarray(config.alphas,dtype=np.float64),center,scale,mask,policy.clip)
