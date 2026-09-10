"""Historical-null median/MAD calibration of the new causal statistic bank."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from numba import njit
from src.next.engine import EngineConfig,SequentialState,FEATURE_NAMES,FEATURE_GROUPS,feature_hash


@njit(cache=True)
def calibrated_one(values,location,scale,apply_mask,output):
    for j in range(len(values)):
        output[j]=min(max((float(values[j])-location[j])/scale[j],-12.),12.) if apply_mask[j] else values[j]
    return output


class CalibratedSequentialState:
    def __init__(self,historical,config=EngineConfig()):
        self.inner=SequentialState(historical,config)
        calibration=copy.deepcopy(self.inner)
        null=calibration.replay(np.asarray(historical,dtype=np.float32))[160:].astype(np.float64)
        if len(null)<128:
            raise ValueError('Insufficient historical null replay')
        self.location=np.median(null,axis=0)
        self.scale=np.maximum(1.4826*np.median(np.abs(null-self.location),axis=0),.05)
        # Elapsed time, hypothetical maximizing age and persistent-memory state
        # are nonstationary by construction; their values remain unchanged.
        self.apply_mask=np.asarray([group!='M' and name not in ('elapsed_log1p','glr_mean_argmax_log_age','glr_variance_argmax_log_age')
                                    for name,group in zip(FEATURE_NAMES,FEATURE_GROUPS)],dtype=np.bool_)
        self.output=np.empty(len(FEATURE_NAMES),dtype=np.float32)

    def update(self,point):
        return calibrated_one(self.inner.update(point),self.location,self.scale,self.apply_mask,self.output)

    def replay(self,points):
        values=self.inner.replay(points)
        for j in range(len(FEATURE_NAMES)):
            if self.apply_mask[j]:
                values[:,j]=np.clip((values[:,j].astype(float)-self.location[j])/self.scale[j],-12,12).astype(np.float32)
        return values

    @property
    def state_array_bytes(self):
        return self.inner.state_array_bytes+self.location.nbytes+self.scale.nbytes+self.apply_mask.nbytes+self.output.nbytes


def feature_names(settings):
    return FEATURE_NAMES


def feature_groups(settings):
    return FEATURE_GROUPS


def make_state(historical,settings):
    return CalibratedSequentialState(historical,EngineConfig(**settings.get('engine_config',{})))


def implementation_hash(settings):
    config=EngineConfig(**settings.get('engine_config',{}))
    return hashlib.sha256(Path(__file__).read_bytes()+feature_hash(config).encode()+json.dumps(settings,sort_keys=True).encode()).hexdigest()
