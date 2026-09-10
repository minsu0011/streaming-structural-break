"""Two unchanged historical AR representations for the fixed equal blend."""
import hashlib
from pathlib import Path
import numpy as np
from src.features.config import CONFIG
from src.features.streaming import feature_names as base_names,FEATURE_FAMILIES
from src.features.historical_calibration import CalibrationPolicy
from src.features.ar_residual_input import ARResidualCalibratedState,implementation_hash as ar4_hash
from src.features.ar_order_input import AROrderCalibratedState,implementation_hash as ar8_hash

REPRESENTATION='cross_order_ar4_ar8_equal'
POLICY=CalibrationPolicy(method='historical_median_mad')


def feature_names(config=CONFIG):
    return tuple(f'ar{order}__{name}' for order in (4,8) for name in base_names(config))


def implementation_hash(config=CONFIG):
    return hashlib.sha256(Path(__file__).read_bytes()+ar4_hash(POLICY,config).encode()+ar8_hash(POLICY,config,8).encode()).hexdigest()


class CrossOrderFeatureState:
    def __init__(self,historical,*,config=CONFIG):
        self.states=(ARResidualCalibratedState(historical,config=config,policy=POLICY),AROrderCalibratedState(historical,config=config,policy=POLICY,input_order=8))
        self.output=np.zeros(2*len(FEATURE_FAMILIES),dtype=np.float32)
    def update_and_get(self,point):
        width=len(FEATURE_FAMILIES)
        for k,state in enumerate(self.states):self.output[k*width:(k+1)*width]=state.update_and_get(point)
        return self.output
    @property
    def state_array_bytes(self):return self.output.nbytes+sum(s.state_array_bytes for s in self.states)
