"""Historical ARCH(1) variance forecast after the unchanged AR8 innovation filter."""
import hashlib,json
from pathlib import Path
import numpy as np
from numba import njit
from src.features.config import CONFIG
from src.features.ar_order_input import historical_filter,implementation_hash as ar_hash
from src.features.ar_residual_input import innovation_one
from src.features.historical_calibration import CalibrationPolicy,HistoricalCalibratedState

ARCH_POLICY=json.loads((Path(__file__).resolve().parents[2]/'configs/arch_input.json').read_text(encoding='utf-8'))


def implementation_hash(policy,config=CONFIG):
    return hashlib.sha256(Path(__file__).read_bytes()+json.dumps(ARCH_POLICY,sort_keys=True).encode()+ar_hash(policy,config,8).encode()).hexdigest()


@njit(cache=False)
def arch_one(residual,parameters,previous_squared):
    residual=np.float64(residual)
    intercept,slope,cap,floor,mean_squared=parameters
    forecast=max(intercept+slope*min(previous_squared[0],cap),floor)
    if not np.isfinite(residual):
        previous_squared[0]=mean_squared
        return np.float32(np.nan)
    result=np.float32(residual/np.sqrt(forecast))
    previous_squared[0]=min(residual*residual,cap)
    return result


@njit(cache=False)
def replay_variance_normalization(residual,parameters):
    previous=np.array([min(float(residual[0])**2,parameters[2])],dtype=np.float64)
    output=np.empty(len(residual)-1,dtype=np.float32)
    for t in range(1,len(residual)):output[t-1]=arch_one(float(residual[t]),parameters,previous)
    return output,previous


def historical_variance_filter(residual,config=CONFIG):
    r=np.asarray(residual,dtype=np.float32)
    if len(r)<2:
        return np.array([1.,0.,1.,config.moment_floor,1.]),np.array([float(r[-1])**2 if len(r) else 0.]),r.copy()
    squared=r.astype(np.float64)**2
    cap=max(float(np.quantile(squared,ARCH_POLICY['historical_squared_residual_winsor_quantile'])),config.moment_floor)
    squared=np.minimum(squared,cap);mean=max(float(squared.mean()),config.moment_floor)
    previous=squared[:-1];current=squared[1:];centered=previous-previous.mean()
    variance=float(np.mean(centered*centered));covariance=float(np.mean(centered*(current-current.mean())))
    slope=float(np.clip(covariance/max(variance*(1.+ARCH_POLICY['ridge_variance_fraction']),config.moment_floor),0.,ARCH_POLICY['slope_upper_bound']))
    parameters=np.array([mean*(1.-slope),slope,cap,config.moment_floor,mean],dtype=np.float64)
    transformed,previous_squared=replay_variance_normalization(r,parameters)
    return parameters,previous_squared,transformed


class ARCHCalibratedState:
    def __init__(self,historical,*,config=CONFIG,policy=CalibrationPolicy(method='historical_median_mad')):
        self.reference,self.coefficients,self.ring,residual=historical_filter(historical,config,8)
        self.counter=np.zeros(1,dtype=np.int64)
        self.variance_parameters,self.previous_squared,transformed=historical_variance_filter(residual,config)
        self.inner=HistoricalCalibratedState(transformed,config=config,policy=policy)
    def update_and_get(self,point):
        residual=innovation_one(float(point),self.reference,self.coefficients,self.ring,self.counter)
        standardized=arch_one(float(residual),self.variance_parameters,self.previous_squared)
        return self.inner.update_and_get(standardized)
    @property
    def state_array_bytes(self):
        return sum(x.nbytes for x in (self.reference,self.coefficients,self.ring,self.counter,self.variance_parameters,self.previous_squared))+self.inner.state_array_bytes
