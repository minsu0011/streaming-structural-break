"""Extend AR order/variance conditioning while preserving the legacy ABCD bank.

The AR8/ARCH1 anchor is bitwise identical to the frozen historical pipeline.
AR12 uses the same ridge and coefficient-L2 policy in this separate module.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.features.config import CONFIG
from src.features.streaming import StreamingFeatureState,feature_names as old_names,FEATURE_FAMILIES,_update
from src.features.historical_calibration import CalibrationPolicy,HistoricalCalibratedState,_transform
from src.features.ar_residual_input import AR_POLICY,historical_normal_equations,historical_innovations,innovation_one
from src.features.ar_order_input import implementation_hash as order_hash
from src.features.arch_input import ARCH_POLICY,historical_variance_filter,arch_one,implementation_hash as arch_hash
from src.features.rank_gaussian import historical_mapping,map_one,implementation_hash as gaussian_hash


OLD_CONFIG = CONFIG.variant(normalization='median_mad',scales=(5,20,160))
OLD_POLICY = CalibrationPolicy(method='historical_median_mad')


@dataclass(frozen=True)
class LegacyConfig:
    order:int=8
    normalization:str='arch1'


def _config(settings):
    config = LegacyConfig(**settings)
    if config.order not in (4,8,12) or config.normalization not in ('none','arch1','arch2','gaussian'):
        raise ValueError('Unregistered legacy-bank extension')
    return config


def extended_historical_filter(historical,order):
    if order not in (4,8,12):
        raise ValueError('Unregistered legacy AR order')
    h = np.asarray(historical,dtype=np.float32)
    reference = StreamingFeatureState(h,config=OLD_CONFIG).reference
    location,scale = reference[:2]
    z = np.clip((np.where(np.isfinite(h),h,location).astype(np.float64)-location)/scale,-OLD_CONFIG.clip_z,OLD_CONFIG.clip_z)
    coefficients = np.zeros(order,dtype=np.float64)
    if len(z)>order:
        gram,rhs = historical_normal_equations(z,order)
        ridge = max(float(np.trace(gram))/order*AR_POLICY['ridge_trace_fraction'],OLD_CONFIG.moment_floor)
        coefficients = np.linalg.solve(gram+ridge*np.eye(order),rhs)
        norm = float(np.linalg.norm(coefficients))
        if norm>AR_POLICY['coefficient_l2_limit']:
            coefficients *= AR_POLICY['coefficient_l2_limit']/norm
        residual = historical_innovations(z,coefficients)
    else:
        residual = z.astype(np.float32)
    ring = np.zeros(order,dtype=np.float64)
    if len(z):
        ring[-min(order,len(z)):] = z[-order:]
    return np.array([location,scale,OLD_CONFIG.clip_z]),coefficients,ring,residual


@njit(cache=True)
def arch2_one(residual,parameters,previous):
    omega,a1,a2,cap,floor,mean = parameters
    variance = max(omega+a1*previous[0]+a2*previous[1],floor)
    previous[1] = previous[0]
    if not np.isfinite(residual):
        previous[0] = mean
        return np.float32(np.nan)
    previous[0] = min(residual*residual,cap)
    return np.float32(residual/np.sqrt(variance))


@njit(cache=True)
def arch2_replay(residual,parameters,previous):
    result = np.empty(len(residual),dtype=np.float32)
    for j in range(len(residual)):
        result[j] = arch2_one(float(residual[j]),parameters,previous)
    return result


def historical_arch2(residual):
    r = np.asarray(residual,dtype=np.float32)
    if len(r)<3:
        raise ValueError('ARCH2 needs at least three historical innovations')
    squared = r.astype(float)**2
    cap = max(float(np.quantile(squared,ARCH_POLICY['historical_squared_residual_winsor_quantile'])),OLD_CONFIG.moment_floor)
    squared = np.minimum(squared,cap)
    mean = max(float(squared.mean()),OLD_CONFIG.moment_floor)
    x = np.column_stack([squared[1:-1],squared[:-2]])
    centered = x-x.mean(axis=0)
    gram = centered.T@centered
    ridge = max(float(np.trace(gram))/2*ARCH_POLICY['ridge_variance_fraction'],OLD_CONFIG.moment_floor)
    slopes = np.maximum(np.linalg.solve(gram+ridge*np.eye(2),centered.T@(squared[2:]-squared[2:].mean())),0.)
    if slopes.sum()>ARCH_POLICY['slope_upper_bound']:
        slopes *= ARCH_POLICY['slope_upper_bound']/slopes.sum()
    parameters = np.array([mean*(1-slopes.sum()),*slopes,cap,OLD_CONFIG.moment_floor,mean])
    previous = np.array([squared[1],squared[0]])
    transformed = arch2_replay(r[2:],parameters,previous)
    return parameters,previous,transformed


@njit(cache=True)
def legacy_step(point,ar_args,mode,parameters,previous,knots,scores,inner_args,center,scale,mask,clip,transformed,output):
    residual = innovation_one(point,*ar_args)
    if mode==1:
        residual = arch_one(float(residual),parameters,previous)
    elif mode==2:
        residual = arch2_one(float(residual),parameters,previous)
    elif mode==3:
        residual = map_one(float(residual),knots,scores)
    _update(float(residual),*inner_args)
    _transform(inner_args[6],center,scale,mask,clip,transformed)
    output[:] = transformed[:51]
    return output


@njit(cache=True)
def legacy_replay(points,args):
    result = np.empty((len(points),51),dtype=np.float32)
    for j in range(len(points)):
        result[j] = legacy_step(float(points[j]),*args)
    return result


class LegacyState:
    def __init__(self,historical,config=LegacyConfig()):
        config = _config(asdict(config))
        reference,coefficients,ring,residual = extended_historical_filter(historical,config.order)
        ar_args = reference,coefficients,ring,np.zeros(1,dtype=np.int64)
        mode = ('none','arch1','arch2','gaussian').index(config.normalization)
        parameters,previous = np.zeros(6),np.zeros(2)
        knots,scores = np.zeros(1),np.zeros(1)
        if mode==1:
            parameters,previous,h_input = historical_variance_filter(residual,OLD_CONFIG)
        elif mode==2:
            parameters,previous,h_input = historical_arch2(residual)
        elif mode==3:
            knots,scores,h_input = historical_mapping(residual)
        else:
            h_input = residual
        inner = HistoricalCalibratedState(h_input,config=OLD_CONFIG,policy=OLD_POLICY)
        state = inner.state
        inner_args = tuple(getattr(state,key) for key in ('reference','baseline','thresholds','ring','ew','counters','output','parameters','alphas','lags'))
        self.args = (ar_args,mode,parameters,previous,knots,scores,inner_args,inner.center,inner.scale,inner.mask,
            inner.policy.clip,inner.output,np.empty(51,dtype=np.float32))

    def update(self,point):
        return legacy_step(float(point),*self.args)

    def replay(self,points):
        return legacy_replay(np.asarray(points,dtype=np.float32),self.args)

    @property
    def state_array_bytes(self):
        def size(item):
            return item.nbytes if isinstance(item,np.ndarray) else sum(size(v) for v in item) if isinstance(item,tuple) else 0
        return size(self.args)


def feature_names(settings):
    _config(settings)
    return tuple(old_names(OLD_CONFIG)[:51])


def feature_groups(settings):
    _config(settings)
    return tuple(FEATURE_FAMILIES[:51])


def make_state(historical,settings):
    return LegacyState(historical,_config(settings))


def implementation_hash(settings):
    config = _config(settings)
    old = order_hash(OLD_POLICY,OLD_CONFIG,config.order)+arch_hash(OLD_POLICY,OLD_CONFIG)+gaussian_hash(OLD_POLICY,OLD_CONFIG)
    return hashlib.sha256(Path(__file__).read_bytes()+old.encode()+json.dumps(asdict(config),sort_keys=True).encode()).hexdigest()
