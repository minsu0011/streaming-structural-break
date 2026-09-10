"""Whole-pipeline ARCH-off control, preserving both AR(8) whitening contracts.

The previous D208 removed conditional normalization in the new energy bank;
its legacy ABCD base still used ARCH. This distinct experiment removes ARCH
from both branches and performs no historical ARCH slope fit.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.features.ar_order_input import historical_filter
from src.features.ar_residual_input import innovation_one
from src.features.historical_calibration import HistoricalCalibratedState,CalibrationPolicy,_transform
from src.features.streaming import _update,feature_names as old_names
from src.next.fast_variance_initialization import OLD_CONFIG,conditional_replay_only
from src.next.whitening import fit_historical
from src.next2.variance_bank import historical_channels,step as variance_step,evidence_args,config as variance_config,feature_names as variance_names

SETTINGS={'order':8,'normalization':'none','energy':'raw','cap':4,'ages':'all','memory':'original','age_normalization':'none'}


def validate(settings):
    if settings!={'version':'full_pipeline_no_arch_v1'}:
        raise ValueError('Exact whole-pipeline ARCH-off control required')


@njit(cache=True)
def step(point,ar_args,state_args,calibration_args,new_args,output):
    residual=innovation_one(point,*ar_args)
    _update(float(residual),*state_args)
    _transform(state_args[6],*calibration_args)
    old=calibration_args[-1]
    new=variance_step(point,*new_args)
    output[:51]=old[:51]
    output[51:]=new
    return output


@njit(cache=True)
def replay(points,args):
    result=np.empty((len(points),77),dtype=np.float32)
    for j in range(len(points)):
        result[j]=step(float(points[j]),*args)
    return result


class FullNoARCHState:
    def __init__(self,historical,settings):
        validate(settings)
        reference,coefficients,ring,residual=historical_filter(historical,OLD_CONFIG,8)
        legacy=HistoricalCalibratedState(residual,config=OLD_CONFIG,policy=CalibrationPolicy(method='historical_median_mad'))
        old=legacy.state
        state_args=tuple(getattr(old,k) for k in ('reference','baseline','thresholds','ring','ew','counters','output','parameters','alphas','lags'))
        calibration=(legacy.center,legacy.scale,legacy.mask,float(legacy.policy.clip),legacy.output)
        fit=fit_historical(historical,8)
        r=fit['historical_innovations']
        mean=float(r.mean())
        sd=max(float(r.std()),1e-8)
        squared=(r-mean)**2
        cap=max(float(np.quantile(squared,.99)),1e-12)
        variance=max(float(np.minimum(squared,cap).mean()),1e-12)
        parameters=np.asarray([mean,sd,variance,0.,0.,cap,max(variance*.0001,1e-12),2./33.])
        conditional=conditional_replay_only(r,parameters,variance)
        empty=np.empty(0)
        robust=np.asarray([0.,1.])
        channels=historical_channels(r,parameters,np.full(3,variance),empty,robust,3,0,4)
        channels=channels[min(160,len(channels)//4):]
        channel_calibration=np.stack([channels.mean(axis=0),np.maximum(channels.std(axis=0),1e-6)])
        new_args=(tuple(fit[k] for k in ('reference','coefficients','ring','counter')),
            parameters,conditional,empty,robust,3,0,4,channel_calibration,np.empty(2),
            evidence_args(variance_config(SETTINGS)),0,0)
        self.args=((reference,coefficients,ring,np.zeros(1,dtype=np.int64)),state_args,calibration,new_args,np.empty(77,dtype=np.float32))

    def update(self,point):
        return step(float(point),*self.args)

    def replay(self,points):
        return replay(np.asarray(points,dtype=np.float32),self.args)


def make_state(historical,settings):
    return FullNoARCHState(historical,settings)


def feature_names(settings):
    validate(settings)
    return tuple('legacy_ar_only_'+n for n in old_names(OLD_CONFIG)[:51])+tuple('energy_ar_only_'+n for n in variance_names(SETTINGS))


def feature_groups(settings):
    validate(settings)
    return ('NEXT2_FULL_NO_ARCH',)*77


def implementation_hash(settings):
    validate(settings)
    root=Path(__file__).parents[2]
    sources=[Path(__file__),root/'src/next/next2_full_no_arch.py',root/'src/features/ar_order_input.py',
        root/'src/features/historical_calibration.py',root/'src/next2/variance_bank.py',root/'src/next/whitening.py']
    return hashlib.sha256(b''.join(p.read_bytes() for p in sources)+json.dumps(settings,sort_keys=True).encode()).hexdigest()
