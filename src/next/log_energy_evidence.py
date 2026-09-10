"""Matched log-energy channels to separate compression from ARCH forecasting.

The old variance-forecast channel is a monotone transformation of lagged clipped
energy when ARCH1 alpha is positive. These explicit alternatives test current
versus lagged evidence and historical tail clipping without fitting another
online model or changing the sequential summaries.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next.whitening import fit_historical,innovation_update,source_hash as whitening_hash
from src.next.channel_evidence import CHANNEL_STATS,channel_step,make_channel_args,source_hash as channel_hash


@dataclass(frozen=True)
class LogEnergyConfig:
    order:int=8
    alignment:str='current'
    cap_at_h99:bool=False
    max_age:int=128


def _config(settings):
    config = LogEnergyConfig(**settings)
    if config.order!=8 or config.alignment not in ('current','lagged') or type(config.cap_at_h99) is not bool or config.max_age not in (128,512):
        raise ValueError('Unregistered log-energy evidence setting')
    return config


@njit(cache=True)
def energy_channels(residual,parameters,previous,lagged,cap,output):
    centered = residual-parameters[0]
    squared = centered*centered
    energy = min(squared,parameters[2]) if cap else squared
    current = np.log1p(energy/parameters[1])
    output[0] = squared/parameters[1]-1.
    output[1] = previous[0] if lagged else current
    previous[0] = current
    return output


@njit(cache=True)
def historical_channels(residuals,parameters,previous,lagged,cap):
    result = np.empty((len(residuals),2))
    for j,residual in enumerate(residuals):
        energy_channels(residual,parameters,previous,lagged,cap,result[j])
    return result


@njit(cache=True)
def log_energy_step(point,filter_args,parameters,previous,lagged,cap,calibration,work,evidence_args):
    residual = innovation_update(point,*filter_args)
    energy_channels(residual,parameters,previous,lagged,cap,work)
    for j in range(2):
        work[j] = (work[j]-calibration[0,j])/calibration[1,j]
    return channel_step(work,evidence_args)


@njit(cache=True)
def replay_energy(points,filter_args,parameters,previous,lagged,cap,calibration,work,evidence_args):
    output = np.empty((len(points),len(evidence_args[-1])),dtype=np.float32)
    for j,point in enumerate(points):
        output[j] = log_energy_step(point,filter_args,parameters,previous,lagged,cap,calibration,work,evidence_args)
    return output


class LogEnergyState:
    def __init__(self,historical,config=LogEnergyConfig()):
        config = _config(asdict(config))
        fit = fit_historical(historical,config.order)
        residuals = fit['historical_innovations']
        center = float(residuals.mean())
        squares = (residuals-center)**2
        cap = max(float(np.quantile(squares,.99)),1e-12)
        variance = max(float(np.minimum(squares,cap).mean()),1e-12)
        self.parameters = np.array([center,variance,cap])
        self.previous = np.array([np.log(2.)])
        self.lagged,self.cap = config.alignment=='lagged',config.cap_at_h99
        channels = historical_channels(residuals,self.parameters,self.previous,self.lagged,self.cap)
        channels = channels[min(160,len(channels)//4):]
        self.calibration = np.stack([channels.mean(axis=0),np.maximum(channels.std(axis=0),1e-6)])
        self.filter_args = tuple(fit[key] for key in ('reference','coefficients','ring','counter'))
        self.work = np.empty(2)
        self.evidence_args = make_channel_args(2,config.max_age)
        self.config = config

    @property
    def args(self):
        return self.filter_args,self.parameters,self.previous,self.lagged,self.cap,self.calibration,self.work,self.evidence_args

    def update(self,point):
        return log_energy_step(float(point),*self.args)

    def replay(self,points):
        return replay_energy(np.asarray(points,dtype=np.float32),*self.args)

    @property
    def state_array_bytes(self):
        return sum(a.nbytes for a in self.filter_args+self.evidence_args)+self.parameters.nbytes+self.previous.nbytes+self.calibration.nbytes+self.work.nbytes


def feature_names(settings):
    _config(settings)
    return tuple(channel+'_'+stat for channel in ('raw_energy','log_energy') for stat in CHANNEL_STATS)


def feature_groups(settings):
    _config(settings)
    return tuple('ENERGY_GLR' if stat.startswith('glr') else 'ENERGY_BAYES' if stat.startswith('bayes') else 'ENERGY_LOCAL' for channel in range(2) for stat in CHANNEL_STATS)


def make_state(historical,settings):
    return LogEnergyState(historical,_config(settings))


def implementation_hash(settings):
    return hashlib.sha256(Path(__file__).read_bytes()+whitening_hash().encode()+channel_hash().encode()+json.dumps(asdict(_config(settings)),sort_keys=True).encode()).hexdigest()
