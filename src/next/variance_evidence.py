"""H-fitted variance forecasts and conditional Gaussian variance-score evidence."""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next.whitening import fit_historical,innovation_update,source_hash as whitening_hash
from src.next.normalization import fit_normalization
from src.next.channel_evidence import CHANNEL_STATS,channel_step,make_channel_args,source_hash as channel_hash


@dataclass(frozen=True)
class VarianceConfig:
    order:int=8
    normalization:str='arch1'
    max_age:int=128


CHANNELS = ('raw_energy','log_variance_forecast','standardized_energy','variance_intercept_score','variance_lag1_score','variance_lag2_score')


def _config(settings):
    config = VarianceConfig(**settings)
    if config.order!=8 or config.normalization not in ('arch1','arch2','ewma') or config.max_age not in (128,512):
        raise ValueError('Unregistered conditional variance evidence policy')
    return config


@njit(cache=True)
def variance_channel_update(residual,parameters,conditional,mode,output):
    h_variance = parameters[2]/max(1.-parameters[3]-parameters[4],.05)
    q1,q2 = conditional[0],conditional[1]
    variance = conditional[2] if mode==5 else parameters[2]+parameters[3]*q1+parameters[4]*q2
    variance = max(variance,parameters[6])
    centered = residual-parameters[0]
    squared = centered*centered
    normalized_energy = squared/variance-1.
    output[0] = squared/h_variance-1.
    output[1] = np.log(variance/h_variance)
    output[2] = normalized_energy
    output[3] = .5*normalized_energy*h_variance/variance
    output[4] = .5*normalized_energy*q1/variance
    output[5] = .5*normalized_energy*q2/variance
    capped = min(squared,parameters[5])
    conditional[1] = q1
    conditional[0] = capped
    conditional[2] = (1.-parameters[7])*conditional[2]+parameters[7]*capped
    return output


@njit(cache=True)
def historical_variance_channels(residuals,parameters,conditional,mode):
    result = np.empty((len(residuals),len(CHANNELS)))
    for j in range(len(residuals)):
        variance_channel_update(residuals[j],parameters,conditional,mode,result[j])
    return result


@njit(cache=True)
def variance_step(point,filter_args,parameters,conditional,mode,calibration,work,evidence_args):
    residual = innovation_update(point,*filter_args)
    variance_channel_update(residual,parameters,conditional,mode,work)
    for j in range(len(work)):
        work[j] = (work[j]-calibration[0,j])/calibration[1,j]
    return channel_step(work,evidence_args)


@njit(cache=True)
def replay_variance(points,filter_args,parameters,conditional,mode,calibration,work,evidence_args):
    result = np.empty((len(points),len(evidence_args[-1])),dtype=np.float32)
    for j in range(len(points)):
        result[j] = variance_step(points[j],filter_args,parameters,conditional,mode,calibration,work,evidence_args)
    return result


class VarianceState:
    def __init__(self,historical,config=VarianceConfig()):
        config = _config(asdict(config))
        self.config = config
        fit = fit_historical(historical,config.order)
        profile = fit_normalization(fit['historical_innovations'],config.normalization)
        self.parameters = profile['parameters']
        self.mode = profile['mode']
        h_variance = self.parameters[2]/max(1.-self.parameters[3]-self.parameters[4],.05)
        conditional = np.full(3,h_variance)
        channels = historical_variance_channels(fit['historical_innovations'],self.parameters,conditional,self.mode)
        np.testing.assert_allclose(conditional,profile['conditional'],rtol=1e-14,atol=1e-14)
        # Burn-in depends on known H length only, never online total length.
        burn = min(160,len(channels)//4)
        channels = channels[burn:]
        self.calibration = np.stack([channels.mean(axis=0),np.maximum(channels.std(axis=0),1e-6)])
        self.conditional = profile['conditional'].copy()
        self.filter_args = tuple(fit[key] for key in ('reference','coefficients','ring','counter'))
        self.work = np.empty(len(CHANNELS))
        self.evidence_args = make_channel_args(len(CHANNELS),config.max_age)

    def update(self,point):
        return variance_step(float(point),self.filter_args,self.parameters,self.conditional,self.mode,self.calibration,self.work,self.evidence_args)

    def replay(self,points):
        return replay_variance(np.asarray(points,dtype=np.float32),self.filter_args,self.parameters,self.conditional,self.mode,self.calibration,self.work,self.evidence_args)

    @property
    def state_array_bytes(self):
        return sum(a.nbytes for a in self.filter_args+self.evidence_args)+self.parameters.nbytes+self.conditional.nbytes+self.calibration.nbytes+self.work.nbytes


def feature_names(settings):
    _config(settings)
    return tuple(channel+'_'+stat for channel in CHANNELS for stat in CHANNEL_STATS)


def feature_groups(settings):
    _config(settings)
    return tuple('VARIANCE_GLR' if stat.startswith('glr') else 'VARIANCE_BAYES' if stat.startswith('bayes') else 'VARIANCE_LOCAL' for channel in CHANNELS for stat in CHANNEL_STATS)


def make_state(historical,settings):
    return VarianceState(historical,_config(settings))


def implementation_hash(settings):
    normalization = Path(__file__).with_name('normalization.py').read_bytes()
    return hashlib.sha256(Path(__file__).read_bytes()+normalization+whitening_hash().encode()+channel_hash().encode()+json.dumps(asdict(_config(settings)),sort_keys=True).encode()).hexdigest()
