"""Fit a historical prefix and calibrate variance evidence on its forward tail.

Only the candidate series' observed H is used. Parameters are frozen after the
prefix; the tail advances causal AR/variance state and estimates channel null
location/scale. Online data never refit parameters or calibration.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next.whitening import fit_historical,innovation_update,source_hash as whitening_hash
from src.next.normalization import fit_normalization
from src.next.variance_evidence import VarianceState,CHANNELS,variance_channel_update,implementation_hash as variance_hash
from src.next.channel_evidence import CHANNEL_STATS,make_channel_args


@dataclass(frozen=True)
class ForwardVarianceConfig:
    order:int=8
    normalization:str='arch1'
    max_age:int=128
    fit_fraction:float=.75


def _config(settings):
    config = ForwardVarianceConfig(**settings)
    if config.order!=8 or config.normalization!='arch1' or config.max_age!=128 or config.fit_fraction not in (.5,.75):
        raise ValueError('Unregistered forward-H variance policy')
    return config


@njit(cache=True)
def forward_channels(points,filter_args,parameters,conditional,mode):
    result = np.empty((len(points),len(CHANNELS)))
    for j,point in enumerate(points):
        residual = innovation_update(point,*filter_args)
        variance_channel_update(residual,parameters,conditional,mode,result[j])
    return result


class ForwardVarianceState(VarianceState):
    def __init__(self,historical,config=ForwardVarianceConfig()):
        config = _config(asdict(config))
        h = np.asarray(historical,dtype=np.float32)
        if h.ndim!=1 or not np.isfinite(h).all():
            raise ValueError('A finite one-dimensional historical sequence is required')
        cut = int(len(h)*config.fit_fraction)
        if cut<32 or len(h)-cut<16:
            raise ValueError('Forward-H fit needs at least 32 prefix and 16 calibration points')
        fit = fit_historical(h[:cut],config.order)
        profile = fit_normalization(fit['historical_innovations'],config.normalization)
        self.parameters = profile['parameters']
        self.conditional = profile['conditional'].copy()
        self.mode = profile['mode']
        self.filter_args = tuple(fit[key] for key in ('reference','coefficients','ring','counter'))
        channels = forward_channels(h[cut:],self.filter_args,self.parameters,self.conditional,self.mode)
        channels = channels[min(32,len(channels)//4):]
        self.calibration = np.stack([channels.mean(axis=0),np.maximum(channels.std(axis=0),1e-6)])
        self.work = np.empty(len(CHANNELS))
        self.evidence_args = make_channel_args(len(CHANNELS),config.max_age)
        self.config = config


def feature_names(settings):
    _config(settings)
    return tuple(channel+'_'+stat for channel in CHANNELS for stat in CHANNEL_STATS)


def feature_groups(settings):
    _config(settings)
    return tuple('VARIANCE_GLR' if stat.startswith('glr') else 'VARIANCE_BAYES' if stat.startswith('bayes') else 'VARIANCE_LOCAL' for channel in CHANNELS for stat in CHANNEL_STATS)


def make_state(historical,settings):
    return ForwardVarianceState(historical,_config(settings))


def implementation_hash(settings):
    config = _config(settings)
    reference = {'order':8,'normalization':'arch1','max_age':128}
    return hashlib.sha256(Path(__file__).read_bytes()+whitening_hash().encode()+variance_hash(reference).encode()+json.dumps(asdict(config),sort_keys=True).encode()).hexdigest()
