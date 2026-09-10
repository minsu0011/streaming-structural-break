"""Prediction-error mismatch for the two predeclared frozen AR-order pairs."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next.whitening import fit_historical, innovation_update, source_hash as whitening_hash
from src.next.channel_evidence import CHANNEL_STATS, channel_step, make_channel_args, source_hash as channel_hash


@dataclass(frozen=True)
class MismatchConfig:
    low_order: int = 4
    high_order: int = 8
    max_age: int = 128


CHANNELS = ('forecast_gap','energy_advantage','log_energy_advantage','conditional_score')


def _config(settings):
    config = MismatchConfig(**settings)
    if (config.low_order,config.high_order) not in ((4,8),(8,12)) or config.max_age not in (128,512):
        raise ValueError('Unregistered AR mismatch pair or age grid')
    return config


@njit(cache=True)
def mismatch_channels(low_error,high_error,scale,output):
    low,high = low_error/scale,high_error/scale
    gap = low-high
    output[0] = gap
    output[1] = low*low-high*high
    output[2] = np.log1p(low*low)-np.log1p(high*high)
    output[3] = high*gap
    return output


@njit(cache=True)
def mismatch_step(point,filter_args,calibration,work,evidence_args):
    low_ref,low_coef,low_ring,low_count,high_ref,high_coef,high_ring,high_count = filter_args
    low = innovation_update(point,low_ref,low_coef,low_ring,low_count)
    high = innovation_update(point,high_ref,high_coef,high_ring,high_count)
    mismatch_channels(low,high,calibration[0,0],work)
    for j in range(len(work)):
        work[j] = (work[j]-calibration[1,j])/calibration[2,j]
    return channel_step(work,evidence_args)


@njit(cache=True)
def replay_mismatch(points,filter_args,calibration,work,evidence_args):
    output = np.empty((len(points),len(evidence_args[-1])),dtype=np.float32)
    for j in range(len(points)):
        output[j] = mismatch_step(points[j],filter_args,calibration,work,evidence_args)
    return output


class MismatchState:
    def __init__(self,historical,config=MismatchConfig()):
        config = _config(asdict(config))
        self.config = config
        low = fit_historical(historical,config.low_order)
        high = fit_historical(historical,config.high_order)
        np.testing.assert_array_equal(low['reference'],high['reference'])
        high_errors = high['historical_innovations']
        low_errors = low['historical_innovations'][-len(high_errors):]
        scale = max(float(high_errors.std()),1e-6)
        channels = np.empty((len(high_errors),len(CHANNELS)))
        for j in range(len(high_errors)):
            mismatch_channels(low_errors[j],high_errors[j],scale,channels[j])
        self.calibration = np.zeros((3,len(CHANNELS)))
        self.calibration[0] = scale
        self.calibration[1] = channels.mean(axis=0)
        self.calibration[2] = np.maximum(channels.std(axis=0),1e-6)
        self.filter_args = tuple(fit[key] for fit in (low,high) for key in ('reference','coefficients','ring','counter'))
        self.work = np.empty(len(CHANNELS))
        self.evidence_args = make_channel_args(len(CHANNELS),config.max_age)

    def update(self,point):
        return mismatch_step(float(point),self.filter_args,self.calibration,self.work,self.evidence_args)

    def replay(self,points):
        return replay_mismatch(np.asarray(points,dtype=np.float32),self.filter_args,self.calibration,self.work,self.evidence_args)

    @property
    def state_array_bytes(self):
        return sum(a.nbytes for a in self.filter_args+self.evidence_args)+self.calibration.nbytes+self.work.nbytes


def feature_names(settings):
    _config(settings)
    return tuple(channel+'_'+stat for channel in CHANNELS for stat in CHANNEL_STATS)


def feature_groups(settings):
    _config(settings)
    return tuple('MISMATCH_GLR' if stat.startswith('glr') else 'MISMATCH_BAYES' if stat.startswith('bayes') else 'MISMATCH_LOCAL' for channel in CHANNELS for stat in CHANNEL_STATS)


def make_state(historical,settings):
    return MismatchState(historical,_config(settings))


def implementation_hash(settings):
    return hashlib.sha256(Path(__file__).read_bytes()+whitening_hash().encode()+channel_hash().encode()+json.dumps(asdict(_config(settings)),sort_keys=True).encode()).hexdigest()
