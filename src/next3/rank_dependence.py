"""H-fixed mid-rank innovation amplitudes before the four C205 lag products."""
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next.whitening import fit_historical,innovation_update
from src.next.fast_variance_initialization import fast_variance_parameters
from src.next.normalization import ecdf_uniform
from src.next2.complement_bank import normalized_one,historical_raw,raw_complement
from src.next2.dependence_runtime import historical_z,evidence,STATS

SETTINGS={'version':'historical_rank_dependence_v1','order':8}


def validate(settings):
    if settings!=SETTINGS:raise ValueError('Exact H-fixed rank dependence contract required')


@njit(cache=False)
def rank_values(values,sorted_h):
    result=np.empty(len(values),dtype=np.float64)
    for j in range(len(values)):result[j]=2.*ecdf_uniform(values[j],sorted_h)-1.
    return result


@njit(cache=False)
def step(point,filter_args,parameters,conditional,sorted_h,ring,counter,center,calibration,work,empty,evidence_args):
    residual=innovation_update(point,*filter_args)
    z=normalized_one(residual,parameters,conditional)
    rank=2.*ecdf_uniform(z,sorted_h)-1.
    raw_complement(rank,empty,ring,counter,center,3,work)
    for j in range(4):work[j]=(work[j]-calibration[0,j])/calibration[1,j]
    return evidence(work,*evidence_args)


@njit(cache=False)
def replay(points,args):
    result=np.empty((len(points),24),dtype=np.float32)
    for j in range(len(points)):result[j]=step(float(points[j]),*args)
    return result


class RankDependenceState:
    def __init__(self,historical,settings):
        validate(settings);fit=fit_historical(historical,8);residuals=fit['historical_innovations']
        parameters,conditional=fast_variance_parameters(residuals,'arch1')
        variance=max(float(np.minimum((residuals-parameters[0])**2,parameters[5]).mean()),1e-12)
        z,replayed=historical_z(residuals,parameters,variance)
        np.testing.assert_array_equal(replayed,conditional)
        sorted_h=np.sort(z);rank=rank_values(z,sorted_h)
        center=np.asarray([rank.mean(),(rank*rank).mean(),np.abs(rank).mean()]);empty=np.empty(0)
        raw,ring,counter=historical_raw(rank,empty,center,3,4)
        raw=raw[min(160,len(raw)//4):]
        calibration=np.stack([raw.mean(axis=0),np.maximum(raw.std(axis=0),1e-6)])
        ages=np.asarray([1,2,4,8,16,32,64,128],dtype=np.int64)
        evidence_args=(ages,np.diff(np.r_[0,ages]).astype(float),np.zeros((4,2)),np.zeros((4,2)),
            np.zeros(4),np.zeros((129,4)),np.zeros(1,dtype=np.int64),np.empty(24,dtype=np.float32))
        self.args=(tuple(fit[k] for k in ('reference','coefficients','ring','counter')),
            parameters,conditional,sorted_h,ring,counter,center,calibration,np.empty(4),empty,evidence_args)

    def update(self,point):return step(float(point),*self.args)

    def replay(self,points):return replay(np.asarray(points,dtype=np.float32),self.args)


def make_state(historical,settings):return RankDependenceState(historical,settings)


def feature_names(settings):
    validate(settings)
    return tuple('lag'+str(lag)+'_'+stat for lag in (1,2,4,8) for stat in STATS)


def feature_groups(settings):
    validate(settings);return ('NEXT3_RANK_DEPENDENCE',)*24


def implementation_hash(settings):
    validate(settings);root=Path(__file__).resolve().parents[2]
    sources=('src/next3/rank_dependence.py','src/next/next3_rank_dependence.py','src/next2/dependence_runtime.py',
        'src/next2/complement_bank.py','src/next/normalization.py','src/next/whitening.py',
        'src/next/fast_variance_initialization.py','src/next/channel_evidence.py')
    return hashlib.sha256(b''.join((root/p).read_bytes() for p in sources)+json.dumps(settings,sort_keys=True).encode()).hexdigest()
