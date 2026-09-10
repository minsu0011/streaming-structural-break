"""C205 evidence with one intervention: sign before lag-product construction."""
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next.whitening import fit_historical,innovation_update
from src.next.fast_variance_initialization import fast_variance_parameters
from src.next2.complement_bank import normalized_one,historical_raw,raw_complement
from src.next2.dependence_runtime import historical_z,evidence,STATS


def validate(settings):
    if settings!={'version':'sign_dependence_v1','order':8}:raise ValueError('Frozen sign-dependence recipe required')


@njit(cache=False)
def step(point,filter_args,parameters,conditional,ring,counter,center,calibration,work,empty,evidence_args):
    residual=innovation_update(point,*filter_args)
    z=normalized_one(residual,parameters,conditional)
    z=1. if z>0. else -1. if z<0. else 0.
    raw_complement(z,empty,ring,counter,center,3,work)
    for j in range(4):work[j]=(work[j]-calibration[0,j])/calibration[1,j]
    return evidence(work,*evidence_args)


@njit(cache=False)
def replay(points,args):
    matrix=np.empty((len(points),24),dtype=np.float32)
    for j in range(len(points)):matrix[j]=step(float(points[j]),*args)
    return matrix


class SignDependenceState:
    def __init__(self,historical,settings):
        validate(settings);fit=fit_historical(historical,8);r=fit['historical_innovations']
        parameters,conditional=fast_variance_parameters(r,'arch1')
        variance=max(float(np.minimum((r-parameters[0])**2,parameters[5]).mean()),1e-12)
        z,replayed=historical_z(r,parameters,variance);np.testing.assert_array_equal(replayed,conditional)
        z=np.sign(z)
        center=np.asarray([z.mean(),(z*z).mean(),np.abs(z).mean()]);empty=np.empty(0)
        raw,ring,counter=historical_raw(z,empty,center,3,4);raw=raw[min(160,len(raw)//4):]
        calibration=np.stack([raw.mean(axis=0),np.maximum(raw.std(axis=0),1e-6)])
        ages=np.asarray([1,2,4,8,16,32,64,128],dtype=np.int64)
        evidence_args=(ages,np.diff(np.r_[0,ages]).astype(float),np.zeros((4,2)),np.zeros((4,2)),
            np.zeros(4),np.zeros((129,4)),np.zeros(1,dtype=np.int64),np.empty(24,dtype=np.float32))
        self.args=(tuple(fit[k] for k in ('reference','coefficients','ring','counter')),
            parameters,conditional,ring,counter,center,calibration,np.empty(4),empty,evidence_args)

    def update(self,point):return step(float(point),*self.args)

    def replay(self,points):return replay(np.asarray(points,dtype=np.float32),self.args)


def make_state(historical,settings):return SignDependenceState(historical,settings)


def feature_names(settings):
    validate(settings)
    return tuple('sign_dependence__lag'+str(lag)+'_'+stat for lag in (1,2,4,8) for stat in STATS)


def feature_groups(settings):
    validate(settings);return ('NEXT2_SIGN_DEPENDENCE',)*24


def implementation_hash(settings):
    validate(settings);root=Path(__file__).resolve().parents[2]
    files=('src/next2/sign_dependence.py','src/next/next2_sign_dependence.py','src/next2/dependence_runtime.py',
        'src/next2/complement_bank.py','src/next/whitening.py','src/next/fast_variance_initialization.py','src/next/channel_evidence.py')
    return hashlib.sha256(b''.join((root/f).read_bytes() for f in files)+json.dumps(settings,sort_keys=True).encode()).hexdigest()
