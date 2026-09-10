"""Four causal dependence summaries added to the exact J102 feature bank."""
from pathlib import Path
from dataclasses import dataclass,asdict
import json
import hashlib
import numpy as np
from numba import njit
from src.next2.pruned_runtime import CompactPairState
from src.next2.complement_bank import ComplementState,step as complement_step
from src.next.serial_channel_evidence import serial_channel_step
from src.next2.pruned_runtime import compact_serial_step
from src.next.channel_evidence import CHANNEL_STATS


@dataclass(frozen=True)
class JointConfig:
    summary:str='ewma32'


def config(settings):
    c=JointConfig(**settings)
    if c.summary not in ('ewma32','bayes025'):
        raise ValueError('Only two prospectively bounded dependence summaries allowed')
    return c


@njit(cache=True)
def step(point,parent_args,complement_args,columns,output):
    parent=compact_serial_step(point,*parent_args)
    extra=complement_step(point,*complement_args)
    output[:26]=parent
    for j in range(4):
        output[26+j]=extra[columns[j]]
    return output


@njit(cache=True)
def replay(points,args):
    result=np.empty((len(points),30),dtype=np.float32)
    for j in range(len(points)):
        result[j]=step(points[j],*args)
    return result


class JointState:
    def __init__(self,historical,settings):
        c=config(settings)
        parent=CompactPairState(historical,{'order':8,'normalization':'arch1','max_age':512},'serial')
        complement=ComplementState(historical,{'family':'dependence','order':8,'cdf_bins':8,'cap':4})
        index=CHANNEL_STATS.index(c.summary)
        columns=np.asarray([13*k+index for k in range(4)],dtype=np.int32)
        self.args=(parent.call_args,complement.args,columns,np.empty(30,dtype=np.float32))

    def update(self,point):
        return step(float(point),*self.args)

    def replay(self,points):
        return replay(np.asarray(points,dtype=np.float32),self.args)


def feature_names(settings):
    c=config(settings)
    names=tuple(channel+'_'+stat for channel in ('raw_energy','log_variance_forecast') for stat in CHANNEL_STATS)
    return names+tuple('conditional_dependence_lag'+str(lag)+'_'+c.summary for lag in (1,2,4,8))


def feature_groups(settings):
    return ('NEXT2_JOINT_DEPENDENCE',)*len(feature_names(settings))


def make_state(historical,settings):
    return JointState(historical,settings)


def implementation_hash(settings):
    paths=[Path(__file__),Path(__file__).with_name('pruned_runtime.py'),Path(__file__).with_name('complement_bank.py'),
        Path(__file__).parents[1]/'next/next2_joint_dependence.py',Path(__file__).parents[1]/'next/serial_channel_evidence.py']
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)+json.dumps(asdict(config(settings)),sort_keys=True).encode()).hexdigest()
