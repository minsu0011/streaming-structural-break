"""Finite directional confirmation, occupancy and decaying-shock states."""
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next5 import innovation
from src.next4.scale_asymmetry import branch_reference,branch_values

KINDS=('two_timescale','soft_count','shock_recovery')


def validate(settings):
    if set(settings)!={'version','kind'} or settings['version']!='next5_persistence_v1' or settings['kind'] not in KINDS:
        raise ValueError('Exact persistence recipe required')


@njit(cache=False)
def persistence_step(point,innovation_args,reference,h_stats,ewma,count_ring,count_sum,shock,counter,kind,output):
    z=innovation.step(point,*innovation_args);raw=np.zeros(6);raw[0]=z[0]*z[0]-1.;raw[2]=z[1]*z[1]-1.
    branches=branch_values(raw,reference,1,np.empty(4));t=counter[0]+1;position=counter[0]%32
    for channel in range(4):
        value=branches[channel];mean,sd,threshold,probability=h_stats[channel];offset=channel*4
        standardized=(value-mean)/sd
        if kind==0:
            for j,half in enumerate((8.,64.)):
                alpha=1.-np.exp(-np.log(2.)/half)
                ewma[channel,j]=(1.-alpha)*ewma[channel,j]+alpha*standardized
                normalizer=np.sqrt(max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)),1e-8))
                output[offset+j]=ewma[channel,j]/normalizer
            fast=output[offset];slow=output[offset+1]
            output[offset+2]=min(max(fast,0.),max(slow,0.));output[offset+3]=fast-slow
        elif kind==1:
            indicator=1. if value>threshold else 0.
            count_sum[channel]+=indicator-count_ring[channel,position];count_ring[channel,position]=indicator
            output[offset]=indicator;output[offset+1]=count_sum[channel]/min(t,32)
            for j,half in enumerate((8.,64.)):
                alpha=1.-np.exp(-np.log(2.)/half)
                ewma[channel,j]=(1.-alpha)*ewma[channel,j]+alpha*(indicator-probability)
                output[offset+2+j]=ewma[channel,j]/max(np.sqrt(probability*(1.-probability)),.1)
        else:
            shock[channel]=max(value,np.exp(-np.log(2.)/4.)*shock[channel])
            alpha=1.-np.exp(-np.log(2.)/64.)
            ewma[channel,1]=(1.-alpha)*ewma[channel,1]+alpha*value
            output[offset]=np.log1p(shock[channel]);output[offset+1]=(ewma[channel,1]-mean)/sd
            output[offset+2]=shock[channel]-ewma[channel,1]
            output[offset+3]=ewma[channel,1]/(1.+shock[channel])
    counter[0]=t;return output


@njit(cache=False)
def replay(points,args):
    values=np.empty((len(points),16),dtype=np.float32)
    for j in range(len(points)):values[j]=persistence_step(float(points[j]),*args)
    return values


class PersistenceState:
    def __init__(self,historical,settings):
        validate(settings);args,h=innovation.prepare(historical)
        raw=np.zeros((len(h),6));raw[:,0]=h[:,0]**2-1.;raw[:,2]=h[:,1]**2-1.
        reference,branches=branch_reference(raw,'log_semitail');threshold=np.quantile(branches,.9,axis=0)
        h_stats=np.column_stack([branches.mean(axis=0),np.maximum(branches.std(axis=0),.1),threshold,(branches>threshold).mean(axis=0)])
        ewma=np.zeros((4,2));shock=np.zeros(4);kind=KINDS.index(settings['kind'])
        if kind==2:ewma[:,1]=h_stats[:,0];shock[:]=h_stats[:,0]
        self.args=(args,reference,h_stats,ewma,np.zeros((4,32)),np.zeros(4),shock,np.zeros(1,dtype=np.int64),kind,np.empty(16,dtype=np.float32))

    def update(self,point):return persistence_step(float(point),*self.args)
    def replay(self,points):return replay(np.asarray(points,dtype=np.float32),self.args)


def make_state(historical,settings):return PersistenceState(historical,settings)


def feature_names(settings):
    validate(settings)
    statistics={'two_timescale':('fast8','slow64','positive_confirmation','fast_minus_slow'),
        'soft_count':('above_H_q90','fraction32','centered_count_ewma8','centered_count_ewma64'),
        'shock_recovery':('log_peak_decay4','centered_persistence64','peak_minus_persistence','persistence_over_one_plus_peak')}[settings['kind']]
    return tuple('next5_'+settings['kind']+'_'+channel+'_'+direction+'_'+stat
        for channel in ('raw','conditional') for direction in ('lower','upper') for stat in statistics)


def implementation_hash(settings):
    validate(settings);root=Path(__file__).resolve().parents[2]
    source=Path(__file__).read_bytes()+innovation.source_hash().encode()+(root/'src/next4/scale_asymmetry.py').read_bytes()
    return hashlib.sha256(source+json.dumps(settings,sort_keys=True).encode()).hexdigest()
