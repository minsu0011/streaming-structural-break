"""Fixed working-likelihood mixtures; no real-DGP anytime-validity claim."""
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next5 import innovation,scale

KINDS=('cumulative','sparse_starts')
AGES=(1,2,4,8,16,32,64,128,256)


def validate(settings):
    if set(settings)!={'version','kind'} or settings['version']!='next5_fixed_mixture_v1' or settings['kind'] not in KINDS:
        raise ValueError('Exact finite mixture recipe required')


@njit(cache=False)
def logmean(values):
    maximum=np.max(values)
    return maximum+np.log(np.exp(values-maximum).mean())


@njit(cache=False)
def mixture_step(point,innovation_args,cumulative,running,counter,kind,output):
    z=innovation.step(point,*innovation_args);t=counter[0]+1;old=counter[1];position=(old+1)%257;offset=0
    for channel in range(2):
        increment=scale.likelihoods(z[channel],False,np.empty(4));running[channel]+=increment
        cumulative[channel,:,position]=cumulative[channel,:,old]+increment
        if kind==0:
            evidence=running[channel];lower=logmean(evidence[:2]);upper=logmean(evidence[2:]);total=logmean(evidence)
            weights=np.exp(evidence-np.max(evidence));weights/=weights.sum()
            entropy=-np.sum(weights*np.log(np.maximum(weights,1e-300)))
            output[offset]=lower;output[offset+1]=upper;output[offset+2]=total;output[offset+3]=upper-lower
            output[offset+4]=entropy;output[offset+5]=np.sum(weights*np.log(scale.VARIANCES));offset+=6
        else:
            evidence=np.empty((9,4));effective_age=np.empty(9)
            for j,age in enumerate(AGES):
                length=min(t,age);previous=(position-length)%257;effective_age[j]=length
                evidence[j]=cumulative[channel,:,position]-cumulative[channel,:,previous]
            lower=logmean(evidence[:,:2]);upper=logmean(evidence[:,2:]);total=logmean(evidence)
            weights=np.exp(evidence-np.max(evidence));weights/=weights.sum()
            entropy=-np.sum(weights*np.log(np.maximum(weights,1e-300)))
            age_posterior=weights.sum(axis=1);variance_posterior=weights.sum(axis=0)
            output[offset]=lower;output[offset+1]=upper;output[offset+2]=total;output[offset+3]=upper-lower
            output[offset+4]=entropy;output[offset+5]=np.sum(age_posterior*np.log(effective_age))
            output[offset+6]=np.sum(variance_posterior*np.log(scale.VARIANCES))
            output[offset+7]=logmean(evidence[:4]);output[offset+8]=logmean(evidence[4:]);offset+=9
    counter[0]=t;counter[1]=position;return output


@njit(cache=False)
def replay(points,args):
    values=np.empty((len(points),len(args[-1])),dtype=np.float32)
    for j in range(len(points)):values[j]=mixture_step(float(points[j]),*args)
    return values


class MixtureState:
    def __init__(self,historical,settings):
        validate(settings);args,_=innovation.prepare(historical);kind=KINDS.index(settings['kind'])
        self.args=(args,np.zeros((2,4,257)),np.zeros((2,4)),np.zeros(2,dtype=np.int64),kind,np.empty(12 if kind==0 else 18,dtype=np.float32))
    def update(self,point):return mixture_step(float(point),*self.args)
    def replay(self,points):return replay(np.asarray(points,dtype=np.float32),self.args)


def make_state(historical,settings):return MixtureState(historical,settings)


def feature_names(settings):
    validate(settings)
    statistics=('lower','upper','total','upper_minus_lower','alternative_entropy','posterior_log_variance') if settings['kind']=='cumulative' else (
        'lower','upper','total','upper_minus_lower','alternative_entropy','posterior_log_age','posterior_log_variance','short_age_mixture','long_age_mixture')
    return tuple('next5_mixture_'+settings['kind']+'_'+channel+'_'+stat for channel in ('raw','conditional') for stat in statistics)


def implementation_hash(settings):
    validate(settings)
    source=Path(__file__).read_bytes()+innovation.source_hash().encode()+scale.implementation_hash({'version':'next5_scale_unification_v1','kind':'directional'}).encode()
    return hashlib.sha256(source+json.dumps(settings,sort_keys=True).encode()).hexdigest()
