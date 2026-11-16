"""H-adaptive tails, energy PIT and quantile self-normalization, without targets."""
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next5 import innovation,scale

KINDS=('adaptive_t','energy_pit','quantile_scale')


def validate(settings):
    if set(settings)!={'version','kind'} or settings['version']!='next5_robust_null_v1' or settings['kind'] not in KINDS:
        raise ValueError('Exact robust-null recipe required')


@njit(cache=False)
def energy_pit(value,sorted_h):
    left=np.searchsorted(sorted_h,value,side='left');right=np.searchsorted(sorted_h,value,side='right')
    return (left+right+1.)/(2.*(len(sorted_h)+1.))


@njit(cache=False)
def t_likelihoods(z,nu,output):
    for j in range(4):
        variance=scale.VARIANCES[j]
        output[j]=-.5*np.log(variance)-.5*(nu+1.)*(np.log1p(z*z/((nu-2.)*variance))-np.log1p(z*z/(nu-2.)))
    return output


@njit(cache=False)
def robust_step(point,innovation_args,nu,sorted_energy,quantiles,branch_reference,cumulative,ewma,counter,kind,output):
    z=innovation.step(point,*innovation_args);t=counter[0]+1;old=counter[1];position=(old+1)%129;offset=0
    for channel in range(2):
        if kind==0:
            current=t_likelihoods(z[channel],nu[channel],np.empty(4))
            cumulative[channel,:,position]=cumulative[channel,:,old]+current
            for window in (8,32,128):
                age=min(t,window);previous=(position-age)%129
                values=cumulative[channel,:,position]-cumulative[channel,:,previous]
                scale.four_summaries(values,output,offset);offset+=4
            scale.four_summaries(current,output,offset);offset+=4
        else:
            if kind==1:
                u=energy_pit(z[channel]*z[channel],sorted_energy[channel])
                values=(-np.log(u),-np.log1p(-u))
            else:
                normalized=min(max((z[channel]-quantiles[channel,1])/quantiles[channel,3],-12.),12.)
                centered=np.log1p(normalized*normalized)-quantiles[channel,4]
                values=(max(-centered,0.),max(centered,0.))
            for direction in range(2):
                branch=channel*2+direction;value=values[direction]
                standardized=(value-branch_reference[branch,0])/branch_reference[branch,1]
                output[offset]=standardized
                for j,half in enumerate((8.,32.,128.)):
                    alpha=1.-np.exp(-np.log(2.)/half)
                    ewma[branch,j]=(1.-alpha)*ewma[branch,j]+alpha*standardized
                    normalizer=np.sqrt(max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)),1e-8))
                    output[offset+j+1]=ewma[branch,j]/normalizer
                offset+=4
    counter[0]=t;counter[1]=position;return output


@njit(cache=False)
def replay(points,args):
    values=np.empty((len(points),len(args[-1])),dtype=np.float32)
    for j in range(len(points)):values[j]=robust_step(float(points[j]),*args)
    return values


class RobustNullState:
    def __init__(self,historical,settings):
        validate(settings);args,h=innovation.prepare(historical);kind=KINDS.index(settings['kind'])
        second=np.maximum(np.mean(h*h,axis=0),1e-8);kurtosis=np.mean(h**4,axis=0)/(second*second)
        nu=np.clip(4.+6./np.maximum(kurtosis-3.,1e-6),4.,30.)
        energy=h*h;sorted_energy=np.sort(energy,axis=0).T.copy();quantiles=np.zeros((2,5));branches=np.zeros((len(h),4))
        for channel in range(2):
            q=np.quantile(h[:,channel],[.25,.5,.75]);spread=max(float((q[2]-q[0])/1.349),.1)
            normalized=np.clip((h[:,channel]-q[1])/spread,-12.,12.)
            log_energy=np.log1p(normalized*normalized);center=float(np.median(log_energy))
            quantiles[channel]=[q[0],q[1],q[2],spread,center]
            if kind==1:
                u=np.asarray([energy_pit(v,sorted_energy[channel]) for v in energy[:,channel]])
                branches[:,channel*2]=-np.log(u);branches[:,channel*2+1]=-np.log1p(-u)
            elif kind==2:
                branches[:,channel*2]=np.maximum(center-log_energy,0.);branches[:,channel*2+1]=np.maximum(log_energy-center,0.)
        reference=np.column_stack([branches.mean(axis=0),np.maximum(branches.std(axis=0),.1)])
        self.args=(args,nu,sorted_energy,quantiles,reference,np.zeros((2,4,129)),np.zeros((4,3)),np.zeros(2,dtype=np.int64),kind,np.empty(32 if kind==0 else 16,dtype=np.float32))
    def update(self,point):return robust_step(float(point),*self.args)
    def replay(self,points):return replay(np.asarray(points,dtype=np.float32),self.args)


def make_state(historical,settings):return RobustNullState(historical,settings)


def feature_names(settings):
    validate(settings);kind=settings['kind']
    if kind=='adaptive_t':return tuple('next5_adaptive_t_'+channel+'_'+window+'_'+stat for channel in ('raw','conditional')
        for window in ('w8','w32','w128','current') for stat in ('lower','upper','upper_minus_lower','total'))
    return tuple('next5_'+kind+'_'+channel+'_'+direction+'_'+stat for channel in ('raw','conditional')
        for direction in ('lower','upper') for stat in ('current','ewma8','ewma32','ewma128'))


def implementation_hash(settings):
    validate(settings)
    source=Path(__file__).read_bytes()+innovation.source_hash().encode()+scale.implementation_hash({'version':'next5_scale_unification_v1','kind':'directional'}).encode()
    return hashlib.sha256(source+json.dumps(settings,sort_keys=True).encode()).hexdigest()
