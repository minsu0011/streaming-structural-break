"""Separate lower/upper working scale likelihoods, robust tails and semitails.

Likelihoods are ranking features. H-estimated dependent nulls do not establish
an e-process or anytime-valid alpha guarantee on the competition distribution.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next5 import innovation,reference_features
from src.next4.scale_asymmetry import branch_reference,branch_values

KINDS=('directional','student','augmented_j','unified')
VARIANCES=np.asarray([.5,.7,1.5,2.])
WINDOWS=(8,32,128)


def validate(settings):
    if set(settings)!={'version','kind'} or settings['version']!='next5_scale_unification_v1' or settings['kind'] not in KINDS:
        raise ValueError('Exact scale-unification recipe required')


@njit(cache=False)
def likelihoods(z,student,output):
    square=z*z
    for j in range(4):
        variance=VARIANCES[j]
        if student:output[j]=-.5*np.log(variance)-3.*(np.log1p(square/(3.*variance))-np.log1p(square/3.))
        else:output[j]=-.5*np.log(variance)+.5*square*(1.-1./variance)
    return output


@njit(cache=False)
def logmean_pair(a,b):return np.logaddexp(a,b)-np.log(2.)


@njit(cache=False)
def four_summaries(values,output,offset):
    lower=logmean_pair(values[0],values[1]);upper=logmean_pair(values[2],values[3])
    output[offset]=lower;output[offset+1]=upper;output[offset+2]=upper-lower
    output[offset+3]=logmean_pair(lower,upper)


@njit(cache=False)
def scale_step(point,innovation_args,cumulative,counter,reference,branch_ewma,branch_cusum,kind,output):
    z=innovation.step(point,*innovation_args);t=counter[0]+1;old=counter[1];position=(old+1)%129
    work=np.empty(4);offset=0
    for distribution in range(2):
        for channel in range(2):
            likelihoods(z[channel],distribution==1,work)
            for j in range(4):cumulative[distribution,channel,j,position]=cumulative[distribution,channel,j,old]+work[j]
            included=kind==3 or distribution==(1 if kind in (1,2) else 0)
            if included:
                for window in WINDOWS:
                    age=min(t,window);previous=(position-age)%129
                    values=cumulative[distribution,channel,:,position]-cumulative[distribution,channel,:,previous]
                    four_summaries(values,output,offset);offset+=4
                four_summaries(work,output,offset);offset+=4
    if kind==3:
        raw=np.zeros(6);raw[0]=z[0]*z[0]-1.;raw[2]=z[1]*z[1]-1.
        branch_values(raw,reference,1,work)
        for channel in range(4):
            j=channel//2;direction=channel%2
            value=(work[channel]-reference[j,3+direction])/reference[j,5+direction]
            output[offset]=work[channel]
            for k,half in enumerate((8.,32.)):
                alpha=1.-np.exp(-np.log(2.)/half)
                branch_ewma[channel,k]=(1.-alpha)*branch_ewma[channel,k]+alpha*value
                normalizer=np.sqrt(max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)),1e-8))
                output[offset+1+k]=branch_ewma[channel,k]/normalizer
            branch_cusum[channel]=max(0.,branch_cusum[channel]+min(max(value,-12.),12.)-.25)
            output[offset+3]=np.log1p(branch_cusum[channel]);offset+=4
    counter[0]=t;counter[1]=position
    return output


@njit(cache=False)
def replay_scale(points,args):
    values=np.empty((len(points),len(args[-1])),dtype=np.float32)
    for j in range(len(points)):values[j]=scale_step(float(points[j]),*args)
    return values


class ScaleState:
    def __init__(self,historical,settings):
        validate(settings);self.kind=settings['kind'];args,h=innovation.prepare(historical)
        raw=np.zeros((len(h),6));raw[:,0]=h[:,0]**2-1.;raw[:,2]=h[:,1]**2-1.
        reference,_=branch_reference(raw,'log_semitail')
        size=80 if self.kind=='unified' else 32
        self.scale_args=(args,np.zeros((2,2,4,129)),np.zeros(2,dtype=np.int64),reference,
            np.zeros((4,2)),np.zeros(4),KINDS.index(self.kind),np.empty(size,dtype=np.float32))
        self.original=reference_features.make_state(historical,{'version':'next5_reference_features_v1','kind':'J102'}) if self.kind=='augmented_j' else None
        self.args=(self.scale_args,self.original.args) if self.original else self.scale_args

    def update(self,point):
        values=scale_step(float(point),*self.scale_args)
        return np.r_[self.original.update(point),values].astype(np.float32) if self.original else values

    def replay(self,points):
        points=np.asarray(points,dtype=np.float32);values=replay_scale(points,self.scale_args)
        return np.column_stack([self.original.replay(points),values]).astype(np.float32) if self.original else values


def make_state(historical,settings):return ScaleState(historical,settings)


def feature_names(settings):
    validate(settings);kind=settings['kind']
    distributions=('gaussian','student5') if kind=='unified' else (('gaussian',) if kind=='directional' else ('student5',))
    names=tuple('next5_'+distribution+'_'+channel+'_'+window+'_'+stat
        for distribution in distributions for channel in ('raw','conditional')
        for window in ('w8','w32','w128','current') for stat in ('lower','upper','upper_minus_lower','total'))
    if kind=='unified':names+=tuple('next5_semitail_'+channel+'_'+direction+'_'+stat
        for channel in ('raw','conditional') for direction in ('lower','upper') for stat in ('current','ewma8','ewma32','cusum'))
    if kind=='augmented_j':names=tuple(reference_features.feature_names({'version':'next5_reference_features_v1','kind':'J102'}))+names
    return names


def implementation_hash(settings):
    validate(settings);root=Path(__file__).resolve().parents[2]
    source=(root/'src/next5/scale.py').read_bytes()+innovation.source_hash().encode()+(root/'src/next4/scale_asymmetry.py').read_bytes()
    if settings['kind']=='augmented_j':source+=reference_features.implementation_hash({'version':'next5_reference_features_v1','kind':'J102'}).encode()
    return hashlib.sha256(source+json.dumps(settings,sort_keys=True).encode()).hexdigest()
