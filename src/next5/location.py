"""Low-budget robust signed location and sign-run evidence, distinct from D1."""
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next5 import innovation

KINDS=('student_location','sign_run')
SHIFTS=(-.75,.75)


def validate(settings):
    if set(settings)!={'version','kind'} or settings['version']!='next5_location_v1' or settings['kind'] not in KINDS:
        raise ValueError('Exact low-budget location recipe required')


@njit(cache=False)
def location_lr(z,shift):return -3.*(np.log1p((z-shift)**2/3.)-np.log1p(z*z/3.))


@njit(cache=False)
def location_step(point,innovation_args,reference,sign_parameters,ewma,last_sign,runs,counter,kind,output):
    z=innovation.step(point,*innovation_args);t=counter[0]+1;offset=0
    for channel in range(2):
        if kind==0:
            for direction in range(2):
                branch=channel*2+direction;value=location_lr(z[channel],SHIFTS[direction])
                standardized=(value-reference[branch,0])/reference[branch,1];output[offset]=standardized
                for j,half in enumerate((8.,32.,128.)):
                    alpha=1.-np.exp(-np.log(2.)/half);ewma[branch,j]=(1.-alpha)*ewma[branch,j]+alpha*standardized
                    normalizer=np.sqrt(max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)),1e-8));output[offset+j+1]=ewma[branch,j]/normalizer
                offset+=4
        else:
            median,mean,sd,same_probability=sign_parameters[channel]
            sign=1 if z[channel]>median else -1 if z[channel]<median else 0
            runs[channel]=runs[channel]+1 if sign!=0 and sign==last_sign[channel] else 1 if sign!=0 else 0
            last_sign[channel]=sign;standardized=(sign-mean)/sd;output[offset]=standardized
            for j,half in enumerate((8.,32.,128.)):
                alpha=1.-np.exp(-np.log(2.)/half);ewma[channel,j]=(1.-alpha)*ewma[channel,j]+alpha*standardized
                normalizer=np.sqrt(max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)),1e-8));output[offset+j+1]=ewma[channel,j]/normalizer
            output[offset+4]=sign*np.log1p(runs[channel])
            output[offset+5]=sign*(1.-np.exp(runs[channel]*np.log(same_probability)));offset+=6
    counter[0]=t;return output


@njit(cache=False)
def replay(points,args):
    values=np.empty((len(points),len(args[-1])),dtype=np.float32)
    for j in range(len(points)):values[j]=location_step(float(points[j]),*args)
    return values


class LocationState:
    def __init__(self,historical,settings):
        validate(settings);args,h=innovation.prepare(historical);kind=KINDS.index(settings['kind'])
        reference=np.empty((4,2));sign_parameters=np.empty((2,4))
        for channel in range(2):
            for direction,shift in enumerate(SHIFTS):
                values=-3.*(np.log1p((h[:,channel]-shift)**2/3.)-np.log1p(h[:,channel]**2/3.))
                reference[channel*2+direction]=[float(values.mean()),max(float(values.std()),.1)]
            median=float(np.median(h[:,channel]));signs=np.sign(h[:,channel]-median)
            same=(float(np.sum(signs[1:]==signs[:-1]))+1.)/(len(signs)+1.)
            sign_parameters[channel]=[median,float(signs.mean()),max(float(signs.std()),.1),same]
        self.args=(args,reference,sign_parameters,np.zeros((4,3)),np.zeros(2,dtype=np.int8),np.zeros(2,dtype=np.int64),
            np.zeros(1,dtype=np.int64),kind,np.empty(16 if kind==0 else 12,dtype=np.float32))
    def update(self,point):return location_step(float(point),*self.args)
    def replay(self,points):return replay(np.asarray(points,dtype=np.float32),self.args)


def make_state(historical,settings):return LocationState(historical,settings)


def feature_names(settings):
    validate(settings)
    if settings['kind']=='student_location':return tuple('next5_t_location_'+channel+'_'+direction+'_'+stat
        for channel in ('raw','conditional') for direction in ('negative','positive') for stat in ('current','ewma8','ewma32','ewma128'))
    return tuple('next5_sign_run_'+channel+'_'+stat for channel in ('raw','conditional')
        for stat in ('current','ewma8','ewma32','ewma128','signed_log_run','signed_geometric_run_evidence'))


def implementation_hash(settings):
    validate(settings)
    return hashlib.sha256(Path(__file__).read_bytes()+innovation.source_hash().encode()+json.dumps(settings,sort_keys=True).encode()).hexdigest()
