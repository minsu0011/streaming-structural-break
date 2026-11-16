"""Bounded rolling sums for raw/trimmed/log scale and multiwindow agreement."""
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next5 import innovation

KINDS=('scale_ratio','trimmed_energy','disagreement')
WINDOWS=(16,32,64,128)


def validate(settings):
    if set(settings)!={'version','kind'} or settings['version']!='next5_windowed_v1' or settings['kind'] not in KINDS:
        raise ValueError('Exact windowed evidence recipe required')


@njit(cache=False)
def window_step(point,innovation_args,reference,cumulative,counter,kind,output):
    z=innovation.step(point,*innovation_args);t=counter[0]+1;old=counter[1];position=(old+1)%129;offset=0
    for channel in range(2):
        energy=z[channel]*z[channel]
        current=(energy,min(energy,reference[channel,1]),np.log(energy+1e-4))
        for stat in range(3):cumulative[channel,stat,position]=cumulative[channel,stat,old]+current[stat]
        ratios=np.empty(4)
        for j,window in enumerate(WINDOWS):
            age=min(t,window);previous=(position-age)%129
            means=(cumulative[channel,:,position]-cumulative[channel,:,previous])/age
            ratios[j]=np.log(max(means[0],1e-8)/reference[channel,0])
            if kind==0:
                output[offset]=ratios[j];output[offset+1]=max(-ratios[j],0.);output[offset+2]=max(ratios[j],0.);offset+=3
            elif kind==1:
                output[offset]=np.log(max(means[1],1e-8)/reference[channel,2]);output[offset+1]=means[2]-reference[channel,3];offset+=2
        if kind==2:
            for fast,slow in ((0,3),(1,2)):
                difference=ratios[fast]-ratios[slow]
                output[offset]=difference;output[offset+1]=abs(difference)
                output[offset+2]=min(max(ratios[fast],0.),max(ratios[slow],0.))
                output[offset+3]=min(max(-ratios[fast],0.),max(-ratios[slow],0.));offset+=4
    counter[0]=t;counter[1]=position;return output


@njit(cache=False)
def replay(points,args):
    values=np.empty((len(points),len(args[-1])),dtype=np.float32)
    for j in range(len(points)):values[j]=window_step(float(points[j]),*args)
    return values


class WindowState:
    def __init__(self,historical,settings):
        validate(settings);args,h=innovation.prepare(historical);energy=h*h
        cap=np.maximum(np.quantile(energy,.95,axis=0),1e-8)
        reference=np.column_stack([np.maximum(energy.mean(axis=0),1e-8),cap,
            np.maximum(np.minimum(energy,cap).mean(axis=0),1e-8),np.log(energy+1e-4).mean(axis=0)])
        kind=KINDS.index(settings['kind']);size=24 if kind==0 else 16
        self.args=(args,reference,np.zeros((2,3,129)),np.zeros(2,dtype=np.int64),kind,np.empty(size,dtype=np.float32))

    def update(self,point):return window_step(float(point),*self.args)
    def replay(self,points):return replay(np.asarray(points,dtype=np.float32),self.args)


def make_state(historical,settings):return WindowState(historical,settings)


def feature_names(settings):
    validate(settings);kind=settings['kind']
    if kind=='disagreement':
        return tuple('next5_windows_'+channel+'_'+pair+'_'+stat for channel in ('raw','conditional')
            for pair in ('16vs128','32vs64') for stat in ('difference','absolute_difference','increase_agreement','decrease_agreement'))
    stats=('log_energy_ratio','decrease','increase') if kind=='scale_ratio' else ('log_trimmed_energy_ratio','mean_log_energy_shift')
    return tuple('next5_'+kind+'_'+channel+'_w'+str(window)+'_'+stat
        for channel in ('raw','conditional') for window in WINDOWS for stat in stats)


def implementation_hash(settings):
    validate(settings)
    return hashlib.sha256(Path(__file__).read_bytes()+innovation.source_hash().encode()+json.dumps(settings,sort_keys=True).encode()).hexdigest()
