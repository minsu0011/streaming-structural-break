"""Small mean, conditional-dependence and incremental CDF complements."""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next.whitening import fit_historical,innovation_update
from src.next.normalization import fit_normalization,ecdf_uniform
from src.next.channel_evidence import channel_step,make_channel_args,CHANNEL_STATS

FAMILIES=('mean','rank_sign','page_hinkley','dependence','squared_dependence','absolute_dependence','cdf')


@dataclass(frozen=True)
class ComplementConfig:
    family:str='mean'
    order:int=8
    cdf_bins:int=8
    cap:int=4


def config(settings):
    c=ComplementConfig(**settings)
    if c.family not in FAMILIES or c.order!=8 or c.cdf_bins not in (8,16) or c.cap!=4:
        raise ValueError('Complement configuration outside fixed small family')
    return c


def n_channels(c):
    return 4 if c.family=='dependence' else 2 if c.family in ('squared_dependence','absolute_dependence') else 1


@njit(cache=True)
def normalized_one(residual,parameters,conditional):
    variance=max(parameters[2]+parameters[3]*conditional[0]+parameters[4]*conditional[1],parameters[6])
    centered=residual-parameters[0]
    z=min(max(centered/np.sqrt(variance),-4.),4.)
    squared=min(centered*centered,parameters[5])
    conditional[1]=conditional[0]
    conditional[0]=squared
    conditional[2]=(1.-parameters[7])*conditional[2]+parameters[7]*squared
    return z


@njit(cache=True)
def raw_complement(z,sorted_h,ring,counter,center,family,work):
    t=counter[0]
    if family==0 or family==2:
        work[0]=z
    elif family==1:
        work[0]=2.*ecdf_uniform(z,sorted_h)-1.
    elif family==3:
        for k,lag in enumerate((1,2,4,8)):
            work[k]=(z-center[0])*(ring[(t-lag)%8]-center[0])
    elif family==4:
        for k,lag in enumerate((1,4)):
            previous=ring[(t-lag)%8]
            work[k]=(z*z-center[1])*(previous*previous-center[1])
    elif family==5:
        for k,lag in enumerate((1,4)):
            previous=ring[(t-lag)%8]
            work[k]=(abs(z)-center[2])*(abs(previous)-center[2])
    ring[t%8]=z
    counter[0]=t+1
    return work


@njit(cache=True)
def historical_raw(z,sorted_h,center,family,channels):
    ring=np.zeros(8)
    counter=np.zeros(1,dtype=np.int64)
    output=np.empty((len(z),channels))
    for j in range(len(z)):
        raw_complement(z[j],sorted_h,ring,counter,center,family,output[j])
    return output,ring,counter


@njit(cache=True)
def cdf_step(u,baseline,ewma,output):
    bins=len(baseline)
    cell=min(int(u*bins),bins-1)
    for k,half in enumerate((8.,32.,128.)):
        alpha=1.-np.exp(-np.log(2.)/half)
        for b in range(bins):
            ewma[k,b]=(1.-alpha)*ewma[k,b]+alpha*(1. if b==cell else 0.)
        chi,cvm,cumulative,tail,center,imbalance,maximum=0.,0.,0.,0.,0.,0.,0.
        for b in range(bins):
            d=ewma[k,b]-baseline[b]
            chi+=d*d/max(baseline[b],1e-8)
            cumulative+=d
            cvm+=cumulative*cumulative/bins
            if b<bins//8 or b>=7*bins//8:
                tail+=d
            if 3*bins//8<=b<5*bins//8:
                center+=d
            imbalance+=d*(1. if b>=bins//2 else -1.)
            maximum=max(maximum,abs(d))
        output[k*6:k*6+6]=np.array([chi,cvm,tail,center,imbalance,maximum])
    return output


@njit(cache=True)
def step(point,filter_args,parameters,conditional,sorted_h,ring,counter,center,family,calibration,work,evidence_args,baseline,bin_ewma,output):
    residual=innovation_update(point,*filter_args)
    z=normalized_one(residual,parameters,conditional)
    if family==6:
        return cdf_step(ecdf_uniform(z,sorted_h),baseline,bin_ewma,output)
    raw_complement(z,sorted_h,ring,counter,center,family,work)
    for j in range(len(work)):
        work[j]=(work[j]-calibration[0,j])/calibration[1,j]
    if family==2:
        # H-fixed center, clipped innovation, two Page-Hinkley drift settings.
        for k,drift in enumerate((.1,.25)):
            output[k*2]=max(0.,output[k*2]+work[0]-drift)
            output[k*2+1]=max(0.,output[k*2+1]-work[0]-drift)
        return output
    return channel_step(work,evidence_args)


@njit(cache=True)
def replay(points,args,size):
    result=np.empty((len(points),size),dtype=np.float32)
    for j in range(len(points)):
        result[j]=step(points[j],*args)
    return result


class ComplementState:
    def __init__(self,historical,settings):
        c=config(settings); family=FAMILIES.index(c.family)
        fit=fit_historical(historical,c.order)
        profile=fit_normalization(fit['historical_innovations'],'arch1')
        z=np.clip(profile['historical_z'],-4.,4.)
        sorted_h=np.sort(z)
        center=np.asarray([z.mean(),(z*z).mean(),np.abs(z).mean()])
        raw,ring,counter=historical_raw(z,sorted_h,center,family,n_channels(c)) if c.family!='cdf' else (np.zeros((len(z),1)),np.zeros(8),np.zeros(1,dtype=np.int64))
        raw=raw[min(160,len(raw)//4):]
        calibration=np.stack([raw.mean(axis=0),np.maximum(raw.std(axis=0),1e-6)])
        uniform=np.asarray([ecdf_uniform(v,sorted_h) for v in z])
        counts=np.bincount(np.minimum((uniform*c.cdf_bins).astype(int),c.cdf_bins-1),minlength=c.cdf_bins)
        baseline=(counts+.5)/(len(z)+.5*c.cdf_bins)
        bin_ewma=np.tile(baseline,(3,1))
        size=18 if c.family=='cdf' else 4 if c.family=='page_hinkley' else n_channels(c)*13
        self.args=(tuple(fit[k] for k in ('reference','coefficients','ring','counter')),
            profile['parameters'],profile['conditional'].copy(),sorted_h,ring,counter,center,family,
            calibration,np.empty(n_channels(c)),make_channel_args(n_channels(c),128),
            baseline,bin_ewma,np.zeros(size,dtype=np.float32))
        self.size=size

    def update(self,point):
        return step(float(point),*self.args)

    def replay(self,points):
        return replay(np.asarray(points,dtype=np.float32),self.args,self.size)


def feature_names(settings):
    c=config(settings)
    if c.family=='cdf':
        return tuple('cdf'+str(c.cdf_bins)+'_'+str(h)+'_'+s for h in (8,32,128) for s in ('chi_square','cvm','tail_mass','center_mass','imbalance','max_bin'))
    if c.family=='page_hinkley':
        return ('ph01_positive','ph01_negative','ph025_positive','ph025_negative')
    return tuple(c.family+'_'+str(k)+'_'+s for k in range(n_channels(c)) for s in CHANNEL_STATS)


def feature_groups(settings):
    return ('NEXT2_COMPLEMENT',)*len(feature_names(settings))


def make_state(historical,settings):
    return ComplementState(historical,settings)


def implementation_hash(settings):
    sources=[Path(__file__),Path(__file__).parents[1]/'next/normalization.py',
        Path(__file__).parents[1]/'next/whitening.py',Path(__file__).parents[1]/'next/channel_evidence.py']
    return hashlib.sha256(b''.join(p.read_bytes() for p in sources)+json.dumps(asdict(config(settings)),sort_keys=True).encode()).hexdigest()
