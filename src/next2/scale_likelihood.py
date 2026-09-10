"""Directional scale likelihood and conjugate variance-prior evidence."""
from pathlib import Path
from dataclasses import dataclass,asdict
import hashlib
import json
import math
import numpy as np
from numba import njit
from src.next.whitening import fit_historical,innovation_update
from src.next.fast_variance_initialization import fast_variance_parameters
from src.next.channel_evidence import _logadd,CHANNEL_STATS
from src.next.variance_evidence import VarianceState,VarianceConfig,variance_step

STATS=('lower_glr_log1p','upper_glr_log1p','variance_bayes2_signed_log1p','variance_bayes8_signed_log1p','short_logratio','short_long_logratio')


@dataclass(frozen=True)
class ScaleConfig:
    channel_pair:str='raw_forecast'
    cap:int=0
    include_parent:str='none'


def config(settings):
    c=ScaleConfig(**settings)
    if c.channel_pair not in ('raw_forecast','raw_conditional') or c.cap not in (0,4) or c.include_parent not in ('none','lower','ratios'):
        raise ValueError('Unregistered directional scale likelihood configuration')
    if c.include_parent!='none' and (c.channel_pair!='raw_forecast' or c.cap!=0):
        raise ValueError('Parent-addition contrasts preserve raw/forecast and no clipping')
    return c


@njit(cache=True)
def variance_log_bayes(total,n,alpha):
    beta=alpha-1.
    return alpha*np.log(beta)-math.lgamma(alpha)+math.lgamma(alpha+.5*n)-(alpha+.5*n)*np.log(beta+.5*total)+.5*total


@njit(cache=True)
def directional_glr(total,n):
    ratio=max(total/n,1e-8)
    value=max(0.,.5*(total-n-n*np.log(ratio)))
    return (value,0.) if ratio<1. else (0.,value)


@njit(cache=True)
def energies_one(residual,parameters,conditional,cap,pair,work):
    reference=parameters[2]/max(1.-parameters[3]-parameters[4],.05)
    variance=max(parameters[2]+parameters[3]*conditional[0]+parameters[4]*conditional[1],parameters[6])
    centered=residual-parameters[0]
    squared=centered*centered
    work[0]=min(squared/reference,float(cap*cap)) if cap else squared/reference
    work[1]=variance/reference if pair==0 else squared/variance
    if cap and pair==1:
        work[1]=min(work[1],float(cap*cap))
    capped=min(squared,parameters[5])
    conditional[1]=conditional[0]
    conditional[0]=capped
    conditional[2]=(1.-parameters[7])*conditional[2]+parameters[7]*capped
    return work


@njit(cache=True)
def historical_energies(residuals,parameters,conditional,cap,pair):
    result=np.empty((len(residuals),2))
    for j in range(len(residuals)):
        energies_one(residuals[j],parameters,conditional,cap,pair,result[j])
    return result


@njit(cache=True)
def likelihood_step(values,ages,widths,cumulative,prefix,counter,output):
    t=counter[0]+1
    for channel in range(2):
        # Fixed numerical bound, separate from the robust |z|=4 intervention.
        cumulative[channel]+=min(max(values[channel],1e-12),1e6)
        lower,upper=0.,0.
        b2,b8,width=-1e300,-1e300,0.
        for k in range(len(ages)):
            n=ages[k]
            if n>t:
                break
            total=max(cumulative[channel]-prefix[(t-n)%513,channel],1e-12)
            lo,up=directional_glr(total,n)
            lower=max(lower,lo); upper=max(upper,up)
            b2=_logadd(b2,np.log(widths[k])+variance_log_bayes(total,n,2.))
            b8=_logadd(b8,np.log(widths[k])+variance_log_bayes(total,n,8.))
            width+=widths[k]
        short=min(t,32); long=min(t,512)
        short_mean=max((cumulative[channel]-prefix[(t-short)%513,channel])/short,1e-8)
        long_mean=max((cumulative[channel]-prefix[(t-long)%513,channel])/long,1e-8)
        cursor=channel*6
        output[cursor]=np.log1p(lower)
        output[cursor+1]=np.log1p(upper)
        value2=b2-np.log(width)
        value8=b8-np.log(width)
        output[cursor+2]=np.sign(value2)*np.log1p(abs(value2))
        output[cursor+3]=np.sign(value8)*np.log1p(abs(value8))
        output[cursor+4]=np.log(short_mean)
        output[cursor+5]=np.log(short_mean/long_mean)
        prefix[t%513,channel]=cumulative[channel]
    counter[0]=t
    return output


@njit(cache=True)
def step(point,filter_args,parameters,conditional,cap,pair,reference,work,evidence):
    residual=innovation_update(point,*filter_args)
    energies_one(residual,parameters,conditional,cap,pair,work)
    for j in range(2):
        work[j]/=reference[j]
    return likelihood_step(work,*evidence)


@njit(cache=True)
def plus_step(point,parent_args,scale_args,columns,output):
    parent=variance_step(point,*parent_args)
    extra=step(point,*scale_args)
    output[:26]=parent[:26]
    for j in range(len(columns)):
        output[26+j]=extra[columns[j]]
    return output


@njit(cache=True)
def replay_single(points,args):
    result=np.empty((len(points),12),dtype=np.float32)
    for j in range(len(points)):
        result[j]=step(points[j],*args)
    return result


@njit(cache=True)
def replay_plus(points,args):
    result=np.empty((len(points),len(args[-1])),dtype=np.float32)
    for j in range(len(points)):
        result[j]=plus_step(points[j],*args)
    return result


class ScaleState:
    def __init__(self,historical,settings):
        c=config(settings)
        fit=fit_historical(historical,8)
        r=fit['historical_innovations']
        parameters,conditional=fast_variance_parameters(r,'arch1')
        initial=parameters[2]/max(1.-parameters[3]-parameters[4],.05)
        pair=0 if c.channel_pair=='raw_forecast' else 1
        energy=historical_energies(r,parameters,np.full(3,initial),c.cap,pair)
        reference=np.maximum(energy[min(160,len(energy)//4):].mean(axis=0),1e-8)
        ages=np.asarray([1,2,4,8,16,32,64,128,256,512],dtype=np.int64)
        evidence=(ages,np.diff(np.r_[0,ages]).astype(float),np.zeros(2),np.zeros((513,2)),
                  np.zeros(1,dtype=np.int64),np.empty(12,dtype=np.float32))
        args=(tuple(fit[k] for k in ('reference','coefficients','ring','counter')),
              parameters,conditional,c.cap,pair,reference,np.empty(2),evidence)
        self.function=step
        self.replay_function=replay_single
        if c.include_parent!='none':
            parent=VarianceState(historical,VarianceConfig(max_age=512))
            pargs=(parent.filter_args,parent.parameters,parent.conditional,parent.mode,
                   parent.calibration,parent.work,parent.evidence_args)
            columns=np.asarray([0,6] if c.include_parent=='lower' else [4,5,10,11],dtype=np.int64)
            args=(pargs,args,columns,np.empty(26+len(columns),dtype=np.float32))
            self.function=plus_step
            self.replay_function=replay_plus
        self.args=args

    def update(self,point):
        return self.function(float(point),*self.args)

    def replay(self,points):
        return self.replay_function(np.asarray(points,dtype=np.float32),self.args)


def feature_names(settings):
    c=config(settings)
    names=tuple(channel+'_'+s for channel in ('raw_energy','forecast' if c.channel_pair=='raw_forecast' else 'conditional_energy') for s in STATS)
    if c.include_parent=='none':
        return names
    old=tuple(channel+'_'+s for channel in ('raw_energy','log_variance_forecast') for s in CHANNEL_STATS)
    columns=(0,6) if c.include_parent=='lower' else (4,5,10,11)
    return old+tuple('scale_extra_'+names[j] for j in columns)


def feature_groups(settings):
    return ('NEXT2_SCALE_LIKELIHOOD',)*len(feature_names(settings))


def make_state(historical,settings):
    return ScaleState(historical,settings)


def implementation_hash(settings):
    sources=[Path(__file__),Path(__file__).parents[1]/'next/next2_scale_likelihood.py',
        Path(__file__).parents[1]/'next/variance_evidence.py',Path(__file__).parents[1]/'next/fast_variance_initialization.py']
    return hashlib.sha256(b''.join(p.read_bytes() for p in sources)+json.dumps(asdict(config(settings)),sort_keys=True).encode()).hexdigest()
