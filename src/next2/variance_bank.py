"""Small causal interventions on energy, age integration and evidence memory."""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next.whitening import fit_historical,innovation_update
from src.next.fast_variance_initialization import fast_variance_parameters
from src.next.channel_evidence import CHANNEL_STATS,_logadd
from src.next.normalization import ecdf_uniform

ENERGIES=('raw','winsor','huber','log1p','abs','clipped_abs','robust_scale','ecdf_tail','log_ratio','log_raw')
MEMORIES=('original','ewma','leaky','resettable','posterior_like','decayed_only')
AGE_GRIDS={'all':(1,2,4,8,16,32,64,128,256,512),'short':(1,2,4,8,16,32),
           'medium':(16,32,64,128),'long':(128,256,512),
           'sparse32':(32,64,128,256,512),'sparse64':(64,128,256,512)}


@dataclass(frozen=True)
class BankConfig:
    order:int=8
    normalization:str='arch1'
    energy:str='raw'
    cap:int=4
    ages:str='all'
    memory:str='original'
    age_normalization:str='none'


def config(settings):
    c=BankConfig(**settings)
    if c.order!=8 or c.normalization not in ('arch1','arch2','ewma','none') or c.energy not in ENERGIES or c.cap not in (3,4,6) or c.ages not in AGE_GRIDS or c.memory not in MEMORIES or c.age_normalization not in ('none','sqrt_age','log_age'):
        raise ValueError('Unregistered NEXT2 variance-bank setting')
    return c


@njit(cache=True)
def channels_one(residual,parameters,conditional,sorted_h,robust,mode,energy,cap,output):
    reference=parameters[2]/max(1.-parameters[3]-parameters[4],.05)
    variance=conditional[2] if mode==5 else parameters[2]+parameters[3]*conditional[0]+parameters[4]*conditional[1]
    variance=max(variance,parameters[6])
    centered=residual-parameters[0]
    square=centered*centered
    z=centered/np.sqrt(reference)
    a=abs(z)
    if energy==0:
        value=square/reference-1.
    elif energy==1:
        value=min(z*z,cap*cap)
    elif energy==2:
        value=z*z if a<=cap else 2.*cap*a-cap*cap
    elif energy==3:
        value=np.log1p(z*z)
    elif energy==4:
        value=a
    elif energy==5:
        value=min(a,cap)
    elif energy==6:
        value=min(abs((residual-robust[0])/robust[1]),cap)
    elif energy==7:
        u=ecdf_uniform(residual,sorted_h)
        value=-np.log(max(2.*min(u,1.-u),1e-8))
    elif energy==8:
        value=np.log((square+reference*1e-6)/(variance+reference*1e-6))
    else:
        value=np.log(square/reference+1e-6)
    output[0]=value
    output[1]=np.log(variance/reference)
    capped=min(square,parameters[5])
    conditional[1]=conditional[0]
    conditional[0]=capped
    conditional[2]=(1.-parameters[7])*conditional[2]+parameters[7]*capped
    return output


@njit(cache=True)
def historical_channels(residuals,parameters,conditional,sorted_h,robust,mode,energy,cap):
    output=np.empty((len(residuals),2))
    for j in range(len(residuals)):
        channels_one(residuals[j],parameters,conditional,sorted_h,robust,mode,energy,cap,output[j])
    return output


def evidence_args(c):
    ages=np.asarray(AGE_GRIDS[c.ages],dtype=np.int64)
    return (ages,np.diff(np.r_[0,ages]).astype(float),np.zeros((2,3)),np.zeros((2,2)),
            np.zeros(2),np.zeros((int(ages[-1])+1,2)),np.zeros(1,dtype=np.int64),np.zeros((2,2)),np.empty(26,dtype=np.float32))


@njit(cache=True)
def evidence_step(values,args,memory_mode,normalization):
    ages,widths,ewma,cusum,cumulative,prefix,counter,memory,output=args
    t=counter[0]+1
    halves=(8.,32.,128.)
    for channel in range(2):
        value=min(max(values[channel],-12.),12.)
        cursor=channel*13
        output[cursor]=value
        for k in range(3):
            alpha=1.-np.exp(-np.log(2.)/halves[k])
            ewma[channel,k]=(1.-alpha)*ewma[channel,k]+alpha*value
            variance=max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)),1e-12)
            output[cursor+1+k]=ewma[channel,k]/np.sqrt(variance)
        cusum[channel,0]=max(0.,cusum[channel,0]+value-.25)
        cusum[channel,1]=max(0.,cusum[channel,1]-value-.25)
        output[cursor+4]=np.log1p(cusum[channel,0])
        output[cursor+5]=np.log1p(cusum[channel,1])
        cumulative[channel]+=value
        maximum,age_at_max=0.,1
        mix,b025,b1=-1e300,-1e300,-1e300
        count,width=0,0.
        for k in range(len(ages)):
            age=ages[k]
            if age>t:
                break
            count+=1
            width+=widths[k]
            total=cumulative[channel]-prefix[(t-age)%len(prefix),channel]
            energy=total*total
            glr=.5*energy/age
            b0=-.5*np.log1p(.25*age)+.5*.25*energy/(1.+.25*age)
            b=-.5*np.log1p(age)+.5*energy/(1.+age)
            if normalization==1:
                glr/=np.sqrt(age)
                b0/=np.sqrt(age)
                b/=np.sqrt(age)
            elif normalization==2:
                glr-=np.log1p(age)
                b0-=np.log1p(age)
                b-=np.log1p(age)
            if glr>maximum:
                maximum,age_at_max=glr,age
            mix=_logadd(mix,glr)
            b025=_logadd(b025,np.log(widths[k])+b0)
            b1=_logadd(b1,np.log(widths[k])+b)
        prefix[t%len(prefix),channel]=cumulative[channel]
        if memory_mode==0:
            memory[channel,0]=max(memory[channel,0],maximum)
            memory[channel,1]=max(np.exp(-np.log(2.)/32.)*memory[channel,1],maximum)
        elif memory_mode==1:
            memory[channel,0]=np.exp(-np.log(2.)/8.)*memory[channel,0]+(1.-np.exp(-np.log(2.)/8.))*maximum
            memory[channel,1]=np.exp(-np.log(2.)/32.)*memory[channel,1]+(1.-np.exp(-np.log(2.)/32.))*maximum
        elif memory_mode==2:
            memory[channel,0]=max(0.,.95*memory[channel,0]+maximum-.5)
            memory[channel,1]=max(0.,.99*memory[channel,1]+maximum-.5)
        elif memory_mode==3:
            memory[channel,0]=0. if abs(value)<.25 else max(0.,memory[channel,0]+maximum-.5)
            memory[channel,1]=0. if abs(value)<.5 else max(0.,memory[channel,1]+maximum-.5)
        elif memory_mode==4:
            # Bounded evidence-driven persistence. This is not a calibrated posterior.
            prob=1./(1.+np.exp(-min(maximum-2.,50.)))
            memory[channel,0]=.95*memory[channel,0]+.05*prob
            memory[channel,1]=.99*memory[channel,1]+.01*prob
        else:
            memory[channel,0]=max(np.exp(-np.log(2.)/8.)*memory[channel,0],maximum)
            memory[channel,1]=max(np.exp(-np.log(2.)/32.)*memory[channel,1],maximum)
        output[cursor+6]=maximum
        output[cursor+7]=mix-np.log(count) if count else 0.
        output[cursor+8]=np.log2(age_at_max) if count else 0.
        output[cursor+9]=b025-np.log(width) if count else 0.
        output[cursor+10]=b1-np.log(width) if count else 0.
        output[cursor+11]=memory[channel,0]
        output[cursor+12]=memory[channel,1]
    counter[0]=t
    return output


@njit(cache=True)
def step(point,filter_args,parameters,conditional,sorted_h,robust,mode,energy,cap,calibration,work,args,memory_mode,normalization):
    residual=innovation_update(point,*filter_args)
    channels_one(residual,parameters,conditional,sorted_h,robust,mode,energy,cap,work)
    for j in range(2):
        work[j]=(work[j]-calibration[0,j])/calibration[1,j]
    return evidence_step(work,args,memory_mode,normalization)


@njit(cache=True)
def replay(points,args):
    result=np.empty((len(points),26),dtype=np.float32)
    for j in range(len(points)):
        result[j]=step(points[j],*args)
    return result


class BankState:
    def __init__(self,historical,settings):
        c=config(settings)
        fit=fit_historical(historical,c.order)
        r=fit['historical_innovations']
        parameters,conditional=fast_variance_parameters(r,'arch1' if c.normalization=='none' else c.normalization)
        if c.normalization=='none':
            stationary=parameters[2]/max(1.-parameters[3]-parameters[4],.05)
            parameters[2:5]=[stationary,0.,0.]
        mode={'arch1':3,'arch2':4,'ewma':5,'none':3}[c.normalization]
        reference=parameters[2]/max(1.-parameters[3]-parameters[4],.05)
        sorted_h=np.sort(r) if c.energy=='ecdf_tail' else np.empty(0)
        median=float(np.median(r))
        robust=np.array([median,max(float(1.4826*np.median(np.abs(r-median))),float(r.std())*.01,1e-8)])
        channels=historical_channels(r,parameters,np.full(3,reference),sorted_h,robust,mode,ENERGIES.index(c.energy),c.cap)
        channels=channels[min(160,len(channels)//4):]
        calibration=np.stack([channels.mean(axis=0),np.maximum(channels.std(axis=0),1e-6)])
        self.args=(tuple(fit[k] for k in ('reference','coefficients','ring','counter')),
            parameters,conditional,sorted_h,robust,mode,ENERGIES.index(c.energy),c.cap,calibration,
            np.empty(2),evidence_args(c),MEMORIES.index(c.memory),('none','sqrt_age','log_age').index(c.age_normalization))

    def update(self,point):
        return step(float(point),*self.args)

    def replay(self,points):
        return replay(np.asarray(points,dtype=np.float32),self.args)


def feature_names(settings):
    config(settings)
    return tuple(c+'_'+s for c in ('energy','log_variance_forecast') for s in CHANNEL_STATS)


def feature_groups(settings):
    config(settings)
    return ('NEXT2_VARIANCE',)*26


def make_state(historical,settings):
    return BankState(historical,settings)


def implementation_hash(settings):
    paths=[Path(__file__),Path(__file__).parents[1]/'next/whitening.py',
           Path(__file__).parents[1]/'next/fast_variance_initialization.py',
           Path(__file__).parents[1]/'next/normalization.py',Path(__file__).parents[1]/'next/channel_evidence.py']
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)+json.dumps(asdict(config(settings)),sort_keys=True).encode()).hexdigest()
