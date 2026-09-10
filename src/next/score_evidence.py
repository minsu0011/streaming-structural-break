"""AR coefficient score evidence: innovation times past original regressors.

The empirical historical score covariance provides a fixed sandwich scaling.
No online AR fit, inferred true change label, or growing history is used.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next.whitening import fit_historical,innovation_update
from src.next.engine import ALPHAS,AGES,AGE_WIDTHS,_logadd


@dataclass(frozen=True)
class ScoreConfig:
    order:int=8
    covariance:str='full'
    ridge_fraction:float=.01


SCORE_NAMES=('score_instant_q',)+tuple(f'score_{stat}_{half}' for half in [8,32,128] for stat in ['ewma_q','ewma_max_abs','ewma_signed_sum'])+(
    'score_cusum_max','score_cusum_energy','score_glr_max','score_glr_logmean','score_glr_argmax_logage',
    'score_bayes025','score_bayes1','score_memory_max','score_memory_decay')


def score_hash(config):
    return hashlib.sha256(Path(__file__).read_bytes()+json.dumps(asdict(config),sort_keys=True).encode()).hexdigest()


@njit(cache=True)
def score_step(point,args):
    (reference,coefficients,ring,ar_counter,residual_scale,score_center,inverse_root,
     ewma,cusum,cumulative,prefix,counter,memory,work,output)=args
    p=len(coefficients)
    position=ar_counter[0]
    for j in range(p):
        work[0,j]=ring[(position-1-j)%p]
    residual=innovation_update(point,reference,coefficients,ring,ar_counter)
    standardized=(residual-residual_scale[0])/residual_scale[1]
    for j in range(p):
        work[1,j]=standardized*work[0,j]-score_center[j]
    for j in range(p):
        total=0.
        for k in range(p):
            total+=inverse_root[j,k]*work[1,k]
        work[2,j]=total
    t=counter[0]+1
    output[0]=np.sum(work[2]**2)/p
    cursor=1
    for half in range(3):
        alpha=ALPHAS[half]
        variance=max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)),1e-12)
        energy,maximum,signed=0.,0.,0.
        for j in range(p):
            ewma[half,j]=(1.-alpha)*ewma[half,j]+alpha*work[2,j]
            value=ewma[half,j]/np.sqrt(variance)
            energy+=value*value
            maximum=max(maximum,abs(value))
            signed+=value
        output[cursor]=energy/p
        output[cursor+1]=maximum
        output[cursor+2]=signed/np.sqrt(p)
        cursor+=3
    cmax,cenergy=0.,0.
    for j in range(p):
        cusum[0,j]=max(0.,cusum[0,j]+work[2,j]-.25)
        cusum[1,j]=max(0.,cusum[1,j]-work[2,j]-.25)
        cmax=max(cmax,cusum[0,j],cusum[1,j])
        cenergy+=cusum[0,j]**2+cusum[1,j]**2
        cumulative[j]+=work[2,j]
    output[cursor]=np.log1p(cmax)
    output[cursor+1]=np.log1p(cenergy/p)
    cursor+=2
    maximum,age_at_max=0.,1.
    logmix,b025,b1=-1e300,-1e300,-1e300
    count,width=0,0.
    for k in range(len(AGES)):
        age=AGES[k]
        if age>t:
            break
        count+=1
        width+=AGE_WIDTHS[k]
        energy=0.
        for j in range(p):
            total=cumulative[j]-prefix[(t-age)%129,j]
            energy+=total*total
        glr=.5*energy/age
        if glr>maximum:
            maximum=glr;age_at_max=age
        logmix=_logadd(logmix,glr)
        b025=_logadd(b025,np.log(AGE_WIDTHS[k])-.5*p*np.log1p(.25*age)+.5*.25*energy/(1.+.25*age))
        b1=_logadd(b1,np.log(AGE_WIDTHS[k])-.5*p*np.log1p(age)+.5*energy/(1.+age))
    for j in range(p):
        prefix[t%129,j]=cumulative[j]
    output[cursor]=maximum
    output[cursor+1]=logmix-np.log(count)
    output[cursor+2]=np.log2(age_at_max)
    output[cursor+3]=b025-np.log(width)
    output[cursor+4]=b1-np.log(width)
    memory[0]=max(memory[0],maximum)
    memory[1]=max(np.exp(-np.log(2.)/32.)*memory[1],maximum)
    output[cursor+5]=memory[0]
    output[cursor+6]=memory[1]
    counter[0]=t
    return output


@njit(cache=True)
def replay_score(points,args):
    result=np.empty((len(points),len(SCORE_NAMES)),dtype=np.float32)
    for j in range(len(points)):
        result[j]=score_step(points[j],args)
    return result


class ARScoreState:
    def __init__(self,historical,config=ScoreConfig()):
        if config.covariance not in ('full','diagonal') or config.ridge_fraction!=.01:
            raise ValueError('Unregistered AR score covariance policy')
        self.config=config
        fit=fit_historical(historical,config.order)
        reference=fit['reference']
        h=np.asarray(historical,dtype=np.float32).astype(float)
        z=np.clip((h-reference[0])/reference[1],-12.,12.)-reference[2]
        residual=fit['historical_innovations']
        location,scale=float(residual.mean()),max(float(residual.std()),1e-6)
        design=np.lib.stride_tricks.sliding_window_view(z,config.order+1)[:,:-1][:,::-1]
        scores=((residual-location)/scale)[:,None]*design
        center=scores.mean(axis=0)
        centered=scores-center
        covariance=centered.T@centered/len(centered)
        if config.covariance=='diagonal':
            covariance=np.diag(np.diag(covariance))
        ridge=max(float(np.trace(covariance))/config.order*config.ridge_fraction,1e-8)
        values,vectors=np.linalg.eigh(covariance+ridge*np.eye(config.order))
        inverse_root=(vectors*(1./np.sqrt(np.maximum(values,1e-10))))@vectors.T
        p=config.order
        self.args=(fit['reference'],fit['coefficients'],fit['ring'],fit['counter'],np.array([location,scale]),
            center,inverse_root,np.zeros((3,p)),np.zeros((2,p)),np.zeros(p),np.zeros((129,p)),
            np.zeros(1,dtype=np.int64),np.zeros(2),np.zeros((3,p)),np.empty(len(SCORE_NAMES),dtype=np.float32))

    def update(self,point):
        return score_step(float(point),self.args)

    def replay(self,points):
        return replay_score(np.asarray(points,dtype=np.float32),self.args)

    @property
    def state_array_bytes(self):
        return sum(a.nbytes for a in self.args)


def feature_names(settings):
    return SCORE_NAMES


def feature_groups(settings):
    return ('SCORE',)*len(SCORE_NAMES)


def make_state(historical,settings):
    return ARScoreState(historical,ScoreConfig(**settings))


def implementation_hash(settings):
    from src.next.engine import feature_hash,EngineConfig
    return hashlib.sha256(score_hash(ScoreConfig(**settings)).encode()+feature_hash(EngineConfig()).encode()).hexdigest()
