"""H-centered short/long joint correlation departures of AR8 innovations.

These are bounded empirical ranking features, not calibrated Ljung-Box tests.
Only clean H fits the reference; update receives one current scalar.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next.whitening import fit_historical,innovation_update
from src.next.fast_variance_initialization import fast_variance_parameters
from src.next2.dependence_runtime import historical_z
from src.next2.complement_bank import normalized_one

KINDS=('innovation','magnitude','sign')
LAGS=(1,2,4,8)
HALVES=(8.,32.)
STATISTICS=('joint_energy','signed_departure','lag12_energy','lag48_energy',
    'maximum_energy','absolute_departure','correlation_energy_change','local_variance_log_ratio')


def validate(settings):
    if set(settings)!={'version','representation'} or settings['version']!='joint_corr_departure_v1' or settings['representation'] not in KINDS:
        raise ValueError('Exact frozen dependence settings required')


@njit(cache=False)
def transform(z,mode,center,scale):
    value=z if mode==0 else z*z if mode==1 else (1. if z>0 else -1. if z<0 else 0.)
    return min(max((value-center)/scale,-8.),8.)


@njit(cache=False)
def correlation(moments):
    variance=max(moments[2]-moments[0]*moments[0],1e-8)
    lag_variance=max(moments[3]-moments[1]*moments[1],1e-8)
    return min(max((moments[4]-moments[0]*moments[1])/np.sqrt(variance*lag_variance),-1.),1.)


@njit(cache=False)
def step(point,filter_args,parameters,conditional,transform_parameters,mode,
         ring,counter,moments,baseline,noise,alphas,output):
    residual=innovation_update(point,*filter_args)
    z=normalized_one(residual,parameters,conditional)
    value=transform(z,mode,transform_parameters[0],transform_parameters[1])
    t=counter[0]+1;position=counter[1]
    for half in range(2):
        alpha=alphas[half]
        warm_scale=np.sqrt(max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)),1e-8))
        q=0.;signed=0.;q12=0.;q48=0.;maximum=0.;absolute=0.;energy_change=0.;variance_ratio=0.
        for j,lag in enumerate((1,2,4,8)):
            previous=ring[(position-lag)%8];m=moments[half,j]
            m[0]=(1.-alpha)*m[0]+alpha*value
            m[1]=(1.-alpha)*m[1]+alpha*previous
            m[2]=(1.-alpha)*m[2]+alpha*value*value
            m[3]=(1.-alpha)*m[3]+alpha*previous*previous
            m[4]=(1.-alpha)*m[4]+alpha*value*previous
            rho=correlation(m)
            departure=min(max((rho-baseline[j])/(noise[j]*warm_scale),-128.),128.)
            sq=departure*departure;q+=sq;signed+=departure;absolute+=abs(departure);maximum=max(maximum,sq)
            if j<2:q12+=sq
            else:q48+=sq
            energy_change+=rho*rho-baseline[j]*baseline[j]
            variance_ratio+=np.log(max(m[2]-m[0]*m[0],1e-8)/transform_parameters[2])
        k=half*8
        output[k]=np.log1p(q);output[k+1]=signed/2.;output[k+2]=np.log1p(q12)
        output[k+3]=np.log1p(q48);output[k+4]=np.log1p(maximum);output[k+5]=absolute/4.
        output[k+6]=energy_change;output[k+7]=variance_ratio/4.
    output[16]=output[0]-output[8];output[17]=output[1]-output[9]
    ring[position]=value;counter[0]=t;counter[1]=(position+1)%8
    return output


@njit(cache=False)
def replay(points,args):
    result=np.empty((len(points),18),dtype=np.float32)
    for j in range(len(points)):result[j]=step(float(points[j]),*args)
    return result


class DependenceState:
    def __init__(self,historical,settings):
        validate(settings);mode=KINDS.index(settings['representation'])
        fit=fit_historical(historical,8);residuals=fit['historical_innovations']
        parameters,conditional=fast_variance_parameters(residuals,'arch1')
        variance=max(float(np.minimum((residuals-parameters[0])**2,parameters[5]).mean()),1e-12)
        z,replayed=historical_z(residuals,parameters,variance)
        np.testing.assert_array_equal(replayed,conditional)
        raw=z if mode==0 else z*z if mode==1 else np.sign(z)
        center=float(raw.mean());scale=max(float(raw.std()),1e-6)
        x=np.clip((raw-center)/scale,-8.,8.)
        reference_variance=max(float(x.var()),1e-8)
        moments=np.empty((2,4,5));baseline=np.empty(4);noise=np.empty(4)
        for j,lag in enumerate(LAGS):
            current,previous=x[lag:],x[:-lag]
            values=np.array([current.mean(),previous.mean(),np.mean(current*current),
                np.mean(previous*previous),np.mean(current*previous)])
            moments[:,j]=values;baseline[j]=correlation(values)
            cross=(current-current.mean())*(previous-previous.mean())
            noise[j]=max(float(cross.std())/max(float(current.std()*previous.std()),1e-8),.1)
        self.args=(tuple(fit[k] for k in ('reference','coefficients','ring','counter')),
            parameters,conditional,np.array([center,scale,reference_variance]),mode,
            x[-8:].copy(),np.zeros(2,dtype=np.int64),moments,baseline,noise,
            np.asarray([1.-np.exp(-np.log(2.)/h) for h in HALVES]),np.empty(18,dtype=np.float32))

    def update(self,point):return step(float(point),*self.args)

    def replay(self,points):return replay(np.asarray(points,dtype=np.float32),self.args)


def make_state(historical,settings):return DependenceState(historical,settings)


def feature_names(settings):
    validate(settings)
    return tuple('next4_'+settings['representation']+'_h'+str(int(half))+'_'+stat
        for half in HALVES for stat in STATISTICS)+('next4_joint_short_minus_long','next4_signed_short_minus_long')


def implementation_hash(settings):
    validate(settings);root=Path(__file__).resolve().parents[2]
    paths=['src/next4/dependence.py','src/next/whitening.py','src/next/fast_variance_initialization.py',
        'src/next2/dependence_runtime.py','src/next2/complement_bank.py','src/next/normalization.py']
    return hashlib.sha256(b''.join((root/p).read_bytes() for p in paths)+json.dumps(settings,sort_keys=True).encode()).hexdigest()
