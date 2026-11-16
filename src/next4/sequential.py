"""Small fixed likelihood alternatives with sparse windows or hazard filtering.

Empirical ranking features under a working Gaussian observation model. These
are not calibrated posterior probabilities for the real competition process.
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

KINDS=('mean_mixture','scale_mixture','hazard_mixture')
AGES=(4,8,16,32,64,128,256,512)
MEANS=np.array([-1.5,-.75,-.25,.25,.75,1.5])
VARIANCES=np.array([.5,.7,1.5,2.])
HAZARDS=np.array([1./32,1./128,1./512])


def validate(settings):
    if set(settings)!={'version','kind'} or settings['version']!='discrete_likelihood_v1' or settings['kind'] not in KINDS:
        raise ValueError('Exact frozen sequential-likelihood settings required')


@njit(cache=False)
def logsum(values):
    maximum=np.max(values)
    if maximum==-np.inf:return -np.inf
    return maximum+np.log(np.exp(values-maximum).sum())


@njit(cache=False)
def logmix(values):return logsum(values)-np.log(len(values))


@njit(cache=False)
def window_features(sums,squares,counter,mode,output):
    t=counter[0];position=counter[1];totals=np.empty(8)
    for j,window in enumerate(AGES):
        age=min(t,window);previous=(position-age)%513
        sx=sums[position]-sums[previous];sx2=squares[position]-squares[previous]
        if mode==0:
            ll=MEANS*sx-.5*age*MEANS*MEANS
            lower=logmix(ll[:3]);upper=logmix(ll[3:])
        else:
            ll=-.5*age*np.log(VARIANCES)+.5*(1.-1./VARIANCES)*sx2
            lower=logmix(ll[:2]);upper=logmix(ll[2:])
        value=logmix(ll);totals[j]=value
        output[4*j]=value;output[4*j+1]=lower;output[4*j+2]=upper;output[4*j+3]=upper-lower
    output[32]=logmix(totals);output[33]=np.max(totals)
    output[34]=logmix(totals[:4]);output[35]=logmix(totals[4:])


@njit(cache=False)
def hazard_features(z,states,output):
    ll=np.empty(10);ll[:6]=MEANS*z-.5*MEANS*MEANS
    ll[6:]=-.5*np.log(VARIANCES)+.5*(1.-1./VARIANCES)*z*z
    for j in range(3):
        old_null=states[j,0];hazard=HAZARDS[j]
        for k in range(10):
            states[j,k+1]=np.logaddexp(states[j,k+1],old_null+np.log(hazard/10.))+ll[k]
        states[j,0]=old_null+np.log1p(-hazard)
        states[j]-=logsum(states[j])
        null=states[j,0];changed=logsum(states[j,1:])
        offset=j*6
        output[offset]=min(max(changed-null,-80.),80.)
        output[offset+1]=min(max(logsum(states[j,4:7])-null,-80.),80.)
        output[offset+2]=min(max(logsum(states[j,1:4])-null,-80.),80.)
        output[offset+3]=min(max(logsum(states[j,7:9])-null,-80.),80.)
        output[offset+4]=min(max(logsum(states[j,9:11])-null,-80.),80.)
        entropy=0.
        for k in range(10):
            lp=states[j,k+1]-changed
            entropy-=np.exp(lp)*lp
        output[offset+5]=entropy
    output[18]=output[0]-output[12];output[19]=output[6]-output[12]


@njit(cache=False)
def step(point,filter_args,parameters,conditional,h_transform,sums,squares,counter,hazard_states,mode,output):
    z=normalized_one(innovation_update(point,*filter_args),parameters,conditional)
    z=min(max((z-h_transform[0])/h_transform[1],-6.),6.)
    position=counter[1];following=(position+1)%513
    sums[following]=sums[position]+z;squares[following]=squares[position]+z*z
    counter[0]+=1;counter[1]=following
    if mode<2:window_features(sums,squares,counter,mode,output)
    else:hazard_features(z,hazard_states,output)
    return output


@njit(cache=False)
def replay(points,args):
    result=np.empty((len(points),len(args[-1])),dtype=np.float32)
    for j in range(len(points)):result[j]=step(float(points[j]),*args)
    return result


class SequentialState:
    def __init__(self,historical,settings):
        validate(settings);fit=fit_historical(historical,8);residuals=fit['historical_innovations']
        parameters,conditional=fast_variance_parameters(residuals,'arch1')
        variance=max(float(np.minimum((residuals-parameters[0])**2,parameters[5]).mean()),1e-12)
        z,replayed=historical_z(residuals,parameters,variance)
        np.testing.assert_array_equal(replayed,conditional)
        mode=KINDS.index(settings['kind']);hazard_states=np.full((3,11),-np.inf);hazard_states[:,0]=0.
        self.args=(tuple(fit[k] for k in ('reference','coefficients','ring','counter')),parameters,conditional,
            np.array([z.mean(),max(float(z.std()),1e-6)]),np.zeros(513),np.zeros(513),np.zeros(2,dtype=np.int64),
            hazard_states,mode,np.empty(20 if mode==2 else 36,dtype=np.float32))

    def update(self,point):return step(float(point),*self.args)

    def replay(self,points):return replay(np.asarray(points,dtype=np.float32),self.args)


def make_state(historical,settings):return SequentialState(historical,settings)


def feature_names(settings):
    validate(settings);kind=settings['kind']
    if kind=='hazard_mixture':
        return tuple('next4_hazard_'+str(int(round(1/h)))+'_'+name for h in HAZARDS
            for name in ('any_log_odds','mean_positive_log_odds','mean_negative_log_odds','scale_decrease_log_odds','scale_increase_log_odds','alternative_entropy'))+(
            'next4_short_minus_long_hazard','next4_medium_minus_long_hazard')
    return tuple('next4_'+kind+'_age'+str(age)+'_'+name for age in AGES
        for name in ('log_mixture','lower_log_mixture','upper_log_mixture','upper_minus_lower'))+tuple(
            'next4_'+kind+'_'+name for name in ('all_age_mixture','max_age_evidence','short_age_mixture','long_age_mixture'))


def implementation_hash(settings):
    validate(settings);root=Path(__file__).resolve().parents[2]
    paths=['src/next4/sequential.py','src/next/whitening.py','src/next/fast_variance_initialization.py',
        'src/next2/dependence_runtime.py','src/next2/complement_bank.py','src/next/normalization.py']
    return hashlib.sha256(b''.join((root/p).read_bytes() for p in paths)+json.dumps(settings,sort_keys=True).encode()).hexdigest()
