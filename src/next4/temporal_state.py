"""Fixed stable second-order evidence states with a train-fold linear readout."""
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next.whitening import fit_historical,innovation_update
from src.next.fast_variance_initialization import fast_variance_parameters
from src.next2.dependence_runtime import historical_z
from src.next2.complement_bank import normalized_one

CHANNELS=('innovation','energy','lag1_product','signed_root_magnitude')
HALVES=(8.,32.,128.)


def validate(settings):
    if settings!={'version':'coupled_linear_state_v1','kind':'two_pole_readout'}:raise ValueError('Exact fixed temporal-state configuration required')


@njit(cache=False)
def filter_step(values,states,variances,alphas,output):
    for pole in range(3):
        a=alphas[pole];b=1.-a;vu,vv,cov=variances[pole]
        new_u=b*b*vu+a*a
        new_v=b*b*vv+a*a*b*b*vu+2*a*b*b*cov+a**4
        new_cov=b*b*cov+a*b*b*vu+a**3
        variances[pole,0]=new_u;variances[pole,1]=new_v;variances[pole,2]=new_cov
        for channel in range(4):
            states[channel,pole,0]=b*states[channel,pole,0]+a*values[channel]
            states[channel,pole,1]=b*states[channel,pole,1]+a*states[channel,pole,0]
            cursor=(channel*3+pole)*2
            output[cursor]=np.tanh(states[channel,pole,0]/np.sqrt(max(new_u,1e-12)))
            output[cursor+1]=np.tanh(states[channel,pole,1]/np.sqrt(max(new_v,1e-12)))
    for channel in range(4):
        output[24+channel]=np.tanh(values[channel])
        short=states[channel,0,1]/np.sqrt(max(variances[0,1],1e-12))
        long=states[channel,2,1]/np.sqrt(max(variances[2,1],1e-12))
        output[28+channel]=np.tanh((short-long)/2.)
    return output


@njit(cache=False)
def step(point,filter_args,parameters,conditional,calibration,previous,work,states,variances,alphas,output):
    z=normalized_one(innovation_update(point,*filter_args),parameters,conditional)
    work[0]=z;work[1]=z*z;work[2]=z*previous[0]
    work[3]=(1. if z>0 else -1. if z<0 else 0.)*np.sqrt(abs(z))
    previous[0]=z
    for channel in range(4):work[channel]=min(max((work[channel]-calibration[channel,0])/calibration[channel,1],-8.),8.)
    return filter_step(work,states,variances,alphas,output)


@njit(cache=False)
def replay(points,args):
    result=np.empty((len(points),32),dtype=np.float32)
    for j in range(len(points)):result[j]=step(float(points[j]),*args)
    return result


class TemporalState:
    def __init__(self,historical,settings):
        validate(settings);fit=fit_historical(historical,8);residuals=fit['historical_innovations']
        parameters,conditional=fast_variance_parameters(residuals,'arch1')
        variance=max(float(np.minimum((residuals-parameters[0])**2,parameters[5]).mean()),1e-12)
        z,replayed=historical_z(residuals,parameters,variance);np.testing.assert_array_equal(replayed,conditional)
        raw=np.column_stack([z,z*z,z*np.r_[z[0],z[:-1]],np.sign(z)*np.sqrt(np.abs(z))])
        calibration=np.column_stack([raw.mean(axis=0),np.maximum(raw.std(axis=0),1e-6)])
        self.args=(tuple(fit[k] for k in ('reference','coefficients','ring','counter')),parameters,conditional,
            calibration,np.array([z[-1]]),np.empty(4),np.zeros((4,3,2)),np.zeros((3,3)),
            np.array([1.-np.exp(-np.log(2.)/half) for half in HALVES]),np.empty(32,dtype=np.float32))

    def update(self,point):return step(float(point),*self.args)

    def replay(self,points):return replay(np.asarray(points,dtype=np.float32),self.args)


def make_state(historical,settings):return TemporalState(historical,settings)


def feature_names(settings):
    validate(settings)
    return tuple('next4_state_'+channel+'_h'+str(int(half))+'_'+order for channel in CHANNELS for half in HALVES
        for order in ('first_pole','second_pole'))+tuple('next4_state_'+channel+'_current' for channel in CHANNELS)+tuple(
        'next4_state_'+channel+'_short_long' for channel in CHANNELS)


def implementation_hash(settings):
    validate(settings);root=Path(__file__).resolve().parents[2]
    paths=['src/next4/temporal_state.py','src/next/whitening.py','src/next/fast_variance_initialization.py',
        'src/next2/dependence_runtime.py','src/next2/complement_bank.py','src/next/normalization.py']
    return hashlib.sha256(b''.join((root/p).read_bytes() for p in paths)+json.dumps(settings,sort_keys=True).encode()).hexdigest()
