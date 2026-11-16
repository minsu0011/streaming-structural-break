"""H-only AR(8)/ARCH working null, exposing two current innovation channels."""
from pathlib import Path
import hashlib
import numpy as np
from numba import njit
from src.next.whitening import fit_historical,innovation_update
from src.next.fast_variance_initialization import fast_variance_parameters


@njit(cache=False)
def residual_channels(residual,parameters,conditional,raw_scale):
    centered=residual-parameters[0]
    variance=max(parameters[2]+parameters[3]*conditional[0]+parameters[4]*conditional[1],parameters[6])
    raw=centered/raw_scale;conditional_z=centered/np.sqrt(variance)
    squared=min(centered*centered,parameters[5]);conditional[1]=conditional[0];conditional[0]=squared
    conditional[2]=(1.-parameters[7])*conditional[2]+parameters[7]*squared
    return raw,conditional_z


@njit(cache=False)
def historical_channels(residuals,parameters,initial_variance,raw_scale):
    conditional=np.full(3,initial_variance);values=np.empty((len(residuals),2))
    for j in range(len(residuals)):values[j]=residual_channels(residuals[j],parameters,conditional,raw_scale)
    return values,conditional


@njit(cache=False)
def step(point,filter_args,parameters,conditional,raw_scale,calibration,output):
    residual=innovation_update(point,*filter_args)
    raw,conditional_z=residual_channels(residual,parameters,conditional,raw_scale)
    output[0]=min(max((raw-calibration[0,0])/calibration[1,0],-12.),12.)
    output[1]=min(max((conditional_z-calibration[0,1])/calibration[1,1],-12.),12.)
    return output


def prepare(historical):
    h=np.asarray(historical,dtype=np.float32)
    if h.ndim!=1 or len(h)<16 or not np.isfinite(h).all():raise ValueError('Finite one-dimensional H of at least 16 points required')
    fit=fit_historical(h,8);residuals=fit['historical_innovations']
    parameters,expected=fast_variance_parameters(residuals,'arch1')
    initial=max(float(np.minimum((residuals-parameters[0])**2,parameters[5]).mean()),1e-12)
    raw_scale=max(float(np.std(residuals)),1e-8)
    values,conditional=historical_channels(residuals,parameters,initial,raw_scale)
    np.testing.assert_array_equal(conditional,expected)
    trimmed=values[min(160,len(values)//4):]
    calibration=np.stack([np.mean(trimmed,axis=0),np.maximum(np.std(trimmed,axis=0),1e-6)])
    standardized=np.clip((trimmed-calibration[0])/calibration[1],-12.,12.)
    args=(tuple(fit[k] for k in ('reference','coefficients','ring','counter')),parameters,conditional,
        raw_scale,calibration,np.empty(2,dtype=np.float64))
    return args,standardized


def source_hash():
    root=Path(__file__).resolve().parents[2]
    names=('src/next5/innovation.py','src/next/whitening.py','src/next/fast_variance_initialization.py')
    return hashlib.sha256(b''.join((root/n).read_bytes() for n in names)).hexdigest()
