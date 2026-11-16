"""Small current/event/scale correction coordinates; no truth or future length."""
import numpy as np

NAMES=('current_score','event_score','episodes_over7','first_age_log','last_age_log',
       'recovering','confirmed_direction','signed_scale','scale_magnitude_score','scale_confirmation_ratio')
CURRENT_EVENT=tuple(range(7))
FULL=tuple(range(10))
EVENT_SCALE_TREE=(1,2,3,4,6,7,8,9)


def from_state(values):
    v=np.asarray(values,dtype=np.float64)
    scalar=v.ndim==1
    if scalar:
        v=v[None,:]
    if v.ndim!=2 or v.shape[1]!=30:
        raise ValueError('NEXT7 state coordinate matrix required')
    out=np.empty((len(v),10),dtype=np.float64)
    out[:,0]=v[:,1]
    out[:,1]=v[:,3]
    out[:,2]=v[:,4]/7.
    out[:,3:5]=np.log1p(np.minimum(np.maximum(v[:,5:7],0.),16384.))
    out[:,5]=v[:,7]
    out[:,6]=v[:,8]
    out[:,7]=v[:,9]
    out[:,8]=v[:,10]
    out[:,9]=v[:,11]
    return out[0] if scalar else out
