"""Secondary synthetic rank and first-alarm summaries with explicit support."""
import math
import numpy as np
from src.next.alarm_calibration import wilson_interval
from src.scoring.ts_auc import ts_auc


def background(spec):
    return spec['phi_before'],spec['arch_before'],spec['t_df_before']


def calibration_map(lock):
    mapping={}
    for spec in lock['scenarios']:
        mapping.setdefault(background(spec),spec['id'])
    return mapping


def threshold(maxima,alpha):
    values=np.asarray(maxima,dtype=np.float64)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or not 0<alpha<1:
        raise ValueError('Finite calibration maxima and valid alpha required')
    rank=math.ceil((len(values)+1)*(1-alpha))
    return float(np.sort(values)[rank-1]) if rank<=len(values) else None


def alarm_summary(prediction,tau,level):
    p=np.asarray(prediction)
    tau=np.asarray(tau)
    if p.ndim!=2 or tau.shape!=(len(p),) or not np.isfinite(p).all():
        raise ValueError('Complete finite synthetic series required')
    alarm=p>level if level is not None else np.zeros(p.shape,dtype=bool)
    crossed=alarm.any(axis=1)
    first=np.where(crossed,alarm.argmax(axis=1),-1)
    null=tau<0
    changed=~null
    pre=changed&crossed&(first<tau)
    detected=changed&crossed&(first>=tau)
    count=int(crossed[null].sum())
    lo,hi=wilson_interval(count,int(null.sum()))
    r={'null_series':int(null.sum()),'break_series':int(changed.sum()),
        'null_alarm_count':count,'null_eventual_alarm_rate':float(crossed[null].mean()) if null.any() else None,
        'null_alarm_wilson95_low':lo,'null_alarm_wilson95_high':hi,
        'prebreak_alarm_rate':float(pre[changed].mean()) if changed.any() else None,
        'postbreak_first_detection_rate':float(detected[changed].mean()) if changed.any() else None,
        'conditional_detection_delay_median':float(np.median(first[detected]-tau[detected])) if detected.any() else None}
    for age in (128,256,512):
        mask=tau==age
        r['tau'+str(age)+'_break_series']=int(mask.sum())
        r['tau'+str(age)+'_clean_detection_rate']=float(detected[mask].mean()) if mask.any() else None
        r['tau'+str(age)+'_premature_alarm_rate']=float(pre[mask].mean()) if mask.any() else None
    for window in (10,50,128):
        mask=changed&(p.shape[1]-tau>=window)
        r['clean_detection_within_'+str(window)]=float((detected&(first-tau<window))[mask].mean()) if mask.any() else None
    return r,crossed,first


def score(prediction,tau):
    p=np.asarray(prediction)
    t=np.broadcast_to(np.arange(p.shape[1],dtype=np.int32),p.shape)
    y=(t>=np.asarray(tau)[:,None])&(np.asarray(tau)[:,None]>=0)
    return ts_auc(y.ravel(),p.ravel(),t.ravel()) if y.any() else None


def age_normalize(prediction,times,beta):
    if beta not in (0.,.5,1.):
        raise ValueError('Age exponent outside the prospective diagnostic menu')
    p=np.asarray(prediction,dtype=np.float64)
    t=np.asarray(times)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)) or np.any(t<0):
        raise ValueError('Finite probabilities and observed age required')
    factor=(1.+t/128.)**beta
    return p/(p+(1.-p)*factor)
