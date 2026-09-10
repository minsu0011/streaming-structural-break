"""Offline first-alarm diagnostics; thresholds never enter model inference."""
import numpy as np


def training_threshold(null_series_maxima,alpha):
    values=np.asarray(null_series_maxima,dtype=np.float32)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or not 0<alpha<1:raise ValueError('Finite null series maxima and a valid tail fraction required')
    # An alarm uses strict >. Ties at the training quantile do not inflate the
    # empirical fraction beyond the nominal alpha.
    return float(np.quantile(values,1-alpha,method='higher'))


def first_alarm_outcome(prediction,tau,threshold):
    values=np.asarray(prediction,dtype=np.float32)
    if values.ndim!=1 or not np.isfinite(values).all():raise ValueError('Finite one-dimensional predictions required')
    crossings=np.flatnonzero(values>threshold);alarm=int(crossings[0]) if len(crossings) else None
    broken=tau is not None and tau!=-1
    if broken and (int(tau)!=tau or not 0<=tau<len(values)):raise ValueError('Break position outside observed evaluation prefix')
    premature=bool(broken and alarm is not None and alarm<int(tau))
    valid=bool(broken and alarm is not None and not premature)
    return {'has_break':broken,'online_points_observed':len(values),'first_alarm':alarm,
        'any_alarm':alarm is not None,'premature_alarm':premature,'valid_post_break_detection':valid,
        'post_break_points_observed':len(values)-int(tau) if broken else 0,
        'detection_delay':alarm-int(tau) if valid else None}


def summarize_outcomes(rows):
    null=[r for r in rows if not r['has_break']];broken=[r for r in rows if r['has_break']]
    detected=[r for r in broken if r['valid_post_break_detection']]
    def ratio(n,d):return n/d if d else None
    result={'series':len(rows),'null_series':len(null),'break_series':len(broken),
        'null_any_alarm_count':sum(r['any_alarm'] for r in null),'break_premature_alarm_count':sum(r['premature_alarm'] for r in broken),
        'valid_detected_break_count':len(detected),'break_no_alarm_count':sum(not r['any_alarm'] for r in broken)}
    result.update(null_any_alarm_rate=ratio(result['null_any_alarm_count'],len(null)),break_premature_alarm_rate=ratio(result['break_premature_alarm_count'],len(broken)),
        valid_detected_break_rate=ratio(len(detected),len(broken)),detected_only_median_delay=float(np.median([r['detection_delay'] for r in detected])) if detected else None)
    for age in (0,4,9,19,49,99):
        eligible=[r for r in broken if r['post_break_points_observed']>=age+1]
        count=sum(r['valid_post_break_detection'] and r['detection_delay']<=age for r in eligible)
        result[f'delay_le_{age}_eligible_break_series']=len(eligible);result[f'delay_le_{age}_valid_detections']=count
        result[f'delay_le_{age}_rate']=ratio(count,len(eligible))
    return result
