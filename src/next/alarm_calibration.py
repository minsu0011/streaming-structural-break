"""Secondary train-only null-alarm calibration and first-crossing diagnostics."""
import math
import numpy as np
import pandas as pd


def null_maximum_threshold(maxima,alpha):
    values = np.asarray(maxima,dtype=np.float32)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Nonempty finite calibration-series maxima required')
    if not 0.<alpha<1.:
        raise ValueError('Nominal alarm probability must lie strictly between zero and one')
    rank = math.ceil((len(values)+1)*(1.-alpha))
    # Strict exceedance handles score ties conservatively. None means no finite
    # threshold at the requested finite-sample rank, so no alarm is emitted.
    threshold = float(np.sort(values)[rank-1]) if rank<=len(values) else None
    return {'nominal_alpha':alpha,'null_calibration_series':len(values),'one_based_order_statistic':rank,
        'threshold':threshold,'strict_exceedance':True,'no_finite_threshold':threshold is None,
        'calibration_alarm_fraction':float(np.mean(values>threshold)) if threshold is not None else 0.}


def series_maxima(ids,prediction):
    ids,prediction = np.asarray(ids),np.asarray(prediction,dtype=np.float32)
    if len(ids)!=len(prediction) or not len(ids) or not np.isfinite(prediction).all():
        raise ValueError('Finite equal-length nonempty vectors required')
    cuts = np.r_[0,np.flatnonzero(np.diff(ids))+1,len(ids)]
    unique = ids[cuts[:-1]]
    if len(np.unique(unique))!=len(unique):
        raise ValueError('Each complete series must occupy one contiguous block')
    return unique,np.maximum.reduceat(prediction,cuts[:-1])


def first_alarm_records(ids,times,prediction,tau_by_id,threshold):
    ids,times,prediction = np.asarray(ids),np.asarray(times),np.asarray(prediction,dtype=np.float32)
    unique,_ = series_maxima(ids,prediction)
    cuts = np.r_[0,np.flatnonzero(np.diff(ids))+1,len(ids)]
    records = []
    for sid,start,end in zip(unique,cuts[:-1],cuts[1:]):
        sid = int(sid)
        np.testing.assert_array_equal(times[start:end],np.arange(end-start))
        tau = int(tau_by_id[sid])
        if tau>=end-start or tau < -1:
            raise ValueError('Invalid break index for observed online sequence')
        crossings = np.flatnonzero(prediction[start:end]>threshold) if threshold is not None else np.empty(0,dtype=int)
        first = int(crossings[0]) if len(crossings) else -1
        clean = tau>=0 and first>=tau
        records.append({'dataset_id':sid,'online_length':end-start,'tau':tau,'has_break':tau>=0,
            'first_alarm_time':first,'null_false_alarm':tau<0 and first>=0,
            'premature_break_alarm':tau>=0 and 0<=first<tau,
            'clean_post_break_detection':clean,'delay':first-tau if clean else np.nan,
            'post_break_observed_points':end-start-tau if tau>=0 else 0})
    return pd.DataFrame(records)


def wilson_interval(successes,total,z=1.959963984540054):
    if total<=0:
        return None,None
    proportion = successes/total
    denominator = 1.+z*z/total
    center = (proportion+z*z/(2*total))/denominator
    radius = z*np.sqrt(proportion*(1.-proportion)/total+z*z/(4*total*total))/denominator
    return max(0.,center-radius),min(1.,center+radius)


def summarize_alarms(frame):
    null = frame.loc[~frame.has_break]
    broken = frame.loc[frame.has_break]
    false_count = int(null.null_false_alarm.sum())
    lo,hi = wilson_interval(false_count,len(null))
    detected = broken.loc[broken.clean_post_break_detection]
    result = {'null_series':len(null),'break_series':len(broken),'null_false_alarms':false_count,
        'null_false_alarm_fraction':false_count/len(null) if len(null) else None,
        'null_false_alarm_wilson_95_low':lo,'null_false_alarm_wilson_95_high':hi,
        'premature_break_alarms':int(broken.premature_break_alarm.sum()),
        'premature_break_alarm_fraction':float(broken.premature_break_alarm.mean()) if len(broken) else None,
        'clean_detections':len(detected),'clean_detection_fraction':len(detected)/len(broken) if len(broken) else None,
        'conditional_detected_delay_median':float(detected.delay.median()) if len(detected) else None,
        'conditional_detected_delay_p90':float(detected.delay.quantile(.9)) if len(detected) else None}
    for horizon in (5,10,20,50):
        eligible = broken.loc[broken.post_break_observed_points>=horizon]
        detected_in_time = eligible.clean_post_break_detection & (eligible.delay<horizon)
        result[f'eligible_for_{horizon}_point_window'] = len(eligible)
        result[f'clean_detection_within_{horizon}_points'] = float(detected_in_time.mean()) if len(eligible) else None
    return result
