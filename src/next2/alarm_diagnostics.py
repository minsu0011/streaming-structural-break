"""Float64 alarm boundaries preserve the age-normalization rank contract."""
import numpy as np
import pandas as pd
from src.next2.synthetic_analysis import age_normalize,threshold


def series_maxima64(ids,prediction):
    ids,p=np.asarray(ids),np.asarray(prediction,dtype=np.float64)
    if ids.ndim!=1 or p.shape!=ids.shape or not len(ids) or not np.isfinite(p).all():
        raise ValueError('Nonempty aligned finite vectors required')
    cuts=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1,len(ids)]
    unique=ids[cuts[:-1]]
    if len(np.unique(unique))!=len(unique):
        raise ValueError('Complete contiguous series required')
    return unique,np.maximum.reduceat(p,cuts[:-1])


def first_alarm_records64(ids,times,prediction,tau_by_id,level):
    ids,times,p=np.asarray(ids),np.asarray(times),np.asarray(prediction,dtype=np.float64)
    unique,_=series_maxima64(ids,p)
    if times.shape!=ids.shape:
        raise ValueError('Aligned observed ages required')
    cuts=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1,len(ids)]
    records=[]
    for sid,start,end in zip(unique,cuts[:-1],cuts[1:]):
        sid=int(sid)
        np.testing.assert_array_equal(times[start:end],np.arange(end-start))
        tau=int(tau_by_id[sid])
        if tau>=end-start or tau < -1:
            raise ValueError('Invalid observed break index')
        crossings=np.flatnonzero(p[start:end]>level) if level is not None else []
        first=int(crossings[0]) if len(crossings) else -1
        clean=tau>=0 and first>=tau
        records.append({'dataset_id':sid,'online_length':end-start,'tau':tau,'has_break':tau>=0,
            'first_alarm_time':first,'null_false_alarm':tau<0 and first>=0,
            'premature_break_alarm':tau>=0 and 0<=first<tau,'clean_post_break_detection':clean,
            'delay':first-tau if clean else np.nan,'post_break_observed_points':end-start-tau if tau>=0 else 0})
    return pd.DataFrame(records)


def verify_within_time_ranks(original,transformed,times):
    p,q,t=np.asarray(original),np.asarray(transformed),np.asarray(times)
    if p.shape!=q.shape or p.shape!=t.shape or not np.isfinite(q).all():
        raise ValueError('Aligned rank-verification vectors required')
    order=np.lexsort((p,t))
    a,b,age=p[order],q[order],t[order]
    same_time=age[1:]==age[:-1]
    tied=same_time&(a[1:]==a[:-1])
    distinct=same_time&~tied
    if np.any(b[1:][tied]!=b[:-1][tied]) or np.any(b[1:][distinct]<=b[:-1][distinct]):
        raise AssertionError('Observed-age transform changed within-time ranks or ties')
    return {'points':len(p),'within_time_adjacent_ties':int(tied.sum()),'within_time_adjacent_distinct':int(distinct.sum()),'exact_rank_and_tie_parity':True}
