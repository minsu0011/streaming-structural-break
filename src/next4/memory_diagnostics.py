"""Fixed-prediction memory diagnostics; none of these quantities enter inference."""
import numpy as np


def local_recovery(excess, *, sustain=32, tolerance=.01):
    """First beginning of a fully observed quiet run, censored at last start + 1.

    A quiet run only bounds positive contamination; negative score changes can
    qualify. This retrospective quantity is not an online stopping rule.
    """
    values=np.asarray(excess,dtype=np.float64)
    if values.ndim!=2 or not np.isfinite(values).all() or not 1<=sustain<=values.shape[1]:
        raise ValueError('Finite row-by-age values and an observed quiet window required')
    quiet=values<=tolerance
    cumulative=np.pad(np.cumsum(quiet,axis=1,dtype=np.int32),((0,0),(1,0)))
    runs=(cumulative[:,sustain:]-cumulative[:,:-sustain])==sustain
    observed=runs.any(axis=1)
    limit=runs.shape[1]
    time=np.where(observed,runs.argmax(axis=1),limit).astype(np.int32)
    return time,observed,limit


def aligned_outlier_excess(prediction,tau_cycle,*,burst_length,window=256):
    values=np.asarray(prediction)
    if values.ndim!=2 or len(values)%2 or not np.isfinite(values).all() or burst_length<1:
        raise ValueError('Finite paired clean/altered predictions required')
    n=len(values)//2
    if n<1 or not tau_cycle:raise ValueError('Nonempty paired cohort and fixed intervention cycle required')
    start=np.asarray([tau_cycle[i%len(tau_cycle)]+burst_length for i in range(n)],dtype=np.int64)
    if np.any(start<0) or np.any(start+window>values.shape[1]):
        raise ValueError('The entire fixed post-intervention window must be observed')
    index=start[:,None]+np.arange(window)[None,:]
    clean=values[np.arange(n)[:,None],index]
    altered=values[(np.arange(n)+n)[:,None],index]
    excess=altered.astype(np.float64)-clean.astype(np.float64)
    return clean,altered,excess


def null_segments(prediction,cutoff,*,segment_length=1024):
    values=np.asarray(prediction)
    if values.ndim!=2 or not len(values) or not np.isfinite(values).all() or values.shape[1]%segment_length:
        raise ValueError('Complete equally sized finite null segments required')
    prior=np.zeros(len(values),dtype=bool);rows=[]
    for begin in range(0,values.shape[1],segment_length):
        p=values[:,begin:begin+segment_length];event=(p>cutoff).any(axis=1)
        risk=~prior;new=event&risk
        rows.append({'segment_start':begin,'segment_end_exclusive':begin+segment_length,
            'series':len(p),'at_risk_at_segment_start':int(risk.sum()),'new_first_alarms':int(new.sum()),
            'conditional_first_alarm_fraction':float(new.sum()/risk.sum()) if risk.any() else None,
            'any_alarm_in_segment_fraction':float(event.mean()),
            'point_alarm_fraction':float((p>cutoff).mean()),
            'score_median':float(np.median(p)),'score_q99':float(np.quantile(p,.99)),
            'median_segment_maximum':float(np.median(p.max(axis=1))),
            'eventual_alarm_fraction':float((prior|event).mean())})
        prior|=event
    return rows


def paired_mean_interval(candidate,reference,counts):
    a=np.asarray(candidate,dtype=np.float64);b=np.asarray(reference,dtype=np.float64)
    multiplicity=np.asarray(counts)
    if a.ndim!=1 or b.shape!=a.shape or not len(a) or not np.isfinite(a-b).all():
        raise ValueError('Finite aligned generating-seed measurements required')
    if multiplicity.ndim!=2 or multiplicity.shape[1]!=len(a) or len(multiplicity)<2 or np.any(multiplicity<0):
        raise ValueError('Original and nonnegative paired bootstrap rows required')
    if not np.array_equal(multiplicity[0],np.ones(len(a))) or not np.all(multiplicity.sum(axis=1)==len(a)):
        raise ValueError('Fixed whole-cohort multiplicities required')
    draws=(multiplicity@(a-b))/len(a)
    return {'estimate_mean_delta':float(draws[0]),'ci_2_5':float(np.quantile(draws[1:],.025)),
        'ci_97_5':float(np.quantile(draws[1:],.975)),
        'probability_delta_positive':float(np.mean(draws[1:]>0)),'replicates':len(draws)-1}
