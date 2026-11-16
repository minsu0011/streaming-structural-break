"""Evaluation-only fixed scopes; post-hoc mechanism labels never enter models."""
import numpy as np

TIME_BINS=((0,9),(10,24),(25,49),(50,99),(100,199),(200,399),(400,None))
AGE_BINS=((0,4),(5,9),(10,19),(20,49),(50,None))
MECHANISMS=('DEPENDENCE_DOMINANT','SCALE_DOMINANT','MIXED','MEAN_DOMINANT','TAIL_DOMINANT','WEAK_OR_AMBIGUOUS')


def label(prefix,low,high):return prefix+'_'+str(low)+'_'+('plus' if high is None else str(high))


def masks(target,times,tau,regime,signature,direction):
    y=np.asarray(target);times=np.asarray(times);tau=np.asarray(tau)
    result={'all_time':np.ones(len(y),dtype=bool)}
    for low,high in TIME_BINS:result[label('time',low,high)]=(times>=low)&(True if high is None else times<=high)
    age=times-tau
    for low,high in AGE_BINS:
        # Restrict the positive side by break age while preserving all negative
        # comparators at their original online times. These pair sets partition
        # all-time positive-negative pairs, unlike the mechanism subsets.
        result[label('age',low,high)]=(y==0)|((y==1)&(age>=low)&(True if high is None else age<=high))
    for group in range(5):result['regime_'+str(group)]=regime==group
    for name in MECHANISMS:result['mechanism_'+name]=(signature==name)|(tau<0)
    for name in ('decrease','increase'):result['mechanism_observed_scale_'+name]=(direction==name)|(tau<0)
    return result


def pair_mass(target,times):
    y=np.asarray(target);t=np.asarray(times,dtype=np.int64)
    if not len(y):return 0
    count=int(t.max())+1
    positive=np.bincount(t[y==1],minlength=count);negative=np.bincount(t[y==0],minlength=count)
    return int(np.sum(positive*negative,dtype=np.int64))
