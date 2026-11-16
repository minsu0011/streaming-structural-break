"""Support-aware heldout diagnostics with original-group uncertainty."""
import numpy as np
from src.next8.statistics import rank_influence,cluster_difference,series_mean
from src.next8.protocol import TIME_BUCKETS,H_BUCKETS


def original_groups(cache,rows):
    ids=cache.meta['dataset_id'][rows]
    return np.asarray([cache.data.guard.split.groups[int(s)] for s in ids])


def safe_rank(y,p,t,groups):
    if not len(y):return {'status':'INSUFFICIENT_SUPPORT','TS_AUC':None,'ci_low':None,'ci_high':None,
        'pairs':0,'series':0,'positive_series':0,'negative_series':0}
    record,_,_=rank_influence(y,p,t,groups)
    if record['pairs']==0:record.update({'TS_AUC':None,'ci_low':None,'ci_high':None})
    supported=record['pairs']>0 and min(record['positive_series'],record['negative_series'])>=32 and record['series']>=64
    return {**record,'status':'SUPPORTED' if supported else 'INSUFFICIENT_SUPPORT',
        'unit':'Original dependency group; point TS-AUC uses original rows, approximate U-statistic cluster interval'}


def time_H_domains(cache,rows):
    rows=np.asarray(rows);t=cache.meta['time_online'][rows]
    h=cache.series['H_length'][cache.series_index[rows]]
    yield 'ALL','ALL',np.ones(len(rows),dtype=bool)
    for family,values,buckets in (('TIME',t,TIME_BUCKETS),('H_LENGTH',h,H_BUCKETS)):
        for k,(lo,hi) in enumerate(buckets):yield family,f'{family}_{k}',(values>=lo)&(True if hi is None else values<=hi)


def loss_evidence(groups,reference,candidate):
    if not len(groups):return {'status':'INSUFFICIENT_SUPPORT','series':0,'gain':None,'ci_low':None,'ci_high':None,'p_normal_cluster':1.},np.asarray([]),np.asarray([])
    if np.array_equal(reference,candidate):
        keys=np.unique(groups);delta=np.zeros(len(keys))
        return {'status':'SUPPORTED' if len(keys)>=64 else 'INSUFFICIENT_SUPPORT','series':len(keys),
            'gain':0.,'ci_low':0. if len(keys)>=2 else None,'ci_high':0. if len(keys)>=2 else None,
            'p_normal_cluster':1.,'draws':2000,'positive_resample_fraction':0.,
            'unit':'Whole original dependency group','exact_zero_every_resample':'Identical losses; analytic bootstrap equivalence avoids redundant draws'},keys,delta
    result=cluster_difference(groups,reference,candidate,draws=2000)
    keys,delta=series_mean(groups,np.asarray(reference)-np.asarray(candidate))
    result.update({'unit':'Whole original dependency group, equal mean within group',
        'effect':'Reference loss minus candidate loss; positive is improvement','training_fixed':True})
    return result,keys,delta


def equal_fold_loss_summary(parts,*,seed=8101604):
    """Each input is a vector of mean loss gains per original heldout group."""
    if not parts or any(len(a)<64 for a in parts):return {'status':'INSUFFICIENT_SUPPORT','folds':len(parts)}
    rng=np.random.default_rng(seed);samples=[];fold=[]
    for values in parts:
        values=np.asarray(values,dtype=np.float64);fold.append(float(values.mean()));draws=[]
        for start in range(0,2000,64):
            ix=rng.integers(0,len(values),(min(64,2000-start),len(values)))
            draws.append(values[ix].mean(axis=1))
        samples.append(np.concatenate(draws))
    mean=np.mean(samples,axis=0);effect=float(np.mean(fold))
    from scipy.stats import norm
    variance=sum(np.var(a,ddof=1)/len(a) for a in parts)/(len(parts)**2)
    p=float(2*norm.sf(abs(effect)/np.sqrt(variance))) if variance else (0. if effect else 1.)
    return {'status':'SUPPORTED','folds':len(parts),'fold_gains':fold,'mean_gain':effect,
        'positive_folds':sum(v>0 for v in fold),'worst_fold_gain':min(fold),
        'ci_low':float(np.quantile(mean,.025)),'ci_high':float(np.quantile(mean,.975)),
        'positive_resample_fraction':float(np.mean(mean>0)),'positive_fraction_is_posterior':False,
        'p_normal_cluster':p,'p_method':'Normal approximation to independent original-group loss means, not randomization',
        'draws':2000,'seed':seed,'unit':'Whole original groups within original folds, equal-fold average; fixed models'}
