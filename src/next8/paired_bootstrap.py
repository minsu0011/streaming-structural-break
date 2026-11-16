"""Exact paired whole-group bootstrap of pair-weighted same-time AUC.

Sorting is reused, but every resample recomputes pair counts and tied wins.
This is not a first-order influence approximation or a row bootstrap.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from numba import njit
from src.scoring.ts_auc import ts_auc


@njit(cache=True)
def _weighted_auc_many(code,y,end_score,end_time,weights):
    output=np.empty(len(weights),dtype=np.float64)
    pairs_out=np.empty(len(weights),dtype=np.float64)
    for b in range(len(weights)):
        total_pairs=0.;total_wins=0.;negative_seen=0.;positive_total=0.;negative_total=0.
        positive_tie=0.;negative_tie=0.;time_wins=0.
        for k in range(len(code)):
            w=float(weights[b,code[k]])
            if y[k]:positive_tie+=w
            else:negative_tie+=w
            if end_score[k]:
                time_wins+=positive_tie*(negative_seen+.5*negative_tie)
                positive_total+=positive_tie;negative_total+=negative_tie
                negative_seen+=negative_tie;positive_tie=0.;negative_tie=0.
            if end_time[k]:
                total_pairs+=positive_total*negative_total;total_wins+=time_wins
                negative_seen=0.;positive_total=0.;negative_total=0.;time_wins=0.
        pairs_out[b]=total_pairs
        output[b]=total_wins/total_pairs if total_pairs>0 else np.nan
    return output,pairs_out


@dataclass
class PreparedRank:
    groups: np.ndarray
    code: np.ndarray
    y: np.ndarray
    end_score: np.ndarray
    end_time: np.ndarray
    point_auc: float
    point_pairs: int

    @classmethod
    def prepare(cls,y,p,t,group):
        y=np.asarray(y);p=np.asarray(p,dtype=np.float64);t=np.asarray(t);group=np.asarray(group)
        if y.ndim!=1 or not(len(y)==len(p)==len(t)==len(group)) or not len(y):raise ValueError('Equal nonempty rank vectors')
        if not np.isfinite(p).all() or not np.isin(y,[0,1]).all() or np.any(t<0) or np.any(t!=np.floor(t)):
            raise ValueError('Finite causal ranking inputs required')
        groups,code=np.unique(group,return_inverse=True);order=np.lexsort((p,t))
        yy=y[order].astype(np.uint8);pp=p[order];tt=t[order]
        end_time=np.r_[tt[:-1]!=tt[1:],True]
        end_score=np.r_[(tt[:-1]!=tt[1:])|(pp[:-1]!=pp[1:]),True]
        result,pairs=_weighted_auc_many(code[order].astype(np.int32),yy,end_score,end_time,np.ones((1,len(groups)),dtype=np.uint16))
        official=ts_auc(y,p,t)
        if pairs[0]>0 and abs(result[0]-official)>1e-12:raise AssertionError('Canonical scorer parity failure')
        return cls(groups,code[order].astype(np.int32),yy,end_score,end_time,float(result[0]),int(pairs[0]))

    def evaluate(self,weights):
        weights=np.asarray(weights)
        if weights.ndim!=2 or weights.shape[1]!=len(self.groups) or np.any(weights<0):
            raise ValueError('Exact same group-weight matrix required')
        return _weighted_auc_many(self.code,self.y,self.end_score,self.end_time,weights)


def group_resample_weights(groups,draws=2000,seed=8101604,partition=''):
    groups=np.asarray(groups)
    if len(groups)!=len(np.unique(groups)) or not len(groups):raise ValueError('Unique nonempty dependency groups')
    key=hashlib.sha256(('NEXT8_GROUP_BOOTSTRAP_V1|'+str(partition)+'|'+','.join(map(str,groups))).encode()).hexdigest()
    rng=np.random.default_rng(int(seed)^int(key[:8],16));weights=np.empty((draws,len(groups)),dtype=np.uint16)
    for k in range(draws):weights[k]=np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups))
    return weights,{'draws':draws,'seed':seed,'partition':partition,'group_count':len(groups),
        'resample_key_sha256':key,'weight_matrix_sha256':hashlib.sha256(weights.tobytes()).hexdigest(),
        'unit':'Whole original dependency group, resampled within original heldout partition',
        'parallel_worker_time_not_a_statistical_sample':True}


def evaluate_partition(y,predictions,t,group,*,partition,draws=2000):
    groups=np.unique(group);weights,receipt=group_resample_weights(groups,draws,partition=partition)
    points={};samples={};pair_counts=None
    for method,prediction in predictions.items():
        prepared=PreparedRank.prepare(y,prediction,t,group)
        assert np.array_equal(prepared.groups,groups)
        sample,pairs=prepared.evaluate(weights)
        if pair_counts is None:pair_counts=pairs
        else:np.testing.assert_array_equal(pair_counts,pairs)
        points[method]={'TS_AUC':prepared.point_auc if prepared.point_pairs else None,'pairs':prepared.point_pairs}
        samples[method]=sample
    receipt.update({'methods':list(predictions),'same_weights_for_every_method':True,
        'zero_pair_draws':int(np.sum(pair_counts==0)),'zero_pair_draws_not_imputed_as_half':True})
    return points,samples,receipt


def summarize_paired(points,samples,reference,candidate):
    """Average exact partition scores with equal fold weight, as preregistered."""
    if len(points)!=len(samples) or not points:raise ValueError('Matched nonempty partitions required')
    if any(p[reference]['TS_AUC'] is None or p[candidate]['TS_AUC'] is None for p in points):
        return {'status':'INSUFFICIENT_SUPPORT','reason':'A required partition has zero same-time pairs'}
    fold=np.asarray([p[candidate]['TS_AUC']-p[reference]['TS_AUC'] for p in points])
    array=np.stack([s[candidate]-s[reference] for s in samples])
    valid=np.isfinite(array).all(axis=0)
    result={'reference':reference,'candidate':candidate,'mean_delta':float(fold.mean()),
        'fold_deltas':fold.tolist(),'positive_folds':int(np.sum(fold>0)),'worst_fold_delta':float(fold.min()),
        'draws':array.shape[1],'valid_draws':int(valid.sum()),'invalid_draws':int((~valid).sum()),
        'uncertainty_unit':'Paired whole dependency groups within original folds; equal-fold mean',
        'training_models_held_fixed_during_bootstrap':True,'positive_resample_fraction_is_posterior':False}
    if valid.sum()<.95*array.shape[1]:return {**result,'status':'INSUFFICIENT_SUPPORT','ci_low':None,'ci_high':None}
    value=array[:,valid].mean(axis=0)
    return {**result,'status':'SUPPORTED','ci_low':float(np.quantile(value,.025)),
        'ci_high':float(np.quantile(value,.975)),'positive_resample_fraction':float(np.mean(value>0))}
