"""Bounded within-time RankNet approximations and official ranking objectives.

Only training targets/times define queries. The evaluation metric remains exact.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
from numba import njit,prange
from scipy.special import expit
from threadpoolctl import threadpool_limits
from lightgbm import LGBMRegressor,LGBMRanker
from src.models.learners import make_estimator
from src.models.pairwise import WithinTimePairObjective,export_custom_lightgbm
from src.models.compact_trees import _arrays,export_lightgbm
from src.next.fitting import weights_from_training
from src.next.predictor import NextBundle,predictor_hash,arch_contract
from src.next.engine import feature_hash


def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@njit(cache=True,parallel=True)
def histogram_pair_gradients(target,prediction,order,cuts,bins,normalizer):
    n=len(target)
    gradient=np.zeros(n,dtype=np.float64)
    hessian=np.full(n,1e-12,dtype=np.float64)
    for query in prange(len(cuts)-1):
        start,end=cuts[query],cuts[query+1]
        minimum,maximum=1e300,-1e300
        for j in range(start,end):
            value=prediction[order[j]]
            minimum=min(minimum,value);maximum=max(maximum,value)
        width=max(maximum-minimum,1e-10)
        count=np.zeros((2,bins),dtype=np.int64)
        total=np.zeros((2,bins),dtype=np.float64)
        for j in range(start,end):
            row=order[j]
            cell=min(int((prediction[row]-minimum)/width*bins),bins-1)
            label=target[row]
            count[label,cell]+=1
            total[label,cell]+=prediction[row]
        g=np.zeros((2,bins),dtype=np.float64)
        h=np.zeros((2,bins),dtype=np.float64)
        for positive in range(bins):
            if count[1,positive]==0:
                continue
            pos_margin=total[1,positive]/count[1,positive]
            for negative in range(bins):
                if count[0,negative]==0:
                    continue
                neg_margin=total[0,negative]/count[0,negative]
                difference=min(max(pos_margin-neg_margin,-50.),50.)
                probability=1./(1.+np.exp(difference))
                curvature=probability*(1.-probability)
                g[1,positive]-=count[0,negative]*probability
                g[0,negative]+=count[1,positive]*probability
                h[1,positive]+=count[0,negative]*curvature
                h[0,negative]+=count[1,positive]*curvature
        for j in range(start,end):
            row=order[j]
            cell=min(int((prediction[row]-minimum)/width*bins),bins-1)
            gradient[row]=g[target[row],cell]*normalizer
            hessian[row]=max(h[target[row],cell]*normalizer,1e-12)
    return gradient,hessian


@dataclass
class HistogramPairObjective:
    target:np.ndarray
    order:np.ndarray
    cuts:np.ndarray
    bins:int
    normalizer:float
    calls:int=0

    @classmethod
    def build(cls,target,time_online,bins=64):
        if bins not in (64,128):
            raise ValueError('Only two preregistered gradient resolutions are permitted')
        y=np.asarray(target,dtype=np.uint8)
        t=np.asarray(time_online,dtype=np.int32)
        order=np.argsort(t,kind='stable').astype(np.int32)
        cuts=np.r_[0,np.flatnonzero(np.diff(t[order]))+1,len(t)].astype(np.int32)
        positive=np.bincount(t,weights=y)
        negative=np.bincount(t,weights=1-y,minlength=len(positive))
        pairs=float(np.dot(positive,negative))
        if pairs<=0:
            raise ValueError('No within-time training pairs')
        return cls(y.copy(),order,cuts,bins,len(y)/(2.*pairs))

    def __call__(self,y_true,prediction):
        if not self.calls and not np.array_equal(y_true,self.target):
            raise RuntimeError('Within-time training row identity changed')
        self.calls+=1
        return histogram_pair_gradients(self.target,np.asarray(prediction,dtype=float),self.order,self.cuts,self.bins,self.normalizer)


def export_ranking(booster):
    dump=booster.dump_model()
    if dump.get('objective') not in ('lambdarank','rank_xendcg') or dump['average_output'] or dump['num_tree_per_iteration']!=1:
        raise ValueError('Unexpected scalar ranking model contract')
    roots,features,thresholds,left,right,values,missing=[],[],[],[],[],[],[]
    def add(node):
        index=len(features)
        features.append(-1);thresholds.append(0.);left.append(-1);right.append(-1);values.append(0.);missing.append(False)
        if 'leaf_value' in node:
            values[index]=float(node['leaf_value'])
            return index
        if node['decision_type']!='<=' or node['missing_type'] not in ('None','NaN'):
            raise ValueError('Unsupported ranking tree split')
        features[index]=int(node['split_feature']);thresholds[index]=float(node['threshold']);missing[index]=bool(node['default_left'])
        left[index]=add(node['left_child']);right[index]=add(node['right_child'])
        return index
    for tree in dump['tree_info']:
        roots.append(add(tree['tree_structure']))
    return _arrays(roots,features,thresholds,left,right,values,missing,0.)


def fit_ranking(design,metadata,training_rows,validation_rows,*,guard,config,names,objective):
    train=np.asarray(training_rows,dtype=np.int64)
    valid=np.asarray(validation_rows,dtype=np.int64)
    guard.partition(np.unique(metadata['dataset_id'][train]),np.unique(metadata['dataset_id'][valid]))
    kind=objective['kind']
    grouped=kind in ('binary_query_sorted','rank_xendcg','lambdarank')
    if grouped:
        train=train[np.argsort(metadata['time_online'][train],kind='stable')]
    matrix=np.asarray(design[train],dtype=np.float32)
    columns=np.flatnonzero(np.ptp(matrix,axis=0)>1e-12).astype(np.int32)
    x=matrix[:,columns]
    y=metadata['target'][train]
    t=metadata['time_online'][train]
    if not np.isfinite(x).all() or set(np.unique(y))!={0,1}:
        raise ValueError('Finite features and both classes required')
    settings=dict(n_estimators=100,num_leaves=15,max_depth=4,min_child_samples=30,reg_lambda=10,
        learning_rate=.05,n_jobs=4,random_state=20260908,deterministic=True,force_col_wise=True,verbosity=-1)
    callback=None
    fit_arguments={}
    if kind=='binary_query_sorted':
        estimator=make_estimator('M4_lightgbm')
        fit_arguments['sample_weight']=weights_from_training(y,t,metadata['dataset_id'][train],'W3')
    elif kind=='sampled_ranknet':
        callback=WithinTimePairObjective.build(y,t,pairs_per_positive=objective['pairs_per_positive'],seed=20260908)
        estimator=LGBMRegressor(objective=callback,boost_from_average=False,metric='None',**settings)
    elif kind=='histogram_ranknet':
        callback=HistogramPairObjective.build(y,t,bins=objective['bins'])
        estimator=LGBMRegressor(objective=callback,boost_from_average=False,metric='None',**settings)
    elif kind in ('rank_xendcg','lambdarank'):
        parameters={'objective_seed':20260908,'label_gain':[0,1]}
        if kind=='lambdarank':
            parameters['lambdarank_truncation_level']=64
        estimator=LGBMRanker(objective=kind,**settings,**parameters)
        _,counts=np.unique(t,return_counts=True)
        fit_arguments['group']=counts
    else:
        raise ValueError('Unregistered ranking objective')
    with threadpool_limits(limits=4):
        estimator.fit(x,y,**fit_arguments)
    booster=estimator.booster_
    if kind=='binary_query_sorted':
        compact=export_lightgbm(estimator)
    elif kind in ('rank_xendcg','lambdarank'):
        compact=export_ranking(booster)
    else:
        compact=export_custom_lightgbm(booster)
    probe=x[:min(2048,len(x))]
    np.testing.assert_allclose(compact.predict(probe),expit(booster.predict(probe,raw_score=True,num_threads=1)),rtol=0,atol=1e-12)
    # Persist the native forest, not the sklearn wrapper's potentially very large
    # training-pair objective object. Inference uses the checked compact forest.
    booster.free_dataset()
    model=NextBundle(config,tuple(names),columns,compact,booster,kind,feature_hash(config),predictor_hash(),
                     arch_contract() if any(name.startswith('arch8__') for name in names) else '')
    prediction=model.predict(design[valid]).astype(np.float32)
    details={'training_rows':len(train),'validation_rows':len(valid),'retained_columns':len(columns),
        'ranking_objective':objective,'objective_source_sha256':implementation_hash(),'queries':len(np.unique(t)),
        'query_group':'same absolute online time','validation_labels_in_objective':False,
        'training_sorted_by_query':grouped,'native_compact_max_abs_diff_bound':1e-12}
    if callback is not None:
        details['objective_calls']=callback.calls
    return model,prediction,details
