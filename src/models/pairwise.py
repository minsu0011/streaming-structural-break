"""Sampled within-time RankNet loss with pair-count contribution weights."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
from scipy.special import expit
from lightgbm import LGBMRegressor
from threadpoolctl import threadpool_limits
from src.models.learners import ModelBundle,implementation_hash as predictor_hash
from src.models.compact_trees import _arrays
from src.features.streaming import FEATURE_FAMILIES,feature_version
from src.features.config import CONFIG

def implementation_hash():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

@dataclass
class WithinTimePairObjective:
    positive: np.ndarray
    negative: np.ndarray
    weight: np.ndarray
    target: np.ndarray
    calls: int = 0

    @classmethod
    def build(cls,target,time_online,*,pairs_per_positive=8,seed=20260908):
        if pairs_per_positive not in (8,32):raise ValueError('Only two predeclared pair-sampling budgets are permitted')
        y=np.asarray(target,dtype=np.uint8);t=np.asarray(time_online,dtype=np.int64)
        if not np.isin(y,[0,1]).all() or y.shape!=t.shape:raise ValueError('Binary labels and matching online steps required')
        order=np.argsort(t,kind='stable');cuts=np.r_[0,np.flatnonzero(np.diff(t[order]))+1,len(t)]
        rng=np.random.default_rng(seed);positives=[];negatives=[];weights=[]
        for start,end in zip(cuts[:-1],cuts[1:]):
            indices=order[start:end];pos=indices[y[indices]==1];neg=indices[y[indices]==0]
            if not len(pos) or not len(neg):continue
            count=len(pos)*pairs_per_positive
            positives.append(np.repeat(pos,pairs_per_positive).astype(np.int32))
            negatives.append(neg[rng.integers(0,len(neg),size=count)].astype(np.int32))
            weights.append(np.full(count,len(neg)/pairs_per_positive,dtype=np.float64))
        if not positives:raise ValueError('No within-time training pairs')
        positive=np.concatenate(positives);negative=np.concatenate(negatives);weight=np.concatenate(weights)
        weight*=len(y)/(2*weight.sum())
        return cls(positive,negative,weight,y.copy())

    def loss(self,prediction):
        p=np.asarray(prediction,dtype=np.float64)
        return float(np.dot(self.weight,np.logaddexp(0.,-(p[self.positive]-p[self.negative]))))

    def __call__(self,y_true,prediction):
        if not self.calls and not np.array_equal(y_true,self.target):raise RuntimeError('Pair objective row order or training labels changed')
        self.calls+=1
        p=np.asarray(prediction,dtype=np.float64);q=expit(-(p[self.positive]-p[self.negative]))
        gradient_weight=self.weight*q;hessian_weight=gradient_weight*(1-q)
        gradient=np.bincount(self.negative,weights=gradient_weight,minlength=len(p))-np.bincount(self.positive,weights=gradient_weight,minlength=len(p))
        hessian=np.bincount(self.negative,weights=hessian_weight,minlength=len(p))+np.bincount(self.positive,weights=hessian_weight,minlength=len(p))
        return gradient,np.maximum(hessian,1e-12)

    @property
    def array_bytes(self):return self.positive.nbytes+self.negative.nbytes+self.weight.nbytes+self.target.nbytes

def export_custom_lightgbm(booster):
    dump=booster.dump_model()
    if dump.get('objective') not in ('custom','none',None) or dump['average_output'] or dump['num_tree_per_iteration']!=1:raise ValueError('Unexpected custom scalar GBDT contract')
    roots=[];features=[];thresholds=[];left=[];right=[];values=[];missing=[]
    def add(node):
        index=len(features);features.append(-1);thresholds.append(0.);left.append(-1);right.append(-1);values.append(0.);missing.append(False)
        if 'leaf_value' in node:values[index]=float(node['leaf_value']);return index
        if node['decision_type']!='<=' or node['missing_type'] not in ('None','NaN'):raise ValueError('Unsupported custom GBDT node')
        features[index]=int(node['split_feature']);thresholds[index]=float(node['threshold']);missing[index]=bool(node['default_left'])
        left[index]=add(node['left_child']);right[index]=add(node['right_child']);return index
    for tree in dump['tree_info']:roots.append(add(tree['tree_structure']))
    return _arrays(roots,features,thresholds,left,right,values,missing,0.)

def fit_pairwise(features,target,time_online,*,families='ABCD',config=CONFIG,pairs_per_positive=8,seed=20260908,params=None):
    matrix=np.asarray(features,dtype=np.float32)
    columns=np.array([j for j,family in enumerate(FEATURE_FAMILIES) if family in families and np.ptp(matrix[:,j])>1e-12],dtype=np.int32)
    x=matrix[:,columns];y=np.asarray(target,dtype=np.uint8)
    if not np.isfinite(x).all():raise ValueError('Finite causal features required')
    objective=WithinTimePairObjective.build(y,time_online,pairs_per_positive=pairs_per_positive,seed=seed)
    settings=dict(n_estimators=100,num_leaves=15,max_depth=4,min_child_samples=30,reg_lambda=10,learning_rate=.05,
        n_jobs=4,random_state=20260908,deterministic=True,force_col_wise=True,verbosity=-1,boost_from_average=False,metric='None')
    settings.update(params or {})
    estimator=LGBMRegressor(objective=objective,**settings)
    with threadpool_limits(limits=4):estimator.fit(x,y)
    booster=estimator.booster_;compact=export_custom_lightgbm(booster)
    probe=x[:min(2048,len(x))]
    np.testing.assert_allclose(compact.predict(probe),expit(booster.predict(probe,raw_score=True,num_threads=1)),rtol=0,atol=1e-12)
    metadata={'pair_count':len(objective.positive),'pair_array_bytes':objective.array_bytes,'objective_calls':objective.calls,
        'pairs_per_positive':pairs_per_positive,'pair_seed':seed,'training_rows':len(y),'objective_hash':implementation_hash(),
        'time_step_is_a_pair_group_not_a_feature':True,'validation_labels_used':False,'full_pair_matrix_created':False}
    booster.free_dataset()
    model=ModelBundle('M6_pairwise',feature_version(config),columns,booster,compact=compact,predictor_hash=predictor_hash(),feature_config=config)
    return model,metadata
