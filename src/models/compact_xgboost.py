"""Optional numeric binary XGBoost exporter with explicit float32 accumulation.

Source format: official Booster.save_raw(raw_format='json'), not a textual dump.
Unsupported objectives, categorical trees and vector leaves fail closed.
"""
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
import numpy as np
from numba import njit

def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

@njit(cache=False)
def margin_one(x,roots,features,thresholds,left,right,values,missing_left,bias):
    result=np.float32(bias)
    for root in roots:
        node=root
        while features[node]>=0:
            value=x[features[node]]
            go_left=missing_left[node] if np.isnan(value) else value<thresholds[node]
            node=left[node] if go_left else right[node]
        result=np.float32(result+values[node])
    return result

@njit(cache=False)
def probability_one(x,roots,features,thresholds,left,right,values,missing_left,bias):
    margin=margin_one(x,roots,features,thresholds,left,right,values,missing_left,bias)
    return np.float32(1.)/(np.float32(1.)+np.exp(-margin))

@njit(cache=False)
def probabilities(x,roots,features,thresholds,left,right,values,missing_left,bias):
    output=np.empty(len(x),dtype=np.float32)
    for i in range(len(x)):output[i]=probability_one(x[i],roots,features,thresholds,left,right,values,missing_left,bias)
    return output

@dataclass
class CompactXGBoost:
    roots: np.ndarray
    features: np.ndarray
    thresholds: np.ndarray
    left: np.ndarray
    right: np.ndarray
    values: np.ndarray
    missing_left: np.ndarray
    bias: np.float32
    source_hash: str

    def args(self):return self.roots,self.features,self.thresholds,self.left,self.right,self.values,self.missing_left,self.bias
    def predict(self,x):return probabilities(np.asarray(x,dtype=np.float32),*self.args())
    def predict_one(self,x):return float(probability_one(x,*self.args()))

def export_xgboost(estimator):
    booster=estimator.get_booster()
    raw=booster.save_raw(raw_format='json');dump=json.loads(raw);learner=dump['learner']
    if learner['objective']['name']!='binary:logistic' or learner['gradient_booster']['name']!='gbtree':
        raise ValueError('Only binary logistic gbtree is supported')
    params=learner['learner_model_param']
    if int(params['num_target'])!=1 or int(params['num_class'])!=0:raise ValueError('Only scalar binary output is supported')
    base=json.loads(params['base_score'])
    if isinstance(base,list):
        if len(base)!=1:raise ValueError('Vector intercept unsupported')
        base=base[0]
    base=np.float32(base)
    if not 0<base<1:raise ValueError('Invalid logistic base score')
    bias=np.float32(-np.log(np.float32(1.)/base-np.float32(1.)))
    roots=[];features=[];thresholds=[];left=[];right=[];values=[];missing=[]
    model=learner['gradient_booster']['model']
    if any(int(i)!=0 for i in model['tree_info']):raise ValueError('Multi-output tree assignment unsupported')
    for tree in model['trees']:
        if any(tree['split_type']) or int(tree['tree_param']['size_leaf_vector'])!=1 or int(tree['tree_param']['num_deleted'])!=0:
            raise ValueError('Categorical, vector-leaf or deleted-node export unsupported')
        offset=len(features);roots.append(offset)
        for j,(lc,rc) in enumerate(zip(tree['left_children'],tree['right_children'])):
            leaf=lc==-1
            if leaf!=(rc==-1):raise ValueError('Malformed scalar tree')
            features.append(-1 if leaf else int(tree['split_indices'][j]))
            thresholds.append(tree['split_conditions'][j]);values.append(tree['split_conditions'][j] if leaf else 0.)
            left.append(-1 if leaf else offset+lc);right.append(-1 if leaf else offset+rc);missing.append(bool(tree['default_left'][j]))
    return CompactXGBoost(np.asarray(roots,dtype=np.int32),np.asarray(features,dtype=np.int32),np.asarray(thresholds,dtype=np.float32),
        np.asarray(left,dtype=np.int32),np.asarray(right,dtype=np.int32),np.asarray(values,dtype=np.float32),np.asarray(missing,dtype=np.bool_),bias,hashlib.sha256(raw).hexdigest())
