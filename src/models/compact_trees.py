"""Validated numeric binary-tree inference, avoiding per-row library wrappers.

    Export is intentionally limited to the exact HistGB and LightGBM candidates.
    Unsupported categorical splits, link functions or averaging fail closed.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit
from scipy.special import expit

@njit(cache=False)
def tree_margin(x,roots,features,thresholds,left,right,values,missing_left,bias):
    result=bias
    for root in roots:
        node=root
        while features[node]>=0:
            value=x[features[node]]
            go_left=missing_left[node] if np.isnan(value) else value<=thresholds[node]
            node=left[node] if go_left else right[node]
        result+=values[node]
    return result

@njit(cache=False)
def tree_margins(x,roots,features,thresholds,left,right,values,missing_left,bias):
    output=np.empty(len(x),dtype=np.float64)
    for i in range(len(x)):
        output[i]=tree_margin(x[i],roots,features,thresholds,left,right,values,missing_left,bias)
    return output

@dataclass
class CompactTrees:
    roots: np.ndarray
    features: np.ndarray
    thresholds: np.ndarray
    left: np.ndarray
    right: np.ndarray
    values: np.ndarray
    missing_left: np.ndarray
    bias: float

    def args(self):
        return self.roots,self.features,self.thresholds,self.left,self.right,self.values,self.missing_left,self.bias

    def predict_one(self,x):
        return float(expit(tree_margin(x,*self.args())))

    def predict(self,x):
        return expit(tree_margins(x,*self.args()))

def export_histgb(estimator):
    if estimator.n_trees_per_iteration_!=1:raise ValueError('Binary HistGB required')
    roots=[];features=[];thresholds=[];left=[];right=[];values=[];missing=[]
    for iteration in estimator._predictors:
        nodes=iteration[0].nodes
        if np.any(nodes['is_categorical']):raise ValueError('Categorical export unsupported')
        start=len(features);roots.append(start)
        for node in nodes:
            features.append(-1 if node['is_leaf'] else int(node['feature_idx']))
            thresholds.append(float(node['num_threshold']))
            left.append(start+int(node['left']));right.append(start+int(node['right']))
            values.append(float(node['value']));missing.append(bool(node['missing_go_to_left']))
    return _arrays(roots,features,thresholds,left,right,values,missing,float(estimator._baseline_prediction.ravel()[0]))

def export_lightgbm(estimator):
    dump=estimator.booster_.dump_model()
    if dump['objective']!='binary sigmoid:1' or dump['average_output'] or dump['num_tree_per_iteration']!=1:
        raise ValueError('Unsupported LightGBM objective or averaging')
    roots=[];features=[];thresholds=[];left=[];right=[];values=[];missing=[]
    def add(node):
        index=len(features);features.append(-1);thresholds.append(0.);left.append(-1);right.append(-1);values.append(0.);missing.append(False)
        if 'leaf_value' in node:
            values[index]=float(node['leaf_value'])
            return index
        if node['decision_type']!='<=' or node['missing_type'] not in ('None','NaN'):
            raise ValueError('Unsupported LightGBM split/missing policy')
        features[index]=int(node['split_feature']);thresholds[index]=float(node['threshold']);missing[index]=bool(node['default_left'])
        left[index]=add(node['left_child']);right[index]=add(node['right_child'])
        return index
    for tree in dump['tree_info']:roots.append(add(tree['tree_structure']))
    return _arrays(roots,features,thresholds,left,right,values,missing,0.)

def _arrays(roots,features,thresholds,left,right,values,missing,bias):
    return CompactTrees(np.asarray(roots,dtype=np.int32),np.asarray(features,dtype=np.int32),np.asarray(thresholds,dtype=np.float64),
        np.asarray(left,dtype=np.int32),np.asarray(right,dtype=np.int32),np.asarray(values,dtype=np.float64),np.asarray(missing,dtype=np.bool_),bias)
