"""Separate exact feature and tree dispatch for measurement, never production selection."""
from functools import lru_cache
from pathlib import Path
import hashlib
import numpy as np
from numba import njit,typeof
from src.next.runtime_components import next_feature_kernel,next_tree,resident_parts
from src.next.serial_variance_runtime import SerialVarianceDetector,pruned_serial_step
from src.next.fast_variance_initialization import FastVarianceDetector
from src.next.pruned_variance_runtime import pruned_variance_step
from src.next2.pruned_runtime import CompactDetector
from src.next2.dependence_runtime import DependenceDetector,step as dependence_step
from src.next2.used_feature_runtime import UsedDetector,used_arch_step
from src.next2.runtime import BlendDetector,BankDetector
from src.next.extensions import extension_module
from src.models.compact_trees import tree_margin


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@lru_cache(maxsize=16)
def used_feature_kernel(new_function):
    @njit(cache=False)
    def feature(point,args):
        old_args,new_args,old_positions,new_positions,new_columns,output,tree_args=args
        old=used_arch_step(point,old_args)
        new=new_function(point,*new_args)
        for j in range(len(old_positions)):
            output[old_positions[j]]=old[j]
        for j in range(len(new_positions)):
            output[new_positions[j]]=new[new_columns[j]]
    return feature


@njit(cache=False)
def used_tree(args):
    margin=tree_margin(args[5],*args[6])
    return np.float32(1./(1.+np.exp(-margin)))


@lru_cache(maxsize=16)
def blend_parts(left_feature,right_feature,left_tree,right_tree,weight):
    @njit(cache=False)
    def feature(point,args):
        left_feature(point,args[0])
        right_feature(point,args[1])
    @njit(cache=False)
    def model(args):
        left=np.float64(np.float32(left_tree(args[0])))
        right=np.float64(np.float32(right_tree(args[1])))
        return np.float32(weight*left+(1.-weight)*right)
    return feature,model


def functions(detector):
    if isinstance(detector,BlendDetector):
        lf,lt=functions(detector.primary);rf,rt=functions(detector.complement)
        return blend_parts(lf,rf,lt,rt,detector.prepared.model.weight)
    if isinstance(detector,UsedDetector):
        return used_feature_kernel(detector.new.function),used_tree
    if isinstance(detector,SerialVarianceDetector):
        function=pruned_serial_step
    elif isinstance(detector,FastVarianceDetector):
        function=pruned_variance_step
    elif isinstance(detector,CompactDetector):
        function=detector.new.function
    elif isinstance(detector,DependenceDetector):
        function=dependence_step
    elif isinstance(detector,BankDetector):
        module=extension_module(detector.prepared.model.extension)
        function=getattr(detector.new,'function',None) or module.step
    else:
        raise ValueError('Unreviewed component profiling runtime')
    return next_feature_kernel(function,len(detector.call_args[2])>0),next_tree


class Parts:
    def __init__(self,detector):
        self.detector=detector
        feature,model=functions(detector)
        self.resident=resident_parts(feature,model,typeof(detector.call_args))(detector.call_args)

    def feature_update(self,point):
        self.resident.feature_update(float(point))

    def model_predict(self):
        return float(self.resident.model_predict())
