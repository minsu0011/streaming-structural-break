"""Separate feature/tree calls for timing attribution, preserving predictions.

The additional Python dispatch is intentional. The sum of separately measured
parts need not equal the production runtime's single combined call.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import numpy as np
from numba import njit,typeof
from numba.experimental import jitclass
from src.features.ar_residual_input import innovation_one
from src.features.arch_input import arch_one
from src.features.streaming import _update
from src.features.historical_calibration import _transform
from src.models.compact_trees import tree_margin
from src.models.cross_order import CrossOrderBlendBundle
from src.models.arch_calibrated import ARCHCalibratedBundle
from src.streaming.prepared import prepare_model
from src.next.fused_runtime import PreparedNextPredictor,_extension_call,_arch_feature_step,source_hash as fused_hash
from src.next.pruned_variance_runtime import supports_pruned_pair,pruned_variance_step
from src.next.fast_variance_initialization import PreparedFastVariancePredictor,source_hash as fast_hash


def source_hash():
    root = Path(__file__).resolve().parents[2]
    files = sorted((root/'src/streaming').glob('*.py'))
    return hashlib.sha256(Path(__file__).read_bytes()+fused_hash().encode()+fast_hash().encode()+b''.join(p.read_bytes() for p in files)).hexdigest()


@lru_cache(maxsize=16)
def next_feature_kernel(function,has_old):
    @njit(cache=False)
    def update(point,args):
        extension_args,old_args,old_positions,old_columns,new_positions,new_columns,output,tree_args = args
        new = function(point,*extension_args)
        if has_old:
            old = _arch_feature_step(point,old_args)
            for j in range(len(old_positions)):
                output[old_positions[j]] = old[old_columns[j]]
        for j in range(len(new_positions)):
            output[new_positions[j]] = new[new_columns[j]]
    return update


@njit(cache=False)
def next_tree(args):
    margin = tree_margin(args[6],*args[7])
    return np.float32(1./(1.+np.exp(-margin)))


@njit(cache=False)
def ar_feature(point,args):
    ar,state,calibration,trees,is_xgboost = args
    residual = innovation_one(point,*ar)
    _update(float(residual),*state)
    _transform(state[6],calibration[0],calibration[1],calibration[2],calibration[3],calibration[4])


@njit(cache=False)
def ar_tree(args):
    margin = tree_margin(args[2][4],*args[3])
    return np.float32(1./(1.+np.exp(-margin)))


@njit(cache=False)
def cross_feature(point,args):
    ar_feature(point,args[0])
    ar_feature(point,args[1])


@njit(cache=False)
def cross_tree(args):
    left = np.float64(ar_tree(args[0]))
    right = np.float64(ar_tree(args[1]))
    return np.float32(.5*left+.5*right)


@njit(cache=False)
def arch_feature(point,args):
    ar,variance,state,calibration,trees = args
    residual = innovation_one(point,*ar)
    standardized = arch_one(np.float64(residual),*variance)
    _update(np.float64(standardized),*state)
    _transform(state[6],calibration[0],calibration[1],calibration[2],calibration[3],calibration[4])


@njit(cache=False)
def arch_tree(args):
    margin = tree_margin(args[3][4],*args[4])
    return np.float32(1./(1.+np.exp(-margin)))


@lru_cache(maxsize=16)
def resident_parts(feature_function,model_function,argument_type):
    @jitclass([('args',argument_type)])
    class ResidentParts:
        def __init__(self,args):
            self.args = args
        def feature_update(self,point):
            feature_function(point,self.args)
        def model_predict(self):
            return model_function(self.args)
    return ResidentParts


class PreparedComponents:
    def __init__(self,model):
        self.model = model
        self.is_next = hasattr(model,'extension')
        self.is_pair = self.is_next and supports_pruned_pair(model)
        if self.is_next:
            self.prepared = PreparedFastVariancePredictor(model) if self.is_pair else PreparedNextPredictor(model)
        else:
            self.prepared = prepare_model(model,fast_initialization=True)

    def new_state(self,historical):
        return ComponentState(self,historical)


class ComponentState:
    def __init__(self,prepared,historical):
        self.detector = prepared.prepared.new_state(historical)
        detector = self.detector
        if prepared.is_next:
            args = detector.call_args
            function = pruned_variance_step if prepared.is_pair else _extension_call(detector.original.new,prepared.model.extension['kind'])[0]
            feature,predict = next_feature_kernel(function,bool(len(detector.call_args[2]))),next_tree
        elif isinstance(prepared.model,CrossOrderBlendBundle):
            args,feature,predict = detector.call_args,cross_feature,cross_tree
        elif isinstance(prepared.model,ARCHCalibratedBundle):
            args = detector.ar_args,detector.variance_args,detector.state_args,detector.calibration_args,detector.prepared.tree_args
            feature,predict = arch_feature,arch_tree
        else:
            if detector.prepared.is_xgboost:
                raise ValueError('Separate XGBoost component timing is not registered')
            args = detector.ar_args,detector.state_args,detector.calibration_args,detector.prepared.tree_args,False
            feature,predict = ar_feature,ar_tree
        self.resident = resident_parts(feature,predict,typeof(args))(args)

    def feature_update(self,point):
        self.resident.feature_update(float(point))

    def model_predict(self):
        return float(self.resident.model_predict())

    def predict_one(self,point):
        self.feature_update(point)
        return self.model_predict()

    @property
    def state_array_bytes(self):
        return self.detector.state_array_bytes
