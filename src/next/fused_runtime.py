"""Prepared scalar inference for NEXT feature banks without changing learners.

One compiled call updates the bank, optional legacy ARCH features and trees.
The optional resident jitclass holds array arguments in memory only; it is never
serialized. Scientific model/source hashes remain those of the original bank.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import numpy as np
from numba import njit,typeof
from numba.experimental import jitclass
from src.models.compact_trees import tree_margin
from src.features.ar_residual_input import innovation_one
from src.features.arch_input import arch_one
from src.features.streaming import _update
from src.features.historical_calibration import _transform
from src.next.extensions import extension_module,names_and_groups
from src.next.late_blend import validate_next_component


def source_hash():
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__),root/'src/next/late_blend.py',root/'src/models/compact_trees.py']
    return hashlib.sha256(b''.join(path.read_bytes() for path in paths)).hexdigest()


@njit(cache=False)
def _arch_feature_step(point,args):
    ar_args,variance_args,state_args,calibration_args = args
    residual = innovation_one(point,*ar_args)
    standardized = arch_one(float(residual),*variance_args)
    _update(float(standardized),*state_args)
    _transform(state_args[6],*calibration_args)
    return calibration_args[-1]


@lru_cache(maxsize=32)
def _kernel(update_function,has_old):
    @njit(cache=False)
    def fused(point,args):
        extension_args,old_args,old_positions,old_columns,new_positions,new_columns,output,tree_args = args
        new = update_function(point,*extension_args)
        if has_old:
            old = _arch_feature_step(point,old_args)
            for j in range(len(old_positions)):
                output[old_positions[j]] = old[old_columns[j]]
        for j in range(len(new_positions)):
            output[new_positions[j]] = new[new_columns[j]]
        margin = tree_margin(output,*tree_args)
        return np.float32(1./(1.+np.exp(-margin)))
    return fused


@lru_cache(maxsize=32)
def _resident_class(kernel,argument_type):
    @jitclass([('args',argument_type)])
    class ResidentNext:
        def __init__(self,args):
            self.args = args
        def predict_one(self,point):
            return kernel(point,self.args)
    return ResidentNext


def _extension_call(state,kind):
    if kind=='ar_score':
        from src.next.score_evidence import score_step
        return score_step,(state.args,)
    if kind=='score_coordinates':
        from src.next.score_coordinates import coordinate_step
        return coordinate_step,(state.base.args,state.output)
    if kind=='markov_transition':
        from src.next.markov_evidence import transition_step
        return transition_step,(state.args,)
    if kind=='ar_mismatch':
        from src.next.mismatch_evidence import mismatch_step
        return mismatch_step,(state.filter_args,state.calibration,state.work,state.evidence_args)
    if kind=='conditional_variance':
        from src.next.variance_evidence import variance_step
        return variance_step,(state.filter_args,state.parameters,state.conditional,state.mode,state.calibration,state.work,state.evidence_args)
    if kind=='legacy_bank':
        from src.next.legacy_extension import legacy_step
        return legacy_step,state.args
    raise ValueError('This feature bank has no audited prepared runtime yet')


def _arch_args(state):
    ar_args = state.reference,state.coefficients,state.ring,state.counter
    variance_args = state.variance_parameters,state.previous_squared
    inner = state.inner
    s = inner.state
    state_args = tuple(getattr(s,key) for key in ('reference','baseline','thresholds','ring','ew','counters','output','parameters','alphas','lags'))
    calibration_args = inner.center,inner.scale,inner.mask,float(inner.policy.clip),inner.output
    return ar_args,variance_args,state_args,calibration_args


def unique_array_bytes(value):
    seen = set()
    def visit(item):
        if id(item) in seen:
            return 0
        seen.add(id(item))
        if isinstance(item,np.ndarray):
            return item.nbytes
        if isinstance(item,(tuple,list)):
            return sum(visit(v) for v in item)
        return 0
    return visit(value)


class PreparedNextPredictor:
    def __init__(self,model,*,resident=True):
        validate_next_component(model)
        if not getattr(model,'extension',None):
            raise ValueError('Prepared NEXT runtime requires an explicit feature bank')
        self.model = model
        self.resident = resident
        tree = model.compact
        args = list(tree.args())
        mapped = tree.features.copy()
        nodes = mapped>=0
        mapped[nodes] = model.columns[mapped[nodes]]
        args[1] = mapped
        self.tree_args = tuple(args)
        self.runtime_source_sha256 = source_hash()

    def new_state(self,historical):
        return FusedNextDetector(self,historical)


class FusedNextDetector:
    def __init__(self,prepared,historical):
        self.prepared = prepared
        self.original = prepared.model.make_state(historical)
        state = self.original
        function,args = _extension_call(state.new,prepared.model.extension['kind'])
        old_args = _arch_args(state.old) if state.old is not None else ()
        self.call_args = (args,old_args,state.old_positions,state.old_columns,state.new_positions,state.new_columns,state.output,prepared.tree_args)
        self.kernel = _kernel(function,state.old is not None)
        self.resident = _resident_class(self.kernel,typeof(self.call_args))(self.call_args) if prepared.resident else None

    def predict_one(self,point):
        return float(self.resident.predict_one(float(point)) if self.resident is not None else self.kernel(float(point),self.call_args))

    @property
    def state_array_bytes(self):
        # Include mutable online arrays; shared immutable trees are reported separately.
        return unique_array_bytes(self.call_args[:-1])

    @property
    def immutable_tree_array_bytes(self):
        return unique_array_bytes(self.prepared.tree_args)
