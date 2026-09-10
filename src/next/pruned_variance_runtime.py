"""Exact runtime specialization for models using only variance channels 0/1.

The six-channel historical fit is unchanged. Online GLR/Bayes updates for four
unconsumed channels are omitted. This specialization refuses a model that reads
any omitted feature; its output contract is the trained scalar prediction.
"""
from pathlib import Path
import hashlib
import numpy as np
from numba import njit,typeof
from src.next.whitening import innovation_update
from src.next.variance_evidence import variance_channel_update
from src.next.channel_evidence import channel_step
from src.next.fused_runtime import PreparedNextPredictor,_kernel,_resident_class,_arch_args,unique_array_bytes,source_hash as fused_hash


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+fused_hash().encode()).hexdigest()


def supports_pruned_pair(model):
    if getattr(model,'extension',{}).get('kind')!='conditional_variance':
        return False
    names = [name for name in model.names if not name.startswith('arch8__')]
    return bool(names) and all(name.startswith(('conditional_variance__raw_energy_','conditional_variance__log_variance_forecast_')) for name in names)


@njit(cache=False)
def pruned_variance_step(point,filter_args,parameters,conditional,mode,calibration,work,evidence_args):
    residual = innovation_update(point,*filter_args)
    variance_channel_update(residual,parameters,conditional,mode,work)
    for j in range(2):
        work[j] = (work[j]-calibration[0,j])/calibration[1,j]
    # channel_step updates each channel independently; t advances once per point.
    return channel_step(work[:2],evidence_args)


class PreparedPrunedVariancePredictor(PreparedNextPredictor):
    def __init__(self,model,*,resident=True):
        if not supports_pruned_pair(model):
            raise ValueError('Pruning requires a model using only raw-energy and variance-forecast channels')
        super().__init__(model,resident=resident)
        self.runtime_source_sha256 = source_hash()

    def new_state(self,historical):
        return PrunedVarianceDetector(self,historical)


class PrunedVarianceDetector:
    def __init__(self,prepared,historical):
        self.prepared = prepared
        self.original = prepared.model.make_state(historical)
        state = self.original
        if np.any(state.new_columns>=26):
            raise RuntimeError('An omitted variance channel is required by this model')
        new = state.new
        extension_args = (new.filter_args,new.parameters,new.conditional,new.mode,new.calibration,new.work,new.evidence_args)
        old_args = _arch_args(state.old) if state.old is not None else ()
        self.call_args = (extension_args,old_args,state.old_positions,state.old_columns,state.new_positions,state.new_columns,state.output,prepared.tree_args)
        self.kernel = _kernel(pruned_variance_step,state.old is not None)
        self.resident = _resident_class(self.kernel,typeof(self.call_args))(self.call_args) if prepared.resident else None

    def predict_one(self,point):
        return float(self.resident.predict_one(float(point)) if self.resident is not None else self.kernel(float(point),self.call_args))

    @property
    def state_array_bytes(self):
        return unique_array_bytes(self.call_args[:-1])

    @property
    def immutable_tree_array_bytes(self):
        return unique_array_bytes(self.prepared.tree_args)
