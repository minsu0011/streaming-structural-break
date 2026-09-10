"""Prepared NEXT2 banks and exact float32 fixed-blend streaming inference.

Scientific model objects stay unchanged. All runtime specialization is checked
against their named feature contracts and recorded separately in descriptors.
"""
from pathlib import Path
import hashlib
import numpy as np
from numba import typeof, njit
from functools import lru_cache
from src.next.extensions import extension_module, names_and_groups
from src.next.fused_runtime import PreparedNextPredictor, _kernel, _resident_class, _arch_args, unique_array_bytes
from src.next.fast_variance_initialization import FastARCHFeatureState, OLD_CONFIG
from src.features.streaming import feature_names
from src.next2.io import MODULES, register
from src.next2.blend import FixedBlendBundle
from src.next2.pruned_runtime import PreparedCompactPredictor


def source_hash():
    paths=[Path(__file__),Path(__file__).with_name('pruned_runtime.py'),Path(__file__).with_name('dependence_runtime.py')]
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()


class PreparedBankPredictor(PreparedNextPredictor):
    def __init__(self,model,*,resident=True):
        if model.extension['kind'] in MODULES:
            register(model.extension)
        super().__init__(model,resident=resident)
        self.old_positions=np.asarray([j for j,n in enumerate(model.names) if n.startswith('arch8__')],dtype=np.int32)
        self.new_positions=np.asarray([j for j,n in enumerate(model.names) if not n.startswith('arch8__')],dtype=np.int32)
        old_names=['arch8__'+n for n in feature_names(OLD_CONFIG)]
        new_names=names_and_groups(model.extension)[0]
        self.old_columns=np.asarray([old_names.index(model.names[j]) for j in self.old_positions],dtype=np.int32)
        self.new_columns=np.asarray([new_names.index(model.names[j]) for j in self.new_positions],dtype=np.int32)
        self.runtime_source_sha256=source_hash()

    def new_state(self,historical):
        return BankDetector(self,historical)


class BankDetector:
    def __init__(self,prepared,historical):
        self.prepared=prepared
        module=extension_module(prepared.model.extension)
        self.new=module.make_state(historical,prepared.model.extension['settings'])
        self.old=FastARCHFeatureState(historical) if len(prepared.old_positions) else None
        self.output=np.empty(len(prepared.model.names),dtype=np.float32)
        function=getattr(self.new,'function',None) or module.step
        args=self.new.args
        old_args=_arch_args(self.old) if self.old is not None else ()
        self.call_args=(args,old_args,prepared.old_positions,prepared.old_columns,
            prepared.new_positions,prepared.new_columns,self.output,prepared.tree_args)
        self.kernel=_kernel(function,self.old is not None)
        self.resident=_resident_class(self.kernel,typeof(self.call_args))(self.call_args) if prepared.resident else None

    def predict_one(self,point):
        return float(self.resident.predict_one(float(point)) if self.resident is not None else self.kernel(float(point),self.call_args))

    @property
    def state_array_bytes(self):
        return unique_array_bytes(self.call_args[:-1])

    @property
    def immutable_tree_array_bytes(self):
        return unique_array_bytes(self.prepared.tree_args)


@lru_cache(maxsize=16)
def blend_kernel(left,right,weight):
    @njit(cache=False)
    def predict(point,args):
        # Each component is rounded exactly as its saved OOF vector was.
        a=np.float64(np.float32(left(point,args[0])))
        b=np.float64(np.float32(right(point,args[1])))
        return np.float32(weight*a+(1.-weight)*b)
    return predict


class PreparedBlendPredictor:
    def __init__(self,model,*,resident=True):
        model.validate()
        self.model=model
        self.resident=resident
        self.primary=prepare(model.primary,resident=False)
        self.complement=prepare(model.complement,resident=False)
        self.runtime_source_sha256=source_hash()

    def new_state(self,historical):
        return BlendDetector(self,historical)


class BlendDetector:
    def __init__(self,prepared,historical):
        self.prepared=prepared
        self.primary=prepared.primary.new_state(historical)
        self.complement=prepared.complement.new_state(historical)
        self.call_args=(self.primary.call_args,self.complement.call_args)
        self.kernel=blend_kernel(self.primary.kernel,self.complement.kernel,prepared.model.weight)
        self.resident=_resident_class(self.kernel,typeof(self.call_args))(self.call_args) if prepared.resident else None

    def predict_one(self,point):
        return float(self.resident.predict_one(float(point)) if self.resident is not None else self.kernel(float(point),self.call_args))

    @property
    def state_array_bytes(self):
        return unique_array_bytes((self.primary.call_args[:-1],self.complement.call_args[:-1]))

    @property
    def immutable_tree_array_bytes(self):
        return unique_array_bytes((self.primary.prepared.tree_args,self.complement.prepared.tree_args))


def prepare(model,*,resident=True):
    if isinstance(model,FixedBlendBundle):
        return PreparedBlendPredictor(model,resident=resident)
    if model.extension['kind'] in ('conditional_variance','serial_variance'):
        return PreparedCompactPredictor(model,resident=resident)
    from src.next2.dependence_runtime import NAMES,PreparedDependencePredictor
    if model.extension['kind']=='next2_complement' and tuple(model.names)==NAMES:
        return PreparedDependencePredictor(model,resident=resident)
    if model.extension['kind'] not in (*MODULES,'next2_scale_likelihood','next2_joint_dependence'):
        raise ValueError('No reviewed NEXT2 runtime for this model')
    return PreparedBankPredictor(model,resident=resident)
