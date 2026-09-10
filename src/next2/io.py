"""Versioned NEXT2 feature registration without edits to frozen NEXT sources."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
from src.next.extensions import EXTENSION_MODULES,ExtendedBundle,extension_hash,load_extended_design
from src.next.predictor import NextBundle
from src.next.candidate_io import candidate_design as old_design,fit_candidate as old_fit
from src.next.fitting import fit_binary
from src.next.engine import EngineConfig

MODULES={'next2_variance':'src.next2.variance_bank','next2_complement':'src.next2.complement_bank'}


def registration_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def register(extension):
    kind=extension['kind']
    if kind not in MODULES:
        raise ValueError('Unknown NEXT2 bank')
    target=MODULES[kind]
    if kind in EXTENSION_MODULES and EXTENSION_MODULES[kind]!=target:
        raise RuntimeError('Cannot replace an existing bank registration')
    EXTENSION_MODULES.setdefault(kind,target)


def is_next2(candidate):
    return candidate.get('extension',{}).get('kind') in MODULES


def validate(candidate):
    if is_next2(candidate):
        register(candidate['extension'])
        if candidate['next2_registration_sha256']!=registration_hash() or candidate['extension_frozen_hash']!=extension_hash(candidate['extension']):
            raise RuntimeError('NEXT2 candidate source changed after preregistration')


@dataclass
class Next2Bundle(ExtendedBundle):
    next2_registration_sha256:str=''

    @classmethod
    def wrap(cls,model,extension):
        register(extension)
        return cls(**vars(model),extension=extension,extension_implementation_hash=extension_hash(extension),
                   next2_registration_sha256=registration_hash())

    def make_state(self,historical):
        register(self.extension)
        return super().make_state(historical)

    @classmethod
    def load(cls,path):
        model=NextBundle.load(path)
        register(model.extension)
        if model.next2_registration_sha256!=registration_hash() or model.extension_implementation_hash!=extension_hash(model.extension):
            raise RuntimeError('NEXT2 model implementation changed')
        return model


def design(data,candidate,scope='full'):
    if not is_next2(candidate):
        return old_design(data,candidate,scope)
    validate(candidate)
    return load_extended_design(data,candidate,scope)


def fit(data,candidate,x,metadata,names,train,valid):
    if not is_next2(candidate):
        return old_fit(data,candidate,x,metadata,names,train,valid)
    validate(candidate)
    model,prediction,detail=fit_binary(x,metadata,train,valid,guard=data.guard,
        config=EngineConfig(**candidate['engine_config']),names=names,learner=candidate['learner'],
        weighting=candidate['weighting'],params=candidate['params'])
    return Next2Bundle.wrap(model,candidate['extension']),prediction,detail


def model_class(candidate):
    if is_next2(candidate):
        return Next2Bundle
    from src.next.candidate_io import model_class as old_class
    return old_class(candidate)
