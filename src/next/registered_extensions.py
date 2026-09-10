"""Additive, pinned local feature-bank registration without changing old banks."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
from src.next.extensions import EXTENSION_MODULES,ExtendedBundle,extension_hash
from src.next.predictor import NextBundle


def registration_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def register_extension(extension,registration):
    kind,module=registration['kind'],registration['module']
    if kind!=extension['kind'] or not kind.isidentifier() or not module.isidentifier() or module.startswith('_'):
        raise ValueError('Invalid local feature-bank registration')
    path=Path(__file__).parent/(module+'.py')
    if not path.is_file():
        raise ValueError('Local feature-bank source is missing')
    target='src.next.'+module
    if kind in EXTENSION_MODULES and EXTENSION_MODULES[kind]!=target:
        raise RuntimeError('Existing feature-bank registrations cannot be overwritten')
    EXTENSION_MODULES.setdefault(kind,target)


@dataclass
class RegisteredExtendedBundle(ExtendedBundle):
    registration:dict=None
    registration_source_sha256:str=''

    @classmethod
    def wrap_registered(cls,model,extension,registration):
        register_extension(extension,registration)
        return cls(**vars(model),extension=extension,extension_implementation_hash=extension_hash(extension),
                   registration=registration,registration_source_sha256=registration_hash())

    def make_state(self,historical):
        register_extension(self.extension,self.registration)
        return super().make_state(historical)

    @classmethod
    def load(cls,path):
        model=NextBundle.load(path)
        if model.registration_source_sha256!=registration_hash():
            raise RuntimeError('Feature-bank registration implementation changed')
        register_extension(model.extension,model.registration)
        if model.extension_implementation_hash!=extension_hash(model.extension):
            raise RuntimeError('Registered feature-bank implementation changed')
        return model
