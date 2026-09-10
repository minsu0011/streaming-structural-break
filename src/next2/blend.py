"""Fixed scalar blend of two honestly cross-fitted component predictions."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import joblib
import numpy as np
from src.next.late_blend import validate_next_component
from src.next2.io import register,registration_hash

WEIGHTS=(.9,.8,.67,.5)


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def combine(primary,complement,weight):
    if weight not in WEIGHTS:
        raise ValueError('Blend weight outside prospective NEXT2 grid')
    left=np.asarray(primary,dtype=np.float32).astype(np.float64)
    right=np.asarray(complement,dtype=np.float32).astype(np.float64)
    if left.shape!=right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Finite aligned component predictions required')
    return np.asarray(weight*left+(1.-weight)*right,dtype=np.float32)


class BlendState:
    def __init__(self,primary,complement,historical):
        self.primary=primary.make_state(historical)
        self.complement=complement.make_state(historical)

    def update(self,point):
        return self.primary.update(point),self.complement.update(point)


@dataclass
class FixedBlendBundle:
    primary:object
    complement:object
    weight:float
    implementation_sha256:str=''

    def validate(self):
        if self.weight not in WEIGHTS or self.implementation_sha256!=source_hash():
            raise RuntimeError('NEXT2 fixed blend changed')
        register(self.complement.extension)
        if self.complement.next2_registration_sha256!=registration_hash():
            raise RuntimeError('Complement registration changed')
        validate_next_component(self.primary)
        validate_next_component(self.complement)

    def make_state(self,historical):
        self.validate()
        return BlendState(self.primary,self.complement,historical)

    def predict_one(self,features):
        left,right=features
        return float(combine(self.primary.predict_one(left),self.complement.predict_one(right),self.weight))

    def save(self,path):
        self.implementation_sha256=source_hash()
        self.validate()
        joblib.dump(self,path)

    @classmethod
    def load(cls,path):
        model=joblib.load(path)
        model.validate()
        return model
