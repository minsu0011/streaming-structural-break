"""Fixed late combination with explicit float32 component quantization."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import joblib
import numpy as np
from src.next.engine import feature_hash
from src.next.predictor import predictor_hash, arch_contract
from src.next.extensions import extension_hash
from src.next.registered_extensions import register_extension, registration_hash


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def combine_scores(primary, partner, primary_weight):
    if primary_weight not in (.5,.67,.33):
        raise ValueError('Weight is outside the predeclared late-combination grid')
    left = np.asarray(primary,dtype=np.float32).astype(np.float64)
    right = np.asarray(partner,dtype=np.float32).astype(np.float64)
    if left.shape != right.shape:
        raise ValueError('Component score shapes differ')
    return np.asarray(primary_weight*left+(1.-primary_weight)*right,dtype=np.float32)


def validate_next_component(model):
    if model.engine_hash != feature_hash(model.config) or model.implementation_hash != predictor_hash():
        raise RuntimeError('Late partner implementation changed')
    if model.old_arch_hash and model.old_arch_hash != arch_contract():
        raise RuntimeError('Late partner ARCH representation changed')
    if getattr(model,'registration',None):
        if model.registration_source_sha256 != registration_hash():
            raise RuntimeError('Late partner feature registration changed')
        register_extension(model.extension,model.registration)
    if getattr(model,'extension',None) and model.extension_implementation_hash != extension_hash(model.extension):
        raise RuntimeError('Late partner feature bank changed')


class LateBlendState:
    def __init__(self,primary,partner,historical):
        self.primary = primary.make_state(historical)
        self.partner = partner.make_state(historical)

    def update(self,point):
        return self.primary.update_and_get(point),self.partner.update(point)


@dataclass
class LateBlendBundle:
    primary: object
    partner: object
    primary_weight: float
    implementation_sha256: str = ''

    def validate(self):
        if self.primary_weight not in (.5,.67,.33) or self.implementation_sha256 != source_hash():
            raise RuntimeError('Fixed late-combination definition changed')
        self.primary.validate()
        validate_next_component(self.partner)

    def make_state(self,historical):
        return LateBlendState(self.primary,self.partner,historical)

    def predict(self,features):
        left,right = features
        return combine_scores(self.primary.predict(left),self.partner.predict(right),self.primary_weight)

    def predict_one(self,features):
        left,right = features
        return float(combine_scores(self.primary.predict_one(left),self.partner.predict_one(right),self.primary_weight))

    def save(self,path):
        if not self.implementation_sha256:
            self.implementation_sha256 = source_hash()
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(self,path)

    @classmethod
    def load(cls,path):
        model = joblib.load(path)
        model.validate()
        return model
