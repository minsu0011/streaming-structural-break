"""Fixed two-learner average sharing exactly the same causal feature bank."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import joblib
import numpy as np
from src.next.late_blend import combine_scores,validate_next_component,source_hash as combination_hash
from src.next.extensions import ExtendedFeatureState


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+combination_hash().encode()).hexdigest()


@dataclass
class SameBankBlendBundle:
    left:object
    right:object
    left_weight:float
    implementation_sha256:str=''

    @property
    def names(self):
        return self.left.names

    @property
    def right_positions(self):
        return np.asarray([self.names.index(name) for name in self.right.names],dtype=np.int32)

    def validate(self):
        if self.left_weight not in (.5,.67,.33) or self.implementation_sha256!=source_hash():
            raise RuntimeError('Fixed same-bank blend implementation changed')
        for model in (self.left,self.right):
            validate_next_component(model)
        if self.left.extension!=self.right.extension or self.left.config!=self.right.config or self.left.old_arch_hash!=self.right.old_arch_hash:
            raise RuntimeError('Shared feature bank requires identical H and sequential feature recipes')
        if not set(self.right.names).issubset(self.left.names):
            raise RuntimeError('The left feature vector must contain every right feature')

    def make_state(self,historical):
        self.validate()
        return ExtendedFeatureState(historical,self.names,self.left.extension)

    def predict(self,features):
        matrix = np.asarray(features,dtype=np.float32)
        return combine_scores(self.left.predict(matrix),self.right.predict(matrix[:,self.right_positions]),self.left_weight)

    def predict_one(self,features):
        return float(combine_scores(self.left.predict_one(features),self.right.predict_one(np.asarray(features)[self.right_positions]),self.left_weight))

    def save(self,path):
        if not self.implementation_sha256:
            self.implementation_sha256 = source_hash()
        self.validate()
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(self,path)

    @classmethod
    def load(cls,path):
        model = joblib.load(path)
        model.validate()
        return model

    @classmethod
    def create(cls,left,right,weight):
        result = cls(left,right,weight,source_hash())
        result.validate()
        return result
