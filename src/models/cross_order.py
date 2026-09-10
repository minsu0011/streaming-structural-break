"""Immutable 50:50 average; child probabilities are quantized before averaging."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,joblib
import numpy as np
from src.features.config import CONFIG
from src.features.streaming import feature_version
from src.features.cross_order import CrossOrderFeatureState,POLICY,implementation_hash as feature_hash
from src.features.ar_residual_input import implementation_hash as ar4_hash
from src.features.ar_order_input import implementation_hash as ar8_hash
from src.models.learners import fit_candidate,implementation_hash as predictor_hash
from src.models.ar_calibrated import ARCalibratedBundle
from src.models.ar_order_calibrated import AROrderCalibratedBundle

MODEL_NAME='M7_cross_order_equal'


def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+predictor_hash().encode()).hexdigest()


@dataclass
class CrossOrderBlendBundle:
    components:tuple
    feature_config:object=CONFIG
    feature_hash:str=''
    predictor_hash:str=''
    @property
    def name(self):return MODEL_NAME
    def make_state(self,historical):return CrossOrderFeatureState(historical,config=self.feature_config)
    def predict(self,features):
        x=np.asarray(features,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=114:raise ValueError('Both 57-column AR representations are required')
        left=self.components[0].predict(x[:,:57]).astype(np.float32).astype(np.float64)
        right=self.components[1].predict(x[:,57:]).astype(np.float32).astype(np.float64)
        return (.5*left+.5*right).astype(np.float32)
    def predict_one(self,features):
        if len(features)!=114:raise ValueError('Cross-order feature width differs')
        left=float(np.float32(self.components[0].predict_one(features[:57])))
        right=float(np.float32(self.components[1].predict_one(features[57:])))
        return float(np.float32(.5*left+.5*right))
    def validate(self):
        if len(self.components)!=2 or type(self.components[0]) is not ARCalibratedBundle or type(self.components[1]) is not AROrderCalibratedBundle:raise RuntimeError('Unexpected fixed-blend component types')
        if self.components[1].input_order!=8:raise RuntimeError('Fixed blend requires AR8 as its second component')
        for k,component in enumerate(self.components):
            expected=(ar4_hash(POLICY,self.feature_config) if k==0 else ar8_hash(POLICY,self.feature_config,8))
            if component.feature_config!=self.feature_config or component.policy!=POLICY or component.feature_hash!=expected:raise RuntimeError('Fixed-blend component representation changed')
            if component.base_model.feature_hash!=feature_version(self.feature_config) or component.base_model.predictor_hash!=predictor_hash():raise RuntimeError('Fixed-blend child predictor source changed')
            if component.base_model.name!=('M4_lightgbm','M3_histgb')[k]:raise RuntimeError('Fixed blend learner order changed')
        if self.feature_hash!=feature_hash(self.feature_config) or self.predictor_hash!=implementation_hash():raise RuntimeError('Fixed-blend implementation changed')
    def save(self,path):
        if not self.feature_hash:self.feature_hash=feature_hash(self.feature_config)
        if not self.predictor_hash:self.predictor_hash=implementation_hash()
        self.validate();Path(path).parent.mkdir(parents=True,exist_ok=True);joblib.dump(self,path)
    @staticmethod
    def load(path):
        model=joblib.load(path);model.validate();return model


def fit_cross_order(features,target,*,sample_weight=None,config=CONFIG,families='ABCD',params=None):
    if families!='ABCD' or params:raise ValueError('The frozen equal blend does not support family or parameter tuning')
    x=np.asarray(features,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=114:raise ValueError('Cross-order training feature width differs')
    components=[]
    for k,name in enumerate(('M4_lightgbm','M3_histgb')):
        matrix=x[:,k*57:(k+1)*57]
        excluded=tuple(np.flatnonzero(np.ptp(matrix,axis=0)<=1e-12))
        base=fit_candidate(name,matrix,target,families=families,sample_weight=sample_weight,params={},config=config,excluded_columns=excluded)
        child=ARCalibratedBundle(base,POLICY,config) if k==0 else AROrderCalibratedBundle(base,POLICY,config,input_order=8)
        child.feature_hash=ar4_hash(POLICY,config) if k==0 else ar8_hash(POLICY,config,8)
        components.append(child)
    return CrossOrderBlendBundle(tuple(components),config)
