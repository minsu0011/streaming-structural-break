"""Explicit artifact for historical G context; never interpreted as original E/F/Q."""
from dataclasses import dataclass
from pathlib import Path
import joblib
from src.features.historical_context import HistoricalContextState,implementation_hash
from src.features.config import CONFIG
from src.features.streaming import feature_version
from src.features.historical_calibration import CalibrationPolicy
from src.models.learners import implementation_hash as predictor_hash


@dataclass
class ContextCalibratedBundle:
    base_model:object
    policy:object=CalibrationPolicy(method='historical_median_mad')
    feature_config:object=CONFIG
    feature_hash:str=''
    @property
    def name(self):return self.base_model.name+'_historical_context'
    def make_state(self,historical):return HistoricalContextState(historical,config=self.feature_config,policy=self.policy)
    def predict(self,features):return self.base_model.predict(features)
    def predict_one(self,feature):return self.base_model.predict_one(feature)
    def save(self,path):
        if not self.feature_hash:self.feature_hash=implementation_hash(self.policy,self.feature_config)
        Path(path).parent.mkdir(parents=True,exist_ok=True);joblib.dump(self,path)
    @staticmethod
    def load(path):
        model=joblib.load(path)
        if model.feature_hash!=implementation_hash(model.policy,model.feature_config):raise RuntimeError('Historical context representation changed')
        if model.base_model.feature_hash!=feature_version(model.feature_config) or model.base_model.predictor_hash!=predictor_hash():raise RuntimeError('Base feature or predictor source changed')
        return model
