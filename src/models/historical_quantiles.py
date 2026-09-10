"""Explicit research artifact for the two frozen historical quantile scopes."""
from dataclasses import dataclass
from pathlib import Path
import joblib
from src.features.historical_quantiles import HistoricalQuantileARState,implementation_hash
from src.features.config import CONFIG
from src.features.streaming import feature_version
from src.features.historical_calibration import CalibrationPolicy
from src.models.learners import implementation_hash as predictor_hash


@dataclass
class HistoricalQuantileBundle:
    base_model:object
    quantile_families:str='C'
    policy:object=CalibrationPolicy(method='historical_median_mad')
    feature_config:object=CONFIG
    feature_hash:str=''
    @property
    def name(self):return self.base_model.name+'_historical_quantiles_'+self.quantile_families
    def make_state(self,historical):return HistoricalQuantileARState(historical,families=self.quantile_families,config=self.feature_config,policy=self.policy)
    def predict(self,features):return self.base_model.predict(features)
    def predict_one(self,features):return self.base_model.predict_one(features)
    def save(self,path):
        if not self.feature_hash:self.feature_hash=implementation_hash(self.quantile_families,self.policy,self.feature_config)
        Path(path).parent.mkdir(parents=True,exist_ok=True);joblib.dump(self,path)
    @staticmethod
    def load(path):
        model=joblib.load(path)
        if model.feature_hash!=implementation_hash(model.quantile_families,model.policy,model.feature_config):raise RuntimeError('Historical quantile representation changed')
        if model.base_model.feature_hash!=feature_version(model.feature_config) or model.base_model.predictor_hash!=predictor_hash():raise RuntimeError('Base feature or predictor source changed')
        return model
