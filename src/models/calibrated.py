"""Serialized contract for models trained on historical-null feature coordinates."""
from dataclasses import dataclass
from pathlib import Path
import joblib
from src.features.config import CONFIG
from src.features.streaming import feature_version
from src.features.historical_calibration import POLICY,HistoricalCalibratedState,implementation_hash as representation_hash
from src.models.learners import implementation_hash as predictor_hash

def representation_version(policy,config=CONFIG):
    if policy.method.endswith('_startup'):
        from src.features.startup_calibration import implementation_hash
        return implementation_hash(policy,config)
    return representation_hash(policy,config)

@dataclass
class CalibratedBundle:
    base_model: object
    policy: object = POLICY
    feature_config: object = CONFIG
    feature_hash: str = ''

    @property
    def name(self):return self.base_model.name+'_historical_calibrated'
    def make_state(self,historical):
        if self.policy.method.endswith('_startup'):
            from src.features.startup_calibration import StartupCalibratedState
            return StartupCalibratedState(historical,config=self.feature_config,policy=self.policy)
        return HistoricalCalibratedState(historical,config=self.feature_config,policy=self.policy)
    def predict(self,calibrated_features):return self.base_model.predict(calibrated_features)
    def predict_one(self,calibrated_feature):return self.base_model.predict_one(calibrated_feature)

    def save(self,path):
        if not self.feature_hash:self.feature_hash=representation_version(self.policy,self.feature_config)
        Path(path).parent.mkdir(parents=True,exist_ok=True);joblib.dump(self,path)

    @staticmethod
    def load(path):
        bundle=joblib.load(path)
        if bundle.feature_hash!=representation_version(bundle.policy,bundle.feature_config):raise RuntimeError('Calibration representation changed')
        if bundle.base_model.feature_hash!=feature_version(bundle.feature_config):raise RuntimeError('Base feature representation changed')
        if bundle.base_model.predictor_hash!=predictor_hash():raise RuntimeError('Base predictor implementation changed')
        return bundle
