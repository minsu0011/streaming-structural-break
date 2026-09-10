"""Hash-checked bundle for the historical AR residual input representation."""
from dataclasses import dataclass
from pathlib import Path
import joblib
from src.features.ar_residual_input import ARResidualCalibratedState, implementation_hash
from src.features.config import CONFIG
from src.features.streaming import feature_version
from src.features.historical_calibration import CalibrationPolicy
from src.models.learners import implementation_hash as predictor_hash


@dataclass
class ARCalibratedBundle:
    base_model: object
    policy: object = CalibrationPolicy(method='historical_median_mad')
    feature_config: object = CONFIG
    feature_hash: str = ''

    @property
    def name(self): return self.base_model.name + '_ar_residual_calibrated'
    def make_state(self, historical): return ARResidualCalibratedState(historical, config=self.feature_config, policy=self.policy)
    def predict(self, features): return self.base_model.predict(features)
    def predict_one(self, feature): return self.base_model.predict_one(feature)
    def save(self, path):
        if not self.feature_hash: self.feature_hash = implementation_hash(self.policy, self.feature_config)
        Path(path).parent.mkdir(parents=True, exist_ok=True); joblib.dump(self, path)
    @staticmethod
    def load(path):
        bundle = joblib.load(path)
        if bundle.feature_hash != implementation_hash(bundle.policy, bundle.feature_config): raise RuntimeError('AR input representation changed')
        if bundle.base_model.feature_hash != feature_version(bundle.feature_config): raise RuntimeError('Base feature representation changed')
        if bundle.base_model.predictor_hash != predictor_hash(): raise RuntimeError('Predictor implementation changed')
        return bundle
