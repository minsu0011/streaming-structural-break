"""Separate serialized bundle for the two prospective AR-order refinements."""
from dataclasses import dataclass
from pathlib import Path
import joblib
from src.features.ar_order_input import AROrderCalibratedState, implementation_hash
from src.features.config import CONFIG
from src.features.streaming import feature_version
from src.features.historical_calibration import CalibrationPolicy
from src.models.learners import implementation_hash as predictor_hash


@dataclass
class AROrderCalibratedBundle:
    base_model: object
    policy: object = CalibrationPolicy(method='historical_median_mad')
    feature_config: object = CONFIG
    input_order: int = 4
    feature_hash: str = ''

    @property
    def name(self): return self.base_model.name + f'_ar_order{self.input_order}_calibrated'
    def make_state(self, historical): return AROrderCalibratedState(historical, config=self.feature_config, policy=self.policy, input_order=self.input_order)
    def predict(self, features): return self.base_model.predict(features)
    def predict_one(self, feature): return self.base_model.predict_one(feature)
    def save(self, path):
        if not self.feature_hash: self.feature_hash = implementation_hash(self.policy, self.feature_config, self.input_order)
        Path(path).parent.mkdir(parents=True, exist_ok=True); joblib.dump(self, path)
    @staticmethod
    def load(path):
        bundle = joblib.load(path)
        if bundle.feature_hash != implementation_hash(bundle.policy, bundle.feature_config, bundle.input_order): raise RuntimeError('AR-order input representation changed')
        if bundle.base_model.feature_hash != feature_version(bundle.feature_config): raise RuntimeError('Base feature representation changed')
        if bundle.base_model.predictor_hash != predictor_hash(): raise RuntimeError('Predictor implementation changed')
        return bundle
