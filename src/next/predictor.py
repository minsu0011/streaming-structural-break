"""Persisted NEXT predictor and its causal feature adapter."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import joblib
import numpy as np
from src.next.engine import EngineConfig, SequentialState, FEATURE_NAMES, feature_hash
from src.features.arch_input import ARCHCalibratedState, implementation_hash as arch_hash
from src.features.historical_calibration import CalibrationPolicy
from src.features.config import CONFIG
from src.features.streaming import feature_names as old_feature_names


def predictor_hash():
    compact = Path(__file__).resolve().parents[1]/'models/compact_trees.py'
    return hashlib.sha256(Path(__file__).read_bytes()+compact.read_bytes()).hexdigest()


def arch_contract():
    config = CONFIG.variant(normalization='median_mad', scales=(5,20,160))
    return arch_hash(CalibrationPolicy(method='historical_median_mad'), config)


class NextFeatureState:
    def __init__(self, historical, names, config):
        self.output = np.empty(len(names), dtype=np.float32)
        old_config = CONFIG.variant(normalization='median_mad', scales=(5,20,160))
        old_names = ['arch8__'+name for name in old_feature_names(old_config)]
        self.old_positions = np.asarray([j for j,name in enumerate(names) if name.startswith('arch8__')], dtype=np.int32)
        self.old_columns = np.asarray([old_names.index(names[j]) for j in self.old_positions], dtype=np.int32)
        self.new_positions = np.asarray([j for j,name in enumerate(names) if not name.startswith('arch8__')], dtype=np.int32)
        self.new_columns = np.asarray([FEATURE_NAMES.index(names[j]) for j in self.new_positions], dtype=np.int32)
        self.old = ARCHCalibratedState(historical, config=old_config) if len(self.old_positions) else None
        self.new = SequentialState(historical, config) if len(self.new_positions) else None

    def update(self, point):
        if self.old is not None:
            self.output[self.old_positions] = self.old.update_and_get(point)[self.old_columns]
        if self.new is not None:
            self.output[self.new_positions] = self.new.update(point)[self.new_columns]
        return self.output


@dataclass
class NextBundle:
    config: EngineConfig
    names: tuple
    columns: np.ndarray
    compact: object
    estimator: object
    learner: str
    engine_hash: str
    implementation_hash: str
    old_arch_hash: str

    def predict(self, features):
        matrix = np.asarray(features, dtype=np.float32)[:,self.columns]
        return self.compact.predict(matrix)

    def predict_one(self, feature):
        return self.compact.predict_one(np.asarray(feature, dtype=np.float32)[self.columns])

    def make_state(self, historical):
        return NextFeatureState(historical, self.names, self.config)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        model = joblib.load(path)
        if model.engine_hash != feature_hash(model.config) or model.implementation_hash != predictor_hash():
            raise RuntimeError('NEXT feature/predictor code changed')
        if model.old_arch_hash and model.old_arch_hash != arch_contract():
            raise RuntimeError('Frozen ARCH base contract changed')
        return model
