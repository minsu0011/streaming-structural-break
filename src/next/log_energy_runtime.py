"""Exact log-energy bank plus the already verified fast legacy H initializer."""
from pathlib import Path
import hashlib
import numpy as np
from numba import typeof
from src.next.fused_runtime import PreparedNextPredictor, _kernel, _resident_class, _arch_args, unique_array_bytes, source_hash as fused_hash
from src.next.fast_variance_initialization import FastARCHFeatureState, OLD_CONFIG, source_hash as fast_h_hash
from src.next.log_energy_evidence import LogEnergyState, LogEnergyConfig, log_energy_step, implementation_hash
from src.next.extensions import names_and_groups
from src.features.streaming import feature_names


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+fused_hash().encode()+fast_h_hash().encode()+
        Path(__file__).with_name('log_energy_evidence.py').read_bytes()).hexdigest()


class PreparedLogEnergyPredictor(PreparedNextPredictor):
    def __init__(self, model, *, resident=True):
        if getattr(model, 'extension', {}).get('kind') != 'log_energy':
            raise ValueError('Explicit log-energy model required')
        super().__init__(model, resident=resident)
        old_names = ['arch8__'+name for name in feature_names(OLD_CONFIG)]
        new_names, _ = names_and_groups(model.extension)
        self.old_positions = np.asarray([j for j, name in enumerate(model.names) if name.startswith('arch8__')], dtype=np.int32)
        self.new_positions = np.asarray([j for j, name in enumerate(model.names) if not name.startswith('arch8__')], dtype=np.int32)
        self.old_columns = np.asarray([old_names.index(model.names[j]) for j in self.old_positions], dtype=np.int32)
        self.new_columns = np.asarray([new_names.index(model.names[j]) for j in self.new_positions], dtype=np.int32)
        if not len(self.new_positions) or np.any(self.new_columns >= 26):
            raise ValueError('Log-energy adapter cannot accept a different channel bank')
        self.runtime_source_sha256 = source_hash()

    def new_state(self, historical):
        return LogEnergyDetector(self, historical)


class LogEnergyDetector:
    def __init__(self, prepared, historical):
        self.prepared = prepared
        self.old = FastARCHFeatureState(historical) if len(prepared.old_positions) else None
        # Use the exact original H fit, historical channels and calibration.
        self.new = LogEnergyState(historical, LogEnergyConfig(**prepared.model.extension['settings']))
        self.output = np.empty(len(prepared.model.names), dtype=np.float32)
        old_args = _arch_args(self.old) if self.old is not None else ()
        self.call_args = (self.new.args, old_args, prepared.old_positions, prepared.old_columns,
            prepared.new_positions, prepared.new_columns, self.output, prepared.tree_args)
        self.kernel = _kernel(log_energy_step, self.old is not None)
        self.resident = _resident_class(self.kernel, typeof(self.call_args))(self.call_args) if prepared.resident else None

    def predict_one(self, point):
        return float(self.resident.predict_one(float(point)) if self.resident is not None else self.kernel(float(point), self.call_args))

    @property
    def state_array_bytes(self):
        return unique_array_bytes(self.call_args[:-1])

    @property
    def immutable_tree_array_bytes(self):
        return unique_array_bytes(self.prepared.tree_args)
