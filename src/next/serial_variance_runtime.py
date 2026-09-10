"""Exact two-channel serial-evidence runtime; original H calibration retained."""
from pathlib import Path
import hashlib
import numpy as np
from numba import njit, typeof
from src.next.fused_runtime import PreparedNextPredictor, _kernel, _resident_class, _arch_args, unique_array_bytes, source_hash as fused_hash
from src.next.fast_variance_initialization import FastARCHFeatureState, OLD_CONFIG, source_hash as fast_h_hash
from src.next.serial_variance_evidence import SerialVarianceState, SerialVarianceConfig
from src.next.serial_channel_evidence import serial_channel_step, source_hash as serial_hash
from src.next.whitening import innovation_update
from src.next.variance_evidence import variance_channel_update
from src.next.extensions import names_and_groups
from src.features.streaming import feature_names


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+fused_hash().encode()+fast_h_hash().encode()+serial_hash().encode()+
        Path(__file__).with_name('serial_variance_evidence.py').read_bytes()).hexdigest()


@njit(cache=False)
def pruned_serial_step(point, filter_args, parameters, conditional, mode, calibration, work, evidence_args, age_variances):
    residual = innovation_update(point, *filter_args)
    variance_channel_update(residual, parameters, conditional, mode, work)
    for j in range(2):
        work[j] = (work[j]-calibration[0, j])/calibration[1, j]
    return serial_channel_step(work[:2], evidence_args, age_variances)


class PreparedSerialVariancePredictor(PreparedNextPredictor):
    def __init__(self, model, *, resident=True):
        if getattr(model, 'extension', {}).get('kind') != 'serial_variance':
            raise ValueError('Explicit serial-variance model required')
        super().__init__(model, resident=resident)
        old_names = ['arch8__'+name for name in feature_names(OLD_CONFIG)]
        new_names, _ = names_and_groups(model.extension)
        self.old_positions = np.asarray([j for j, name in enumerate(model.names) if name.startswith('arch8__')], dtype=np.int32)
        self.new_positions = np.asarray([j for j, name in enumerate(model.names) if not name.startswith('arch8__')], dtype=np.int32)
        self.old_columns = np.asarray([old_names.index(model.names[j]) for j in self.old_positions], dtype=np.int32)
        self.new_columns = np.asarray([new_names.index(model.names[j]) for j in self.new_positions], dtype=np.int32)
        if not len(self.new_positions) or np.any(self.new_columns >= 26):
            raise ValueError('Serial runtime requires only the raw-energy and forecast channels')
        self.runtime_source_sha256 = source_hash()

    def new_state(self, historical):
        return SerialVarianceDetector(self, historical)


class SerialVarianceDetector:
    def __init__(self, prepared, historical):
        self.prepared = prepared
        self.old = FastARCHFeatureState(historical) if len(prepared.old_positions) else None
        self.new = SerialVarianceState(historical, SerialVarianceConfig(**prepared.model.extension['settings']))
        self.output = np.empty(len(prepared.model.names), dtype=np.float32)
        new = self.new
        extension_args = (new.filter_args, new.parameters, new.conditional, new.mode, new.calibration,
            new.work, new.evidence_args, new.age_variances)
        old_args = _arch_args(self.old) if self.old is not None else ()
        self.call_args = (extension_args, old_args, prepared.old_positions, prepared.old_columns,
            prepared.new_positions, prepared.new_columns, self.output, prepared.tree_args)
        self.kernel = _kernel(pruned_serial_step, self.old is not None)
        self.resident = _resident_class(self.kernel, typeof(self.call_args))(self.call_args) if prepared.resident else None

    def predict_one(self, point):
        return float(self.resident.predict_one(float(point)) if self.resident is not None else self.kernel(float(point), self.call_args))

    @property
    def state_array_bytes(self):
        return unique_array_bytes(self.call_args[:-1])

    @property
    def immutable_tree_array_bytes(self):
        return unique_array_bytes(self.prepared.tree_args)
