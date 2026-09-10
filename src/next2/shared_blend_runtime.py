"""Exact fixed-blend runtime sharing only identical new-branch AR/ARCH work."""
from pathlib import Path
import hashlib
import numpy as np
from numba import njit, typeof
from src.next.whitening import fit_historical, innovation_update
from src.next.fast_variance_initialization import fast_variance_parameters
from src.next.channel_evidence import make_channel_args
from src.next.serial_channel_evidence import historical_correlations, sum_variances, serial_channel_step
from src.next.fused_runtime import _resident_class, unique_array_bytes
from src.models.compact_trees import tree_margin
from src.next2.pruned_runtime import historical_pair, pair_channel_update
from src.next2.complement_bank import historical_raw, raw_complement
from src.next2.dependence_runtime import historical_z, evidence, PreparedDependencePredictor
from src.next2.used_feature_runtime import PreparedUsedPredictor, UsedOldState, used_arch_step
from src.next2.blend import FixedBlendBundle
from src.next.runtime_components import resident_parts


def source_hash():
    here = Path(__file__)
    names = ('shared_blend_runtime.py', 'used_feature_runtime.py', 'pruned_runtime.py',
             'dependence_runtime.py', 'complement_bank.py')
    return hashlib.sha256(b''.join((here.parent/name).read_bytes() for name in names)).hexdigest()


@njit(cache=False)
def shared_step(point, filter_args, parameters, conditional, pair_calibration, pair_work,
                pair_evidence, age_variances, ring, counter, center, calibration, work, empty, dependence_evidence):
    residual = innovation_update(point, *filter_args)
    # normalized_one's pre-update variance and centered residual, without a
    # second mutation of the recurrence shared by both branches.
    variance = max(parameters[2]+parameters[3]*conditional[0]+parameters[4]*conditional[1], parameters[6])
    centered = residual-parameters[0]
    z = min(max(centered/np.sqrt(variance), -4.), 4.)
    pair_channel_update(residual, parameters, conditional, 3, pair_work)
    for j in range(2):
        pair_work[j] = (pair_work[j]-pair_calibration[0, j])/pair_calibration[1, j]
    left = serial_channel_step(pair_work, pair_evidence, age_variances)
    raw_complement(z, empty, ring, counter, center, 3, work)
    for j in range(4):
        work[j] = (work[j]-calibration[0, j])/calibration[1, j]
    right = evidence(work, *dependence_evidence)
    return left, right


class SharedChannels:
    def __init__(self, historical):
        fit = fit_historical(historical, 8)
        residuals = fit['historical_innovations']
        parameters, conditional = fast_variance_parameters(residuals, 'arch1')
        pair = historical_pair(residuals, parameters, 3)
        pair = pair[min(160, len(pair)//4):]
        pair_calibration = np.stack([pair.mean(axis=0), np.maximum(pair.std(axis=0), 1e-6)])
        pair_evidence = make_channel_args(2, 512)
        correlations = historical_correlations((pair-pair_calibration[0])/pair_calibration[1])
        age_variances = sum_variances(pair_evidence[0], correlations)
        # Do not initialize C205's H replay by inverting omega/(1-alpha).
        variance = max(float(np.minimum((residuals-parameters[0])**2, parameters[5]).mean()), 1e-12)
        z, replayed = historical_z(residuals, parameters, variance)
        np.testing.assert_array_equal(replayed, conditional)
        center = np.asarray([z.mean(), (z*z).mean(), np.abs(z).mean()])
        empty = np.empty(0)
        raw, ring, counter = historical_raw(z, empty, center, 3, 4)
        raw = raw[min(160, len(raw)//4):]
        calibration = np.stack([raw.mean(axis=0), np.maximum(raw.std(axis=0), 1e-6)])
        ages = np.asarray([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int64)
        dependence_evidence = (ages, np.diff(np.r_[0, ages]).astype(float),
            np.zeros((4, 2)), np.zeros((4, 2)), np.zeros(4), np.zeros((129, 4)),
            np.zeros(1, dtype=np.int64), np.empty(24, dtype=np.float32))
        filter_args = tuple(fit[k] for k in ('reference', 'coefficients', 'ring', 'counter'))
        self.call_args = (filter_args, parameters, conditional, pair_calibration, np.empty(2),
            pair_evidence, age_variances, ring, counter, center, calibration, np.empty(4), empty, dependence_evidence)
        self.pair_output = pair_evidence[-1]
        self.dependence_output = dependence_evidence[-1]

    def update(self, point):
        return shared_step(float(point), *self.call_args)


@njit(cache=False)
def shared_feature_step(point, args):
    old_args, shared_args, old_positions, new_positions, new_columns, output, models = args
    old = used_arch_step(point, old_args)
    new, dependence = shared_step(point, *shared_args)
    for j in range(len(old_positions)):
        output[old_positions[j]] = old[j]
    for j in range(len(new_positions)):
        output[new_positions[j]] = new[new_columns[j]]
    return output


@njit(cache=False)
def shared_model_step(args):
    old_args, shared_args, old_positions, new_positions, new_columns, output, models = args
    dependence = shared_args[-1][-1]
    primary_tree, complement_tree, weight = models
    margin_a = tree_margin(output, *primary_tree)
    margin_b = tree_margin(dependence, *complement_tree)
    a = np.float64(np.float32(1./(1.+np.exp(-margin_a))))
    b = np.float64(np.float32(1./(1.+np.exp(-margin_b))))
    return np.float32(weight*a+(1.-weight)*b)


@njit(cache=False)
def shared_kernel(point, args):
    shared_feature_step(point, args)
    return shared_model_step(args)


class SharedParts:
    def __init__(self, state):
        self.resident = resident_parts(shared_feature_step, shared_model_step, typeof(state.call_args))(state.call_args)

    def feature_update(self, point):
        return self.resident.feature_update(float(point))

    def model_predict(self):
        return float(self.resident.model_predict())


class PreparedSharedBlendPredictor:
    def __init__(self, model, *, resident=True):
        if not isinstance(model, FixedBlendBundle):
            raise ValueError('Frozen fixed blend required')
        model.validate()
        expected = {'kind': 'serial_variance', 'settings': {'order': 8, 'normalization': 'arch1', 'max_age': 512}}
        if model.primary.extension != expected:
            raise ValueError('Exact J102 serial ARCH1 contract required')
        self.model = model
        self.resident = resident
        self.primary = PreparedUsedPredictor(model.primary, resident=False)
        self.complement = PreparedDependencePredictor(model.complement, resident=False)
        self.runtime_source_sha256 = source_hash()

    def new_state(self, historical):
        return SharedBlendDetector(self, historical)


class SharedBlendDetector:
    def __init__(self, prepared, historical):
        self.prepared = prepared
        self.old = UsedOldState(historical, prepared.primary.old_columns)
        self.shared = SharedChannels(historical)
        self.output = np.empty(len(prepared.primary.names), dtype=np.float32)
        self.call_args = (self.old.call_args, self.shared.call_args,
            prepared.primary.old_positions, prepared.primary.new_positions, prepared.primary.new_columns,
            self.output, (prepared.primary.tree_args, prepared.complement.tree_args, prepared.model.weight))
        self.kernel = shared_kernel
        self.resident = _resident_class(self.kernel, typeof(self.call_args))(self.call_args) if prepared.resident else None

    def predict_one(self, point):
        return float(self.resident.predict_one(float(point)) if self.resident is not None else self.kernel(float(point), self.call_args))

    @property
    def state_array_bytes(self):
        return unique_array_bytes(self.call_args[:-1])

    @property
    def immutable_tree_array_bytes(self):
        return unique_array_bytes((self.prepared.primary.tree_args, self.prepared.complement.tree_args))
