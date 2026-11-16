"""Exact shared streaming features with a bounded packed forest representation."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
from numba import njit, typeof
from src.next3.packed_trees import pack_tree_args, packed_tree_margin
from src.next3.precomputed_shared_runtime import (
    feature_step as precomputed_feature_step, precompute_dependence,
)
from src.next2.precomputed_serial_runtime import precompute
from src.next2.shared_blend_runtime import (
    PreparedSharedBlendPredictor, SharedBlendDetector, shared_feature_step,
)
from src.next.fused_runtime import _resident_class, unique_array_bytes
from src.next.runtime_components import resident_parts
from src.next2.numeric_memory import inventory


VARIANTS = ('PACKED_SHARED', 'PACKED_PRECOMPUTED')


def source_hash():
    from src.next3.precomputed_shared_runtime import source_hash as feature_hash
    here = Path(__file__).parent
    return hashlib.sha256((here/'packed_trees.py').read_bytes() +
        Path(__file__).read_bytes() + feature_hash().encode()).hexdigest()


@njit(cache=False)
def packed_model_step(args):
    old_args, shared_args, old_positions, new_positions, new_columns, output, models = args
    primary_tree, complement_tree, weight = models
    margin_a = packed_tree_margin(output, *primary_tree)
    margin_b = packed_tree_margin(shared_args[-1][-1], *complement_tree)
    a = np.float64(np.float32(1./(1.+np.exp(-margin_a))))
    b = np.float64(np.float32(1./(1.+np.exp(-margin_b))))
    return np.float32(weight*a+(1.-weight)*b)


@njit(cache=False)
def packed_kernel(point, args):
    shared_feature_step(point, args)
    return packed_model_step(args)


@njit(cache=False)
def packed_precomputed_model_step(args):
    return packed_model_step(args[0])


@njit(cache=False)
def packed_precomputed_kernel(point, args):
    precomputed_feature_step(point, args)
    return packed_precomputed_model_step(args)


def readonly_copy(array):
    result = np.array(array, copy=True, order='C')
    result.setflags(write=False)
    return result


class PreparedPackedSharedPredictor:
    def __init__(self, model, *, variant='PACKED_SHARED', resident=True):
        if variant not in VARIANTS:
            raise ValueError('Unknown prospectively fixed packed variant')
        reference = PreparedSharedBlendPredictor(model, resident=False)
        original = reference.primary
        self.primary = SimpleNamespace(
            names=original.names,
            old_columns=readonly_copy(original.old_columns),
            old_positions=readonly_copy(original.old_positions),
            new_positions=readonly_copy(original.new_positions),
            new_columns=readonly_copy(original.new_columns),
            tree_args=pack_tree_args(original.tree_args),
        )
        self.complement = SimpleNamespace(tree_args=pack_tree_args(reference.complement.tree_args))
        # Keep only the scalar used by SharedBlendDetector. Neither this object
        # nor its streaming state retains the original forest/model arrays.
        self.model = SimpleNamespace(weight=float(model.weight))
        self.variant = variant
        self.resident = resident
        self.runtime_source_sha256 = source_hash()
        self.dependence_table = np.empty((0, 4), dtype=np.float64)
        self.dependence_logs = np.empty(0, dtype=np.float64)
        if variant == 'PACKED_PRECOMPUTED':
            ages = np.asarray([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int64)
            self.dependence_table, self.dependence_logs = precompute_dependence(
                ages, np.diff(np.r_[0, ages]).astype(float))
        self.dependence_table.setflags(write=False)
        self.dependence_logs.setflags(write=False)

    def new_state(self, historical):
        return PackedSharedDetector(self, historical)

    def numeric_arrays(self):
        return (vars(self.primary), vars(self.complement),
                self.dependence_table, self.dependence_logs)


class PackedSharedDetector:
    def __init__(self, prepared, historical):
        self.prepared = prepared
        adapter = SimpleNamespace(primary=prepared.primary, complement=prepared.complement,
                                  model=prepared.model, resident=False)
        self.base = SharedBlendDetector(adapter, historical)
        self.base_args = self.base.call_args
        if prepared.variant == 'PACKED_PRECOMPUTED':
            shared = self.base_args[1]
            serial_table, serial_logs = precompute(shared[5][0], shared[5][1], shared[6])
            self.call_args = (self.base_args, serial_table, serial_logs,
                             prepared.dependence_table, prepared.dependence_logs)
            self.private_and_selector_args = (self.base_args[:-1], *self.call_args[1:])
            self.kernel = packed_precomputed_kernel
        else:
            self.call_args = self.base_args
            self.private_and_selector_args = self.base_args[:-1]
            self.kernel = packed_kernel
        self.resident = _resident_class(self.kernel, typeof(self.call_args))(self.call_args) if prepared.resident else None

    def predict_one(self, point):
        return float(self.resident.predict_one(float(point)) if self.resident is not None
                     else self.kernel(float(point), self.call_args))

    @property
    def state_array_bytes(self):
        return unique_array_bytes(self.private_and_selector_args)

    @property
    def immutable_tree_array_bytes(self):
        return unique_array_bytes(self.base_args[-1])


class PackedSharedParts:
    def __init__(self, state):
        precomputed = state.prepared.variant == 'PACKED_PRECOMPUTED'
        feature = precomputed_feature_step if precomputed else shared_feature_step
        model = packed_precomputed_model_step if precomputed else packed_model_step
        self.resident = resident_parts(feature, model, typeof(state.call_args))(state.call_args)

    def feature_update(self, point):
        self.resident.feature_update(float(point))

    def model_predict(self):
        return float(self.resident.model_predict())


def accounting(state):
    if not isinstance(state, PackedSharedDetector):
        from src.next3.runtime_memory import accounting as original_accounting
        return original_accounting(state)
    state_alloc = inventory(state.private_and_selector_args)
    shared = inventory(state.prepared.numeric_arrays())
    private = set(state_alloc) - set(shared)
    overlap = set(state_alloc) & set(shared)
    return {
        'retained_state_numeric_bytes_including_shared_selectors': sum(v['bytes'] for v in state_alloc.values()),
        'private_per_series_numeric_bytes': sum(state_alloc[k]['bytes'] for k in private),
        'shared_selector_bytes_reachable_from_state': sum(state_alloc[k]['bytes'] for k in overlap),
        'model_and_prepared_shared_numeric_bytes': sum(v['bytes'] for v in shared.values()),
        'total_one_stream_plus_one_model_numeric_bytes': sum(v['bytes'] for v in {**shared, **state_alloc}.values()),
        'private_allocation_count': len(private), 'state_allocation_count': len(state_alloc),
    }, state_alloc, private
