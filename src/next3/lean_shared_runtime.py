"""Frozen packed shared arithmetic without importing training bundle classes."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
from numba import typeof
from src.next3.lean_model import tree_args, feature_contract, validate_record
from src.next3.lean_numeric import (
    OLD_CONFIG, feature_names, UsedOldState, SharedChannels, packed_kernel,
    _resident_class, unique_array_bytes,
)
from src.next3.packed_trees import pack_tree_args
from src.next2.numeric_memory import inventory


def source_hash():
    root = Path(__file__).resolve().parents[2]
    paths = ('src/next3/lean_model.py', 'src/next3/lean_numeric.py', 'src/next3/lean_shared_runtime.py',
        'src/next3/packed_trees.py', 'src/next2/numeric_memory.py',
        'configs/next3/LEAN_FEATURE_CONTRACT.json', 'configs/next3/LEAN_NUMERIC_BUILD.json',
        'configs/next3/LEAN_RUNTIME_INPUTS.json')
    return hashlib.sha256(b''.join((root/path).read_bytes() for path in paths)).hexdigest()


def readonly_copy(array):
    result = np.array(array, copy=True, order='C'); result.setflags(write=False)
    return result


class PreparedLeanSharedPredictor:
    def __init__(self, record, *, resident=True, contract=None):
        contract = feature_contract() if contract is None else contract
        validate_record(record, contract)
        primary = record['primary']; args = tree_args(primary)
        used = np.unique(args[1][args[1] >= 0]); names = tuple(primary['names'][k] for k in used)
        old_positions = np.asarray([k for k, name in enumerate(names) if name.startswith('arch8__')], dtype=np.int32)
        new_positions = np.asarray([k for k, name in enumerate(names) if not name.startswith('arch8__')], dtype=np.int32)
        old_names = ['arch8__'+name for name in feature_names(OLD_CONFIG)]
        full_new = [name for name in primary['names'] if not name.startswith('arch8__')]
        if old_names[:51] != primary['names'][:51] or len(full_new) != 26: raise ValueError('Frozen 51+26 feature ordering required')
        old_columns = np.asarray([old_names.index(names[k]) for k in old_positions], dtype=np.int64)
        new_columns = np.asarray([full_new.index(names[k]) for k in new_positions], dtype=np.int32)
        remap = np.full(len(primary['names']), -1, dtype=np.int32); remap[used] = np.arange(len(used), dtype=np.int32)
        mapped = args[1].copy(); live = mapped >= 0; mapped[live] = remap[mapped[live]]
        args = (args[0], mapped, *args[2:])
        self.primary = SimpleNamespace(names=names, old_columns=readonly_copy(old_columns),
            old_positions=readonly_copy(old_positions), new_positions=readonly_copy(new_positions),
            new_columns=readonly_copy(new_columns), tree_args=pack_tree_args(args))
        self.complement = SimpleNamespace(tree_args=pack_tree_args(tree_args(record['complement'])))
        self.model = SimpleNamespace(weight=float(record['weight']))
        self.resident = resident; self.variant = 'LEAN_PACKED_SHARED'
        self.runtime_source_sha256 = source_hash()

    def new_state(self, historical):
        return LeanSharedDetector(self, historical)

    def numeric_arrays(self):
        return vars(self.primary), vars(self.complement)


class LeanSharedDetector:
    def __init__(self, prepared, historical):
        self.prepared = prepared
        self.old = UsedOldState(historical, prepared.primary.old_columns)
        self.shared = SharedChannels(historical)
        self.output = np.empty(len(prepared.primary.names), dtype=np.float32)
        self.call_args = (self.old.call_args, self.shared.call_args, prepared.primary.old_positions,
            prepared.primary.new_positions, prepared.primary.new_columns, self.output,
            (prepared.primary.tree_args, prepared.complement.tree_args, prepared.model.weight))
        self.base_args = self.call_args
        self.private_and_selector_args = self.call_args[:-1]
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


def accounting(state):
    state_alloc = inventory(state.private_and_selector_args); shared = inventory(state.prepared.numeric_arrays())
    private = set(state_alloc)-set(shared); overlap = set(state_alloc)&set(shared)
    return {
        'retained_state_numeric_bytes_including_shared_selectors': sum(v['bytes'] for v in state_alloc.values()),
        'private_per_series_numeric_bytes': sum(state_alloc[k]['bytes'] for k in private),
        'shared_selector_bytes_reachable_from_state': sum(state_alloc[k]['bytes'] for k in overlap),
        'model_and_prepared_shared_numeric_bytes': sum(v['bytes'] for v in shared.values()),
        'total_one_stream_plus_one_model_numeric_bytes': sum(v['bytes'] for v in {**shared, **state_alloc}.values()),
        'private_allocation_count': len(private), 'state_allocation_count': len(state_alloc),
    }, state_alloc, private
