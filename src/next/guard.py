"""Fail-closed admission to NEXT-stage data, fitting, OOF and diagnostics.

This is an application guard, not an OS permission boundary. Parquet scanners
may decode mixed row-group bytes internally; only predicate-filtered DEV rows
are exposed. No seal feature, target, prediction or evaluation is authorized.
"""
from pathlib import Path
import json
import numbers
import numpy as np
import pyarrow.dataset as ds
from src.validation.splits import FrozenSplit
from src.utils.artifacts import sha256


class SealAccessError(RuntimeError):
    pass


class SealGuard:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
        self.split = FrozenSplit(self.root / 'artifacts/validation')
        self.lock_path = self.root / 'SEAL_PARTITION_500x4_LOCK.json'
        self.lock_sha = sha256(self.lock_path)
        self.lock = json.loads(self.lock_path.read_text(encoding='utf-8'))
        if self.lock['original_assignments_sha256'] != self.split.digest:
            raise SealAccessError('Seal partition source changed')
        seals = set()
        for name, entry in self.lock['partitions'].items():
            path = self.root / entry['ids_file']
            if entry['opened'] or sha256(path) != entry['ids_sha256']:
                raise SealAccessError(f'{name} state or membership changed')
            values = {int(v) for v in path.read_text(encoding='ascii').splitlines()}
            if len(values) != 500 or seals.intersection(values):
                raise SealAccessError('Invalid seal partition')
            seals.update(values)
        original = {i for i, (r, _) in self.split.mapping.items() if r == 'LOCAL_FINAL_SEAL'}
        if seals != original:
            raise SealAccessError('Original seal membership changed')
        self.seal_ids = frozenset(seals)
        self.dev_ids = frozenset(set(self.split.mapping) - seals)
        self.ledger_sha = sha256(self.root / 'REDUCED_TEST_USAGE_LEDGER.json')
        ledger = json.loads((self.root / 'REDUCED_TEST_USAGE_LEDGER.json').read_text(encoding='utf-8'))
        if ledger['scored_uses'] != 2 or ledger['max_scored_uses'] != 2:
            raise SealAccessError('Unexpected reduced quota state')

    def check_state(self):
        if sha256(self.lock_path) != self.lock_sha:
            raise SealAccessError('Seal lock modified during operation')
        if sha256(self.root / 'REDUCED_TEST_USAGE_LEDGER.json') != self.ledger_sha:
            raise SealAccessError('Reduced ledger modified during operation')

    def admit(self, ids=None, *, purpose='DEV access'):
        values = self.dev_ids if ids is None else list(ids)
        if any(isinstance(i, bool) or not isinstance(i, numbers.Integral) for i in values):
            raise SealAccessError(f'{purpose}: integral dataset IDs required')
        requested = frozenset(int(i) for i in values)
        if not requested or not requested.issubset(self.dev_ids):
            raise SealAccessError(f'{purpose}: only frozen DEV IDs are permitted')
        self.check_state()
        return sorted(requested)

    def partition(self, train_ids, validation_ids):
        train = set(self.admit(train_ids, purpose='training'))
        valid = set(self.admit(validation_ids, purpose='validation'))
        if train & valid:
            raise SealAccessError('Training and validation series overlap')
        if {self.split.groups[i] for i in train} & {self.split.groups[i] for i in valid}:
            raise SealAccessError('Training and validation historical groups overlap')
        return train, valid

    def scanner(self, filename, ids=None, *, columns=None, batch_size=262144):
        requested = self.admit(ids, purpose=f'read {filename}')
        if filename not in ('X_train.parquet', 'y_train_index.parquet'):
            raise SealAccessError('Only canonical DEV training inputs are admitted')
        path = self.root / 'data/raw' / filename
        return ds.dataset(path, format='parquet').scanner(
            columns=columns, filter=ds.field('id').isin(requested), batch_size=batch_size,
            use_threads=False)

    def labels(self, ids=None):
        requested = self.admit(ids, purpose='target access')
        table = self.scanner('y_train_index.parquet', requested).to_table().to_pandas()
        if 'id' in table.columns:
            table = table.set_index('id')
        if set(map(int, table.index)) != set(requested) or not table.index.is_unique:
            raise SealAccessError('DEV label coverage mismatch')
        return table

    def record_rows(self, ids, *, purpose):
        values = np.asarray(ids)
        self.admit(np.unique(values), purpose=purpose)
        return {'purpose': purpose, 'rows': int(len(values)),
                'series': int(len(np.unique(values))), 'seal_rows': 0,
                'split_sha256': self.split.digest, 'seal_lock_sha256': self.lock_sha}
