"""Scoped selective label projection, admitted only after target-free freeze.

The trusted RP ZIP_STORED NPY member is mapped in place. No eager NPZ target
decode, tau member read, raw-data load, or full-cohort target copy is needed.
"""
from pathlib import Path
import struct
import zipfile
import numpy as np
from src.next9.common import BASE, ROOT, read_json, digest, atomic, utc_now, ids_hash


def map_stored_vector(path, member='target.npy'):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1:
            raise ValueError('Selective mapping requires an unencrypted stored member')
        offset = info.header_offset
    with path.open('rb') as f:
        f.seek(offset)
        header = f.read(30)
        signature, *_, name_len, extra_len = struct.unpack('<IHHHHHIIIHH', header)
        if signature != 0x04034b50:
            raise ValueError('Bad ZIP local header')
        f.seek(name_len+extra_len, 1)
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError('Unsupported NPY header version')
        data_offset = f.tell()
    if len(shape) != 1 or fortran or dtype.hasobject:
        raise ValueError('A simple stored vector is required')
    if data_offset+shape[0]*dtype.itemsize > path.stat().st_size:
        raise ValueError('Stored vector exceeds archive')
    return np.memmap(path, mode='r', dtype=dtype, offset=data_offset, shape=shape)


def validate_rows(bank, rows, scope, role, *, early=False, prediction_receipt=None):
    rows = np.asarray(rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or np.any(rows < 0) or np.any(rows >= len(bank.row_ids)):
        raise ValueError('Valid global row indices required')
    ids = bank.row_ids[rows]
    if role == 'TRAIN':
        scope.admit_training(ids)
    elif role == 'INTERNAL_VALIDATION':
        scope.admit_validation(ids)
        if set(map(int, ids)) & scope.outer_excluded_ids:
            raise ValueError('Outer labels cannot enter internal selection')
    elif role == 'FROZEN_OUTER_EVALUATION':
        scope.admit_validation(ids)
        if prediction_receipt is None or prediction_receipt.get('status') != 'FROZEN':
            raise ValueError('Outer predictions must be globally frozen before label projection')
        if prediction_receipt.get('scope') != scope.provenance():
            raise ValueError('Prediction scope mismatch')
    else:
        raise ValueError('Unregistered label purpose')
    if early and role != 'FROZEN_OUTER_EVALUATION' and np.any(bank.times[rows] > 199):
        raise ValueError('Strict early selection cannot read late labels')
    return rows


class ScopedLabels:
    def __init__(self, bank):
        gate = read_json(BASE / 'LABEL_ACCESS_GATE_NEXT9.json')
        if gate['status'] != 'OPEN' or gate['candidate_freeze_sha256'] != digest(BASE / 'TARGET_FREE_CANDIDATES_FROZEN_NEXT9.json'):
            raise ValueError('All target-free candidates must be frozen before labels')
        self.bank = bank
        pre = read_json(BASE / 'PREFLIGHT_NEXT9.json')
        directory = ROOT / pre['RP_cache']
        manifest = read_json(directory / 'MANIFEST.json')
        if digest(directory / 'rows.npz') != manifest['files_sha256']['rows.npz']:
            raise ValueError('RP label source hash changed')
        self._vector = map_stored_vector(directory / 'rows.npz')
        if self._vector.shape != bank.row_ids.shape:
            raise ValueError('RP row alignment mismatch')
        self.accesses = []

    def get(self, rows, scope, role, *, early=False, prediction_receipt=None):
        rows = validate_rows(self.bank, rows, scope, role, early=early, prediction_receipt=prediction_receipt)
        result = np.asarray(self._vector[rows], dtype=np.uint8)
        if not np.isin(result, [0, 1]).all():
            raise ValueError('Nonbinary break label')
        self.accesses.append(dict(scope=scope.name, role=role, rows=len(rows), early=early,
            groups_sha256=ids_hash(np.unique(self.bank.row_ids[rows])),
            maximum_observed_time=int(self.bank.times[rows].max()) if len(rows) else None,
            target_member_only=True, tau_read=False))
        return result


def open_label_gate():
    frozen = read_json(BASE / 'TARGET_FREE_CANDIDATES_FROZEN_NEXT9.json')
    if frozen['status'] != 'FROZEN' or frozen['procedure_count'] != 20:
        raise ValueError('All twenty procedures must finish target-free selection')
    for record in frozen['candidates']:
        if digest(BASE / record['path']) != record['sha256']:
            raise ValueError('Frozen candidates changed')
    atomic(BASE / 'LABEL_ACCESS_GATE_NEXT9.json', dict(status='OPEN', timestamp=utc_now(),
        candidate_freeze_sha256=digest(BASE / 'TARGET_FREE_CANDIDATES_FROZEN_NEXT9.json'),
        orientation_details_sha256=digest(BASE / 'ORIENTATION_DETAILS_PROTOCOL_NEXT9.json'),
        method='Selective mapping of RP target.npy; scope and time admission before indexed access',
        outer_quality_released=False, tau_access=0))
