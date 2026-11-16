"""RP-only reads and explicit train/validation membership for NEXT8."""
from dataclasses import dataclass
import hashlib
import numpy as np
from src.next8.common import ROOT, BASE, read_json, digest, ids_hash, require_guard
from src.next5.data import ResearchArrays, admitted_npz


def source():
    guard = require_guard()
    directory = ROOT/'data/next5/research_5d54e316ff5bf25d'
    receipt = read_json(BASE/'PREFLIGHT_NEXT8.json')
    if digest(directory/'MANIFEST.json') != receipt['cache_manifest_sha256']:
        raise RuntimeError('RP manifest changed; no fallback raw loader permitted')
    return ResearchArrays(directory, guard)


@dataclass(frozen=True)
class FitScope:
    training_ids: frozenset
    validation_ids: frozenset
    outer_excluded_ids: frozenset
    name: str

    def __post_init__(self):
        if not self.training_ids or not self.validation_ids:
            raise ValueError('Nonempty training and validation groups required')
        if self.training_ids & (self.validation_ids | self.outer_excluded_ids):
            raise ValueError('Training/validation/outer exclusion overlap')

    def admit_training(self, ids):
        supplied = set(map(int, ids))
        if not supplied or not supplied <= self.training_ids:
            raise ValueError('Selection/fit requested outside its training scope')

    def admit_validation(self, ids):
        supplied = set(map(int, ids))
        if not supplied or not supplied <= self.validation_ids:
            raise ValueError('Prediction requested outside its validation scope')

    def provenance(self):
        return {'scope': self.name, 'training_ids_sha256': ids_hash(self.training_ids),
                'validation_ids_sha256': ids_hash(self.validation_ids),
                'outer_excluded_ids_sha256': ids_hash(self.outer_excluded_ids),
                'training_series': len(self.training_ids),
                'validation_series': len(self.validation_ids),
                'selection_training_only': True}


def outer_scope(fold, data=None):
    data = source() if data is None else data
    valid = frozenset(int(s) for s in data.ids if data.guard.split.assignment(int(s))[1] == fold)
    train = frozenset(map(int, data.ids))-valid
    data.guard.partition(train, valid)
    return FitScope(train, valid, valid, f'outer_{fold}')


def inner_scope(fold, inner, data=None):
    data = source() if data is None else data
    outer = outer_scope(fold, data)
    mapping = read_json(ROOT/'artifacts/next7/NEXT7_PROTOCOL_LOCK.json')['inner_mapping'][str(fold)]
    assert set(map(int, mapping)) == outer.training_ids
    valid = frozenset(int(s) for s, k in mapping.items() if int(k) == inner)
    train = outer.training_ids-valid
    data.guard.partition(train, valid)
    return FitScope(train, valid, outer.validation_ids, f'outer_{fold}_inner_{inner}')


def rows_for(metadata, ids):
    return np.flatnonzero(np.isin(metadata['dataset_id'], list(ids)))


def training_row_sample(metadata, scope, stride=8, max_time=None):
    """Predeclared time-only subsampling. No length, outcome or survival weights."""
    rows = rows_for(metadata, scope.training_ids)
    t = metadata['time_online'][rows]
    keep = t % stride == 0
    if max_time is not None:
        keep &= t <= max_time
    rows = rows[keep]
    scope.admit_training(np.unique(metadata['dataset_id'][rows]))
    return rows


def baseline_oof(fold, inner_training=False, data=None):
    """Provenance-verified saved predictions. No full-DEV deployment model."""
    data = source() if data is None else data
    audit = read_json(BASE/'BASELINE_PROVENANCE_NEXT8.json')
    if audit['status'] != 'PASS':
        raise RuntimeError('Baseline provenance gate')
    if inner_training:
        path = ROOT/f'artifacts/next7/nested_base/fold_{fold}/BL80_INNER_OOF.npz'
        assert digest(path) == audit['files_sha256'][path.relative_to(ROOT).as_posix()]
        return admitted_npz(path, data.guard, 'NEXT8 verified inner BL80 OOF')
    pieces = []
    for kind in ('J102', 'C205'):
        path = ROOT/f'artifacts/next5/references/fits/PRIMARY/full/{kind}/fold_{fold}/predictions.npz'
        assert digest(path) == audit['files_sha256'][path.relative_to(ROOT).as_posix()]
        pieces.append(admitted_npz(path, data.guard, 'NEXT8 verified outer reference OOF'))
    np.testing.assert_array_equal(pieces[0]['global_row'], pieces[1]['global_row'])
    return {**pieces[0], 'prediction': .8*pieces[0]['prediction']+.2*pieces[1]['prediction']}


def membership_numpy_hash(ids):
    return hashlib.sha256(np.asarray(sorted(ids), dtype=np.int32).tobytes()).hexdigest()
