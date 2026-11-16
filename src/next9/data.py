"""Label-free feature bank and preserved original nested grouped partitions."""
from dataclasses import dataclass
import hashlib
import numpy as np
from src.next9.common import ROOT, BASE, read_json, digest, ids_hash, require_guard
from src.next9.boundary import target_free_npz
from src.next9.states import describe_history


@dataclass(frozen=True)
class FitScope:
    training_ids: frozenset
    validation_ids: frozenset
    outer_excluded_ids: frozenset
    name: str

    def __post_init__(self):
        if not self.training_ids or not self.validation_ids or self.training_ids & (self.validation_ids | self.outer_excluded_ids):
            raise ValueError('Nonempty disjoint original group scope required')

    def admit_training(self, ids):
        if not set(map(int, ids)) or not set(map(int, ids)) <= self.training_ids:
            raise ValueError('Training outside admitted groups')

    def admit_validation(self, ids):
        if not set(map(int, ids)) or not set(map(int, ids)) <= self.validation_ids:
            raise ValueError('Validation outside admitted groups')

    def provenance(self):
        return dict(scope=self.name, training_ids_sha256=ids_hash(self.training_ids),
                    validation_ids_sha256=ids_hash(self.validation_ids), outer_excluded_ids_sha256=ids_hash(self.outer_excluded_ids),
                    training_series=len(self.training_ids), validation_series=len(self.validation_ids), selection_training_only=True)


class FeatureBank:
    def __init__(self):
        self.guard = require_guard()
        pre = read_json(BASE / 'PREFLIGHT_NEXT9.json')
        directory = ROOT / pre['RP_cache']
        feature = ROOT / pre['feature_cache']
        assert digest(directory / 'MANIFEST.json') == pre['cache_manifest_sha256']
        assert digest(feature / 'MANIFEST.json') == pre['feature_manifest_sha256']
        with target_free_npz() as read_members:
            with np.load(directory / 'series.npz', allow_pickle=False) as z:
                self.ids = z['dataset_id']
                self.guard.admit(self.ids, purpose='NEXT9 target-free bank membership')
                self.offsets = z['online_offsets']
                self.historical_offsets = z['historical_offsets']
                self.folds = z['fold']
            with np.load(feature / 'series.npz', allow_pickle=False) as z:
                assert not {'target', 'tau', 'breakOccurred'} & set(z.files)
                self.series = {name: z[name] for name in z.files}
            self.decoded_NPZ_members = tuple(read_members)
        assert len(self.ids) == 6400 and set(map(int, self.ids)) == self.guard.research_ids
        np.testing.assert_array_equal(self.series['dataset_id'], self.ids)
        self.values = np.load(feature / 'values.npy', mmap_mode='r', allow_pickle=False)
        self.series_index = np.repeat(np.arange(len(self.ids), dtype=np.int32), np.diff(self.offsets))
        self.row_ids = self.ids[self.series_index]
        self.times = np.arange(self.offsets[-1], dtype=np.int64) - self.offsets[self.series_index]
        assert self.values.shape == (len(self.times), 153)
        self.meta = dict(dataset_id=self.row_ids, time_online=self.times)

    def current(self, rows, columns):
        return np.asarray(self.values[np.asarray(rows)[:, None], np.asarray(columns)[None, :]], dtype=np.float32)

    def history(self, rows, j):
        rows = np.asarray(rows, dtype=np.int64)
        t = self.times[rows]
        out = np.empty((len(rows), 16), dtype=np.float32)
        for k, lag in enumerate(range(16, 0, -1)):
            past = t >= lag
            q = rows[past] - lag
            assert np.array_equal(self.row_ids[q], self.row_ids[rows[past]])
            out[past, k] = self.values[q, j]
            s = self.series_index[rows[~past]]
            out[~past, k] = self.series['seed_values'][s, 16 + t[~past] - lag, j]
        return out

    def pre_state(self, rows, j):
        s = self.series_index[rows]
        # Movement availability is not the previous LEVEL q20-q80-gap test.
        return describe_history(self.history(rows, j), self.series['epsilon'][s, 0, j], self.series['H_length'][s] >= 160)

    def sampled_rows(self, ids, cutoff=None, start=0, stride=8):
        r = np.flatnonzero(np.isin(self.row_ids, list(ids)) & (self.times % stride == 0) & (self.times >= start))
        if cutoff is not None:
            r = r[self.times[r] <= cutoff]
        return r

    def forecast_rows(self, rows, horizon, cutoff=None):
        if horizon not in (1, 4, 16):
            raise ValueError('Unregistered horizon')
        r = np.asarray(rows, dtype=np.int64)
        keep = r + horizon < len(self.times)
        q = np.flatnonzero(keep)
        keep[q] &= self.row_ids[r[q] + horizon] == self.row_ids[r[q]]
        if cutoff is not None:
            keep &= self.times[r] + horizon <= cutoff
        return r[keep]


def outer_scope(bank, fold):
    valid = frozenset(int(s) for s, f in zip(bank.ids, bank.folds) if int(f) == fold)
    train = frozenset(map(int, bank.ids)) - valid
    bank.guard.partition(train, valid)
    return FitScope(train, valid, valid, f'outer_{fold}')


def inner_scope(bank, fold, inner):
    outer = outer_scope(bank, fold)
    mapping = read_json(ROOT / 'artifacts/next7/NEXT7_PROTOCOL_LOCK.json')['inner_mapping'][str(fold)]
    assert set(map(int, mapping)) == outer.training_ids
    valid = frozenset(int(s) for s, k in mapping.items() if int(k) == inner)
    bank.guard.partition(outer.training_ids - valid, valid | outer.validation_ids)
    return FitScope(outer.training_ids - valid, valid, outer.validation_ids, f'outer_{fold}_inner_{inner}')


def selection_splits(bank, fold, inner=None):
    if inner is None:
        return [inner_scope(bank, fold, k) for k in range(3)]
    parent = inner_scope(bank, fold, inner)
    groups = {}
    for sid in sorted(parent.training_ids):
        groups.setdefault(bank.guard.split.groups[sid], []).append(sid)
    key = lambda g: hashlib.sha256(f'NEXT8_SUBINNER_V1|{fold}|{inner}|{g}'.encode()).hexdigest()
    assignments = {g: k % 3 for k, g in enumerate(sorted(groups, key=key))}
    result = []
    for k in range(3):
        valid = frozenset(s for g, members in groups.items() if assignments[g] == k for s in members)
        excluded = parent.validation_ids | parent.outer_excluded_ids
        bank.guard.partition(parent.training_ids - valid, valid | excluded)
        result.append(FitScope(parent.training_ids - valid, valid, excluded, parent.name + f'_subinner_{k}'))
    return result
