"""Predicate-filtered DEV arrays, with immutable raw/split/code identities."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from src.next.guard import SealGuard, SealAccessError
from src.data.labels import labels_from_tau, normalize_tau
from src.utils.artifacts import sha256, write_json, utc_now

RAW_SHA = {
    'X_train.parquet': '2aa61853ce0c3a8f6cdfe137ccde9a59f97cf0872779d2063b549c60a7b7ed54',
    'y_train_index.parquet': '3b0ab253817b9828b8ab3393caed472125963fff3ad2a63de71ea89d12c6dd07'}


def code_hash():
    return hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name('guard.py').read_bytes()).hexdigest()


def iter_dev_series(guard, ids=None):
    requested = set(guard.admit(ids))
    labels = guard.labels(requested)
    pending = None
    seen = set()
    for batch in guard.scanner('X_train.parquet', requested).to_batches():
        frame = batch.to_pandas()
        if 'id' in frame.columns:
            frame = frame.set_index(['id', 'time'])
        if pending is not None:
            frame = pd.concat([pending, frame])
        if frame.empty:
            continue
        row_ids = frame.index.get_level_values('id').to_numpy()
        cuts = np.r_[0, np.flatnonzero(row_ids[1:] != row_ids[:-1]) + 1, len(row_ids)]
        for start, end in zip(cuts[:-2], cuts[1:-1]):
            sid = int(row_ids[start])
            if sid in seen or sid not in requested:
                raise SealAccessError('DEV stream ID admission/disorder failure')
            seen.add(sid)
            yield _extract(sid, frame.iloc[start:end], labels)
        pending = frame.iloc[cuts[-2]:].copy()
    if pending is not None and len(pending):
        sid = int(pending.index.get_level_values('id')[0])
        if sid in seen or sid not in requested:
            raise SealAccessError('DEV stream final ID failure')
        seen.add(sid)
        yield _extract(sid, pending, labels)
    if seen != requested:
        raise SealAccessError('DEV stream coverage mismatch')


def _extract(sid, frame, labels):
    period = frame.period.to_numpy()
    if not frame.index.is_unique or np.any(np.diff(period) < 0) or not np.isin(period, [1, 2]).all():
        raise ValueError('Invalid series period/order')
    if not frame.index.get_level_values('time').is_monotonic_increasing:
        raise ValueError('Invalid chronological order')
    h = frame.loc[period == 1, 'value'].to_numpy(dtype=np.float32)
    o = frame.loc[period == 2, 'value'].to_numpy(dtype=np.float32)
    if not len(h) or not len(o) or not np.isfinite(h).all() or not np.isfinite(o).all():
        raise ValueError('Finite nonempty real sequences required')
    tau = normalize_tau(labels.loc[sid, 'tau_index'])
    labels_from_tau(len(o), tau)
    return sid, h, o, -1 if tau is None else tau


def prepare_dev(root):
    root = Path(root)
    guard = SealGuard(root)
    identity = {'raw_sha256': RAW_SHA, 'split_sha256': guard.split.digest,
                'seal_lock_sha256': guard.lock_sha, 'code_sha256': code_hash(),
                'config': {'dtype': 'float32', 'periods': [1, 2], 'ids': 'all frozen DEV only'}}
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    out = root / 'data/next' / ('dev_' + identity_hash[:16])
    manifest = out / 'MANIFEST.json'
    if manifest.exists():
        stored = json.loads(manifest.read_text(encoding='utf-8'))
        if stored['identity'] != identity or stored['status'] != 'COMPLETE':
            raise RuntimeError('Invalid DEV cache identity')
        for name, digest in stored['files_sha256'].items():
            if sha256(out/name) != digest:
                raise RuntimeError('DEV cache bytes changed')
        return DevArrays(out, guard)
    out.mkdir(parents=True, exist_ok=True)
    # Byte hashing is explicitly distinct from observing seal target/feature rows.
    for name, digest in RAW_SHA.items():
        if sha256(root/'data/raw'/name) != digest:
            raise RuntimeError('Canonical training source changed')
    hs, os, ids, taus, hf, of, folds = [], [], [], [], [0], [0], []
    for sid, historical, online, tau in iter_dev_series(guard):
        hs.append(historical); os.append(online); ids.append(sid); taus.append(tau)
        hf.append(hf[-1] + len(historical)); of.append(of[-1] + len(online))
        folds.append(guard.split.assignment(sid)[1])
        if len(ids) % 1000 == 0:
            print('DEV PREP', len(ids), 'series', flush=True)
    guard.admit(ids, purpose='DEV array construction')
    np.save(out/'historical.npy', np.concatenate(hs), allow_pickle=False)
    np.save(out/'online.npy', np.concatenate(os), allow_pickle=False)
    np.savez(out/'series.npz', dataset_id=np.asarray(ids, dtype=np.int32), tau=np.asarray(taus, dtype=np.int32),
             historical_offsets=np.asarray(hf, dtype=np.int64), online_offsets=np.asarray(of, dtype=np.int64),
             fold=np.asarray(folds, dtype=np.int8))
    row_ids = np.repeat(np.asarray(ids, dtype=np.int32), np.diff(of))
    times = np.concatenate([np.arange(n, dtype=np.int16) for n in np.diff(of)])
    targets = np.concatenate([labels_from_tau(n, tau) for n, tau in zip(np.diff(of), taus)])
    np.savez(out/'rows.npz', dataset_id=row_ids, time_online=times, target=targets,
             fold=np.repeat(np.asarray(folds, dtype=np.int8), np.diff(of)))
    files = ['historical.npy', 'online.npy', 'series.npz', 'rows.npz']
    write_json(manifest, {'status': 'COMPLETE', 'created_utc': utc_now(), 'identity': identity,
        'files_sha256': {n: sha256(out/n) for n in files},
        **guard.record_rows(row_ids, purpose='DEV cache')})
    return DevArrays(out, guard)


class DevArrays:
    def __init__(self, directory, guard):
        self.directory = Path(directory)
        self.guard = guard
        with np.load(self.directory/'series.npz', allow_pickle=False) as source:
            self.series = {k: source[k] for k in source.files}
        guard.admit(self.series['dataset_id'], purpose='load DEV array cache')
        self.historical = np.load(self.directory/'historical.npy', mmap_mode='r', allow_pickle=False)
        self.online = np.load(self.directory/'online.npy', mmap_mode='r', allow_pickle=False)
        self.ids = self.series['dataset_id']
        self.id_to_index = {int(sid): j for j, sid in enumerate(self.ids)}

    def get(self, sid):
        self.guard.admit([sid], purpose='historical/online series')
        j = self.id_to_index[int(sid)]
        ho, oo = self.series['historical_offsets'], self.series['online_offsets']
        return self.historical[ho[j]:ho[j+1]], self.online[oo[j]:oo[j+1]], int(self.series['tau'][j])

    def rows(self):
        with np.load(self.directory/'rows.npz', allow_pickle=False) as source:
            result = {k: source[k] for k in source.files}
        self.guard.record_rows(result['dataset_id'], purpose='DEV row metadata')
        return result
