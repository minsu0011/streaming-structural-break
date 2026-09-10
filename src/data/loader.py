"""Chunked official parquet reader; only one series is retained between batches."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.data.labels import normalize_tau,labels_from_tau
from src.validation.splits import assignment

@dataclass
class TrainingSeries:
    dataset_id: int
    historical: np.ndarray
    online: np.ndarray
    tau: int | None
    online_index: np.ndarray

def series_frames(path, batch_size=65536):
    """Require contiguous IDs and increasing within-ID time; fail on disorder.

    Auditing must expose a contract mismatch instead of silently changing time.
    Arrow preserves parquet pandas MultiIndex metadata in each batch.
    """
    pending=None
    seen=set()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        frame=batch.to_pandas()
        if list(frame.index.names) != ['id','time']:
            if {'id','time'}.issubset(frame.columns):
                frame=frame.set_index(['id','time'])
            else:
                raise ValueError('Expected id,time index')
        if pending is not None:
            frame=pd.concat([pending,frame])
        if frame.empty:
            continue
        ids=frame.index.get_level_values('id').to_numpy()
        cuts=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1,len(ids)]
        for start,end in zip(cuts[:-2],cuts[1:-1]):
            part=frame.iloc[start:end]
            sid=int(ids[start])
            if sid in seen:
                raise ValueError('Noncontiguous series ID')
            seen.add(sid)
            _validate_series(part)
            yield sid,part
        pending=frame.iloc[cuts[-2]:].copy()
    if pending is not None and not pending.empty:
        sid=int(pending.index.get_level_values('id')[0])
        if sid in seen:
            raise ValueError('Noncontiguous series ID')
        _validate_series(pending)
        yield sid,pending

def _validate_series(frame):
    if not frame.index.is_unique or not frame.index.get_level_values('time').is_monotonic_increasing:
        raise ValueError('Duplicate or unordered time')
    if not {'value','period'}.issubset(frame.columns):
        raise ValueError('Missing official value/period columns')
    p=frame.period.to_numpy()
    if not np.isin(p,[1,2]).all() or np.any(np.diff(p)<0):
        raise ValueError('Expected historical period 1 followed by online period 2')

def load_training(directory, *, development_only=True, batch_size=65536, split=None):
    directory=Path(directory)
    canonical=Path(__file__).resolve().parents[2]/'data/raw'
    if development_only and directory.resolve()==canonical.resolve() and split is None:
        raise RuntimeError('Canonical real-data evaluation requires explicit frozen group split')
    development_ids=[sid for sid,role_fold in split.mapping.items() if role_fold[0]=='DEVELOPMENT_POOL'] if split is not None and development_only else None
    filters=[('id','in',development_ids)] if development_ids is not None else None
    labels=pd.read_parquet(directory/'y_train_index.parquet',filters=filters)
    expected_ids=set(split.mapping) if development_ids is not None else set(labels.index)
    if not labels.index.is_unique or 'tau_index' not in labels:
        raise ValueError('Invalid y_train_index')
    encountered=set()
    for sid,frame in series_frames(directory/'X_train.parquet',batch_size):
        encountered.add(sid)
        if development_only and assignment(sid,split)[0]!='DEVELOPMENT_POOL':
            continue
        if sid not in labels.index:
            raise ValueError('Missing training tau')
        tau=normalize_tau(labels.loc[sid,'tau_index'])
        historical=frame.loc[frame.period==1,'value'].to_numpy()
        online=frame.loc[frame.period==2,'value'].to_numpy()
        labels_from_tau(len(online),tau)  # Bounds check only; no feature receives tau.
        yield TrainingSeries(sid,historical,online,tau,frame.loc[frame.period==2].index.get_level_values('time').to_numpy())
    if encountered != expected_ids:
        raise ValueError('Training IDs do not match label IDs')
