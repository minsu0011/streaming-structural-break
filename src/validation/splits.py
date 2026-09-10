"""Immutable series-level split; no labels, lengths, or performance enter allocation."""
import hashlib
import json
import numbers
from pathlib import Path
import pandas as pd
from src.utils.artifacts import sha256, utc_now, write_json

SEED = 20260908
FOLDS = 5
ALGORITHM = "sha256-v1; role=hash(seed:role:int_id)<floor(2**256/5); fold=hash(seed:fold:int_id)%5"

def canonical_id(series_id):
    if isinstance(series_id, bool) or not isinstance(series_id, numbers.Integral):
        raise ValueError("Official dataset_id must be an integer")
    return str(int(series_id))

def assignment(series_id, split=None):
    if split is not None:
        return split.assignment(series_id)
    identifier = canonical_id(series_id)
    def hashed(purpose):
        return int.from_bytes(hashlib.sha256(f"{SEED}:{purpose}:{identifier}".encode()).digest(), "big")
    if hashed("role") < (2**256 // 5):
        return "LOCAL_FINAL_SEAL", -1
    return "DEVELOPMENT_POOL", hashed("fold") % FOLDS

def freeze_split(ids, directory):
    ids = list(ids)
    if len(set(map(canonical_id, ids))) != len(ids):
        raise ValueError("Duplicate series IDs")
    records = [{"dataset_id": int(sid), "role": assignment(sid)[0], "fold": assignment(sid)[1]} for sid in sorted(ids)]
    frame = pd.DataFrame(records, columns=["dataset_id", "role", "fold"])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path, table_path = directory / "SPLIT_LOCK.json", directory / "SPLIT_ASSIGNMENTS.parquet"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get('status') == 'POLICY_FROZEN_IDS_PENDING':
            if lock['algorithm']!=ALGORITHM or lock['seed']!=SEED or table_path.exists():
                raise RuntimeError('Pending split policy modified')
        else:
            if lock["algorithm"] != ALGORITHM or lock["seed"] != SEED or sha256(table_path) != lock["assignments_sha256"]:
                raise RuntimeError("Split lock modified or algorithm mismatch")
            stored = pd.read_parquet(table_path)
            if not stored.equals(frame):
                raise RuntimeError("Refusing to change frozen split")
            return stored
    frame.to_parquet(table_path, index=False)
    write_json(lock_path, {"status": "LOCKED", "seed": SEED, "algorithm": ALGORITHM,
        "created_utc": utc_now(), "ids": [int(i) for i in sorted(ids)], "counts": frame.groupby(["role", "fold"]).size().rename("count").reset_index().to_dict("records"),
        "assignments_sha256": sha256(table_path), "seal_opened": False})
    return frame

def assert_development_ids(ids, split=None):
    if any(assignment(sid, split)[0] != "DEVELOPMENT_POOL" for sid in ids):
        raise RuntimeError("LOCAL_FINAL_SEAL is forbidden in feature/model evaluation")

GROUP_ALGORITHM = 'sha256-v2-group; representative=min integer ID; same seed/role/fold hash as v1'

class FrozenSplit:
    """Explicit immutable real-data allocation; unknown IDs fail closed."""
    def __init__(self, directory):
        directory = Path(directory)
        self.lock = json.loads((directory/'SPLIT_LOCK_REAL.json').read_text(encoding='utf-8'))
        path = directory/'GROUP_ASSIGNMENTS.parquet'
        if self.lock['status'] != 'LOCKED' or self.lock['assignments_sha256'] != sha256(path):
            raise RuntimeError('Real split integrity failure')
        self.frame = pd.read_parquet(path)
        if not self.frame.dataset_id.is_unique:
            raise RuntimeError('Duplicate split IDs')
        for _, group in self.frame.groupby('group_id'):
            if group.role.nunique() != 1 or group.fold.nunique() != 1:
                raise RuntimeError('Duplicate group crosses role or fold')
        self.mapping = {int(r.dataset_id):(r.role,int(r.fold)) for r in self.frame.itertuples()}
        self.groups = dict(zip(self.frame.dataset_id.astype(int),self.frame.group_id.astype(int)))
        self.digest = self.lock['assignments_sha256']

    def assignment(self, series_id):
        canonical_id(series_id)
        if int(series_id) not in self.mapping:
            raise RuntimeError('Unknown ID absent from frozen real split')
        return self.mapping[int(series_id)]

def freeze_group_split(groups, directory, *, group_policy_sha256, raw_sha256):
    """Called only after outcome-blind duplicate audit, before any score."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not groups.dataset_id.is_unique or groups[['dataset_id','group_id']].isna().any().any():
        raise ValueError('Unique integral ID/group required')
    frame = groups[['dataset_id','group_id']].sort_values('dataset_id').reset_index(drop=True).copy()
    roles = [assignment(int(group)) for group in frame.group_id]
    frame['role'] = [r[0] for r in roles]
    frame['fold'] = [r[1] for r in roles]
    lock_path = directory/'SPLIT_LOCK_REAL.json'
    if lock_path.exists():
        split = FrozenSplit(directory)
        if not split.frame.equals(frame) or split.lock['group_policy_sha256'] != group_policy_sha256 or split.lock['raw_sha256'] != raw_sha256:
            raise RuntimeError('Refusing to alter real frozen split')
        return split
    old = directory/'SPLIT_LOCK.json'
    if old.exists() and json.loads(old.read_text(encoding='utf-8')).get('status') != 'POLICY_FROZEN_IDS_PENDING':
        raise RuntimeError('An earlier actual split exists; cannot silently replace')
    table = directory/'GROUP_ASSIGNMENTS.parquet'
    frame.to_parquet(table,index=False)
    lock = {'status':'LOCKED','seed':SEED,'folds':FOLDS,'algorithm':GROUP_ALGORITHM,
        'created_utc':utc_now(),'group_policy_sha256':group_policy_sha256,'raw_sha256':raw_sha256,
        'assignments_sha256':sha256(table),'group_count':int(frame.group_id.nunique()),
        'series_count':len(frame),'counts':frame.groupby(['role','fold']).size().rename('count').reset_index().to_dict('records'),
        'ids_sha256':hashlib.sha256(','.join(map(str,frame.dataset_id)).encode()).hexdigest(),
        'LOCAL_FINAL_SEAL_OPENED':False,'performance_used_for_allocation':False,
        'prior_policy_only_replaced':True}
    write_json(lock_path,lock)
    return FrozenSplit(directory)
