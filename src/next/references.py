"""Read-only restoration of frozen reference caches and OOF coordinates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.artifacts import sha256, write_json
from src.scoring.ts_auc import ts_auc
from src.models.representations import bundle_class

REFERENCES = {
    'PRIMARY': 'M7_cross_order_AR4_LGB_AR8_HGB_equal',
    'HIGHEST_MEAN': 'M4_lightgbm_AR8_ARCH1_ABCD_S3_1cb5e51ac7ae',
    'ROBUST_BACKUP': 'M3_histgb_AR8_ARCH1_ABCD_S3_1cb5e51ac7ae'}
REPRESENTATIONS = {'PRIMARY': 'cross_order_ar4_ar8_equal',
                   'HIGHEST_MEAN': 'ar8_arch1_historical_median_mad',
                   'ROBUST_BACKUP': 'ar8_arch1_historical_median_mad'}


def verify_reference_sources(root):
    root = Path(root)
    session = json.loads((root/'artifacts/next/SESSION_NEXT.json').read_text(encoding='utf-8'))
    for name, digest in session['old_reference_hashes'].items():
        if sha256(root/name) != digest:
            raise RuntimeError(f'Frozen prior reference modified: {name}')


def load_reference_fold(root, guard, role, fold, *, features=False, verify_bytes=True):
    root = Path(root)
    eid = REFERENCES[role]
    oof = root/'data/processed/oof'/eid
    manifest = json.loads((oof/'MANIFEST.json').read_text(encoding='utf-8'))
    cache = root/manifest['cache_directory']
    cm = json.loads((cache/'MANIFEST.json').read_text(encoding='utf-8'))
    if cm['status'] != 'COMPLETE' or cm['split_sha256'] != guard.split.digest or manifest['split_sha256'] != guard.split.digest:
        raise RuntimeError('Reference cache allocation changed')
    path = cache/f'fold_{fold}.parquet'
    if verify_bytes and sha256(path) != cm['fold_sha256'][str(fold)]:
        raise RuntimeError('Reference feature cache bytes changed')
    # Existing loader recomputes the representation/configuration/code hash.
    model_path = root/'artifacts/models/real'/eid/f'fold_{fold}.joblib'
    model = bundle_class(REPRESENTATIONS[role]).load(model_path)
    columns = None if features else ['dataset_id', 'time_online', 'target']
    frame = pd.read_parquet(path, columns=columns)
    guard.record_rows(frame.dataset_id.to_numpy(), purpose=f'{role} reference OOF')
    expected = {i for i, (r, f) in guard.split.mapping.items() if r == 'DEVELOPMENT_POOL' and f == fold}
    if set(map(int, frame.dataset_id.unique())) != expected:
        raise RuntimeError('Reference fold ID coverage changed')
    prediction = np.load(oof/f'fold_{fold}.npy', allow_pickle=False)
    if len(prediction) != len(frame) or prediction.dtype != np.float32:
        raise RuntimeError('Invalid frozen OOF dtype/length')
    expected_score = json.loads((root/'artifacts/next/REFERENCE_RESTORATION.json').read_text(encoding='utf-8'))['references'][role]['fold_scores'][fold]
    actual = ts_auc(frame.target, prediction, frame.time_online)
    if abs(actual - expected_score) > 1e-14:
        raise RuntimeError('Stored reference score cannot be reconstructed')
    return frame, prediction, model
