"""DEV-only AR-input feature caches for explicit configurations, never seal data."""
import gc, json
from pathlib import Path
import numpy as np
import pandas as pd
from src.features.streaming import feature_names
from src.features.ar_residual_input import ARResidualCalibratedState, implementation_hash, AR_POLICY
from src.features.cache import build_cache, read_fold
from src.data.loader import load_training
from src.utils.artifacts import write_json, sha256


def build_ar_cache(raw, cache_root, split, config, policy, *, input_order=None):
    raw = Path(raw); version = implementation_hash(policy, config)
    state_factory=ARResidualCalibratedState;prefix='ar_residual_input_'
    if input_order is not None:
        from functools import partial
        from src.features.ar_order_input import AROrderCalibratedState,implementation_hash as order_hash
        state_factory=partial(AROrderCalibratedState,input_order=input_order)
        version=order_hash(policy,config,input_order);prefix=f'ar_order{input_order}_'
    directory = Path(cache_root) / (prefix + version[:16]); manifest_path = directory / 'MANIFEST.json'
    source_identity = {name: sha256(raw / name) for name in ('X_train.parquet', 'y_train_index.parquet')}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('raw_source_identity', source_identity) != source_identity: raise RuntimeError('Raw bytes changed under an existing AR representation cache')
        if manifest.get('status') == 'COMPLETE' and manifest.get('representation_hash') == version and manifest.get('split_sha256') == split.digest:
            for fold in range(5):
                if sha256(directory / f'fold_{fold}.parquet') != manifest['fold_sha256'][str(fold)]: raise RuntimeError('AR feature cache bytes changed')
            return directory
    source = build_cache(raw, cache_root, split=split, config=config)
    series = {s.dataset_id: s for s in load_training(raw, split=split)}
    names = feature_names(config)
    manifest = {'status': 'BUILDING', 'representation_hash': version, 'split_sha256': split.digest, 'ar_policy': AR_POLICY,
                'calibration_policy': policy.__dict__, 'source_cache_manifest_sha256': sha256(source / 'MANIFEST.json'),
                'raw_source_identity': source_identity, 'seal_rows': 0, 'fold_sha256': {}}
    if input_order is not None:manifest['input_order_override']=input_order
    write_json(manifest_path, manifest)
    for fold in range(5):
        frame = read_fold(source, fold, split=split, config=config); matrix = np.empty((len(frame), len(names)), dtype=np.float32)
        times = frame.time_online.to_numpy()
        for sid, positions in frame.groupby('dataset_id', sort=False).indices.items():
            item = series[int(sid)]; stream = state_factory(item.historical, config=config, policy=policy)
            if not np.array_equal(times[positions], np.arange(len(item.online))): raise RuntimeError('AR cache online row identity mismatch')
            for position, point in zip(positions, item.online): matrix[position] = stream.update_and_get(point)
        if not np.isfinite(matrix).all(): raise RuntimeError('Nonfinite AR feature representation')
        frame.loc[:, names] = matrix; path = directory / f'fold_{fold}.parquet'; frame.to_parquet(path, index=False)
        manifest['fold_sha256'][str(fold)] = sha256(path); write_json(manifest_path, manifest)
        print('AR cache fold', fold, 'complete', flush=True); del frame, matrix; gc.collect()
    manifest['status'] = 'COMPLETE'; write_json(manifest_path, manifest)
    return directory
