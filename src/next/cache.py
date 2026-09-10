"""Float32 memory-mapped feature caches bound to data, split, config and code."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from src.next.engine import SequentialState, EngineConfig, FEATURE_NAMES, FEATURE_GROUPS, feature_hash
from src.next.references import load_reference_fold, verify_reference_sources
from src.features.streaming import feature_names as old_feature_names
from src.features.config import CONFIG
from src.utils.artifacts import sha256, write_json, utc_now


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _verify_or_create(directory, identity):
    path = directory/'MANIFEST.json'
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['status'] != 'COMPLETE' or manifest['identity'] != identity:
            raise RuntimeError('Feature cache identity mismatch')
        for name, digest in manifest['files_sha256'].items():
            if sha256(directory/name) != digest:
                raise RuntimeError('Feature cache content changed')
        return True
    directory.mkdir(parents=True, exist_ok=True)
    return False


def build_engine_cache(data, config=EngineConfig(), scope='screen'):
    if scope not in ('screen', 'full'):
        raise ValueError('Only fixed screen or full DEV cache admitted')
    root = data.guard.root
    lock = json.loads((root/'ROBUSTNESS_SPLIT_LOCK.json').read_text(encoding='utf-8'))
    ids = data.guard.admit(lock['screen']['ids'] if scope == 'screen' else None, purpose='feature cache')
    rows = data.rows()
    global_rows = np.flatnonzero(np.isin(rows['dataset_id'], ids)).astype(np.int64)
    identity = {'scope': scope, 'config': asdict(config), 'feature_hash': feature_hash(config),
                'builder_sha256': sha256(Path(__file__)), 'dev_manifest_sha256': sha256(data.directory/'MANIFEST.json'),
                'split_sha256': data.guard.split.digest, 'seal_lock_sha256': data.guard.lock_sha,
                'ids_sha256': _digest(ids), 'columns': list(FEATURE_NAMES), 'dtype': 'float32'}
    out = root/'data/next/features'/(_digest(identity)[:20])
    if _verify_or_create(out, identity):
        return out
    started = time.perf_counter()
    matrix = np.lib.format.open_memmap(out/'features.npy', mode='w+', dtype=np.float32,
                                     shape=(len(global_rows), len(FEATURE_NAMES)))
    pointer = 0
    initializations = []
    selected = set(ids)
    ho, oo = data.series['historical_offsets'], data.series['online_offsets']
    count = 0
    for j, sid in enumerate(data.ids):
        if int(sid) not in selected:
            continue
        # Feature engine receives exactly H and the next online value; tau is
        # never passed. Output allocation length is not an input to the state.
        h = data.historical[ho[j]:ho[j+1]]
        points = data.online[oo[j]:oo[j+1]]
        tick = time.perf_counter()
        state = SequentialState(h, config)
        initializations.append(time.perf_counter()-tick)
        values = state.replay(points)
        if not np.isfinite(values).all():
            raise RuntimeError('Nonfinite causal feature matrix')
        matrix[pointer:pointer+len(points)] = values
        pointer += len(points)
        count += 1
        if count % 500 == 0:
            print('NEXT FEATURES', config.order, config.normalization, scope, count, round(time.perf_counter()-started,1), flush=True)
    if pointer != len(global_rows):
        raise RuntimeError('Feature row coverage mismatch')
    matrix.flush()
    del matrix
    np.save(out/'global_row_index.npy', global_rows, allow_pickle=False)
    write_json(out/'MANIFEST.json', {'status': 'COMPLETE', 'created_utc': utc_now(), 'identity': identity,
        'files_sha256': {name: sha256(out/name) for name in ['features.npy','global_row_index.npy']},
        'series': count, 'rows': pointer, 'seal_rows': 0, 'elapsed_seconds': time.perf_counter()-started,
        'historical_initialization_median_seconds': float(np.median(initializations)),
        'historical_initialization_p95_seconds': float(np.quantile(initializations,.95)),
        'maximum_single_series_feature_allocation_rows': int(np.max(np.diff(oo)))})
    return out


def build_arch_reference_matrix(data):
    root = data.guard.root
    verify_reference_sources(root)
    reference = 'M4_lightgbm_AR8_ARCH1_ABCD_S3_1cb5e51ac7ae'
    om = root/'data/processed/oof'/reference/'MANIFEST.json'
    old_manifest = json.loads(om.read_text(encoding='utf-8'))
    cm = root/old_manifest['cache_directory']/'MANIFEST.json'
    identity = {'reference': reference, 'old_oof_manifest_sha256': sha256(om),
        'old_feature_manifest_sha256': sha256(cm), 'builder_sha256': sha256(Path(__file__)),
        'dev_manifest_sha256': sha256(data.directory/'MANIFEST.json'),
        'split_sha256': data.guard.split.digest, 'seal_lock_sha256': data.guard.lock_sha,
        'old_config': {'normalization': 'median_mad', 'scales': [5,20,160]}}
    out = root/'data/next/reference_features'/(_digest(identity)[:20])
    if _verify_or_create(out, identity):
        return out
    config = CONFIG.variant(normalization='median_mad', scales=(5,20,160))
    names = old_feature_names(config)
    matrix = np.lib.format.open_memmap(out/'features.npy', mode='w+', dtype=np.float32,
                                     shape=(len(data.online), len(names)))
    offsets = dict(zip(map(int,data.ids), map(int,data.series['online_offsets'][:-1])))
    seen = np.zeros(len(data.online), dtype=bool)
    for fold in range(5):
        frame, _, _ = load_reference_fold(root, data.guard, 'HIGHEST_MEAN', fold, features=True)
        indices = frame.dataset_id.map(offsets).to_numpy()+frame.time_online.to_numpy()
        if np.any(seen[indices]):
            raise RuntimeError('Reference row coordinates overlap')
        matrix[indices] = frame.loc[:,names].to_numpy(dtype=np.float32)
        seen[indices] = True
        print('REFERENCE MATRIX', fold, flush=True)
    if not seen.all():
        raise RuntimeError('Reference matrix missing DEV coordinates')
    matrix.flush(); del matrix
    write_json(out/'MANIFEST.json', {'status': 'COMPLETE', 'created_utc': utc_now(), 'identity': identity,
        'files_sha256': {'features.npy': sha256(out/'features.npy')},
        'columns': ['arch8__'+name for name in names], 'seal_rows': 0, 'rows': len(data.online)})
    return out


def load_design(data, *, config=EngineConfig(), scope='screen', groups=('I','S1','S2','S3','S4'), base='none', keep_names=None):
    if base not in ('none','arch8'):
        raise ValueError('Unknown reference feature base')
    if groups:
        path = build_engine_cache(data, config, scope)
        matrix = np.load(path/'features.npy', mmap_mode='r', allow_pickle=False)
        global_rows = np.load(path/'global_row_index.npy', allow_pickle=False)
        columns = [j for j,(name,group) in enumerate(zip(FEATURE_NAMES,FEATURE_GROUPS))
                   if group in groups and (keep_names is None or name in keep_names)]
        names = [FEATURE_NAMES[j] for j in columns]
        sources = {'engine_manifest_sha256': sha256(path/'MANIFEST.json')}
    else:
        rows = data.rows()
        lock = json.loads((data.guard.root/'ROBUSTNESS_SPLIT_LOCK.json').read_text(encoding='utf-8'))
        global_rows = np.flatnonzero(np.isin(rows['dataset_id'], lock['screen']['ids'])) if scope == 'screen' else np.arange(len(data.online))
        matrix, columns, names, sources = None, [], [], {}
    base_matrix, base_names = None, []
    if base == 'arch8':
        path = build_arch_reference_matrix(data)
        base_matrix = np.load(path/'features.npy', mmap_mode='r', allow_pickle=False)
        manifest = json.loads((path/'MANIFEST.json').read_text(encoding='utf-8'))
        base_names = manifest['columns']
        sources['base_manifest_sha256'] = sha256(path/'MANIFEST.json')
    if not len(columns)+len(base_names):
        raise ValueError('No feature selected')
    design = np.empty((len(global_rows), len(base_names)+len(columns)), dtype=np.float32)
    if base_names:
        design[:,:len(base_names)] = base_matrix[global_rows]
    if columns:
        # Column-by-column assignment avoids a second full fancy-index copy.
        for k,j in enumerate(columns):
            design[:,len(base_names)+k] = matrix[:,j]
    metadata = {k:v[global_rows] for k,v in data.rows().items()}
    data.guard.record_rows(metadata['dataset_id'], purpose='supervised design')
    return design, metadata, base_names+names, sources, global_rows
