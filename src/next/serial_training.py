"""Strict official iterator training for reviewed registered NEXT banks.

The already verified variance trainer remains byte-for-byte untouched; this
explicit adapter keeps the same ID admission and chronological training rules."""
from pathlib import Path
import hashlib
import json
import numpy as np
from src.next.guard import SealGuard
from src.next.extensions import ExtendedFeatureState
from src.next.registered_final_fitting import fit_all_development,training_feature_names,source_hash as fitting_source_hash
from src.next.candidate_io import model_class
from src.next.serial_inference import implementation_hash as inference_hash
from src.next.serial_variance_runtime import source_hash as runtime_hash
from src.data.labels import labels_from_tau
from src.utils.artifacts import write_json,sha256,utc_now


def training_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+fitting_source_hash().encode()).hexdigest()


def train_frozen_candidate(datasets,model_directory_path,spec,*,root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    directory = Path(model_directory_path)
    if (directory/'model.joblib').exists() or (directory/'model.json').exists():
        raise RuntimeError('Refusing to retrain over an existing NEXT artifact')
    guard = SealGuard(root)
    allowed = set(map(int,spec['development_ids']))
    known = set(map(int,spec['known_training_ids']))
    if allowed!=set(guard.dev_ids) or known!=set(guard.split.mapping) or spec['split_sha256']!=guard.split.digest:
        raise RuntimeError('NEXT training requires the complete frozen DEV and known-ID universe')
    if spec.get('status') not in ('FROZEN_NEXT_DEV_PRIMARY','FROZEN_NEXT_DEV_BACKUP','FROZEN_NEXT_RESEARCH_REFIT'):
        raise RuntimeError('NEXT candidate definition is not frozen')
    candidate = spec['candidate']
    names = training_feature_names(candidate)
    matrices,targets = {},{}
    seen,skipped = set(),0
    for sid,historical,online,tau in datasets:
        if isinstance(sid,(bool,np.bool_)) or not isinstance(sid,(int,np.integer)):
            raise RuntimeError('Integral official training ID required')
        sid = int(sid)
        if sid in seen or sid not in known:
            raise RuntimeError('Duplicate or unknown official training ID')
        seen.add(sid)
        # The seal tuple's three other fields are never examined or converted.
        if sid not in allowed:
            skipped += 1
            continue
        h,o = np.asarray(historical,dtype=np.float32),np.asarray(online,dtype=np.float32)
        if h.ndim!=1 or o.ndim!=1 or not len(o) or not np.isfinite(h).all() or not np.isfinite(o).all():
            raise ValueError('Finite one-dimensional training sequences required')
        state = ExtendedFeatureState(h,names,candidate['extension'])
        matrix = np.empty((len(o),len(names)),dtype=np.float32)
        for t,point in enumerate(o):
            matrix[t] = state.update(point)
        matrices[sid] = matrix
        targets[sid] = labels_from_tau(len(o),tau)
        if len(matrices)%500==0:
            print('NEXT OFFICIAL TRAIN FEATURES',len(matrices),flush=True)
    if seen!=known or set(matrices)!=allowed or skipped!=len(guard.seal_ids):
        raise RuntimeError('Official training iterator has incomplete ID coverage')
    ids = sorted(allowed,key=lambda sid:(guard.split.assignment(sid)[1],sid))
    design = np.concatenate([matrices[sid] for sid in ids])
    target = np.concatenate([targets[sid] for sid in ids])
    metadata = {'dataset_id':np.concatenate([np.full(len(targets[sid]),sid,dtype=np.int32) for sid in ids]),
        'target':target,'time_online':np.concatenate([np.arange(len(targets[sid]),dtype=np.int32) for sid in ids])}
    del matrices,targets
    model,details = fit_all_development(design,metadata,candidate,guard,names)
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'model.joblib').exists() or (directory/'model.json').exists():
        raise RuntimeError('Refusing to overwrite an existing NEXT refit artifact')
    model.save(directory/'model.joblib')
    loaded = model_class(candidate).load(directory/'model.joblib')
    probe = design[:min(len(design),4096)]
    np.testing.assert_array_equal(model.predict(probe),loaded.predict(probe))
    descriptor = {'kind':'NEXT_REGISTERED_BINARY','runtime':'serial_variance_fast_h','candidate':candidate,
        'model_sha256':sha256(directory/'model.joblib'),'runtime_sha256':runtime_hash(),'inference_sha256':inference_hash(),
        'status':spec['status'],'created_utc':utc_now(),'training':details,
        'training_source_sha256':training_hash(),'specification_sha256':hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),
        'seal_series_skipped_before_value_or_label_access':skipped,'seal_rows_fitted':0,'reduced_new_usage':0,
        'split_sha256':guard.split.digest,'seal_lock_sha256':guard.lock_sha,
        'input_origin':spec.get('input_origin','OFFICIAL_TRAINING_ITERATOR')}
    write_json(directory/'model.json',descriptor)
    return descriptor
