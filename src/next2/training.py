"""Guarded official-iterator fitting; seal values remain opaque objects."""
from pathlib import Path
import hashlib
import json
import numpy as np
from src.next.guard import SealGuard
from src.next.extensions import ExtendedFeatureState
from src.next2.final_fitting import fit_all_development,training_feature_names,source_hash
from src.next2.io import model_class
from src.data.labels import labels_from_tau
from src.utils.artifacts import write_json,sha256,utc_now


def training_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+source_hash().encode()).hexdigest()


def iterator_design(datasets,spec,guard):
    allowed=set(map(int,spec['development_ids']))
    known=set(map(int,spec['known_training_ids']))
    if allowed!=set(guard.dev_ids) or known!=set(guard.split.mapping) or spec['split_sha256']!=guard.split.digest:
        raise RuntimeError('Complete frozen DEV and official ID universe required')
    if spec['status'] not in ('FROZEN_NEXT2_RESEARCH_REFIT','FROZEN_NEXT2_DEPLOYMENT'):
        raise RuntimeError('NEXT2 training specification is not frozen')
    candidate=spec['candidate']
    names=training_feature_names(candidate)
    matrices,targets={},{}
    seen,skipped=set(),0
    for sid,historical,online,tau in datasets:
        if isinstance(sid,(bool,np.bool_)) or not isinstance(sid,(int,np.integer)):
            raise RuntimeError('Integral official training ID required')
        sid=int(sid)
        if sid in seen or sid not in known:
            raise RuntimeError('Duplicate or unknown official training ID')
        seen.add(sid)
        if sid not in allowed:
            skipped+=1
            continue
        h,o=np.asarray(historical,dtype=np.float32),np.asarray(online,dtype=np.float32)
        if h.ndim!=1 or o.ndim!=1 or not len(o) or not np.isfinite(h).all() or not np.isfinite(o).all():
            raise ValueError('Finite one-dimensional training sequences required')
        state=ExtendedFeatureState(h,names,candidate['extension'])
        matrix=np.empty((len(o),len(names)),dtype=np.float32)
        for t,point in enumerate(o):
            matrix[t]=state.update(point)
        matrices[sid]=matrix
        targets[sid]=labels_from_tau(len(o),tau)
        if len(matrices)%500==0:
            print('NEXT2 RAW TRAIN FEATURES',len(matrices),flush=True)
    if seen!=known or set(matrices)!=allowed or skipped!=len(guard.seal_ids):
        raise RuntimeError('Official iterator ID coverage incomplete')
    ids=sorted(allowed,key=lambda sid:(guard.split.assignment(sid)[1],sid))
    design=np.concatenate([matrices[sid] for sid in ids])
    target=np.concatenate([targets[sid] for sid in ids])
    metadata={'dataset_id':np.concatenate([np.full(len(targets[sid]),sid,dtype=np.int32) for sid in ids]),
        'target':target,'time_online':np.concatenate([np.arange(len(targets[sid]),dtype=np.int32) for sid in ids])}
    return design,metadata,names,skipped


def train_frozen_candidate(datasets,directory,spec,*,root=None):
    root=Path(root) if root is not None else Path(__file__).resolve().parents[2]
    directory=Path(directory)
    if (directory/'model.joblib').exists() or (directory/'TRAINING.json').exists():
        raise RuntimeError('Refusing to overwrite a NEXT2 model')
    guard=SealGuard(root)
    design,metadata,names,skipped=iterator_design(datasets,spec,guard)
    model,detail=fit_all_development(design,metadata,spec['candidate'],guard,names)
    directory.mkdir(parents=True,exist_ok=True)
    model.save(directory/'model.joblib')
    loaded=model_class(spec['candidate']).load(directory/'model.joblib')
    np.testing.assert_array_equal(model.predict(design[:4096]),loaded.predict(design[:4096]))
    report={'status':'PASS','candidate':spec['candidate'],'training':detail,
        'model_sha256':sha256(directory/'model.joblib'),'training_source_sha256':training_hash(),
        'specification_sha256':hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),
        'seal_series_skipped_before_value_or_label_access':skipped,'seal_rows_fitted':0,
        'input_origin':'PREDICATE_FILTERED_RAW_DEV_WITH_OPAQUE_SEAL_PLACEHOLDERS','created_utc':utc_now()}
    write_json(directory/'TRAINING.json',report)
    return report
