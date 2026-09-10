"""Train the fixed single/blend recipe from one opaque official iterator."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
from src.next.guard import SealGuard
from src.data.labels import normalize_tau
from src.next2.training import train_frozen_candidate,training_hash
from src.next2.io import model_class
from src.next2.blend import FixedBlendBundle
from src.next2.inference import write_descriptor
from src.utils.artifacts import sha256


class OpaqueSeal:
    def __array__(self,*args,**kwargs):
        raise AssertionError('Seal conversion attempted')
    def __len__(self):
        raise AssertionError('Seal length accessed')
    def __bool__(self):
        raise AssertionError('Seal value inspected')
    def __int__(self):
        raise AssertionError('Seal label converted')
    def __float__(self):
        raise AssertionError('Seal observation converted')


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+training_hash().encode()).hexdigest()


def materialize_dev(datasets,spec,guard):
    allowed=set(spec['development_ids'])
    known=set(spec['known_training_ids'])
    if allowed!=set(guard.dev_ids) or known!=set(guard.split.mapping) or spec['split_sha256']!=guard.split.digest:
        raise RuntimeError('Complete frozen official ID universe required')
    seen,records=set(),{}
    skipped=0
    for sid,historical,online,tau in datasets:
        if isinstance(sid,(bool,np.bool_)) or not isinstance(sid,(int,np.integer)):
            raise RuntimeError('Integral official ID required')
        sid=int(sid)
        if sid in seen or sid not in known:
            raise RuntimeError('Duplicate or unknown training ID')
        seen.add(sid)
        if sid not in allowed:
            # Discard every seal value without inspection, conversion or saving.
            skipped+=1
            continue
        h=np.array(historical,dtype=np.float32,copy=True)
        o=np.array(online,dtype=np.float32,copy=True)
        if h.ndim!=1 or o.ndim!=1 or not len(o) or not np.isfinite(h).all() or not np.isfinite(o).all():
            raise ValueError('Finite one-dimensional DEV sequences required')
        value=normalize_tau(tau)
        if value is not None and value>=len(o):
            raise ValueError('DEV break index outside observed online sequence')
        records[sid]=(h,o,value)
    if seen!=known or set(records)!=allowed or skipped!=len(guard.seal_ids):
        raise RuntimeError('Complete official training iterator required')
    guard.check_state()
    return records,skipped


def train(datasets,directory,spec,*,root=None):
    root=Path(root) if root is not None else Path(__file__).resolve().parents[2]
    directory=Path(directory)
    if spec.get('status')!='FROZEN_NEXT2_DEPLOYMENT' or spec.get('bundle_kind') not in ('single','fixed_blend'):
        raise RuntimeError('Reviewed frozen NEXT2 recipe required')
    if (directory/'model.joblib').exists() or (directory/'model.json').exists():
        raise RuntimeError('Refusing to overwrite a NEXT2 deployment artifact')
    guard=SealGuard(root)
    records,skipped=materialize_dev(datasets,spec,guard)
    def replay():
        for sid in sorted(records):
            yield sid,*records[sid]
        for sid in sorted(guard.seal_ids):
            yield sid,OpaqueSeal(),OpaqueSeal(),OpaqueSeal()
    candidates=[('single',spec['candidate'])] if spec['bundle_kind']=='single' else [('primary',spec['primary']),('complement',spec['complement'])]
    components,receipts={},{}
    for role,candidate in candidates:
        sub=directory/'training_components'/role
        c_spec={'status':'FROZEN_NEXT2_DEPLOYMENT','candidate':candidate,
            'development_ids':spec['development_ids'],'known_training_ids':spec['known_training_ids'],
            'split_sha256':spec['split_sha256']}
        receipts[role]=train_frozen_candidate(replay(),sub,c_spec,root=root)
        components[role]=model_class(candidate).load(sub/'model.joblib')
    directory.mkdir(parents=True,exist_ok=True)
    if spec['bundle_kind']=='single':
        shutil.copyfile(directory/'training_components/single/model.joblib',directory/'model.joblib')
    else:
        FixedBlendBundle(components['primary'],components['complement'],spec['weight']).save(directory/'model.joblib')
    guard.check_state()
    detail={'component_receipts':receipts,'source_sha256':source_hash(),
        'seal_series_skipped_before_any_value_access':skipped,'training_series':len(records),
        'new_fits':len(candidates),'one_pass_official_iterator':True,'only_DEV_materialized_in_memory':True}
    return write_descriptor(directory,spec,origin='OFFICIAL_ITERATOR_TRAIN',training=detail)


def train_role(datasets,directory,role):
    if role not in ('PERFORMANCE','DISTILLED'):
        raise ValueError('Explicit NEXT2 role required')
    root=Path(__file__).resolve().parents[2]
    path=root/'configs/next2'/('DEPLOYMENT_RESEARCH_'+role+'.json')
    return train(datasets,directory,json.loads(path.read_text(encoding='utf-8')),root=root)
