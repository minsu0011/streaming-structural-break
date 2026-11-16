"""Read-only public entry for the prediction-exact NEXT3 shared runtime."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib
import math
import os
import uuid
import numpy as np
from src.next3.artifacts import atomic_json,read_json,digest,utc_now
from src.next3.precomputed_shared_runtime import PreparedPrecomputedSharedPredictor,source_hash as runtime_hash
from src.next2.canonical_model import load,dumps,source_hash as format_hash
from src.next2.shared_inference import validate_contract
from src.next.readonly_jit import disable_loaded_disk_caches,source_hash as readonly_hash

INFER_PARALLELISM=1
disable_loaded_disk_caches()


def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+readonly_hash().encode()).hexdigest()


def write_artifact(model,directory,specification,*,origin,training=None):
    directory=Path(directory)
    if (directory/'model.json').exists() or (directory/'model.sbnext2').exists():
        raise RuntimeError('Refusing to overwrite a NEXT3 exact deployment')
    validate_contract(model,specification)
    directory.mkdir(parents=True,exist_ok=True)
    content=dumps(model)
    temporary=directory/('model.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(content);stream.flush();os.fsync(stream.fileno())
    try:os.link(temporary,directory/'model.sbnext2')
    finally:temporary.unlink(missing_ok=True)
    descriptor={'kind':'NEXT3_EXACT_SHARED_DEPLOYMENT','runtime':'precomputed_shared_v1',
        'specification':specification,'canonical_format_sha256':format_hash(),'runtime_sha256':runtime_hash(),
        'inference_sha256':implementation_hash(),'model_sha256':digest(directory/'model.sbnext2'),
        'origin':origin,'training':training,'created_utc':utc_now(),'seal_rows_fitted':0,'reduced_new_usage':0,
        'scientific_role':'FROZEN_PERFORMANCE','scientific_quality_change':False}
    atomic_json(directory/'model.json',descriptor,immutable=True)
    load_artifact(directory)
    return descriptor


def load_artifact(directory):
    directory=Path(directory);descriptor=read_json(directory/'model.json')
    if descriptor.get('kind')!='NEXT3_EXACT_SHARED_DEPLOYMENT' or descriptor.get('runtime')!='precomputed_shared_v1':
        raise RuntimeError('Explicit NEXT3 exact shared descriptor required')
    if descriptor['canonical_format_sha256']!=format_hash() or descriptor['runtime_sha256']!=runtime_hash() or descriptor['inference_sha256']!=implementation_hash():
        raise RuntimeError('NEXT3 exact implementation identity changed')
    if digest(directory/'model.sbnext2')!=descriptor['model_sha256']:raise RuntimeError('NEXT3 canonical model bytes changed')
    model=load(directory/'model.sbnext2');validate_contract(model,descriptor['specification'])
    return model,descriptor


def infer(datasets,model_directory_path):
    disable_loaded_disk_caches()
    model,_=load_artifact(model_directory_path);prepared=PreparedPrecomputedSharedPredictor(model)
    index=np.arange(128,dtype=np.float64)
    h=(np.sin(index*.37)+.2*np.cos(index*.11)).astype(np.float32)
    prepared.new_state(h).predict_one(0.)
    yield None
    for historical,online in datasets:
        state=prepared.new_state(historical)
        for point in online:
            value=float(point)
            if not math.isfinite(value):raise ValueError('Finite current online observation required')
            prediction=state.predict_one(value)
            if not math.isfinite(prediction) or not 0.<=prediction<=1.:raise RuntimeError('Invalid NEXT3 streaming probability')
            yield prediction


def train(datasets,directory):
    from src.next2.canonical_inference import train_role,load_artifact as load_training
    directory=Path(directory)
    if (directory/'model.json').exists() or (directory/'model.sbnext2').exists():raise RuntimeError('Refusing to overwrite a NEXT3 exact deployment')
    root=Path(__file__).resolve().parents[2]
    specification=read_json(root/'configs/next2/DEPLOYMENT_RESEARCH_PERFORMANCE.json')
    scratch=directory/'training_canonical'
    descriptor=train_role(datasets,scratch,'PERFORMANCE');model,_=load_training(scratch)
    return write_artifact(model,directory,specification,origin='OFFICIAL_ITERATOR_TRAIN_NEXT3_EXACT_SHARED',training=descriptor['training'])
