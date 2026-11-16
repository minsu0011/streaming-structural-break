"""Lazy frozen-performance inference with training dependencies imported on train only."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import math
import os
import uuid
import numpy as np
from src.next3.artifacts import atomic_json, read_json, digest, utc_now
from src.next3.lean_model import loads, feature_contract
from src.next3.lean_shared_runtime import PreparedLeanSharedPredictor, source_hash as runtime_hash
from src.next.readonly_jit import disable_loaded_disk_caches, source_hash as readonly_hash

INFER_PARALLELISM = 1
disable_loaded_disk_caches()


def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+readonly_hash().encode()).hexdigest()


def checked_specification(specification):
    root = Path(__file__).resolve().parents[2]
    expected = read_json(root/'configs/next2/DEPLOYMENT_RESEARCH_PERFORMANCE.json')
    if specification != expected: raise ValueError('Exact frozen performance training specification required')


def write_artifact(content, directory, specification, *, origin, training=None):
    directory = Path(directory)
    if (directory/'model.json').exists() or (directory/'model.sbnext2').exists():
        raise RuntimeError('Refusing to overwrite a NEXT3 lean deployment')
    contract = feature_contract(); record = loads(content, contract=contract)
    checked_specification(specification)
    PreparedLeanSharedPredictor(record, resident=False, contract=contract)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory/('model.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    try: os.link(temporary, directory/'model.sbnext2')
    finally: temporary.unlink(missing_ok=True)
    descriptor = {'kind': 'NEXT3_LEAN_SHARED_DEPLOYMENT', 'runtime': 'lean_packed_shared_v1',
        'variant': 'LEAN_PACKED_SHARED', 'specification': specification, 'runtime_sha256': runtime_hash(),
        'inference_sha256': implementation_hash(), 'model_sha256': digest(directory/'model.sbnext2'),
        'origin': origin, 'training': training, 'created_utc': utc_now(), 'seal_rows_fitted': 0, 'reduced_new_usage': 0,
        'scientific_role': 'FROZEN_PERFORMANCE', 'scientific_quality_change': False,
        'on_disk_model_format': 'Unchanged SB_NEXT2_MODEL_V1 canonical bytes'}
    atomic_json(directory/'model.json', descriptor, immutable=True)
    load_artifact(directory)
    return descriptor


def load_artifact(directory):
    directory = Path(directory); descriptor = read_json(directory/'model.json')
    if (descriptor.get('kind') != 'NEXT3_LEAN_SHARED_DEPLOYMENT' or
            descriptor.get('runtime') != 'lean_packed_shared_v1' or descriptor.get('variant') != 'LEAN_PACKED_SHARED'):
        raise RuntimeError('Explicit frozen NEXT3 lean descriptor required')
    if descriptor['runtime_sha256'] != runtime_hash() or descriptor['inference_sha256'] != implementation_hash():
        raise RuntimeError('NEXT3 lean implementation identity changed')
    content = (directory/'model.sbnext2').read_bytes()
    if hashlib.sha256(content).hexdigest() != descriptor['model_sha256']:
        raise RuntimeError('NEXT3 lean canonical model bytes changed')
    record = loads(content); checked_specification(descriptor['specification'])
    return record, descriptor


def _prepare_and_warm(directory):
    record, descriptor = load_artifact(directory)
    prepared = PreparedLeanSharedPredictor(record)
    del record, descriptor
    index = np.arange(128, dtype=np.float64)
    historical = (np.sin(index*.37)+.2*np.cos(index*.11)).astype(np.float32)
    prepared.new_state(historical).predict_one(0.)
    return prepared


def infer(datasets, model_directory_path):
    disable_loaded_disk_caches()
    prepared = _prepare_and_warm(model_directory_path)
    yield None
    for historical, online in datasets:
        state = prepared.new_state(historical)
        for point in online:
            value = float(point)
            if not math.isfinite(value): raise ValueError('Finite current online observation required')
            prediction = state.predict_one(value)
            if not math.isfinite(prediction) or not 0. <= prediction <= 1.:
                raise RuntimeError('Invalid NEXT3 lean streaming probability')
            yield prediction


def train(datasets, directory):
    from src.next2.canonical_inference import train_role
    directory = Path(directory)
    if (directory/'model.json').exists() or (directory/'model.sbnext2').exists():
        raise RuntimeError('Refusing to overwrite a NEXT3 lean deployment')
    root = Path(__file__).resolve().parents[2]
    selection = read_json(root/'configs/next3/LEAN_RUNTIME_SELECTION.json')
    if (selection.get('status') not in ('PROVISIONAL_ENGINEERING_ELIGIBLE', 'FINAL_ACCEPTED') or
            selection.get('variant') != 'LEAN_PACKED_SHARED' or selection['runtime_sha256'] != runtime_hash()):
        raise RuntimeError('An engineering-eligible frozen lean runtime selection is required')
    specification = read_json(root/'configs/next2/DEPLOYMENT_RESEARCH_PERFORMANCE.json')
    scratch = directory/'training_canonical'; descriptor = train_role(datasets, scratch, 'PERFORMANCE')
    return write_artifact((scratch/'model.sbnext2').read_bytes(), directory, specification,
        origin='OFFICIAL_RAW_ITERATOR_TRAIN_NEXT3_LEAN_SHARED', training=descriptor['training'])
