"""Lazy read-only public inference for exact packed NEXT3 deployment variants."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import math
import os
import uuid
import numpy as np
from src.next3.artifacts import atomic_json, read_json, digest, utc_now
from src.next3.packed_shared_runtime import PreparedPackedSharedPredictor, VARIANTS, source_hash as runtime_hash
from src.next2.canonical_model import load, dumps, source_hash as format_hash
from src.next2.shared_inference import validate_contract
from src.next.readonly_jit import disable_loaded_disk_caches, source_hash as readonly_hash

INFER_PARALLELISM = 1
disable_loaded_disk_caches()


def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+readonly_hash().encode()).hexdigest()


def write_artifact(model, directory, specification, *, variant, origin, training=None):
    if variant not in VARIANTS: raise ValueError('Unregistered packed variant')
    directory = Path(directory)
    if (directory/'model.json').exists() or (directory/'model.sbnext2').exists():
        raise RuntimeError('Refusing to overwrite a NEXT3 packed deployment')
    validate_contract(model, specification)
    PreparedPackedSharedPredictor(model, variant=variant, resident=False)
    directory.mkdir(parents=True, exist_ok=True)
    content = dumps(model); temporary = directory/('model.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    try: os.link(temporary, directory/'model.sbnext2')
    finally: temporary.unlink(missing_ok=True)
    descriptor = {'kind': 'NEXT3_PACKED_SHARED_DEPLOYMENT', 'runtime': 'packed_shared_v1', 'variant': variant,
        'specification': specification, 'canonical_format_sha256': format_hash(), 'runtime_sha256': runtime_hash(),
        'inference_sha256': implementation_hash(), 'model_sha256': digest(directory/'model.sbnext2'),
        'origin': origin, 'training': training, 'created_utc': utc_now(), 'seal_rows_fitted': 0, 'reduced_new_usage': 0,
        'scientific_role': 'FROZEN_PERFORMANCE', 'scientific_quality_change': False,
        'on_disk_model_format': 'Unchanged frozen canonical bytes; prepared resident arrays use exact packed nodes'}
    atomic_json(directory/'model.json', descriptor, immutable=True)
    load_artifact(directory)
    return descriptor


def load_artifact(directory):
    directory = Path(directory); descriptor = read_json(directory/'model.json')
    if (descriptor.get('kind') != 'NEXT3_PACKED_SHARED_DEPLOYMENT' or descriptor.get('runtime') != 'packed_shared_v1'
            or descriptor.get('variant') not in VARIANTS):
        raise RuntimeError('Explicit registered NEXT3 packed descriptor required')
    if (descriptor['canonical_format_sha256'] != format_hash() or descriptor['runtime_sha256'] != runtime_hash()
            or descriptor['inference_sha256'] != implementation_hash()):
        raise RuntimeError('NEXT3 packed implementation identity changed')
    if digest(directory/'model.sbnext2') != descriptor['model_sha256']:
        raise RuntimeError('NEXT3 canonical model bytes changed')
    model = load(directory/'model.sbnext2'); validate_contract(model, descriptor['specification'])
    return model, descriptor


def _prepare_and_warm(directory):
    # This helper frame, decoded source-model arrays and warm-up H/index arrays
    # disappear before the generator's first yield. Retaining the decoded model
    # in generator locals would invalidate the measured prepared-memory savings.
    model, descriptor = load_artifact(directory)
    prepared = PreparedPackedSharedPredictor(model, variant=descriptor['variant'])
    del model, descriptor
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
                raise RuntimeError('Invalid NEXT3 packed streaming probability')
            yield prediction


def train(datasets, directory):
    from src.next2.canonical_inference import train_role, load_artifact as load_training
    directory = Path(directory)
    if (directory/'model.json').exists() or (directory/'model.sbnext2').exists():
        raise RuntimeError('Refusing to overwrite a NEXT3 packed deployment')
    root = Path(__file__).resolve().parents[2]
    selection = read_json(root/'configs/next3/PACKED_RUNTIME_SELECTION.json')
    if selection.get('status') not in ('PROVISIONAL_ENGINEERING_ELIGIBLE', 'FINAL_ACCEPTED'):
        raise RuntimeError('An engineering-eligible packed runtime selection is required')
    variant = selection['variant']
    if variant not in VARIANTS or selection['runtime_sha256'] != runtime_hash():
        raise RuntimeError('Packed runtime selection identity changed')
    specification = read_json(root/'configs/next2/DEPLOYMENT_RESEARCH_PERFORMANCE.json')
    scratch = directory/'training_canonical'
    descriptor = train_role(datasets, scratch, 'PERFORMANCE'); model, _ = load_training(scratch)
    return write_artifact(model, directory, specification, variant=variant,
        origin='OFFICIAL_RAW_ITERATOR_TRAIN_NEXT3_PACKED_SHARED', training=descriptor['training'])
