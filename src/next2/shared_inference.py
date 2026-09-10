"""Read-only public inference for the verified shared 80/20 runtime."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from src.next2.blend import FixedBlendBundle
from src.next2.canonical_model import load, save, source_hash as format_hash
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor, source_hash as runtime_hash
from src.next.readonly_jit import disable_loaded_disk_caches, source_hash as readonly_hash
from src.utils.artifacts import write_json, sha256, utc_now

INFER_PARALLELISM = 1
disable_loaded_disk_caches()


def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+readonly_hash().encode()).hexdigest()


def validate_contract(model, specification):
    from src.next2.final_fitting import training_feature_names
    if specification.get('status') != 'FROZEN_NEXT2_DEPLOYMENT' or specification.get('bundle_kind') != 'fixed_blend':
        raise RuntimeError('Frozen shared performance recipe required')
    if not isinstance(model, FixedBlendBundle) or model.weight != specification['weight'] or model.weight != .8:
        raise RuntimeError('Exact 80/20 blend contract required')
    for component, candidate in ((model.primary, specification['primary']), (model.complement, specification['complement'])):
        if component.extension != candidate['extension'] or tuple(component.names) != training_feature_names(candidate):
            raise RuntimeError('Shared runtime named feature contract differs')
    PreparedSharedBlendPredictor(model)


def write_artifact(model, directory, specification, *, origin, training=None):
    directory = Path(directory)
    if (directory/'model.json').exists() or (directory/'model.sbnext2').exists():
        raise RuntimeError('Refusing to overwrite a shared canonical model')
    validate_contract(model, specification)
    save(model, directory/'model.sbnext2')
    descriptor = {'kind': 'NEXT2_SHARED_CANONICAL_DEPLOYMENT', 'runtime': 'shared_blend_v1',
        'specification': specification, 'canonical_format_sha256': format_hash(), 'runtime_sha256': runtime_hash(),
        'inference_sha256': implementation_hash(), 'model_sha256': sha256(directory/'model.sbnext2'),
        'origin': origin, 'training': training, 'created_utc': utc_now(), 'seal_rows_fitted': 0, 'reduced_new_usage': 0}
    write_json(directory/'model.json', descriptor)
    load_artifact(directory)
    return descriptor


def load_artifact(directory):
    directory = Path(directory)
    descriptor = json.loads((directory/'model.json').read_text(encoding='utf-8'))
    if descriptor.get('kind') != 'NEXT2_SHARED_CANONICAL_DEPLOYMENT' or descriptor.get('runtime') != 'shared_blend_v1':
        raise RuntimeError('Explicit shared canonical descriptor required')
    if descriptor['canonical_format_sha256'] != format_hash() or descriptor['runtime_sha256'] != runtime_hash() or descriptor['inference_sha256'] != implementation_hash():
        raise RuntimeError('Shared canonical implementation identity changed')
    if sha256(directory/'model.sbnext2') != descriptor['model_sha256']:
        raise RuntimeError('Shared canonical model bytes changed')
    model = load(directory/'model.sbnext2')
    validate_contract(model, descriptor['specification'])
    return model, descriptor


def infer(datasets, model_directory_path):
    disable_loaded_disk_caches()
    model, _ = load_artifact(model_directory_path)
    prepared = PreparedSharedBlendPredictor(model)
    index = np.arange(128, dtype=np.float64)
    h = (np.sin(index*.37)+.2*np.cos(index*.11)).astype(np.float32)
    prepared.new_state(h).predict_one(0.)
    yield None
    for historical, online in datasets:
        state = prepared.new_state(historical)
        for point in online:
            value = float(point)
            if not math.isfinite(value): raise ValueError('Finite current online observation required')
            prediction = state.predict_one(value)
            if not math.isfinite(prediction) or not 0. <= prediction <= 1.: raise RuntimeError('Invalid shared streaming probability')
            yield prediction


def train(datasets, directory):
    from src.next2.canonical_inference import train_role, load_artifact as load_training
    directory = Path(directory)
    if (directory/'model.json').exists() or (directory/'model.sbnext2').exists():
        raise RuntimeError('Refusing to overwrite a shared deployment')
    root = Path(__file__).resolve().parents[2]
    specification = json.loads((root/'configs/next2/DEPLOYMENT_RESEARCH_PERFORMANCE.json').read_text(encoding='utf-8'))
    scratch = directory/'training_canonical'
    descriptor = train_role(datasets, scratch, 'PERFORMANCE')
    model, _ = load_training(scratch)
    return write_artifact(model, directory, specification, origin='OFFICIAL_ITERATOR_TRAIN_SHARED_CANONICAL', training=descriptor['training'])
