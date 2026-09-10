"""Separate production entry for the reviewed log-energy registered bank."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from src.next.candidate_io import model_class, validate_sources
from src.next.log_energy_runtime import PreparedLogEnergyPredictor, source_hash as runtime_hash
from src.next.readonly_jit import disable_loaded_disk_caches, source_hash as readonly_hash
from src.utils.artifacts import sha256

INFER_PARALLELISM = 1
disable_loaded_disk_caches()


def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+readonly_hash().encode()).hexdigest()


def load_artifact(directory):
    directory = Path(directory)
    descriptor = json.loads((directory/'model.json').read_text(encoding='utf-8'))
    if descriptor.get('kind') != 'NEXT_REGISTERED_BINARY' or descriptor.get('runtime') != 'log_energy_fast_h':
        raise RuntimeError('Explicit reviewed log-energy artifact required')
    if descriptor.get('inference_sha256') != implementation_hash() or descriptor.get('runtime_sha256') != runtime_hash():
        raise RuntimeError('Registered inference or runtime source changed')
    if sha256(directory/'model.joblib') != descriptor['model_sha256']:
        raise RuntimeError('Registered model bytes changed')
    candidate = descriptor['candidate']
    validate_sources(candidate)
    if candidate.get('extension', {}).get('kind') != 'log_energy':
        raise RuntimeError('This production adapter accepts only the reviewed log-energy family')
    model = model_class(candidate).load(directory/'model.joblib')
    if model.extension != candidate['extension']:
        raise RuntimeError('Registered model and candidate feature-bank settings differ')
    return model, descriptor


def infer(datasets, model_directory_path):
    disable_loaded_disk_caches()
    model, _ = load_artifact(model_directory_path)
    prepared = PreparedLogEnergyPredictor(model)
    index = np.arange(128, dtype=np.float64)
    warm_h = (np.sin(index*.37)+.2*np.cos(index*.11)).astype(np.float32)
    prepared.new_state(warm_h).predict_one(0.)
    yield None
    for historical, online in datasets:
        detector = prepared.new_state(historical)
        for point in online:
            point = float(point)
            if not math.isfinite(point):
                raise ValueError('Finite online observation required')
            value = detector.predict_one(point)
            if not math.isfinite(value) or not 0. <= value <= 1.:
                raise RuntimeError('Registered streaming probability became invalid')
            yield value


def train(datasets, model_directory_path):
    from src.next.registered_training import train_frozen_candidate
    path = Path(__file__).resolve().parents[2]/'configs/next_registered_deployment.json'
    if not path.is_file():
        raise RuntimeError('Registered NEXT deployment selection is not frozen yet')
    return train_frozen_candidate(datasets, model_directory_path, json.loads(path.read_text(encoding='utf-8')))
