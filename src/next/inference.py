"""Separate NEXT train/infer entry, preserving the previous deployed entry."""
import sys
# Prevent lazy third-party Python imports from writing bytecode in infer().
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from src.next.candidate_io import model_class
from src.next.fast_variance_initialization import PreparedFastVariancePredictor,source_hash as runtime_hash
from src.next.readonly_jit import disable_loaded_disk_caches,source_hash as readonly_hash
from src.next.pruned_variance_runtime import supports_pruned_pair
from src.utils.artifacts import sha256


INFER_PARALLELISM = 1
# Decorator imports happen at module initialization. Disk caching is disabled
# before infer; the readiness warm-up compiles code in memory only.
disable_loaded_disk_caches()


def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+readonly_hash().encode()).hexdigest()


def load_artifact(directory):
    directory = Path(directory)
    descriptor = json.loads((directory/'model.json').read_text(encoding='utf-8'))
    if descriptor.get('kind')!='NEXT_VARIANCE_PAIR' or descriptor.get('runtime')!='pruned_variance_fast_h':
        raise RuntimeError('This entry accepts only the explicitly verified NEXT variance-pair runtime')
    if descriptor.get('inference_sha256')!=implementation_hash() or descriptor.get('runtime_sha256')!=runtime_hash():
        raise RuntimeError('NEXT inference/runtime source changed')
    if sha256(directory/'model.joblib')!=descriptor['model_sha256']:
        raise RuntimeError('NEXT model artifact bytes changed')
    candidate = descriptor['candidate']
    if candidate.get('extension',{}).get('kind')!='conditional_variance':
        raise RuntimeError('An unregistered feature bank cannot enter this production adapter')
    model = model_class(candidate).load(directory/'model.joblib')
    if not supports_pruned_pair(model) or model.extension!=candidate['extension']:
        raise RuntimeError('NEXT candidate and model feature bank differ')
    return model,descriptor


def infer(datasets,model_directory_path):
    # All project dispatchers used by this narrow runtime are imported above.
    # Reapply for a caller that performed training after importing this module.
    disable_loaded_disk_caches()
    model,descriptor = load_artifact(model_directory_path)
    prepared = PreparedFastVariancePredictor(model)
    # Compile in memory before readiness without touching either input iterator.
    index = np.arange(128,dtype=np.float64)
    warm_history = (np.sin(index*.37)+.2*np.cos(index*.11)).astype(np.float32)
    prepared.new_state(warm_history).predict_one(0.)
    yield None
    for historical,online in datasets:
        detector = prepared.new_state(historical)
        for point in online:
            point = float(point)
            if not math.isfinite(point):
                raise ValueError('A finite online observation is required')
            prediction = detector.predict_one(point)
            if not math.isfinite(prediction) or not 0.<=prediction<=1.:
                raise RuntimeError('NEXT streaming probability became invalid')
            yield prediction


def train(datasets,model_directory_path):
    from src.next.training import train_frozen_candidate
    spec_path = Path(__file__).resolve().parents[2]/'configs/next_deployment.json'
    if not spec_path.is_file():
        raise RuntimeError('NEXT deployment selection is not frozen yet')
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    return train_frozen_candidate(datasets,model_directory_path,spec)
