"""Read-only streaming entry for deterministic inference-only model files."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from src.next2.canonical_model import load,save,source_hash as format_hash
from src.next2.runtime import prepare,source_hash as runtime_hash
from src.next2.used_feature_runtime import PreparedUsedPredictor,source_hash as used_hash
from src.next2.blend import FixedBlendBundle
from src.next.readonly_jit import disable_loaded_disk_caches,source_hash as readonly_hash
from src.utils.artifacts import write_json,sha256,utc_now

INFER_PARALLELISM=1
disable_loaded_disk_caches()


def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+readonly_hash().encode()).hexdigest()


def prepare_canonical(model,runtime):
    if runtime=='compact_v1':
        return prepare(model)
    if runtime=='tree_used_v1' and not isinstance(model,FixedBlendBundle):
        return PreparedUsedPredictor(model)
    raise ValueError('Canonical runtime outside the reviewed finite options')


def expected_runtime_hash(runtime):
    if runtime=='compact_v1':
        return runtime_hash()
    if runtime=='tree_used_v1':
        return used_hash()
    raise ValueError('Unknown canonical runtime')


def write_artifact(model,directory,spec,*,runtime,origin,training=None):
    directory=Path(directory)
    if (directory/'model.json').exists():
        raise RuntimeError('Refusing to overwrite a canonical deployment descriptor')
    prepare_canonical(model,runtime)
    save(model,directory/'model.sbnext2')
    descriptor={'kind':'NEXT2_CANONICAL_DEPLOYMENT','specification':spec,'runtime':runtime,
        'canonical_format_sha256':format_hash(),'runtime_sha256':expected_runtime_hash(runtime),
        'inference_sha256':implementation_hash(),'model_sha256':sha256(directory/'model.sbnext2'),
        'origin':origin,'training':training,'created_utc':utc_now(),'seal_rows_fitted':0,'reduced_new_usage':0}
    write_json(directory/'model.json',descriptor)
    load_artifact(directory)
    return descriptor


def load_artifact(directory):
    directory=Path(directory)
    descriptor=json.loads((directory/'model.json').read_text(encoding='utf-8'))
    if descriptor.get('kind')!='NEXT2_CANONICAL_DEPLOYMENT' or descriptor['canonical_format_sha256']!=format_hash():
        raise RuntimeError('Canonical deployment format identity changed')
    if descriptor['inference_sha256']!=implementation_hash() or descriptor['runtime_sha256']!=expected_runtime_hash(descriptor['runtime']):
        raise RuntimeError('Canonical runtime or inference code changed')
    if sha256(directory/'model.sbnext2')!=descriptor['model_sha256']:
        raise RuntimeError('Canonical model bytes changed')
    model=load(directory/'model.sbnext2')
    spec=descriptor['specification']
    if spec.get('status')!='FROZEN_NEXT2_DEPLOYMENT':
        raise RuntimeError('Explicit frozen canonical recipe required')
    from src.next2.final_fitting import training_feature_names
    if spec['bundle_kind']=='fixed_blend':
        if not isinstance(model,FixedBlendBundle) or model.weight!=spec['weight']:
            raise RuntimeError('Fixed canonical blend weight differs')
        pairs=((model.primary,spec['primary']),(model.complement,spec['complement']))
    elif spec['bundle_kind']=='single' and not isinstance(model,FixedBlendBundle):
        pairs=((model,spec['candidate']),)
    else:
        raise RuntimeError('Canonical bundle kind differs')
    for component,candidate in pairs:
        if component.extension!=candidate['extension'] or tuple(component.names)!=training_feature_names(candidate):
            raise RuntimeError('Canonical named feature contract differs')
    return model,descriptor


def infer(datasets,model_directory_path):
    disable_loaded_disk_caches()
    model,descriptor=load_artifact(model_directory_path)
    prepared=prepare_canonical(model,descriptor['runtime'])
    index=np.arange(128,dtype=np.float64)
    historical=(np.sin(index*.37)+.2*np.cos(index*.11)).astype(np.float32)
    prepared.new_state(historical).predict_one(0.)
    yield None
    for historical,online in datasets:
        state=prepared.new_state(historical)
        for point in online:
            value=float(point)
            if not math.isfinite(value):
                raise ValueError('Finite current online observation required')
            prediction=state.predict_one(value)
            if not math.isfinite(prediction) or not 0.<=prediction<=1.:
                raise RuntimeError('Invalid canonical streaming probability')
            yield prediction


def train_role(datasets,directory,role):
    if role not in ('DISTILLED','PERFORMANCE'):
        raise ValueError('Explicit reviewed role required')
    directory=Path(directory)
    if (directory/'model.sbnext2').exists() or (directory/'model.json').exists():
        raise RuntimeError('Refusing to overwrite a canonical model')
    from src.next2.deployment_training import train
    from src.next2.inference import load_artifact as load_training
    root=Path(__file__).resolve().parents[2]
    spec=json.loads((root/'configs/next2'/('DEPLOYMENT_RESEARCH_'+role+'.json')).read_text(encoding='utf-8'))
    scratch=directory/'training_joblib'
    descriptor=train(datasets,scratch,spec,root=root)
    model,_=load_training(scratch)
    return write_artifact(model,directory,spec,runtime='tree_used_v1' if role=='DISTILLED' else 'compact_v1',
        origin='OFFICIAL_ITERATOR_TRAIN_CANONICAL',training=descriptor['training'])
