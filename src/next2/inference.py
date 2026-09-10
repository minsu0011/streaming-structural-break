"""Reviewed read-only NEXT2 single-model and fixed-blend streaming API."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from src.next2.runtime import prepare,source_hash as runtime_hash
from src.next2.io import model_class
from src.next2.blend import FixedBlendBundle
from src.next.readonly_jit import disable_loaded_disk_caches,source_hash as readonly_hash
from src.utils.artifacts import sha256,write_json,utc_now

INFER_PARALLELISM=1
disable_loaded_disk_caches()


def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+readonly_hash().encode()).hexdigest()


def write_descriptor(directory,spec,*,origin,training=None):
    directory=Path(directory)
    if (directory/'model.json').exists():
        raise RuntimeError('Refusing to overwrite a deployment descriptor')
    descriptor={'kind':'NEXT2_REVIEWED_STREAM','bundle_kind':spec['bundle_kind'],'specification':spec,
        'model_sha256':sha256(directory/'model.joblib'),'runtime_sha256':runtime_hash(),
        'inference_sha256':implementation_hash(),'origin':origin,'training':training,
        'created_utc':utc_now(),'seal_rows_fitted':0,'reduced_new_usage':0}
    write_json(directory/'model.json',descriptor)
    load_artifact(directory)
    return descriptor


def load_artifact(directory):
    directory=Path(directory)
    descriptor=json.loads((directory/'model.json').read_text(encoding='utf-8'))
    if descriptor.get('kind')!='NEXT2_REVIEWED_STREAM' or descriptor.get('bundle_kind') not in ('single','fixed_blend'):
        raise RuntimeError('Explicit reviewed NEXT2 artifact required')
    if descriptor['inference_sha256']!=implementation_hash() or descriptor['runtime_sha256']!=runtime_hash():
        raise RuntimeError('NEXT2 runtime or inference source changed')
    if sha256(directory/'model.joblib')!=descriptor['model_sha256']:
        raise RuntimeError('NEXT2 model bytes changed')
    spec=descriptor['specification']
    if spec['bundle_kind']!=descriptor['bundle_kind'] or spec['status']!='FROZEN_NEXT2_DEPLOYMENT':
        raise RuntimeError('Deployment recipe identity differs')
    if descriptor['bundle_kind']=='fixed_blend':
        model=FixedBlendBundle.load(directory/'model.joblib')
        if model.weight!=spec['weight'] or model.primary.extension!=spec['primary']['extension'] or model.complement.extension!=spec['complement']['extension']:
            raise RuntimeError('Blend component feature settings or weight differ')
        from src.next2.final_fitting import training_feature_names
        if tuple(model.primary.names)!=training_feature_names(spec['primary']) or tuple(model.complement.names)!=training_feature_names(spec['complement']):
            raise RuntimeError('Blend named feature contract differs')
    else:
        candidate=spec['candidate']
        model=model_class(candidate).load(directory/'model.joblib')
        from src.next2.final_fitting import training_feature_names
        if model.extension!=candidate['extension'] or tuple(model.names)!=training_feature_names(candidate):
            raise RuntimeError('Single model feature contract differs')
    return model,descriptor


def infer(datasets,model_directory_path):
    disable_loaded_disk_caches()
    model,_=load_artifact(model_directory_path)
    prepared=prepare(model)
    index=np.arange(128,dtype=np.float64)
    warm=(np.sin(index*.37)+.2*np.cos(index*.11)).astype(np.float32)
    prepared.new_state(warm).predict_one(0.)
    yield None
    for historical,online in datasets:
        state=prepared.new_state(historical)
        for point in online:
            value=float(point)
            if not math.isfinite(value):
                raise ValueError('Finite online observation required')
            prediction=state.predict_one(value)
            if not math.isfinite(prediction) or not 0.<=prediction<=1.:
                raise RuntimeError('NEXT2 streaming output is not a finite probability')
            yield prediction
