"""Official train/infer API. Default artifact is an untrained engineering baseline."""
import json
from pathlib import Path
import numpy as np
from src.features.streaming import StreamingFeatureState,feature_version
from src.models.baselines import BaselineDetector
from src.models.learners import ModelBundle
from src.features.config import CONFIG
from src.utils.artifacts import sha256

INFER_PARALLELISM=1

def write_engineering_baseline(model_directory_path):
    """Explicit synthetic-test fixture, independent of any production selection."""
    directory=Path(model_directory_path)
    directory.mkdir(parents=True,exist_ok=True)
    descriptor={'kind':'baseline','name':'B7_manual','feature_hash':feature_version(),'status':'ENGINEERING_BASELINE_NOT_DEV_CHAMPION'}
    (directory/'model.json').write_text(json.dumps(descriptor,indent=2),encoding='utf-8')

def train(datasets,model_directory_path):
    """Refit a frozen DEV selection; use the engineering fixture until final freeze."""
    deployment=Path(__file__).resolve().parents[2]/'configs/deployment.json'
    if deployment.exists():
        from src.streaming.training import train_frozen_candidate
        return train_frozen_candidate(datasets,model_directory_path,json.loads(deployment.read_text(encoding='utf-8')))
    return write_engineering_baseline(model_directory_path)

def infer(datasets,model_directory_path):
    directory=Path(model_directory_path)
    descriptor=json.loads((directory/'model.json').read_text(encoding='utf-8'))
    if descriptor.get('model_sha256') and sha256(directory/'model.joblib')!=descriptor['model_sha256']:
        raise RuntimeError('Model artifact byte hash mismatch')
    if descriptor['kind']=='ar8_arch1_calibrated':
        from src.models.arch_calibrated import ARCHCalibratedBundle
        bundle=ARCHCalibratedBundle.load(directory/'model.joblib')
    elif descriptor['kind']=='cross_order_equal':
        from src.models.cross_order import CrossOrderBlendBundle
        bundle=CrossOrderBlendBundle.load(directory/'model.joblib')
    elif descriptor['kind']=='ar_residual_calibrated':
        from src.models.ar_calibrated import ARCalibratedBundle
        bundle=ARCalibratedBundle.load(directory/'model.joblib')
    elif descriptor['kind']=='ar_order_calibrated':
        from src.models.ar_order_calibrated import AROrderCalibratedBundle
        bundle=AROrderCalibratedBundle.load(directory/'model.joblib')
    elif descriptor['kind']=='historical_calibrated':
        from src.models.calibrated import CalibratedBundle
        bundle=CalibratedBundle.load(directory/'model.joblib')
    elif descriptor['kind']=='supervised':bundle=ModelBundle.load(directory/'model.joblib')
    elif descriptor['kind']=='baseline':bundle=None
    else:raise ValueError('Unknown deployment representation')
    config=bundle.feature_config if bundle is not None else CONFIG
    expected=bundle.feature_hash if bundle is not None else feature_version(config)
    if descriptor['feature_hash']!=expected:
        raise RuntimeError('Feature contract mismatch')
    def make_state(historical):
        return bundle.make_state(historical) if bundle is not None and hasattr(bundle,'make_state') else StreamingFeatureState(historical,config=config)
    prepared=None
    if descriptor.get('use_fused_inference'):
        from src.streaming.prepared import prepare_model,source_hashes
        fast=bool(descriptor.get('use_fast_historical_init'))
        for key,value in source_hashes(bundle,fast_initialization=fast).items():
            if descriptor.get(key)!=value:
                if key=='initialization_source_hash':raise RuntimeError('Historical initialization implementation hash mismatch')
                raise RuntimeError('Production source hash mismatch: '+key)
        prepared=prepare_model(bundle,fast_initialization=fast)
    elif descriptor.get('use_fast_historical_init'):raise RuntimeError('Fast historical initialization requires the fused inference path')
    # Compile the in-memory kernel before ready. cache=False prevents JIT disk writes.
    if bundle is not None or descriptor['name'] not in ('B0_constant','B1_official_ewma'):
        if prepared is not None:prepared.new_state(np.array([0.,1.])).predict_one(0.)
        else:
            dummy=make_state(np.array([0.,1.])).update_and_get(0.)
            if bundle is not None:bundle.predict_one(dummy)
    yield
    for historical,online in datasets:
        if prepared is not None:
            detector=prepared.new_state(historical)
            for point in online:yield detector.predict_one(point)
        elif bundle is None:
            detector=BaselineDetector(descriptor['name'],historical)
            for point in online:
                yield detector.predict_one(point)
        else:
            state=make_state(historical)
            for point in online:
                yield bundle.predict_one(state.update_and_get(point))
