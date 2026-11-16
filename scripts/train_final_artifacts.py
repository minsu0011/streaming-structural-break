"""Fit frozen candidates on DEV only, including a second independent primary fit."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.streaming.inference import train
from src.streaming.training import train_frozen_candidate
from src.models.representations import bundle_class
from src.models.representations import representation_feature_names
from src.features.cache import read_fold
from src.data.loader import load_training
from src.validation.splits import FrozenSplit
from src.utils.artifacts import write_json, sha256, utc_now


class UnopenedSeal:
    def __array__(self,*args,**kwargs):raise AssertionError('Seal values or targets were inspected')
    def __len__(self):raise AssertionError('Seal length was inspected')
    def __iter__(self):raise AssertionError('Seal values were iterated')
    def __float__(self):raise AssertionError('Seal target was inspected')


def development_training_iterator(split):
    # This loader materializes tau metadata for DEV IDs only. Seal placeholders
    # exercise the official train skip path without inspecting sealed metadata.
    for series in load_training(ROOT/'data/raw',split=split):
        yield series.dataset_id,series.historical,series.online,series.tau
    for sid in sorted(split.frame.loc[split.frame.role=='LOCAL_FINAL_SEAL','dataset_id'].astype(int)):
        yield sid,UnopenedSeal(),UnopenedSeal(),UnopenedSeal()


def main():
    frozen_path=ROOT/'DEV_CHAMPION_CANDIDATES.json';frozen=json.loads(frozen_path.read_text())
    if frozen.get('status')!='FROZEN_FOR_FINAL_DIAGNOSTICS':raise RuntimeError('Final DEV selection must already be frozen')
    split=FrozenSplit(ROOT/'artifacts/validation')
    if split.digest!=frozen['split_sha256']:raise RuntimeError('Frozen candidate and split identity differ')
    result_path=ROOT/'artifacts/real/FINAL_REFIT_REAL.json'
    report=json.loads(result_path.read_text()) if result_path.exists() else {'status':'RUNNING','timestamp':utc_now(),'roles':{},
        'selection_sha256':sha256(frozen_path),'split_sha256':split.digest,'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    if report['selection_sha256']!=sha256(frozen_path):raise RuntimeError('Refit selection changed')
    for role,selected in frozen['candidates'].items():
        spec_path=ROOT/selected['specification_path']
        if sha256(spec_path)!=selected['specification_sha256']:raise RuntimeError('Frozen training specification changed')
        spec=json.loads(spec_path.read_text());directory=ROOT/selected['model_directory']
        if role in report['roles']:
            if sha256(directory/'model.joblib')!=report['roles'][role]['model_sha256']:raise RuntimeError('Previously completed refit changed')
            continue
        if (directory/'model.joblib').exists():raise RuntimeError('Unrecorded final model exists; inspect it before retrying')
        started=time.perf_counter()
        if role=='PRIMARY_DEV_CANDIDATE':descriptor=train(development_training_iterator(split),directory)
        else:descriptor=train_frozen_candidate(development_training_iterator(split),directory,spec)
        if descriptor['seal_rows_fitted']!=0 or descriptor['development_series_fitted']!=len(spec['development_ids']):raise RuntimeError('Final fit scope mismatch')
        report['roles'][role]={**descriptor,'elapsed_seconds':time.perf_counter()-started,'model_directory':str(directory.relative_to(ROOT)),
            'descriptor_sha256':sha256(directory/'model.json'),'official_train_entrypoint':role=='PRIMARY_DEV_CANDIDATE'}
        write_json(result_path,report);print('FINAL DEV REFIT',role,descriptor['training_rows'],'rows',flush=True);gc.collect()
    primary=frozen['candidates']['PRIMARY_DEV_CANDIDATE'];primary_id=primary['experiment_id']
    reproduction=ROOT/'artifacts/deployment/final_reproduction/PRIMARY_DEV_CANDIDATE'
    if 'primary_independent_refit' not in report:
        if (reproduction/'model.joblib').exists():raise RuntimeError('Unrecorded reproduction artifact exists; inspect before retrying')
        started=time.perf_counter();second=train(development_training_iterator(split),reproduction)
        original_path=ROOT/primary['model_directory']/'model.joblib';second_path=reproduction/'model.joblib'
        representation=primary['candidate'].get('representation','base');loader=bundle_class(representation)
        first_model=loader.load(original_path);second_model=loader.load(second_path)
        manifest=json.loads((ROOT/'data/processed/oof'/primary_id/'MANIFEST.json').read_text());cache=ROOT/manifest['cache_directory']
        cache_manifest=json.loads((cache/'MANIFEST.json').read_text());rows=0;different=0;maximum=0.
        for k in range(5):
            if representation=='base':frame=read_fold(cache,k,split=split,config=first_model.feature_config)
            else:
                path=cache/f'fold_{k}.parquet'
                if sha256(path)!=cache_manifest['fold_sha256'][str(k)]:raise RuntimeError('Final verification feature cache changed')
                frame=pd.read_parquet(path,columns=list(representation_feature_names(primary['candidate']['representation'],first_model.feature_config)))
            matrix=frame.loc[:,representation_feature_names(primary['candidate']['representation'],first_model.feature_config)].to_numpy(dtype=np.float32)
            left=first_model.predict(matrix).astype(np.float32);right=second_model.predict(matrix).astype(np.float32)
            rows+=len(left);different+=int(np.count_nonzero(left!=right));maximum=max(maximum,float(np.max(abs(left-right))))
            del matrix,frame,left,right;gc.collect()
        if different:raise RuntimeError('Independent final primary refit is not deterministic')
        report['primary_independent_refit']={'status':'PASS','rows_compared':rows,'different_float32_rows':different,'prediction_max_abs_diff':maximum,
            'original_model_sha256':sha256(original_path),'reproduced_model_sha256':sha256(second_path),
            'model_bytes_equal':sha256(original_path)==sha256(second_path),'elapsed_seconds':time.perf_counter()-started,
            'scope':'Two independent all-DEV fits through main train; same frozen configuration and series order. Prediction equality on all DEV cached features; no training-set performance score calculated.'}
    report['status']='PASS';report['completed_utc']=utc_now();write_json(result_path,report)
    print('FINAL REFIT DETERMINISM PASS',report['primary_independent_refit']['rows_compared'],'rows',flush=True)


if __name__=='__main__':main()
