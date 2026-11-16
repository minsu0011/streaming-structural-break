"""Independently refit an immutable selected artifact from DEV raw in a new path.

Useful after extracting the handoff source archive. The archived original
artifact remains the numerical reference; no performance score is computed.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import argparse,gc,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.streaming.inference import train
from src.streaming.training import train_frozen_candidate
from src.models.representations import bundle_class,representation_feature_names
from src.validation.splits import FrozenSplit
from src.utils.artifacts import write_json,sha256,utc_now
from scripts.train_final_artifacts import development_training_iterator


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=('PRIMARY_DEV_CANDIDATE','BACKUP_ROBUST_CANDIDATE','BACKUP_FAST_CANDIDATE'),required=True)
    parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not args.tag.replace('_','').isalnum():raise ValueError('Simple reproduction tag required')
    frozen_path=ROOT/'DEV_CHAMPION_CANDIDATES.json';frozen=json.loads(frozen_path.read_text())
    if frozen['status']!='FROZEN_FOR_FINAL_DIAGNOSTICS':raise RuntimeError('Frozen DEV-only selection is required')
    split=FrozenSplit(ROOT/'artifacts/validation');item=frozen['candidates'][args.role]
    if frozen['split_sha256']!=split.digest:raise RuntimeError('Frozen split changed')
    spec_path=ROOT/item['specification_path']
    if sha256(spec_path)!=item['specification_sha256']:raise RuntimeError('Frozen training specification changed')
    spec=json.loads(spec_path.read_text());source=ROOT/item['model_directory'];destination=ROOT/'artifacts/real/final_model_reproduction'/args.tag/args.role
    if destination.exists():raise RuntimeError('New reproduction output path required')
    destination.mkdir(parents=True);original_descriptor=json.loads((source/'model.json').read_text())
    if sha256(source/'model.joblib')!=original_descriptor['model_sha256']:raise RuntimeError('Original final artifact bytes changed')
    started=time.perf_counter()
    if args.role=='PRIMARY_DEV_CANDIDATE':descriptor=train(development_training_iterator(split),destination)
    else:descriptor=train_frozen_candidate(development_training_iterator(split),destination,spec)
    if descriptor['seal_rows_fitted']!=0 or descriptor['development_series_fitted']!=8000 or descriptor['seal_series_skipped_before_array_or_label_access']!=2000:
        raise RuntimeError('Independent raw refit inspected the wrong data scope')
    representation=item['candidate'].get('representation','base');loader=bundle_class(representation)
    original=loader.load(source/'model.joblib');reproduced=loader.load(destination/'model.joblib')
    names=representation_feature_names(representation,original.feature_config)
    manifest_path=ROOT/'data/processed/oof'/item['experiment_id']/'MANIFEST.json'
    if not manifest_path.exists():manifest_path=ROOT/'artifacts/handoff/selected_oof_manifests'/item['experiment_id']/'MANIFEST.json'
    manifest=json.loads(manifest_path.read_text());cache=ROOT/manifest['cache_directory'];cache_manifest=json.loads((cache/'MANIFEST.json').read_text())
    rows=[]
    for fold in range(5):
        path=cache/f'fold_{fold}.parquet'
        if sha256(path)!=cache_manifest['fold_sha256'][str(fold)]:raise RuntimeError('Reconstructed reference cache bytes changed')
        frame=pd.read_parquet(path,columns=list(names));matrix=frame.to_numpy(dtype=np.float32)
        left=original.predict(matrix).astype(np.float32);right=reproduced.predict(matrix).astype(np.float32)
        np.testing.assert_array_equal(left,right)
        rows.append({'cache_partition':fold,'rows':len(left),'different_float32_rows':0,'prediction_max_abs_diff':0.,'feature_file_sha256':sha256(path)})
        del frame,matrix,left,right;gc.collect()
    report={'status':'PASS','timestamp':utc_now(),'role':args.role,'candidate':item['experiment_id'],'selection_sha256':sha256(frozen_path),
        'original_model_sha256':sha256(source/'model.joblib'),'reproduced_model_sha256':sha256(destination/'model.joblib'),
        'model_bytes_identical':sha256(source/'model.joblib')==sha256(destination/'model.joblib'),'total_rows':sum(r['rows'] for r in rows),'partitions':rows,
        'elapsed_seconds':time.perf_counter()-started,'development_series_fitted':8000,'seal_series_skipped_before_access':2000,'seal_rows_fitted':0,
        'scope':'Independent raw all-DEV refit with frozen specification, compared to packaged original final artifact on all reconstructed DEV features. No performance score; no reduced input; seal placeholders are skipped before access.',
        'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False,'training_source_sha256':sha256(ROOT/'src/streaming/training.py'),'script_sha256':sha256(__file__)}
    write_json(destination/'RESULTS.json',report);print('INDEPENDENT FINAL RAW REFIT PASS',args.role,report['total_rows'],'rows; model bytes equal',report['model_bytes_identical'],flush=True)


if __name__=='__main__':main()
