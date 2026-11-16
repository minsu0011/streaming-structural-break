"""Full actual-artifact stream/batch parity without computing training performance.

These models were fitted on all DEV. The comparison is numerical engineering
only, never an additional validation score or generalization estimate.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.loader import series_frames
from src.validation.splits import FrozenSplit
from src.models.representations import bundle_class,representation_feature_names
from src.streaming.prepared import source_hashes
from src.validation.subprocess_protocol import subprocess_protocol_predictions
from scripts.verify_real_protocol import guarded_multi_series
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--primary-socket',action='store_true');args=parser.parse_args()
    frozen=json.loads((ROOT/'DEV_CHAMPION_CANDIDATES.json').read_text())
    if frozen['status']!='FROZEN_FOR_FINAL_DIAGNOSTICS':raise RuntimeError('Frozen actual DEV artifacts required')
    split=FrozenSplit(ROOT/'artifacts/validation');examples=[];ids=[]
    for sid,frame in series_frames(ROOT/'data/raw/X_train.parquet'):
        if split.assignment(sid)[0]!='DEVELOPMENT_POOL':continue
        examples.append((frame.loc[frame.period==1,'value'].to_numpy(dtype=np.float32),frame.loc[frame.period==2,'value'].to_numpy(dtype=np.float32)));ids.append(sid)
    if set(ids)!=set(split.frame.loc[split.frame.role=='DEVELOPMENT_POOL','dataset_id']):raise AssertionError('DEV input coverage differs')
    for role,item in frozen['candidates'].items():
        eid=item['experiment_id'];directory=ROOT/item['model_directory'];descriptor=json.loads((directory/'model.json').read_text())
        model_path=directory/'model.joblib';representation=item['candidate'].get('representation','base');model=bundle_class(representation).load(model_path)
        receipt=source_hashes(model,fast_initialization=True);descriptor_sha=sha256(directory/'model.json');socket_required=role=='PRIMARY_DEV_CANDIDATE' and args.primary_socket
        if descriptor.get('candidate')!=item['candidate'] or descriptor['model_sha256']!=sha256(model_path) or descriptor['seal_rows_fitted']!=0:
            raise AssertionError('Final artifact identity/scope differs')
        if not descriptor.get('use_fused_inference') or not descriptor.get('use_fast_historical_init') or any(descriptor.get(k)!=v for k,v in receipt.items()):
            raise AssertionError('Actual deployment source mode differs')
        report_path=ROOT/'artifacts/real/final_streaming'/role/'RESULTS.json'
        if report_path.exists():
            old=json.loads(report_path.read_text())
            if old['status']!='PASS' or old['model_descriptor_sha256']!=descriptor_sha or old['inference_source_sha256']!=sha256(ROOT/'src/streaming/inference.py') or any(old.get(k)!=v for k,v in receipt.items()):raise RuntimeError('Existing actual-artifact evidence changed')
            if socket_required and not old.get('primary_socket_pass'):raise RuntimeError('Existing actual-artifact report lacks the requested primary socket audit')
            print('REUSE ACTUAL FINAL STREAM',role,flush=True);continue
        names=representation_feature_names(representation,model.feature_config)
        manifest=json.loads((ROOT/'data/processed/oof'/eid/'MANIFEST.json').read_text());cache=ROOT/manifest['cache_directory'];cache_manifest=json.loads((cache/'MANIFEST.json').read_text())
        if manifest['split_sha256']!=split.digest:raise AssertionError('Reference cache split differs')
        predicted={};batch_started=time.perf_counter()
        for fold in range(5):
            path=cache/f'fold_{fold}.parquet'
            if sha256(path)!=cache_manifest['fold_sha256'][str(fold)]:raise AssertionError('Reference cache bytes changed')
            frame=pd.read_parquet(path,columns=['dataset_id','time_online',*names])
            values=model.predict(frame.loc[:,names].to_numpy(dtype=np.float32)).astype(np.float32)
            for sid,positions in frame.groupby('dataset_id',sort=False).indices.items():
                if sid in predicted or split.assignment(int(sid))!=('DEVELOPMENT_POOL',fold):raise AssertionError('Reference series scope/order differs')
                if not np.array_equal(frame.time_online.to_numpy()[positions],np.arange(len(positions))):raise AssertionError('Reference temporal coordinates differ')
                predicted[int(sid)]=values[positions].copy()
        if set(predicted)!=set(ids):raise AssertionError('Batch reference did not cover all DEV series')
        expected=np.concatenate([predicted[sid] for sid in ids]);batch_seconds=time.perf_counter()-batch_started
        started=time.perf_counter();actual=guarded_multi_series(examples,directory);live_seconds=time.perf_counter()-started
        np.testing.assert_array_equal(actual,expected)
        report={'status':'PASS','timestamp':utc_now(),'role':role,'candidate':eid,'total_series':len(ids),'total_rows':len(expected),
            'batch_reference_seconds_including_cache_io':batch_seconds,'actual_guarded_infer_seconds':live_seconds,'different_float32_rows':0,'max_abs_diff':0.,
            'prediction_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'model_sha256':sha256(model_path),'model_descriptor_sha256':descriptor_sha,
            'inference_source_sha256':sha256(ROOT/'src/streaming/inference.py'),'split_sha256':split.digest,**receipt,
            'scope':'Actual all-DEV-refit artifact, full DEV raw streaming input versus that same artifact on historical-only cached features. No target column is loaded, no score is calculated, and no performance claim is inferred from training inputs.',
            'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False,'target_columns_loaded':False,'primary_socket_pass':False,
            'script_sha256':sha256(__file__)}
        if socket_required:
            wire,timing=subprocess_protocol_predictions(examples,directory);np.testing.assert_array_equal(wire,expected)
            report.update(primary_socket_pass=True,primary_socket_timing=timing,primary_socket_different_float32_rows=0,
                runner_sha256=sha256(ROOT/'vendor/official/runner.py'),
                subprocess_protocol_helper_sha256=sha256(ROOT/'src/validation/subprocess_protocol.py'),
                subprocess_worker_sha256=sha256(ROOT/'scripts/official_socket_worker.py'))
        write_json(report_path,report);print('ACTUAL FINAL STREAM PASS',role,len(expected),'points; socket',report['primary_socket_pass'],flush=True)


if __name__=='__main__':main()
