"""DEV-only, full multi-series API and unchanged official socket-path parity.

No target table is opened and no metric is computed. The wire measurement is
local loopback with a consumer thread, not cloud IPC or crunch_tools.test().
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.loader import series_frames
from src.validation.splits import FrozenSplit
from src.validation.protocol import protocol_predictions
from src.streaming.inference import infer
from src.streaming.prepared import source_hashes
from src.models.representations import bundle_class,deployment_kind
from src.utils.artifacts import write_json,sha256,utc_now
from scripts.verify_selected_contract import GuardedOnline


def guarded_multi_series(examples,directory):
    online=[GuardedOnline(points) for _,points in examples]
    generator=infer(iter((h,guard) for (h,_),guard in zip(examples,online)),directory)
    if next(generator) is not None or any(g.count for g in online):
        raise AssertionError('Ready must consume no online points')
    predictions=[]
    for (_,points),guard in zip(examples,online):
        for _ in points:
            value=next(generator)
            if not isinstance(value,float) or not np.isfinite(value) or not 0<=value<=1:
                raise AssertionError('One finite scalar prediction per point required')
            predictions.append(value);guard.acknowledge()
    try:next(generator)
    except StopIteration:pass
    else:raise AssertionError('Extra output after all series')
    return np.asarray(predictions,dtype=np.float32)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',action='append',required=True)
    parser.add_argument('--tag',required=True);parser.add_argument('--max-series-per-fold',type=int,default=0)
    parser.add_argument('--transport',choices=('thread','subprocess'),default='thread')
    args=parser.parse_args()
    if args.max_series_per_fold<0 or not args.tag.replace('_','').isalnum():raise ValueError('Invalid audit limits')
    split=FrozenSplit(ROOT/'artifacts/validation');examples=[[] for _ in range(5)];ids=[[] for _ in range(5)]
    seen=set()
    for sid,frame in series_frames(ROOT/'data/raw/X_train.parquet'):
        seen.add(sid);role,fold=split.assignment(sid)
        if role!='DEVELOPMENT_POOL':continue
        if args.max_series_per_fold and len(ids[fold])>=args.max_series_per_fold:continue
        historical=frame.loc[frame.period==1,'value'].to_numpy(dtype=np.float32)
        online=frame.loc[frame.period==2,'value'].to_numpy(dtype=np.float32)
        examples[fold].append((historical,online));ids[fold].append(sid)
    if seen!=set(split.mapping):raise AssertionError('Raw ID universe differs from frozen split')
    registry=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text())
    for eid in args.candidate:
        result=next(r for r in registry['results'] if r['experiment_id']==eid)
        representation=result['candidate'].get('representation','base');loader=bundle_class(representation)
        destination=ROOT/'artifacts/real/protocol'/args.tag/eid
        if (destination/'RESULTS.json').exists():raise RuntimeError('Existing protocol audit is immutable; choose another tag')
        report={'status':'RUNNING','candidate':eid,'timestamp':utc_now(),'folds':[],
            'scope':'Real DEV input only; current actual infer generator and pinned official Remote/double-protection over Windows loopback sockets. No target tables, score calculation, reduced input or cloud execution.',
            'transport':args.transport,'socket_topology':('one producer process and one fresh consumer Python process over actual loopback sockets' if args.transport=='subprocess' else 'one producer and one consumer thread in the same Python process, actual loopback sockets; includes GIL scheduling'),
            'timing_includes':'model loading/hash checks, ready preparation, per-series historical initialization, socket send/receive, scalar prediction and collection; input-file loading excluded',
            'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False,'target_tables_opened':False,
            'official_crunch_tools_test_called':False,'runner_sha256':sha256(ROOT/'vendor/official/runner.py'),
            'inference_source_sha256':sha256(ROOT/'src/streaming/inference.py'),'protocol_helper_sha256':sha256(ROOT/'src/validation/protocol.py'),
            'audit_script_sha256':sha256(__file__),'split_sha256':split.digest}
        if args.transport=='subprocess':
            report['subprocess_protocol_helper_sha256']=sha256(ROOT/'src/validation/subprocess_protocol.py')
            report['subprocess_worker_sha256']=sha256(ROOT/'scripts/official_socket_worker.py')
        oof=ROOT/'data/processed/oof'/eid;manifest=json.loads((oof/'MANIFEST.json').read_text());cache=ROOT/manifest['cache_directory']
        cache_manifest=json.loads((cache/'MANIFEST.json').read_text())
        if manifest['split_sha256']!=split.digest or cache_manifest['split_sha256']!=split.digest:raise AssertionError('OOF/cache split mismatch')
        for fold,batch in enumerate(examples):
            original=ROOT/'artifacts/models/real'/eid/f'fold_{fold}.joblib';model=loader.load(original)
            directory=ROOT/'artifacts/deployment/protocol'/args.tag/eid/f'fold_{fold}';directory.mkdir(parents=True,exist_ok=True)
            shutil.copy2(original,directory/'model.joblib')
            descriptor={'kind':deployment_kind(representation),'name':model.name,'feature_hash':model.feature_hash,
                'model_sha256':sha256(original),'use_fused_inference':True,'use_fast_historical_init':True,
                'status':'REAL_DEV_PROTOCOL_VERIFICATION_ONLY',**source_hashes(model,fast_initialization=True)}
            write_json(directory/'model.json',descriptor)
            coords=pd.read_parquet(cache/f'fold_{fold}.parquet',columns=['dataset_id','time_online'])
            count=sum(len(points) for _,points in batch);offset=0
            for sid,(_,points) in zip(ids[fold],batch):
                rows=coords.iloc[offset:offset+len(points)]
                if not np.all(rows.dataset_id.to_numpy()==sid) or not np.array_equal(rows.time_online.to_numpy(),np.arange(len(points))):
                    raise AssertionError('OOF coordinates disagree with the raw series order')
                offset+=len(points)
            if not args.max_series_per_fold and count!=len(coords):raise AssertionError('Incomplete DEV fold coverage')
            expected=np.load(oof/f'fold_{fold}.npy',allow_pickle=False)[:count]
            if len(expected)!=count:raise AssertionError('OOF prediction count mismatch')
            # Compile and import before both timings; each timed call still
            # reloads its artifact, verifies bytes and emits its own ready.
            guarded_multi_series([(batch[0][0],batch[0][1][:1])],directory)
            start=time.perf_counter();direct=guarded_multi_series(batch,directory);direct_seconds=time.perf_counter()-start
            np.testing.assert_array_equal(direct,expected)
            start=time.perf_counter();process_timing=None
            if args.transport=='subprocess':
                from src.validation.subprocess_protocol import subprocess_protocol_predictions
                wire,process_timing=subprocess_protocol_predictions(batch,directory)
            else:wire=protocol_predictions(infer,batch,directory)
            wire_seconds=time.perf_counter()-start
            np.testing.assert_array_equal(wire,expected)
            # Reverse a fixed prefix of series without reloading per series:
            # prediction chunks must move with their inputs, never their position.
            prefix=batch[:32];sizes=[len(points) for _,points in prefix]
            chunks=np.split(expected[:sum(sizes)],np.cumsum(sizes)[:-1])
            reversed_output=guarded_multi_series(list(reversed(prefix)),directory)
            np.testing.assert_array_equal(reversed_output,np.concatenate(list(reversed(chunks))))
            row={'fold':fold,'series':len(batch),'points':count,'direct_guarded_seconds':direct_seconds,'wire_seconds':wire_seconds,
                'wire_us_per_point':wire_seconds/count*1e6,'different_float32_rows':0,'max_abs_diff':0.,
                'prediction_sha256':hashlib.sha256(wire.tobytes()).hexdigest(),'reverse_order_series':len(prefix),'reverse_order_different_rows':0,
                'model_sha256':descriptor['model_sha256'],'model_descriptor_sha256':sha256(directory/'model.json'),**source_hashes(model,fast_initialization=True)}
            if process_timing:row['process_timing']=process_timing
            report['folds'].append(row);write_json(destination/'CHECKPOINT.json',report)
            print('REAL SOCKET FOLD PASS',eid,fold,count,'points',round(wire_seconds,3),'s',flush=True)
        report.update(status='PASS',completed_utc=utc_now(),total_series=sum(r['series'] for r in report['folds']),
            total_rows=sum(r['points'] for r in report['folds']),wire_seconds=sum(r['wire_seconds'] for r in report['folds']),
            direct_guarded_seconds=sum(r['direct_guarded_seconds'] for r in report['folds']),max_abs_diff=0.,different_float32_rows=0)
        report['wire_us_per_point']=report['wire_seconds']/report['total_rows']*1e6
        if args.transport=='subprocess':
            report['cold_start_to_ready_seconds']=[r['process_timing']['process_start_to_ready_seconds'] for r in report['folds']]
            report['after_ready_wire_us_per_point']=sum(r['process_timing']['ready_to_end_wire_seconds'] for r in report['folds'])/report['total_rows']*1e6
        report['ten_million_local_wire_extrapolated_seconds']=report['wire_us_per_point']*10
        report['extrapolation_limitation']='Measured on actual DEV series lengths and the stated loopback topology. Ten-million value is an extrapolation, not a ten-million wire run or cloud estimate.'
        write_json(destination/'RESULTS.json',report);print('REAL SOCKET COMPLETE',eid,report['total_rows'],report['wire_us_per_point'],'us/point',flush=True)


if __name__=='__main__':main()
