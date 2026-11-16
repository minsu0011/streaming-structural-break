"""Replay all DEV series through fused inference and compare untouched saved OOF."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.learners import ModelBundle
from src.models.calibrated import CalibratedBundle
from src.streaming.fused import PreparedFusedPredictor,implementation_hash
from src.features.streaming import StreamingFeatureState
from src.validation.splits import FrozenSplit
from src.data.loader import load_training
from src.utils.artifacts import write_json,utc_now,sha256
from scripts.verify_engineering import benchmark

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);parser.add_argument('--fast-initialization',action='store_true')
    parser.add_argument('--tag',help='Keep an independent engineering replay instead of replacing the current production receipt')
    args=parser.parse_args();eid=args.candidate
    if args.tag and not args.tag.replace('_','').isalnum():raise ValueError('Simple replay tag required')
    output=ROOT/'artifacts/real/fused'
    if args.tag:output=output/args.tag
    output=output/f'{eid}{"_fast_init" if args.fast_initialization else ""}.json'
    if args.tag and output.exists():raise RuntimeError('Tagged replay already exists')
    state=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text());result=next(r for r in state['results'] if r['experiment_id']==eid)
    representation=result.get('representation',result.get('candidate',{}).get('representation','base'))
    from src.models.representations import bundle_class
    loader=bundle_class(representation)
    models=[loader.load(ROOT/'artifacts/models/real'/eid/f'fold_{k}.joblib') for k in range(5)]
    from src.streaming.prepared import prepare_model,source_hashes
    prepared=[prepare_model(m,fast_initialization=args.fast_initialization) for m in models]
    split=FrozenSplit(ROOT/'artifacts/validation');oof=ROOT/'data/processed/oof'/eid
    manifest=json.loads((oof/'MANIFEST.json').read_text());cache=ROOT/manifest['cache_directory'];cache_manifest=json.loads((cache/'MANIFEST.json').read_text())
    expected=[np.load(oof/f'fold_{k}.npy',mmap_mode='r',allow_pickle=False) for k in range(5)]
    identity=[]
    for fold in range(5):
        paths=[cache/s['name'] for s in cache_manifest['shards'] if s['fold']==fold] if 'shards' in cache_manifest else [cache/f'fold_{fold}.parquet']
        identity.append(pd.concat([pd.read_parquet(path,columns=['dataset_id','time_online']) for path in paths],ignore_index=True))
    offsets=[0]*5;counts=[0]*5;differences=[0.]*5;changed=[0]*5;sample=None;start=time.perf_counter();initial_bytes=None;final_bytes=None
    for number,series in enumerate(load_training(ROOT/'data/raw',split=split),1):
        fold=split.assignment(series.dataset_id)[1];h=series.historical.astype(np.float32);o=series.online.astype(np.float32)
        if sample is None:sample=(h,o,fold)
        detector=prepared[fold].new_state(h);initial_bytes=detector.state_array_bytes
        actual=np.array([detector.predict_one(point) for point in o],dtype=np.float32);final_bytes=detector.state_array_bytes
        if initial_bytes!=final_bytes:raise RuntimeError('Fused state grows with online length')
        offset=offsets[fold];reference=expected[fold][offset:offset+len(o)]
        rows=identity[fold].iloc[offset:offset+len(o)]
        if not np.all(rows.dataset_id.to_numpy()==series.dataset_id) or not np.array_equal(rows.time_online.to_numpy(),np.arange(len(o))):raise RuntimeError('Saved OOF row identity mismatch')
        differences[fold]=max(differences[fold],float(np.max(abs(reference-actual))));changed[fold]+=int(np.count_nonzero(reference!=actual));counts[fold]+=1;offsets[fold]+=len(o)
        if number%2000==0:print(f'Fused DEV replay: {number} series; max float32 difference {max(differences)}',flush=True)
    if offsets!=[len(p) for p in expected]:raise RuntimeError('Fused replay did not cover all saved OOF rows')
    elapsed=time.perf_counter()-start;h,o,fold=sample;points=np.tile(o,max(1,10000//len(o)+1))[:10000]
    def native_factory():
        model=models[fold];state=model.make_state(h) if hasattr(model,'make_state') else StreamingFeatureState(h,config=model.feature_config)
        return lambda point:model.predict_one(state.update_and_get(point))
    native_latency=benchmark(native_factory,points,repeats=5)
    fused_latency=benchmark(lambda:prepared[fold].new_state(h).predict_one,points,repeats=5)
    report={'status':'PASS' if max(differences)==0 and sum(changed)==0 else 'FAIL','timestamp':utc_now(),'candidate':eid,'representation':representation,
        'fold_rows':offsets,'fold_series':counts,'fold_max_abs_diff':differences,'different_float32_rows':changed,
        'total_rows':sum(offsets),'total_series':sum(counts),'full_replay_elapsed_seconds':elapsed,
        'fused_source_hash':implementation_hash(),'split_sha256':split.digest,
        'fast_historical_initialization':args.fast_initialization,**source_hashes(models[0],fast_initialization=args.fast_initialization),
        'native_latency':native_latency,'fused_latency':fused_latency,'shared_cpu_load':True,'final_isolated_benchmark_required':True,
        'state_array_bytes':initial_bytes,'state_grows_with_online_length':False,'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    report['numba_cpu_target']=os.environ.get('NUMBA_CPU_NAME','native default')
    report['python_hash_seed']=os.environ.get('PYTHONHASHSEED','interpreter default')
    report['audit_script_sha256']=sha256(__file__)
    write_json(output,report)
    print('FUSED RESULT',report['status'],'rows',sum(offsets),'native_us',native_latency['throughput_median_us_per_point'],'fused_us',fused_latency['throughput_median_us_per_point'],flush=True)
    if report['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
