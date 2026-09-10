"""Bounded float32 parquet cache, fingerprinted by raw data and feature source."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.loader import load_training
from src.data.labels import labels_from_tau
from src.features.streaming import StreamingFeatureState,FEATURE_NAMES,feature_version,feature_names
from src.features.config import CONFIG
from src.models.baselines import BASELINES,OfficialEWMA,from_features
from src.utils.artifacts import sha256,write_json,utc_now
from src.validation.splits import assignment,assert_development_ids

def cache_version(config=CONFIG,split=None):
    root=Path(__file__).resolve().parents[2]
    paths=['src/features/streaming.py','src/features/config.py','src/features/cache.py','src/data/loader.py','src/data/labels.py','src/models/baselines.py','src/validation/splits.py','configs/research.json']
    return hashlib.sha256(b''.join((root/path).read_bytes() for path in paths)+config.digest().encode()+(split.digest.encode() if split is not None else b'SYNTHETIC_UNGROUPED')).hexdigest()

def build_cache(raw_directory,cache_root,*,split=None,config=CONFIG):
    raw_directory=Path(raw_directory); directory=Path(cache_root)/cache_version(config,split)[:16]
    names=feature_names(config)
    source_hashes={name:sha256(raw_directory/name) for name in ('X_train.parquet','y_train_index.parquet')}
    manifest_path=directory/'MANIFEST.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['feature_hash']!=feature_version(config) or manifest.get('cache_hash')!=cache_version(config,split) or manifest['raw_sha256']!=source_hashes:
            raise RuntimeError('Stale cache')
        if manifest['status']=='COMPLETE':
            for shard in manifest['shards']:
                if sha256(directory/shard['name'])!=shard['sha256']: raise RuntimeError('Cache shard hash mismatch')
            return directory
    directory.mkdir(parents=True,exist_ok=True)
    manifest={'status':'BUILDING','feature_hash':feature_version(config),'config_hash':config.digest(),'feature_config':config.to_dict(),
        'split_sha256':split.digest if split else None,'cache_hash':cache_version(config,split),'raw_sha256':source_hashes,'created_utc':utc_now(),'shards':[],
        'columns':list(names),'data_scope':'OFFICIAL_DEVELOPMENT_ONLY' if split else 'SYNTHETIC_ONLY','online_precision':'float32 to match official wire protocol','seal_extracted':False,
        'sampling_policy_version':2}
    write_json(manifest_path,manifest)
    buffers={fold:[] for fold in range(5)}; counts={fold:0 for fold in range(5)}; shard_count={fold:0 for fold in range(5)}
    def flush(fold):
        if not buffers[fold]: return
        frame=pd.concat(buffers[fold],ignore_index=True)
        name=f'fold_{fold}_shard_{shard_count[fold]:04d}.parquet'
        frame.to_parquet(directory/name,index=False)
        manifest['shards'].append({'name':name,'fold':fold,'rows':len(frame),'sha256':sha256(directory/name)})
        write_json(manifest_path,manifest)
        buffers[fold].clear(); counts[fold]=0; shard_count[fold]+=1
    for series_number,series in enumerate(load_training(raw_directory,development_only=True,split=split),1):
        assert_development_ids([series.dataset_id],split)
        h=np.asarray(series.historical,dtype=np.float32); online=np.asarray(series.online,dtype=np.float32)
        state=StreamingFeatureState(h,config=config); official=OfficialEWMA(h)
        matrix=np.empty((len(online),len(FEATURE_NAMES)),dtype=np.float32)
        base=np.empty((len(online),len(BASELINES)),dtype=np.float32)
        for t,point in enumerate(online):
            matrix[t]=state.update_and_get(point)
            base[t,1]=official.predict_one(point)
        base[:,0]=0.5
        evidence=np.empty((len(matrix),5),dtype=np.float64)
        # Calculate from float64 feature scalars like the live baseline, then wire-quantize once.
        f64=matrix.astype(np.float64)
        evidence[:,0]=f64[:,[7,22,37]].max(axis=1)/3
        evidence[:,1]=f64[:,[3,4]].max(axis=1)/4
        evidence[:,2]=np.abs(f64[:,[9,24,39]]).max(axis=1)/2
        evidence[:,3]=np.maximum(np.abs(f64[:,[11,26,41]]).max(axis=1),f64[:,[14,29,44]].max(axis=1))/4
        evidence[:,4]=np.abs(f64[:,[15,30,45,20,35,50]]).max(axis=1)/4
        base[:,2:7]=evidence/(1+evidence)
        combined=evidence[:,[0,2,3,4]].mean(axis=1)
        base[:,7]=combined/(1+combined)
        frame=pd.DataFrame(matrix,columns=names)
        for j,name in enumerate(BASELINES): frame[name]=base[:,j]
        frame['dataset_id']=np.int64(series.dataset_id); frame['time_online']=np.arange(len(online),dtype=np.int32)
        frame['target']=labels_from_tau(len(online),series.tau)
        capped=np.unique(np.linspace(0,len(online)-1,min(256,len(online)),dtype=int)) if len(online) else []
        flags=np.zeros(len(online),dtype=np.uint8); flags[capped]=1; frame['sample_S1']=flags
        stratified=np.zeros(len(online),dtype=np.uint8)
        for lo,hi in zip([0,10,25,50,100,200,400],[10,25,50,100,200,400,len(online)]):
            hi=min(hi,len(online))
            if hi>lo:stratified[np.unique(np.linspace(lo,hi-1,min(32,hi-lo),dtype=int))]=1
        frame['sample_S2']=stratified
        fold=assignment(series.dataset_id,split)[1]; buffers[fold].append(frame); counts[fold]+=len(frame)
        if counts[fold]>=65536: flush(fold)
        if series_number%1000==0:print(f'Cache: {series_number} development series extracted',flush=True)
    for fold in range(5): flush(fold)
    manifest['status']='COMPLETE'; manifest['rows']=sum(s['rows'] for s in manifest['shards'])
    write_json(manifest_path,manifest)
    return directory

def read_fold(directory,fold,*,split=None,config=CONFIG):
    directory=Path(directory)
    manifest=json.loads((directory/'MANIFEST.json').read_text(encoding='utf-8'))
    if manifest['status']!='COMPLETE' or manifest['feature_hash']!=feature_version(config) or manifest.get('cache_hash')!=cache_version(config,split): raise RuntimeError('Incomplete or stale cache')
    frames=[pd.read_parquet(directory/s['name']) for s in manifest['shards'] if s['fold']==fold]
    if not frames: raise RuntimeError(f'Empty fold {fold}')
    frame=pd.concat(frames,ignore_index=True)
    assert_development_ids(frame.dataset_id.unique(),split)
    if any(assignment(sid,split)[1]!=fold for sid in frame.dataset_id.unique()): raise RuntimeError('Fold contamination')
    return frame
