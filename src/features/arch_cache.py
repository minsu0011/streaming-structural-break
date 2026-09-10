"""DEV-only cache for the prospectively fixed conditional-variance experiment."""
import gc,json
from pathlib import Path
import numpy as np
from src.features.arch_input import ARCHCalibratedState,implementation_hash,ARCH_POLICY
from src.features.streaming import feature_names
from src.features.cache import build_cache,read_fold
from src.data.loader import load_training
from src.utils.artifacts import write_json,sha256


def build_arch_cache(raw,cache_root,split,config,policy):
    raw=Path(raw);version=implementation_hash(policy,config);directory=Path(cache_root)/('ar8_arch1_'+version[:16]);manifest_path=directory/'MANIFEST.json'
    identity={name:sha256(raw/name) for name in ('X_train.parquet','y_train_index.parquet')}
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest.get('raw_source_identity')!=identity:raise RuntimeError('Raw identity changed')
        if manifest['status']=='COMPLETE' and manifest['representation_hash']==version and manifest['split_sha256']==split.digest:
            if not all(sha256(directory/f'fold_{k}.parquet')==manifest['fold_sha256'][str(k)] for k in range(5)):raise RuntimeError('ARCH cache bytes changed')
            return directory
    source=build_cache(raw,cache_root,split=split,config=config);series={s.dataset_id:s for s in load_training(raw,split=split)};names=feature_names(config)
    manifest={'status':'BUILDING','representation_hash':version,'split_sha256':split.digest,'arch_policy':ARCH_POLICY,'calibration_policy':policy.__dict__,
        'source_cache_manifest_sha256':sha256(source/'MANIFEST.json'),'raw_source_identity':identity,'seal_rows':0,'fold_sha256':{}}
    write_json(manifest_path,manifest)
    for fold in range(5):
        frame=read_fold(source,fold,split=split,config=config);matrix=np.empty((len(frame),len(names)),dtype=np.float32);times=frame.time_online.to_numpy()
        for sid,positions in frame.groupby('dataset_id',sort=False).indices.items():
            item=series[int(sid)];stream=ARCHCalibratedState(item.historical,config=config,policy=policy)
            if not np.array_equal(times[positions],np.arange(len(item.online))):raise RuntimeError('ARCH row coordinates differ')
            for position,point in zip(positions,item.online):matrix[position]=stream.update_and_get(point)
        if not np.isfinite(matrix).all():raise RuntimeError('Nonfinite ARCH representation')
        frame.loc[:,names]=matrix;path=directory/f'fold_{fold}.parquet';frame.to_parquet(path,index=False)
        manifest['fold_sha256'][str(fold)]=sha256(path);write_json(manifest_path,manifest);print('ARCH CACHE',fold,'complete',flush=True)
        del frame,matrix;gc.collect()
    manifest['status']='COMPLETE';write_json(manifest_path,manifest);return directory
