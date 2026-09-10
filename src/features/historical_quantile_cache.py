"""Quantile variants change only the declared columns of the AR8 DEV cache."""
import gc,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.features.streaming import feature_names,FEATURE_FAMILIES
from src.features.historical_quantiles import HistoricalQuantileARState,implementation_hash,QUANTILE_POLICY
from src.features.ar_cache import build_ar_cache
from src.data.loader import load_training
from src.utils.artifacts import write_json,sha256


def build_quantile_cache(raw,cache_root,split,config,policy,families):
    version=implementation_hash(families,policy,config);directory=Path(cache_root)/('ar8_quantile_'+families+'_'+version[:16]);manifest_path=directory/'MANIFEST.json'
    source=build_ar_cache(raw,cache_root,split,config,policy,input_order=8);source_hash=sha256(source/'MANIFEST.json')
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['status']=='COMPLETE' and manifest['representation_hash']==version and manifest['split_sha256']==split.digest and manifest['source_cache_manifest_sha256']==source_hash:
            if not all(sha256(directory/f'fold_{k}.parquet')==manifest['fold_sha256'][str(k)] for k in range(5)):raise RuntimeError('Quantile feature cache changed')
            return directory
    series={s.dataset_id:s for s in load_training(raw,split=split)};names=feature_names(config)
    unchanged=np.array([f not in families for f in FEATURE_FAMILIES]);unchanged[QUANTILE_POLICY['nonstationary_columns_excluded']]=True
    manifest={'status':'BUILDING','representation_hash':version,'split_sha256':split.digest,'source_cache_manifest_sha256':source_hash,
        'quantile_policy':QUANTILE_POLICY,'quantile_families':families,'calibration_policy':policy.__dict__,'seal_rows':0,'fold_sha256':{},
        'unchanged_columns':[name for name,keep in zip(names,unchanged) if keep],'unchanged_columns_bitwise_equal':True}
    write_json(manifest_path,manifest)
    for fold in range(5):
        frame=pd.read_parquet(source/f'fold_{fold}.parquet');matrix=np.empty((len(frame),len(names)),dtype=np.float32);times=frame.time_online.to_numpy()
        for sid,positions in frame.groupby('dataset_id',sort=False).indices.items():
            item=series[int(sid)];stream=HistoricalQuantileARState(item.historical,families=families,config=config,policy=policy)
            if not np.array_equal(times[positions],np.arange(len(item.online))):raise RuntimeError('Quantile cache row order changed')
            for position,point in zip(positions,item.online):matrix[position]=stream.update_and_get(point)
        if not np.isfinite(matrix).all():raise RuntimeError('Nonfinite historical quantile features')
        np.testing.assert_array_equal(matrix[:,unchanged],frame.loc[:,np.array(names)[unchanged]].to_numpy(dtype=np.float32))
        frame.loc[:,names]=matrix;path=directory/f'fold_{fold}.parquet';frame.to_parquet(path,index=False)
        manifest['fold_sha256'][str(fold)]=sha256(path);write_json(manifest_path,manifest)
        print('QUANTILE CACHE',families,fold,'complete, other columns unchanged',flush=True);del frame,matrix;gc.collect()
    manifest['status']='COMPLETE';write_json(manifest_path,manifest);return directory
