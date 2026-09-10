"""Join two DEV-only caches after exact row-coordinate and target alignment."""
import json,gc
from pathlib import Path
import pandas as pd
import numpy as np
from src.features.streaming import feature_names as base_names
from src.features.cross_order import feature_names,implementation_hash,POLICY
from src.features.ar_cache import build_ar_cache
from src.utils.artifacts import write_json,sha256
from src.validation.splits import assert_development_ids


def build_cross_order_cache(raw,cache_root,split,config):
    version=implementation_hash(config);directory=Path(cache_root)/('cross_order_'+version[:16]);manifest_path=directory/'MANIFEST.json'
    sources=[build_ar_cache(raw,cache_root,split,config,POLICY,input_order=order) for order in (None,8)]
    source_manifests=[json.loads((p/'MANIFEST.json').read_text()) for p in sources]
    identity=[sha256(p/'MANIFEST.json') for p in sources]
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['status']=='COMPLETE' and manifest['representation_hash']==version and manifest['split_sha256']==split.digest and manifest['source_manifest_sha256']==identity:
            if not all(sha256(directory/f'fold_{k}.parquet')==manifest['fold_sha256'][str(k)] for k in range(5)):raise RuntimeError('Cross-order cache changed')
            return directory
    manifest={'status':'BUILDING','representation_hash':version,'split_sha256':split.digest,'source_manifest_sha256':identity,
        'source_cache_directories':[str(p) for p in sources],'fold_sha256':{},'feature_names':feature_names(config),'seal_rows':0}
    write_json(manifest_path,manifest)
    for fold in range(5):
        frames=[pd.read_parquet(p/f'fold_{fold}.parquet') for p in sources]
        for path,source in zip(sources,source_manifests):
            if sha256(path/f'fold_{fold}.parquet')!=source['fold_sha256'][str(fold)]:raise RuntimeError('Component cache bytes changed')
        meta=[c for c in frames[0].columns if c not in base_names(config)]
        for c in meta:np.testing.assert_array_equal(frames[0][c],frames[1][c])
        assert_development_ids(frames[0].dataset_id.unique(),split)
        result=pd.concat([frames[0][meta],*[frame.loc[:,base_names(config)].rename(columns={name:f'ar{order}__{name}' for name in base_names(config)}) for order,frame in zip((4,8),frames)]],axis=1)
        path=directory/f'fold_{fold}.parquet';result.to_parquet(path,index=False)
        manifest['fold_sha256'][str(fold)]=sha256(path);write_json(manifest_path,manifest)
        print('CROSS ORDER CACHE',fold,'complete',flush=True);del frames,result;gc.collect()
    manifest['status']='COMPLETE';write_json(manifest_path,manifest);return directory
