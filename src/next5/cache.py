"""Hash-bound causal features on precisely the admitted ResearchPool coordinate."""
import hashlib
import json
import numpy as np
from src.next5.artifacts import ROOT,BASE,atomic_json,digest,read_json,utc_now,progress
from src.next5.model import feature_module


def build_design(data,registration):
    cid=registration['experiment_id'];module=feature_module(registration['feature_module'])
    settings=registration['feature_settings'];feature_sha=module.implementation_hash(settings)
    if feature_sha!=registration['feature_sha256']:raise RuntimeError('Frozen registered feature source changed')
    regdir=BASE/'references/registrations' if cid.startswith('REF_') else BASE/'registrations'
    if read_json(regdir/(cid+'.json'))!=registration:raise RuntimeError('Persisted registration required')
    metadata=data.rows();data.guard.admit(np.unique(metadata['dataset_id']),purpose='NEXT5 feature design')
    names=tuple(module.feature_names(settings));global_rows=np.arange(len(metadata['dataset_id']),dtype=np.int64)
    identity={'scope':'NEXT5_RESEARCH_POOL_ONLY','candidate':cid,'feature_config':settings,
        'raw_manifest_sha256':digest(data.directory/'MANIFEST.json'),'feature_source_sha256':feature_sha,
        'cache_source_sha256':digest(ROOT/'src/next5/cache.py'),'registration_sha256':digest(regdir/(cid+'.json')),
        'split_sha256':data.guard.split.digest,'shadow_lock_sha256':data.guard.shadow_lock_sha,
        'row_ids_sha256':hashlib.sha256(metadata['dataset_id'].tobytes()).hexdigest()}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out=ROOT/'data/next5'/(cid+'_'+key[:16]);manifest=out/'MANIFEST.json'
    if manifest.exists():
        record=read_json(manifest)
        if record['status']!='COMPLETE' or record['identity']!=identity or digest(out/'features.npy')!=record['features_sha256']:
            raise RuntimeError('Stale or damaged feature cache')
        return np.load(out/'features.npy',mmap_mode='r',allow_pickle=False),metadata,record
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'identity':identity},immutable=True)
    matrix=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype=np.float32,shape=(len(global_rows),len(names)))
    position=0
    for count,sid in enumerate(data.ids,1):
        historical,online,_=data.get(int(sid));values=module.make_state(historical,settings).replay(online)
        end=position+len(online)
        if values.shape!=(len(online),len(names)) or not np.isfinite(values).all():raise RuntimeError('Invalid causal feature output')
        if not np.all(metadata['dataset_id'][position:end]==sid):raise RuntimeError('Series coordinate mismatch')
        matrix[position:end]=values;position=end
        if count%250==0:
            matrix.flush();progress('ResearchPool feature construction',cid+' '+str(count)+'/6400; H and current prefix only')
            print('NEXT5 FEATURES',cid,count,flush=True)
    assert position==len(global_rows)
    matrix.flush();del matrix;data.guard.check_state()
    record={'status':'COMPLETE','created_utc':utc_now(),'identity':identity,'key':key,'directory':out.relative_to(ROOT).as_posix(),
        'series':len(data.ids),'rows':position,'feature_names':names,'feature_count':len(names),'features_sha256':digest(out/'features.npy'),
        'new_fits':0,'shadow_rows':0,'seal_rows':0,'labels_or_tau_passed_to_feature_state':False,'future_length_passed_to_update':False}
    atomic_json(manifest,record,immutable=True)
    return np.load(out/'features.npy',mmap_mode='r',allow_pickle=False),metadata,record
