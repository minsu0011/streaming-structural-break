"""Immutable candidate-specific DEV feature caches, with source and coordinate keys."""
import hashlib
import json
from pathlib import Path
import numpy as np
from src.next4.artifacts import ROOT,BASE,atomic_json,digest,read_json,utc_now,progress
from src.next4.model import feature_module


def cache_hash():return digest(Path(__file__))


def build_design(data,registration,scope):
    if scope not in ('screen','full'):raise ValueError('Existing locked screen/full scope required')
    policy=read_json(ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json')
    ids=data.guard.admit(policy['screen']['ids'] if scope=='screen' else data.ids,purpose='NEXT4 causal feature cache')
    metadata=data.rows();global_rows=np.flatnonzero(np.isin(metadata['dataset_id'],ids))
    metadata={k:v[global_rows] for k,v in metadata.items()}
    module=feature_module(registration['feature_module']);settings=registration['feature_settings']
    feature_sha=module.implementation_hash(settings)
    if feature_sha!=registration['feature_sha256']:raise RuntimeError('Stale registered feature source')
    identity={'candidate_id':registration['experiment_id'],'scope':scope,'feature_config':settings,
        'raw_sha256':read_json(BASE/'preflight/FROZEN_REFERENCES.json')['raw_sha256'],
        'feature_source_sha256':feature_sha,'cache_source_sha256':cache_hash(),
        'split_sha256':data.guard.split.digest,'seal_lock_sha256':data.guard.lock_sha,
        'global_rows_sha256':hashlib.sha256(global_rows.tobytes()).hexdigest(),
        'registration_sha256':digest(BASE/'registrations'/(registration['experiment_id']+'.json'))}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out=ROOT/'data/next4'/(registration['experiment_id']+'_'+scope+'_'+key[:16])
    manifest=out/'MANIFEST.json';names=tuple(module.feature_names(settings))
    if manifest.exists():
        record=read_json(manifest)
        if record['status']!='COMPLETE' or record['identity']!=identity:raise RuntimeError('Stale feature cache identity')
        if digest(out/'features.npy')!=record['features_sha256']:raise RuntimeError('Cached feature bytes changed')
        return np.load(out/'features.npy',mmap_mode='r',allow_pickle=False),metadata,names,global_rows,record
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'identity':identity},immutable=True)
    matrix=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype=np.float32,shape=(len(global_rows),len(names)))
    position=0;seen=0
    for sid in data.ids:
        if sid not in ids:continue
        historical,online,_=data.get(int(sid));state=module.make_state(historical,settings)
        values=state.replay(online)
        if values.shape!=(len(online),len(names)) or not np.isfinite(values).all():raise RuntimeError('Invalid feature output')
        end=position+len(online)
        if not np.all(metadata['dataset_id'][position:end]==sid):raise RuntimeError('Feature/metadata series coordinate mismatch')
        matrix[position:end]=values;position=end;seen+=1
        if seen%250==0:
            matrix.flush();progress('NEXT4 feature construction',registration['experiment_id']+' '+scope+': '+str(seen)+' DEV series')
            print('NEXT4 FEATURES',registration['experiment_id'],scope,seen,flush=True)
    assert position==len(global_rows) and seen==len(ids)
    matrix.flush();del matrix
    record={'status':'COMPLETE','created_utc':utc_now(),'identity':identity,'key':key,'series':seen,
        'rows':position,'feature_names':names,'feature_count':len(names),'features_sha256':digest(out/'features.npy'),
        'new_fits':0,'seal_rows':0,'labels_or_tau_passed_to_feature_state':False,'future_length_passed_to_update':False}
    atomic_json(manifest,record,immutable=True);data.guard.check_state()
    return np.load(out/'features.npy',mmap_mode='r',allow_pickle=False),metadata,names,global_rows,record
