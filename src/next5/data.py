"""Raw-predicate ResearchPool data; old full-DEV caches are never opened."""
import hashlib
import json
from pathlib import Path
import numpy as np
from src.next5.artifacts import ROOT,BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5.guard import ResearchGuard,ShadowAccessError
from src.next.data import iter_dev_series,RAW_SHA
from src.data.labels import labels_from_tau


def source_hash():
    names=('src/next5/data.py','src/next5/guard.py','src/next/data.py','src/data/labels.py')
    return hashlib.sha256(b''.join((ROOT/n).read_bytes() for n in names)).hexdigest()


def admitted_npz(path,guard,purpose):
    with np.load(path,allow_pickle=False) as source:
        identifiers=source['dataset_id']
        guard.admit(np.unique(identifiers),purpose=purpose)
        # Labels/observations in other members are only decoded after admission.
        return {key:(identifiers if key=='dataset_id' else source[key]) for key in source.files}


class ResearchArrays:
    def __init__(self,directory,guard):
        self.directory=Path(directory);self.guard=guard
        self.series=admitted_npz(self.directory/'series.npz',guard,'load ResearchPool series cache')
        self.ids=self.series['dataset_id']
        if set(map(int,self.ids))!=guard.research_ids or len(self.ids)!=len(set(self.ids)):
            raise ShadowAccessError('Exact unique ResearchPool coverage required')
        self.historical=np.load(self.directory/'historical.npy',mmap_mode='r',allow_pickle=False)
        self.online=np.load(self.directory/'online.npy',mmap_mode='r',allow_pickle=False)
        self.id_to_index={int(sid):j for j,sid in enumerate(self.ids)}

    def get(self,sid):
        self.guard.admit([sid],purpose='ResearchPool historical/online series')
        j=self.id_to_index[int(sid)];ho=self.series['historical_offsets'];oo=self.series['online_offsets']
        return self.historical[ho[j]:ho[j+1]],self.online[oo[j]:oo[j+1]],int(self.series['tau'][j])

    def rows(self):
        result=admitted_npz(self.directory/'rows.npz',self.guard,'ResearchPool row metadata')
        self.guard.record_rows(result['dataset_id'],purpose='ResearchPool row metadata')
        return result


def prepare_research(root=ROOT):
    root=Path(root);guard=ResearchGuard(root)
    identity={'scope':'NEXT5_RESEARCH_POOL_ONLY','raw_sha256':RAW_SHA,'source_sha256':source_hash(),
        'research_split_sha256':guard.split.digest,'shadow_lock_sha256':guard.shadow_lock_sha,
        'research_ids_sha256':guard.lock['research_ids_sha256'],'seal_lock_sha256':guard.lock_sha,
        'dtype':'float32','labels_decoded_after_ID_admission':True}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out=root/'data/next5'/('research_'+key[:16]);manifest=out/'MANIFEST.json'
    if manifest.exists():
        old=read_json(manifest)
        if old['status']!='COMPLETE' or old['identity']!=identity:raise ShadowAccessError('Stale ResearchPool cache identity')
        for name,sha in old['files_sha256'].items():
            if digest(out/name)!=sha:raise ShadowAccessError('ResearchPool cache bytes changed')
        return ResearchArrays(out,guard)
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'identity':identity},immutable=True)
    for name,sha in RAW_SHA.items():
        if digest(root/'data/raw'/name)!=sha:raise ShadowAccessError('Canonical raw bytes changed')
    hs=[];os=[];ids=[];taus=[];ho=[0];oo=[0];folds=[]
    for sid,h,o,tau in iter_dev_series(guard):
        hs.append(h);os.append(o);ids.append(sid);taus.append(tau);folds.append(guard.split.assignment(sid)[1])
        ho.append(ho[-1]+len(h));oo.append(oo[-1]+len(o))
        if len(ids)%500==0:
            progress('ResearchPool raw predicate loader',str(len(ids))+'/6400; Shadow and LOCAL SEAL excluded before exposed rows',phase='PREFLIGHT')
            print('NEXT5 RESEARCH RAW',len(ids),flush=True)
    assert set(ids)==guard.research_ids and len(ids)==len(guard.research_ids)
    ids=np.asarray(ids,dtype=np.int32);folds=np.asarray(folds,dtype=np.int8)
    np.save(out/'historical.npy',np.concatenate(hs),allow_pickle=False)
    np.save(out/'online.npy',np.concatenate(os),allow_pickle=False)
    np.savez(out/'series.npz',dataset_id=ids,tau=np.asarray(taus,dtype=np.int32),
             historical_offsets=np.asarray(ho,dtype=np.int64),online_offsets=np.asarray(oo,dtype=np.int64),fold=folds)
    lengths=np.diff(oo);row_ids=np.repeat(ids,lengths)
    np.savez(out/'rows.npz',dataset_id=row_ids,
             time_online=np.concatenate([np.arange(n,dtype=np.int16) for n in lengths]),
             target=np.concatenate([labels_from_tau(n,tau) for n,tau in zip(lengths,taus)]),fold=np.repeat(folds,lengths))
    files=('historical.npy','online.npy','series.npz','rows.npz');guard.check_state()
    atomic_json(manifest,{'status':'COMPLETE','created_utc':utc_now(),'identity':identity,
        'files_sha256':{name:digest(out/name) for name in files},**guard.record_rows(row_ids,purpose='NEXT5 raw ResearchPool cache')},immutable=True)
    atomic_json(BASE/'data/RESEARCH_POINTER.json',{'created_utc':utc_now(),'directory':out.relative_to(root).as_posix(),
        'manifest_sha256':digest(manifest),'rows':len(row_ids),'series':len(ids),'shadow_rows':0,'seal_rows':0},immutable=True)
    return ResearchArrays(out,guard)
