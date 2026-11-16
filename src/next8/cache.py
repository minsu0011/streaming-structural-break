"""Resumable label-free H-only feature cache and causal lag lookup."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from src.next8.common import ROOT, BASE, read_json, digest, atomic, ids_hash
from src.next8.features import N_FEATURES, NAMES, NORMALIZATION, source_hash, categorize
from src.next8.data import source
from src.next5.data import admitted_npz

SHARD_SIZE=64


def identity(data):
    p=read_json(BASE/'PREFLIGHT_NEXT8.json')
    return {'raw_sha256':p['raw_sha256'],'cache_manifest_sha256':p['cache_manifest_sha256'],
            'feature_sha256':source_hash(),'normalization':NORMALIZATION,
            'research_ids_sha256':ids_hash(data.ids),'research_split_sha256':data.guard.split.digest,
            'population_training_membership':'NONE: independent per-series H only; no shared fit',
            'protocol_sha256':digest(BASE/'NEXT8_PROTOCOL_LOCK.json'),
            'dtype':'float32','width':N_FEATURES,'shard_size':SHARD_SIZE}


def directory(data):
    value=identity(data)
    key=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    return BASE/'feature_cache'/key[:20],value


def immutable_npy(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('xb') as f:
        np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())
    os.rename(tmp,path)


def immutable_npz(path,**value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('xb') as f:
        np.savez(f,**value);f.flush();os.fsync(f.fileno())
    os.rename(tmp,path)


class FeatureCache:
    def __init__(self,data=None,*,verify_values=False):
        self.data=source() if data is None else data
        self.directory,self.identity=directory(self.data)
        self.manifest=read_json(self.directory/'MANIFEST.json')
        assert self.manifest['status']=='COMPLETE' and self.manifest['identity']==self.identity
        for name,sha in self.manifest['files_sha256'].items():
            if name!='values.npy' or verify_values:
                assert digest(self.directory/name)==sha,name
        self.series=admitted_npz(self.directory/'series.npz',self.data.guard,'NEXT8 H-only feature cache metadata')
        np.testing.assert_array_equal(self.series['dataset_id'],self.data.ids)
        self.values=np.load(self.directory/'values.npy',mmap_mode='r',allow_pickle=False)
        self.meta=self.data.rows()
        offsets=self.data.series['online_offsets']
        self.series_index=np.repeat(np.arange(len(self.data.ids),dtype=np.int32),np.diff(offsets))
        assert self.values.shape==(len(self.meta['dataset_id']),N_FEATURES)
        np.testing.assert_array_equal(self.data.ids[self.series_index],self.meta['dataset_id'])

    def current(self,rows,columns):
        return np.asarray(self.values[np.asarray(rows)[:,None],np.asarray(columns)[None,:]],dtype=np.float32)

    def lag(self,rows,columns,lag):
        if lag not in (1,4,16):raise ValueError('Frozen lag menu')
        rows=np.asarray(rows,dtype=np.int64);columns=np.asarray(columns,dtype=np.int32)
        t=self.meta['time_online'][rows].astype(np.int64)
        observed=t>=lag
        out=np.empty((len(rows),len(columns)),dtype=np.float32)
        r=rows[observed]-lag
        assert np.array_equal(self.meta['dataset_id'][r],self.meta['dataset_id'][rows[observed]])
        out[observed]=self.current(r,columns)
        s=self.series_index[rows[~observed]]
        out[~observed]=self.series['seed_values'][s[:,None],(16+t[~observed]-lag)[:,None],columns[None,:]]
        return out

    def states(self,rows,columns,*,previous=False):
        rows=np.asarray(rows,dtype=np.int64);columns=np.asarray(columns,dtype=np.int32)
        s=self.series_index[rows]
        values=self.lag(rows,columns,1) if previous else self.current(rows,columns)
        return categorize(values,self.series['low'][s[:,None],columns[None,:]],
                          self.series['high'][s[:,None],columns[None,:]],
                          self.series['valid'][s[:,None],columns[None,:]])

    def epsilon(self,rows,columns,horizon):
        s=self.series_index[np.asarray(rows,dtype=np.int64)]
        k=(1,4,16).index(horizon)
        return self.series['epsilon'][s[:,None],k,np.asarray(columns)[None,:]]

    def context(self,rows):
        s=self.series_index[np.asarray(rows,dtype=np.int64)]
        return np.column_stack([self.series['context'][s],np.log1p(self.meta['time_online'][rows])]).astype(np.float32)

    def forecast_rows(self,rows,horizon,cutoff=None):
        rows=np.asarray(rows,dtype=np.int64)
        t=self.meta['time_online'][rows].astype(np.int64)
        following=rows+horizon
        keep=following<len(self.values)
        q=np.flatnonzero(keep)
        keep[q]&=(self.meta['dataset_id'][following[q]]==self.meta['dataset_id'][rows[q]]) & (self.meta['time_online'][following[q]]==t[q]+horizon)
        if cutoff is not None:keep&=t+horizon<=cutoff
        return rows[keep]
