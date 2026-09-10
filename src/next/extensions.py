"""Versioned extension interface for bounded, independently tested statistics."""
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import time
import numpy as np
from src.next.predictor import NextBundle
from src.next.cache import load_design
from src.next.design_selection import select_reference_families
from src.features.arch_input import ARCHCalibratedState
from src.features.config import CONFIG
from src.features.streaming import feature_names as old_feature_names
from src.utils.artifacts import sha256,write_json,utc_now

EXTENSION_MODULES={
    'historical_null':'src.next.calibrated_engine',
    'ar_score':'src.next.score_evidence',
    'legacy_bank':'src.next.legacy_extension',
    'ar_mismatch':'src.next.mismatch_evidence',
    'conditional_variance':'src.next.variance_evidence'}


def extension_module(extension):
    if extension['kind'] not in EXTENSION_MODULES:
        raise ValueError('Unregistered extension kind')
    return importlib.import_module(EXTENSION_MODULES[extension['kind']])


def extension_hash(extension):
    module=extension_module(extension)
    return hashlib.sha256(Path(__file__).read_bytes()+module.implementation_hash(extension['settings']).encode()+
                          json.dumps(extension,sort_keys=True).encode()).hexdigest()


def names_and_groups(extension):
    module=extension_module(extension)
    names=[extension['kind']+'__'+name for name in module.feature_names(extension['settings'])]
    return names,list(module.feature_groups(extension['settings']))


def build_extension_cache(data,extension,scope):
    root=data.guard.root
    policy=json.loads((root/'ROBUSTNESS_SPLIT_LOCK.json').read_text(encoding='utf-8'))
    ids=set(data.guard.admit(policy['screen']['ids'] if scope=='screen' else None,purpose='extension features'))
    metadata=data.rows()
    global_rows=np.flatnonzero(np.isin(metadata['dataset_id'],list(ids))).astype(np.int64)
    names,groups=names_and_groups(extension)
    identity={'extension':extension,'extension_hash':extension_hash(extension),'scope':scope,
        'dev_manifest_sha256':sha256(data.directory/'MANIFEST.json'),'split_sha256':data.guard.split.digest,
        'seal_lock_sha256':data.guard.lock_sha,'ids_sha256':hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest(),
        'names':names,'groups':groups,'dtype':'float32'}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out=root/'data/next/extensions'/key[:20]
    if (out/'MANIFEST.json').exists():
        manifest=json.loads((out/'MANIFEST.json').read_text(encoding='utf-8'))
        if manifest['identity']!=identity or manifest['status']!='COMPLETE':
            raise RuntimeError('Extension cache identity changed')
        for name,digest in manifest['files_sha256'].items():
            if sha256(out/name)!=digest:
                raise RuntimeError('Extension cache bytes changed')
        return out
    out.mkdir(parents=True,exist_ok=True)
    module=extension_module(extension)
    matrix=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype=np.float32,shape=(len(global_rows),len(names)))
    ho,oo=data.series['historical_offsets'],data.series['online_offsets']
    pointer,count=0,0
    started=time.perf_counter()
    for j,sid in enumerate(data.ids):
        if int(sid) not in ids:
            continue
        historical=data.historical[ho[j]:ho[j+1]]
        points=data.online[oo[j]:oo[j+1]]
        state=module.make_state(historical,extension['settings'])
        values=state.replay(points)
        if values.shape!=(len(points),len(names)) or not np.isfinite(values).all():
            raise RuntimeError('Invalid causal extension output')
        matrix[pointer:pointer+len(points)]=values
        pointer+=len(points);count+=1
        if count%500==0:
            print('EXTENSION FEATURES',extension['kind'],scope,count,round(time.perf_counter()-started,1),flush=True)
    if pointer!=len(global_rows):
        raise RuntimeError('Extension row coverage mismatch')
    matrix.flush();del matrix
    np.save(out/'global_row_index.npy',global_rows,allow_pickle=False)
    write_json(out/'MANIFEST.json',{'status':'COMPLETE','timestamp':utc_now(),'identity':identity,
        'files_sha256':{name:sha256(out/name) for name in ['features.npy','global_row_index.npy']},
        'series':count,'rows':pointer,'seal_rows':0,'elapsed_seconds':time.perf_counter()-started})
    return out


def load_extended_design(data,candidate,scope):
    extension=candidate['extension']
    path=build_extension_cache(data,extension,scope)
    matrix=np.load(path/'features.npy',mmap_mode='r',allow_pickle=False)
    global_rows=np.load(path/'global_row_index.npy',allow_pickle=False)
    names,groups=names_and_groups(extension)
    keep=candidate.get('keep_names')
    columns=[j for j,(name,group) in enumerate(zip(names,groups)) if group in candidate['groups'] and (keep is None or name in keep)]
    selected_names=[names[j] for j in columns]
    base_names=[]
    sources={'extension_manifest_sha256':sha256(path/'MANIFEST.json'),'extension_hash':extension_hash(extension)}
    if candidate['base']=='arch8':
        base,metadata,base_names,base_sources,base_rows=load_design(data,scope=scope,groups=(),base='arch8')
        base,base_names=select_reference_families(base,base_names,'ABCD')
        np.testing.assert_array_equal(base_rows,global_rows)
        sources.update(base_sources)
    elif candidate['base']=='none':
        base=None
        metadata={k:v[global_rows] for k,v in data.rows().items()}
    else:
        raise ValueError('Unsupported extension base')
    design=np.empty((len(global_rows),len(base_names)+len(columns)),dtype=np.float32)
    if base is not None:
        design[:,:len(base_names)]=base
    for k,j in enumerate(columns):
        design[:,len(base_names)+k]=matrix[:,j]
    return design,metadata,base_names+selected_names,sources,global_rows


class ExtendedFeatureState:
    def __init__(self,historical,names,extension):
        config=CONFIG.variant(normalization='median_mad',scales=(5,20,160))
        old_names=['arch8__'+name for name in old_feature_names(config)]
        extension_names,_=names_and_groups(extension)
        self.old_positions=np.asarray([j for j,name in enumerate(names) if name.startswith('arch8__')],dtype=np.int32)
        self.old_columns=np.asarray([old_names.index(names[j]) for j in self.old_positions],dtype=np.int32)
        self.new_positions=np.asarray([j for j,name in enumerate(names) if not name.startswith('arch8__')],dtype=np.int32)
        self.new_columns=np.asarray([extension_names.index(names[j]) for j in self.new_positions],dtype=np.int32)
        self.old=ARCHCalibratedState(historical,config=config) if len(self.old_positions) else None
        self.new=extension_module(extension).make_state(historical,extension['settings'])
        self.output=np.empty(len(names),dtype=np.float32)

    def update(self,point):
        if self.old is not None:
            self.output[self.old_positions]=self.old.update_and_get(point)[self.old_columns]
        self.output[self.new_positions]=self.new.update(point)[self.new_columns]
        return self.output


@dataclass
class ExtendedBundle(NextBundle):
    extension:dict=None
    extension_implementation_hash:str=''

    @classmethod
    def wrap(cls,model,extension):
        return cls(**vars(model),extension=extension,extension_implementation_hash=extension_hash(extension))

    def make_state(self,historical):
        return ExtendedFeatureState(historical,self.names,self.extension)

    @classmethod
    def load(cls,path):
        model=NextBundle.load(path)
        if model.extension_implementation_hash!=extension_hash(model.extension):
            raise RuntimeError('Extension feature implementation changed')
        return model
