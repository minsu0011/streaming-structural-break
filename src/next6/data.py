"""Read only the hash-validated RP source; NEXT6 derived files are isolated."""
import numpy as np
from src.next6.common import ROOT,BASE,read_json,digest
from src.next5.data import ResearchArrays
from src.next5.guard import ResearchGuard

def source():
    audit=read_json(BASE/'PREFLIGHT.json')
    if audit['status']!='PASS':raise RuntimeError('Preflight required')
    guard=ResearchGuard(ROOT)
    if guard.split.digest!=audit['split_sha256']:raise RuntimeError('Split bytes changed')
    directory=ROOT/'data/next5/research_5d54e316ff5bf25d'
    if digest(directory/'MANIFEST.json')!=audit['cache_manifest_sha256']:raise RuntimeError('RP manifest changed')
    return ResearchArrays(directory,guard)

def reference_matrix(kind,data):
    paths=list((ROOT/'data/next5').glob('REF_'+kind+'_*/MANIFEST.json'))
    if len(paths)!=1:raise RuntimeError('Unique reference feature cache required')
    p=paths[0];m=read_json(p);reg=read_json(ROOT/'artifacts/next5/references/registrations'/('REF_'+kind+'.json'))
    from src.next5.reference_features import implementation_hash
    if m['identity']['feature_source_sha256']!=implementation_hash(reg['feature_settings']):raise RuntimeError('Reference source changed')
    if m['identity']['split_sha256']!=data.guard.split.digest or m['series']!=6400:raise RuntimeError('Reference scope mismatch')
    if digest(p.parent/'features.npy')!=m['features_sha256']:raise RuntimeError('Reference feature bytes changed')
    return np.load(p.parent/'features.npy',mmap_mode='r',allow_pickle=False)

def arrays():
    data=source();m=read_json(BASE/'data/MANIFEST.json')
    if m['status']!='COMPLETE' or m['node_source_sha256']!=digest(ROOT/'src/next6/nodes.py'):raise RuntimeError('Node cache identity mismatch')
    values={k:np.load(BASE/'data'/filename,mmap_mode='r',allow_pickle=False) for k,filename in m['arrays'].items()}
    return data,data.rows(),values
