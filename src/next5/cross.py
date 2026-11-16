"""Fixed parent feature banks and eight explicitly frozen bounded products."""
from pathlib import Path
import hashlib
import json
from functools import lru_cache
import numpy as np
from src.next5.artifacts import BASE,read_json,digest
from src.next5.model import feature_module

KINDS=('scale_persistence','scale_tail','dependence_scale')


@lru_cache(maxsize=1)
def locked_settings():return read_json(BASE/'locks/X_SETTINGS_LOCK.json')['settings']


def validate(settings):
    if set(settings)!={'version','kind','parents','interactions'} or settings['version']!='next5_cross_family_v1' or settings['kind'] not in KINDS:
        raise ValueError('Exact frozen cross-family settings required')
    expected=locked_settings()[settings['kind']]
    if settings!=expected:raise ValueError('Cross-family parent or interaction recipe changed')
    if len(settings['parents'])!=2 or len(settings['interactions'])!=8:raise ValueError('Two fixed parents and eight explicit interactions required')


class CrossState:
    def __init__(self,historical,settings):
        validate(settings);self.settings=settings;parents=settings['parents'];modules=[feature_module(p['feature_module']) for p in parents]
        names=[module.feature_names(parent['feature_settings']) for module,parent in zip(modules,parents)]
        self.states=[module.make_state(historical,parent['feature_settings']) for module,parent in zip(modules,parents)]
        self.left=np.asarray([names[0].index(pair['left']) for pair in settings['interactions']],dtype=np.int32)
        self.right=np.asarray([names[1].index(pair['right']) for pair in settings['interactions']],dtype=np.int32)
        self.left_scale=np.asarray([pair['left_scale'] for pair in settings['interactions']],dtype=np.float64)
        self.right_scale=np.asarray([pair['right_scale'] for pair in settings['interactions']],dtype=np.float64)
        self.args=(self.states[0].args,self.states[1].args,self.left,self.right,self.left_scale,self.right_scale)

    def update(self,point):
        a=self.states[0].update(point);b=self.states[1].update(point)
        products=np.tanh(a[self.left]/self.left_scale)*np.tanh(b[self.right]/self.right_scale)
        return np.r_[a,b,products].astype(np.float32)

    def replay(self,points):
        a=self.states[0].replay(points);b=self.states[1].replay(points)
        products=np.tanh(a[:,self.left]/self.left_scale)*np.tanh(b[:,self.right]/self.right_scale)
        return np.column_stack([a,b,products]).astype(np.float32)


def make_state(historical,settings):return CrossState(historical,settings)


def feature_names(settings):
    validate(settings);names=[]
    for parent in settings['parents']:names.extend(feature_module(parent['feature_module']).feature_names(parent['feature_settings']))
    return tuple(names)+tuple('next5_cross_'+settings['kind']+'_'+str(j) for j in range(8))


def implementation_hash(settings):
    validate(settings);source=Path(__file__).read_bytes()+digest(BASE/'locks/X_SETTINGS_LOCK.json').encode()
    for parent in settings['parents']:
        actual=feature_module(parent['feature_module']).implementation_hash(parent['feature_settings'])
        if actual!=parent['feature_sha256']:raise RuntimeError('Frozen X parent feature source changed')
        source+=actual.encode()
    return hashlib.sha256(source+json.dumps(settings,sort_keys=True).encode()).hexdigest()
