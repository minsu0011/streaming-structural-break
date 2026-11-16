"""Exact legacy feature representations rebuilt on admitted ResearchPool only."""
import hashlib
import json
from pathlib import Path
import numpy as np
from numba import njit
from src.next4 import scale_asymmetry,sequential
from src.next2.dependence_runtime import CompactDependenceState,NAMES,step as dependence_step

KINDS=('J102','C205','B2_augmented_J','D2_expert')
B2={'version':'separate_energy_branches_v1','kind':'log_semitail'}
D2={'version':'discrete_likelihood_v1','kind':'scale_mixture'}


def validate(settings):
    if set(settings)!={'version','kind'} or settings['version']!='next5_reference_features_v1' or settings['kind'] not in KINDS:
        raise ValueError('Exact ResearchPool reference settings required')


@njit(cache=False)
def replay_dependence(points,args):
    values=np.empty((len(points),24),dtype=np.float32)
    for j in range(len(points)):values[j]=dependence_step(float(points[j]),*args)
    return values


class ReferenceState:
    def __init__(self,historical,settings):
        validate(settings);self.kind=settings['kind']
        self.state=CompactDependenceState(historical) if self.kind=='C205' else (
            sequential.make_state(historical,D2) if self.kind=='D2_expert' else scale_asymmetry.make_state(historical,B2))
        self.args=self.state.args

    def update(self,point):
        values=self.state.update(float(point))
        return values[:77] if self.kind=='J102' else values

    def replay(self,points):
        values=replay_dependence(np.asarray(points,dtype=np.float32),self.args) if self.kind=='C205' else self.state.replay(points)
        return values[:,:77] if self.kind=='J102' else values


def make_state(historical,settings):return ReferenceState(historical,settings)


def feature_names(settings):
    validate(settings);kind=settings['kind']
    if kind=='C205':return NAMES
    if kind=='D2_expert':return sequential.feature_names(D2)
    names=scale_asymmetry.feature_names(B2)
    return names[:77] if kind=='J102' else names


def implementation_hash(settings):
    validate(settings);root=Path(__file__).resolve().parents[2]
    files=('src/next5/reference_features.py','src/next2/dependence_runtime.py','src/next2/complement_bank.py',
           'src/next/fast_variance_initialization.py','src/next/whitening.py')
    hashes=''.join(hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files)
    return hashlib.sha256((hashes+scale_asymmetry.implementation_hash(B2)+sequential.implementation_hash(D2)+json.dumps(settings,sort_keys=True)).encode()).hexdigest()
