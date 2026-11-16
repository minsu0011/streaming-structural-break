"""Resident scalar inference with exact frozen features and float64 blends.

This is an execution adapter. Fitted artifacts, feature definitions, learner
parameters and blend weights retain their original NEXT5 registration hashes.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np
from numba import njit,typeof
from numba.experimental import jitclass
from src.models.compact_trees import tree_margin
from src.next.fused_runtime import unique_array_bytes
from src.next2.shared_blend_runtime import SharedChannels,shared_feature_step
from src.next2.used_feature_runtime import UsedOldState
from src.next5.model import FeatureModel,feature_module
from src.next5 import reference_features,scale,persistence,windowed,mixture,robust_null,location
from src.next4.scale_asymmetry import appended_step
from src.next4.sequential import step as sequential_step
from src.next2.dependence_runtime import step as dependence_step

FLOAT32_MAX=float(np.finfo(np.float32).max)


def source_hash():
    root=Path(__file__).resolve().parents[2]
    names=('src/next5/runtime.py','src/next2/shared_blend_runtime.py','src/next2/used_feature_runtime.py',
        'src/next/fused_runtime.py','src/models/compact_trees.py')
    return hashlib.sha256(b''.join((root/name).read_bytes() for name in names)).hexdigest()


@njit(cache=False)
def j102_step(point,*args):return appended_step(point,*args)[:77]


@njit(cache=False)
def augmented_scale_step(point,scale_args,original_args):
    new=scale.scale_step(point,*scale_args);old=appended_step(point,*original_args)
    output=np.empty(109,dtype=np.float32);output[:77]=old[:77];output[77:]=new
    return output


@lru_cache(maxsize=16)
def cross_step(left_step,right_step):
    @njit(cache=False)
    def update(point,left_args,right_args,left,right,left_scale,right_scale):
        a=left_step(point,*left_args);b=right_step(point,*right_args)
        output=np.empty(len(a)+len(b)+8,dtype=np.float32);output[:len(a)]=a;output[len(a):len(a)+len(b)]=b
        for j in range(8):output[len(a)+len(b)+j]=np.tanh(a[left[j]]/left_scale[j])*np.tanh(b[right[j]]/right_scale[j])
        return output
    return update


def feature_step(module_name,settings):
    if module_name=='src.next5.reference_features':
        return {'J102':j102_step,'C205':dependence_step,'B2_augmented_J':appended_step,'D2_expert':sequential_step}[settings['kind']]
    if module_name=='src.next5.scale':return augmented_scale_step if settings['kind']=='augmented_j' else scale.scale_step
    direct={'src.next5.persistence':persistence.persistence_step,'src.next5.windowed':windowed.window_step,
        'src.next5.mixture':mixture.mixture_step,'src.next5.robust_null':robust_null.robust_step,'src.next5.location':location.location_step}
    if module_name in direct:return direct[module_name]
    if module_name=='src.next5.cross':
        parents=settings['parents'];return cross_step(*[feature_step(p['feature_module'],p['feature_settings']) for p in parents])
    raise ValueError('No reviewed runtime adapter for this frozen feature module')


def tree_coordinates(model):
    args=list(model.compact.args());features=args[1].copy();used=features>=0
    features[used]=model.columns[features[used]];args[1]=features
    return tuple(args)


@njit(cache=False)
def shared_probabilities(point,args):
    output=shared_feature_step(point,args);dependence=args[1][-1][-1]
    left=1./(1.+np.exp(-tree_margin(output,*args[-1][0])))
    right=1./(1.+np.exp(-tree_margin(dependence,*args[-1][1])))
    return .8*left+.2*right,right


@njit(cache=False)
def baseline_step(point,args):return shared_probabilities(point,args)[0]


@lru_cache(maxsize=32)
def composite_step(update):
    @njit(cache=False)
    def predict(point,args):
        baseline_args,feature_args,tree_args,weights=args
        baseline,dependence=shared_probabilities(point,baseline_args)
        features=update(point,*feature_args)
        expert=1./(1.+np.exp(-tree_margin(features,*tree_args)))
        return weights[0]*baseline+weights[1]*dependence+weights[2]*expert
    return predict


@lru_cache(maxsize=64)
def resident_class(kernel,argument_type):
    @jitclass([('args',argument_type)])
    class Resident:
        def __init__(self,args):self.args=args
        def predict_one(self,point):return kernel(point,self.args)
    return Resident


class SharedResearchBaseline:
    def __init__(self,j102,c205):
        for model,kind in ((j102,'J102'),(c205,'C205')):
            if type(model) is not FeatureModel:raise TypeError('Hash-validated ResearchPool feature model required')
            model.validate()
            if model.module_name!='src.next5.reference_features' or model.settings['kind']!=kind:raise ValueError('Exact J102/C205 reference pair required')
        if j102.training['training_ids_sha256']!=c205.training['training_ids_sha256']:raise ValueError('Reference training memberships differ')
        self.j102=j102;self.c205=c205
        tree=tree_coordinates(j102);used=np.unique(tree[1][tree[1]>=0]);self.used=used
        self.old_positions=np.flatnonzero(used<51).astype(np.int32);self.new_positions=np.flatnonzero(used>=51).astype(np.int32)
        self.old_columns=used[used<51].astype(np.int64);self.new_columns=(used[used>=51]-51).astype(np.int32)
        if not len(self.old_columns):raise ValueError('Current frozen J102 models require old feature columns')
        mapping=np.full(77,-1,dtype=np.int32);mapping[used]=np.arange(len(used),dtype=np.int32)
        trees=list(tree);features=trees[1].copy();features[features>=0]=mapping[features[features>=0]];trees[1]=features
        self.j_tree=tuple(trees);self.c_tree=tree_coordinates(c205)

    def arguments(self,historical):
        old=UsedOldState(historical,self.old_columns);shared=SharedChannels(historical)
        return (old.call_args,shared.call_args,self.old_positions,self.new_positions,self.new_columns,
            np.empty(len(self.used),dtype=np.float32),(self.j_tree,self.c_tree,.8))

    def new_state(self,historical):return ScalarDetector(baseline_step,self.arguments(historical),self)


class PreparedComposite:
    def __init__(self,j102,c205,expert,combination):
        self.baseline=SharedResearchBaseline(j102,c205);expert.validate()
        if expert.training['training_ids_sha256']!=j102.training['training_ids_sha256']:raise ValueError('Expert and baseline training memberships differ')
        if set(combination)-{'ResearchPool_BL80','C205','new_model'}:raise ValueError('Only frozen static combination coordinates allowed')
        weights=np.array([combination.get('ResearchPool_BL80',0.),combination.get('C205',0.),combination['new_model']],dtype=np.float64)
        if np.any(weights<0) or abs(weights.sum()-1)>1e-12:raise ValueError('Convex frozen combination required')
        self.expert=expert;self.weights=weights;self.tree=tree_coordinates(expert)
        self.update=feature_step(expert.module_name,expert.settings);self.kernel=composite_step(self.update)

    def new_state(self,historical):
        state=feature_module(self.expert.module_name).make_state(historical,self.expert.settings)
        args=(self.baseline.arguments(historical),state.args,self.tree,self.weights)
        return ScalarDetector(self.kernel,args,self)


class ScalarDetector:
    def __init__(self,kernel,args,prepared):
        self.prepared=prepared;self.args=args;self.kernel=kernel
        self.resident=resident_class(kernel,typeof(args))(args)

    def predict_one(self,point):
        value=float(point)
        if not np.isfinite(value) or abs(value)>FLOAT32_MAX:raise ValueError('Finite float32 current observation required')
        prediction=float(self.resident.predict_one(float(np.float32(value))))
        if not np.isfinite(prediction) or not 0<=prediction<=1:raise RuntimeError('Invalid scalar probability')
        return prediction

    @property
    def state_array_bytes(self):
        if isinstance(self.prepared,PreparedComposite):return unique_array_bytes((self.args[0][:-1],self.args[1]))
        return unique_array_bytes(self.args[:-1])

    @property
    def immutable_tree_array_bytes(self):
        if isinstance(self.prepared,PreparedComposite):return unique_array_bytes((self.args[0][-1],self.args[2]))
        return unique_array_bytes(self.args[-1])
