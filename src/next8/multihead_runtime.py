"""One causal feature update for several frozen heads with identical H maps.

This adapter is for matched replay/audit throughput. Scalar latency gates still
measure each actually deployed standalone method and its baseline composition.
"""
import copy
import numpy as np
from numba import njit
from src.next8.features import FeatureState,N_FEATURES,h_context
from src.next8.runtime_features import raw_arguments,normalized_step
from src.next8.runtime_head import conditions,coordinates,score,state_arguments
from src.next8.streaming_runtime import points,PreparedJoint
from src.next.fused_runtime import unique_array_bytes


@njit(cache=True)
def finish_multi(normalized,args):
    state,heads,history,counter,context,output=args
    t=counter[0];conditions(history,t,state)
    for k in range(len(heads)):
        coordinate,head=heads[k]
        x=coordinates(normalized,history,t,context,state[-2],state[-1],coordinate)
        output[k]=score(x,head)
    history[t%32]=normalized;counter[0]+=1
    return output


@njit(cache=True)
def multi_step(point,args):
    engine,normalization,models=args
    return finish_multi(normalized_step(point,engine,normalization),models)


@njit(cache=True)
def multi_zero_step(point,args):return finish_multi(args[0],args[1])


@njit(cache=True)
def multi_replay(online,args):
    result=np.empty((len(online),len(args[-1][-1])),dtype=np.float32)
    for k,x in enumerate(online):result[k]=multi_step(float(x),args)
    return result


@njit(cache=True)
def multi_zero_replay(online,args):
    result=np.empty((len(online),len(args[-1][-1])),dtype=np.float32)
    for k,x in enumerate(online):result[k]=multi_zero_step(float(x),args)
    return result


def same_graph(a,b):
    if set(a)!=set(b):return False
    for j in a:
        x,y=a[j],b[j]
        if x.parents!=y.parents or x.training!=y.training:return False
        for key in ('prior','own_table','own_support','pair_tables','pair_support'):
            if not np.array_equal(np.asarray(getattr(x,key)),np.asarray(getattr(y,key))):return False
    return True


class PreparedMultiHead:
    def __init__(self,prepared):
        if not prepared or any(not isinstance(x,PreparedJoint) for x in prepared.values()):
            raise ValueError('Nonempty named frozen joint heads required')
        self.prepared=dict(prepared);self.names=tuple(prepared)
        first=next(iter(prepared.values()));self.selection=copy.deepcopy(first.selection)
        self.graph_models=copy.deepcopy(first.graph_models)
        for item in prepared.values():
            if item.selection!=self.selection or not same_graph(item.graph_models,self.graph_models):
                raise ValueError('Shared execution requires the same selection, H map and graph refit')
            if item.model.training.get('early_training_cutoff')!=first.model.training.get('early_training_cutoff'):
                raise ValueError('Shared heads must have the same training time domain')

    def new_state(self,historical):return MultiHeadState(self,historical)


class MultiHeadState:
    def __init__(self,prepared,historical):
        historical=points(historical);self.prepared=prepared
        direct=set().union(*(p.direct_features for p in prepared.prepared.values()))
        if direct:seed=FeatureState(historical)
        else:
            seed=FeatureState(np.empty(0,dtype=np.float32));seed.context=h_context(historical)
        selection=copy.deepcopy(prepared.selection)
        methods={p.method for p in prepared.prepared.values()}
        if not methods.intersection({'M3','M4','M5'}):selection['variable_features']=[]
        state,required=state_arguments(seed,selection,prepared.graph_models,use_graph=bool(methods.intersection({'M3','M5'})))
        needed=direct|required;self.used_features=tuple(sorted(needed))
        heads=tuple((copy.deepcopy(p.coordinate_args),copy.deepcopy(p.head)) for p in prepared.prepared.values())
        history=np.zeros((32,N_FEATURES),dtype=np.float32);history[16:]=seed.history
        models=(state,heads,history,np.zeros(1,dtype=np.int64),seed.context.copy(),np.zeros(len(heads),dtype=np.float32))
        self.zero_features=not seed.engines or not needed
        if self.zero_features:self.args=(np.zeros(N_FEATURES,dtype=np.float32),models)
        else:
            engine,norm=raw_arguments(seed,needed);self.args=(engine,norm,models)

    @property
    def counter(self):return int(self.args[-1][3][0])

    @property
    def unrounded_probabilities(self):return np.asarray([head[-1][1] for _,head in self.args[-1][1]])

    @property
    def total_instance_array_bytes(self):return unique_array_bytes(self.args)

    def predict_one(self,point):
        value=float(point)
        if not np.isfinite(value) or abs(value)>np.finfo(np.float32).max:raise ValueError('Finite float32 scalar observation required')
        function=multi_zero_step if self.zero_features else multi_step
        return function(float(np.float32(value)),self.args).copy()

    def replay(self,online):
        function=multi_zero_replay if self.zero_features else multi_replay
        return function(points(online),self.args)

    def snapshot(self):return copy.deepcopy(self)

    @classmethod
    def restore(cls,snapshot):
        if not isinstance(snapshot,cls):raise TypeError('Exact multihead snapshot required')
        return copy.deepcopy(snapshot)
