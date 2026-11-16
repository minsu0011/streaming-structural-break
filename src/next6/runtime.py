"""Numerically identical fused scalar NEXT6 inference; no scientific refit."""
import numpy as np
from numba import njit,typeof
from src.next4.scale_asymmetry import appended_step
from src.next2.dependence_runtime import step as dependence_step
from src.next6.nodes import NodeState,B_POS,C_POS,temporal_replay
from src.next6.graph import GraphState,graph_replay
from src.next5.reference_features import ReferenceState
from src.next5.runtime import resident_class
from src.models.compact_trees import tree_margin
from src.next6.model import EXTRA_COLUMNS

FLOAT32_MAX=float(np.finfo(np.float32).max)

@njit(cache=False)
def kernel(point,args):
    if not np.isfinite(point):raise ValueError('Finite current observation required')
    ja,ca,ba,da,center,spread,threshold,degenerate,history,ewma,counter,gring,coef,variance,tables,own_tables,parents,summary,extra,tree_args,output=args
    jf=appended_step(point,*ja);cf=dependence_step(point,*ca)
    for j in range(77):output[j]=jf[j]
    for j in range(24):output[77+j]=cf[j]
    if len(output)>101:
        bf=appended_step(point,*ba);df=dependence_step(point,*da);raw=np.empty(16,dtype=np.float32)
        for j in range(12):raw[j]=bf[B_POS[j]]
        for j in range(4):raw[12+j]=df[C_POS[j]]
        x=np.empty((1,16),dtype=np.float32);s=np.ones((1,16),dtype=np.int8)
        for j in range(16):
            x[0,j]=min(max((raw[j]-center[j])/spread[j],-12.),12.)
            if raw[j]<threshold[0,j]:s[0,j]=0
            if raw[j]>threshold[1,j]:s[0,j]=2
            if degenerate[j]:s[0,j]=3
        temporal=temporal_replay(x,history,ewma,counter);output[101:245]=temporal[0]
        if len(extra):
            value,updated=graph_replay(temporal,s,gring,coef,variance,tables,own_tables,parents,summary)
            gring[:,:]=updated
            for j in range(len(extra)):output[245+j]=value[0,extra[j]]
    return 1./(1.+np.exp(-tree_margin(output,*tree_args)))

@njit(cache=False)
def replay(points,args):
    out=np.empty(len(points))
    for t in range(len(points)):out[t]=kernel(float(points[t]),args)
    return out

class Prepared:
    def __init__(self,model,resident=True):
        model.validate();self.model=model;self.resident=resident
        tree=list(model.compact.args());features=tree[1].copy();used=features>=0;features[used]=model.columns[features[used]];tree[1]=features;self.tree=tuple(tree)
    def new_state(self,historical):return State(self,historical)

class State:
    def __init__(self,prepared,historical):
        j=ReferenceState(historical,{'version':'next5_reference_features_v1','kind':'J102'})
        c=ReferenceState(historical,{'version':'next5_reference_features_v1','kind':'C205'})
        n=NodeState(historical);model=prepared.model;extra=np.asarray(EXTRA_COLUMNS[model.method],dtype=np.int64)
        if model.graph is None:
            gring=n.pseudo_s[-16:].copy();coef=np.zeros((2,16,82));variance=np.ones((2,16));tables=np.full((2,16,27,3),1/3);own=np.full((2,16,3,3),1/3);parents=np.full((16,2,2),-1,dtype=np.int64);summary=np.zeros(9)
        else:
            g=model.graph;s=GraphState(g,n.pseudo_s[-16:]);gring=s.ring;coef=g.coef;variance=g.variance;tables=g.tables;own=g.own_tables;parents=s.parents;summary=s.summary
        output=np.empty((101 if model.method=='BASE_ONLY' else 245)+len(extra),dtype=np.float32)
        self.args=(j.state.args,c.state.args,n.b.state.args,n.c.state.args,n.center,n.spread,n.threshold,n.degenerate,
            n.history,n.ewma,n.counter,gring,coef,variance,tables,own,parents,summary,extra,prepared.tree,output)
        self.resident=resident_class(kernel,typeof(self.args))(self.args) if prepared.resident else None
    def predict_one(self,point):
        value=float(point)
        if not np.isfinite(value) or abs(value)>FLOAT32_MAX:raise ValueError('Finite float32 current observation required')
        value=float(np.float32(value))
        prediction=float(self.resident.predict_one(value) if self.resident is not None else kernel(value,self.args))
        if not np.isfinite(prediction) or not 0<=prediction<=1:raise RuntimeError('Invalid scalar probability')
        return prediction
    def replay(self,points):
        values=np.asarray(points,dtype=np.float64)
        if values.ndim!=1 or not np.isfinite(values).all() or np.any(abs(values)>FLOAT32_MAX):raise ValueError('Finite float32 scalar stream required')
        return replay(values.astype(np.float32),self.args)
