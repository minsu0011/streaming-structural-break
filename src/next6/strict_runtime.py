"""Fused strict nongraph comparator, reusing exact original base feature kernel."""
from types import SimpleNamespace
import numpy as np
from numba import njit,typeof
from src.next6.runtime import State as BaseState,kernel as base_kernel,FLOAT32_MAX
from src.next6.strict_baseline import extras
from src.next6.nodes import B_POS,C_POS
from src.models.compact_trees import CompactTrees,tree_margin
from src.next5.runtime import resident_class,unique_array_bytes

@njit(cache=False)
def kernel(point,args):
    base,ring,ewma,duration,counter,tree,out=args
    base_kernel(point,base) # Constant dummy leaf; reuse unchanged 245 feature calculations.
    raw=np.empty(16,dtype=np.float32);s=np.ones((1,16),dtype=np.int8)
    for j in range(12):raw[j]=base[2][-1][B_POS[j]]
    for j in range(4):raw[12+j]=base[3][-1][-1][C_POS[j]]
    for j in range(16):
        if raw[j]<base[6][0,j]:s[0,j]=0
        if raw[j]>base[6][1,j]:s[0,j]=2
        if base[7][j]:s[0,j]=3
    out[:245]=base[-1];out[245:]=extras(s,ring,ewma,duration,counter)[0]
    return 1./(1.+np.exp(-tree_margin(out,*tree)))

@njit(cache=False)
def replay(points,args):
    out=np.empty(len(points))
    for i in range(len(points)):out[i]=kernel(float(points[i]),args)
    return out

class PreparedStrict:
    def __init__(self,model):
        model.validate();self.model=model;tree=list(model.compact.args());cols=tree[1].copy();use=cols>=0;cols[use]=model.columns[cols[use]];tree[1]=cols;self.tree=tuple(tree)
    def new_state(self,h):return StrictScalar(self,h)

class StrictScalar:
    def __init__(self,p,h):
        dummy=CompactTrees(np.array([0],np.int32),np.array([-1],np.int32),np.zeros(1),np.full(1,-1,np.int32),np.full(1,-1,np.int32),np.zeros(1),np.ones(1,dtype=bool),0.)
        b=BaseState(SimpleNamespace(model=SimpleNamespace(method='M1',graph=None),tree=dummy.args(),resident=False),h)
        self.args=(b.args,b.args[11].copy(),np.zeros((2,16)),np.zeros(16),np.zeros(1),p.tree,np.empty(390,dtype=np.float32))
        self.resident=resident_class(kernel,typeof(self.args))(self.args)
    def predict_one(self,point):
        value=float(point)
        if not np.isfinite(value) or abs(value)>FLOAT32_MAX:raise ValueError('Finite float32 current observation required')
        return float(self.resident.predict_one(float(np.float32(value))))
    def replay(self,points):
        values=np.asarray(points,dtype=np.float64)
        if values.ndim!=1 or not np.isfinite(values).all() or np.any(abs(values)>FLOAT32_MAX):raise ValueError('Finite float32 stream required')
        return replay(values.astype(np.float32),self.args)
    @property
    def state_array_bytes(self):return unique_array_bytes(self.args[0][:12]+(self.args[0][-1],)+self.args[1:5]+(self.args[-1],))
