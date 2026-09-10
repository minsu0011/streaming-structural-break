"""Exact tree-used feature runtime, including sparse legacy EWMA computation.

Model bytes and all split thresholds stay frozen. Historical replay computes
only the old columns referenced by a tree; online EWMA state includes only
their required moment/lag statistics. Reference 77-column names stay intact
inside the scientific model, while the materialized tree input is smaller.
"""
from pathlib import Path
from functools import lru_cache
import hashlib
import numpy as np
from numba import njit,typeof
from src.next.fused_runtime import PreparedNextPredictor,_resident_class,unique_array_bytes
from src.next2.pruned_runtime import PreparedCompactPredictor,CompactPairState
from src.next.fast_variance_initialization import OLD_CONFIG,OLD_POLICY
from src.features.ar_order_input import historical_filter
from src.features.arch_input import historical_variance_filter,arch_one
from src.features.ar_residual_input import innovation_one
from src.features.streaming import StreamingFeatureState,feature_names
from src.models.compact_trees import tree_margin


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('pruned_runtime.py').read_bytes()).hexdigest()


def dependencies(columns):
    pairs=set()
    for column in columns:
        if column<6:
            continue
        scale,kind=divmod(int(column)-6,15)
        needs={0:(0,),1:(0,),2:(1,),3:(0,1),4:(2,),5:(3,),6:(4,),7:(5,),
            8:(12,13,14,15,16),9:(6,),10:(7,),11:(8,),12:(9,),13:(10,),14:(11,)}[kind]
        pairs.update((scale,j) for j in needs)
    return np.asarray(sorted(pairs),dtype=np.int64).reshape(-1,2)


@njit(cache=False)
def used_old_update(point,reference,baseline,thresholds,ring,ew,counters,parameters,alphas,lags,pairs,lookup,columns,work,output):
    clip_z,drift,decay,moment_floor,rate_floor,tail_low,tail_high=parameters
    x=reference[0] if not np.isfinite(point) else point
    z=min(max((x-reference[0])/reference[1],-clip_z),clip_z)
    t=int(counters[0])+1
    counters[0]=t
    # Only requested current summaries are updated or transformed.
    for column in columns:
        if column==3:
            counters[1]=max(0.,counters[1]+z-drift)
        elif column==4:
            counters[2]=max(0.,counters[2]-z-drift)
        elif column==5:
            counters[3]+=(z-counters[3])/t
    for k in range(len(pairs)):
        scale,stat=pairs[k]
        if stat==0:
            value=z
        elif stat==1:
            value=z*z
        elif stat==2:
            value=abs(z)
        elif stat==3:
            value=1. if abs(z)>tail_low else 0.
        elif stat==4:
            value=1. if abs(z)>tail_high else 0.
        elif stat==5:
            value=1. if z>0. else 0.
        elif stat<10:
            value=z*ring[(int(counters[4])-lags[stat-6])%len(ring)]
        elif stat<12:
            error=z-reference[4]*ring[(int(counters[4])-1)%len(ring)]
            value=error if stat==10 else error*error
        else:
            cell=0
            for boundary in thresholds:
                cell+=int(z>boundary)
            value=1. if cell==stat-12 else 0.
        ew[k]+=alphas[scale]*(value-ew[k])
    for scale in range(3):
        counters[5+scale]=(1-alphas[scale])*counters[5+scale]+1.
    for k,column in enumerate(columns):
        if column==0:
            output[k]=z
        elif column==1:
            output[k]=abs(z)
        elif column==2:
            output[k]=min(max((x-reference[2])/reference[3],-clip_z),clip_z)
        elif column==3:
            output[k]=np.log1p(counters[1])
        elif column==4:
            output[k]=np.log1p(counters[2])
        elif column==5:
            output[k]=counters[3]
        else:
            scale,kind=divmod(column-6,15)
            root_n=np.sqrt(counters[5+scale])
            if kind==0 or kind==1:
                value=(ew[lookup[scale,0]]-baseline[0])*root_n
                output[k]=value if kind==0 else abs(value)
            elif kind==2:
                output[k]=np.log(max(ew[lookup[scale,1]],moment_floor)/max(baseline[1],moment_floor))
            elif kind==3:
                output[k]=np.log(max(ew[lookup[scale,1]]-ew[lookup[scale,0]]**2,moment_floor)/max(baseline[1]-baseline[0]**2,moment_floor))
            elif kind==4:
                output[k]=(ew[lookup[scale,2]]-baseline[2])*root_n
            elif kind<=7:
                stat=kind-2
                rate=baseline[stat]
                output[k]=(ew[lookup[scale,stat]]-rate)*root_n/np.sqrt(max(rate*(1-rate),rate_floor))
            elif kind==8:
                for j in range(5):
                    work[j]=abs(ew[lookup[scale,12+j]]-baseline[12+j])
                output[k]=np.sum(work)*root_n
            elif kind<=12:
                stat=kind-3
                output[k]=(ew[lookup[scale,stat]]-baseline[stat])*root_n
            elif kind==13:
                output[k]=(ew[lookup[scale,10]]-baseline[10])*root_n/np.sqrt(max(baseline[11],moment_floor))
            else:
                output[k]=np.log(max(ew[lookup[scale,11]],moment_floor)/max(baseline[11],moment_floor))
    ring[int(counters[4])%len(ring)]=z
    counters[4]+=1
    return output


@njit(cache=False)
def replay_old(points,args):
    result=np.empty((len(points),len(args[-1])),dtype=np.float32)
    for k in range(len(points)):
        result[k]=used_old_update(float(points[k]),*args)
    return result


class UsedOldState:
    def __init__(self,historical,columns):
        ar_ref,coefficients,ar_ring,residual=historical_filter(historical,OLD_CONFIG,8)
        ar_args=(ar_ref,coefficients,ar_ring,np.zeros(1,dtype=np.int64))
        vparams,previous,transformed=historical_variance_filter(residual,OLD_CONFIG)
        original=StreamingFeatureState(transformed,config=OLD_CONFIG)
        columns=np.asarray(columns,dtype=np.int64)
        if not len(columns) or np.any((columns<0)|(columns>=51)) or len(np.unique(columns))!=len(columns):
            raise ValueError('Unique used old ABCD columns required')
        pairs=dependencies(columns)
        lookup=np.full((3,17),-1,dtype=np.int64)
        for k,(scale,stat) in enumerate(pairs):
            lookup[scale,stat]=k
        ew=np.asarray([original.ew[scale,stat] for scale,stat in pairs],dtype=np.float64)
        counters=np.zeros(8)
        counters[4]=len(original.ring)
        args=(original.reference,original.baseline,original.thresholds,original.ring,ew,counters,
            original.parameters,original.alphas,original.lags,pairs,lookup,columns,np.empty(5),np.empty(len(columns),dtype=np.float32))
        temporary=tuple(a.copy() if isinstance(a,np.ndarray) else a for a in args)
        matrix=replay_old(np.asarray(transformed,dtype=np.float32),temporary)
        if not len(matrix):
            matrix=np.zeros((1,len(columns)),dtype=np.float32)
        burn=min(OLD_POLICY.burn_in,max(0,len(matrix)//2))
        values=np.ascontiguousarray(matrix[burn:].T,dtype=np.float64)
        center=np.median(values,axis=1)
        scale=np.maximum(np.median(np.abs(values-center[:,None]),axis=1)*1.4826,OLD_POLICY.standard_deviation_floor)
        mask=~np.isin(columns,[3,4,5])
        self.call_args=(ar_args,(vparams,previous),args,center,scale,mask,float(OLD_POLICY.clip),np.empty(len(columns),dtype=np.float32))
        self.ewma_cells=len(ew)


@njit(cache=False)
def used_arch_step(point,args):
    ar_args,variance_args,old_args,center,scale,mask,clip,output=args
    residual=innovation_one(point,*ar_args)
    value=arch_one(float(residual),*variance_args)
    raw=used_old_update(float(value),*old_args)
    for j in range(len(raw)):
        output[j]=min(max((raw[j]-center[j])/scale[j],-clip),clip) if mask[j] else raw[j]
    return output


@lru_cache(maxsize=16)
def used_kernel(new_function):
    @njit(cache=False)
    def fused(point,args):
        old_args,new_args,old_positions,new_positions,new_columns,output,tree_args=args
        old=used_arch_step(point,old_args)
        new=new_function(point,*new_args)
        for j in range(len(old_positions)):
            output[old_positions[j]]=old[j]
        for j in range(len(new_positions)):
            output[new_positions[j]]=new[new_columns[j]]
        margin=tree_margin(output,*tree_args)
        return np.float32(1./(1.+np.exp(-margin)))
    return fused


class PreparedUsedPredictor(PreparedNextPredictor):
    def __init__(self,model,*,resident=True):
        contract=PreparedCompactPredictor(model,resident=False)
        super().__init__(model,resident=resident)
        raw=model.compact.features
        self.used=np.unique(model.columns[raw[raw>=0]])
        self.names=tuple(model.names[k] for k in self.used)
        self.old_positions=np.asarray([k for k,n in enumerate(self.names) if n.startswith('arch8__')],dtype=np.int32)
        self.new_positions=np.asarray([k for k,n in enumerate(self.names) if not n.startswith('arch8__')],dtype=np.int32)
        old_names=['arch8__'+n for n in feature_names(OLD_CONFIG)]
        self.old_columns=np.asarray([old_names.index(self.names[k]) for k in self.old_positions],dtype=np.int64)
        full_new={model.names[k]:int(column) for k,column in zip(contract.new_positions,contract.new_columns)}
        self.new_columns=np.asarray([full_new[self.names[k]] for k in self.new_positions],dtype=np.int32)
        remap=np.full(len(model.names),-1,dtype=np.int32)
        remap[self.used]=np.arange(len(self.used),dtype=np.int32)
        args=list(self.tree_args)
        mapped=args[1].copy()
        mapped[mapped>=0]=remap[mapped[mapped>=0]]
        args[1]=mapped
        self.tree_args=tuple(args)
        self.mode=contract.mode
        self.runtime_source_sha256=source_hash()

    def new_state(self,historical):
        return UsedDetector(self,historical)


class UsedDetector:
    def __init__(self,prepared,historical):
        self.prepared=prepared
        self.old=UsedOldState(historical,prepared.old_columns)
        self.new=CompactPairState(historical,prepared.model.extension['settings'],prepared.mode)
        self.output=np.empty(len(prepared.names),dtype=np.float32)
        self.call_args=(self.old.call_args,self.new.call_args,prepared.old_positions,prepared.new_positions,
            prepared.new_columns,self.output,prepared.tree_args)
        self.kernel=used_kernel(self.new.function)
        self.resident=_resident_class(self.kernel,typeof(self.call_args))(self.call_args) if prepared.resident else None

    def predict_one(self,point):
        return float(self.resident.predict_one(float(point)) if self.resident is not None else self.kernel(float(point),self.call_args))

    @property
    def state_array_bytes(self):
        return unique_array_bytes(self.call_args[:-1])

    @property
    def immutable_tree_array_bytes(self):
        return unique_array_bytes(self.prepared.tree_args)
