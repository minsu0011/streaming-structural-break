"""Exact H-only precomputation of the fixed serial Bayes arithmetic."""
from pathlib import Path
import hashlib
import numpy as np
from numba import njit,typeof
from src.next.channel_evidence import CHANNEL_STATS,_logadd
from src.next.fused_runtime import _resident_class,unique_array_bytes
from src.next2.pruned_runtime import CompactPairState,_values
from src.next2.used_feature_runtime import PreparedUsedPredictor,UsedOldState,used_kernel,source_hash as used_hash


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+used_hash().encode()).hexdigest()


@njit(cache=False)
def precompute(ages,widths,age_variances):
    table=np.empty((len(age_variances),len(ages),6),dtype=np.float64)
    logs=np.empty((len(ages),3),dtype=np.float64)
    width=0.
    for k in range(len(ages)):
        age=ages[k];width+=widths[k]
        logs[k,0]=np.log(k+1);logs[k,1]=np.log(width);logs[k,2]=np.log2(age)
        for channel in range(len(age_variances)):
            ratio=age/age_variances[channel,k]
            table[channel,k,0]=np.log(widths[k])-.5*np.log1p(.25*age*ratio)
            table[channel,k,1]=.5*.25*ratio*ratio
            table[channel,k,2]=1.+.25*age*ratio
            table[channel,k,3]=np.log(widths[k])-.5*np.log1p(age*ratio)
            table[channel,k,4]=.5*ratio*ratio
            table[channel,k,5]=1.+age*ratio
    return table,logs


@njit(cache=False)
def precomputed_channel_step(values,args,age_variances,table,logs):
    ages,widths,ewma,cusum,cumulative,prefix,counter,memory,output=args
    t=counter[0]+1
    halves=(8.,32.,128.)
    for channel in range(len(values)):
        value=min(max(values[channel],-12.),12.)
        cursor=channel*len(CHANNEL_STATS)
        output[cursor]=value
        for k in range(3):
            alpha=1.-np.exp(-np.log(2.)/halves[k])
            ewma[channel,k]=(1.-alpha)*ewma[channel,k]+alpha*value
            variance=max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)),1e-12)
            output[cursor+1+k]=ewma[channel,k]/np.sqrt(variance)
        cusum[channel,0]=max(0.,cusum[channel,0]+value-.25)
        cusum[channel,1]=max(0.,cusum[channel,1]-value-.25)
        output[cursor+4]=np.log1p(cusum[channel,0])
        output[cursor+5]=np.log1p(cusum[channel,1])
        cumulative[channel]+=value
        maximum,maximum_index=0.,0
        mix,b025,b1=-1e300,-1e300,-1e300
        count=0
        for k in range(len(ages)):
            age=ages[k]
            if age>t:break
            count+=1
            total=cumulative[channel]-prefix[(t-age)%len(prefix),channel]
            energy=total*total
            glr=.5*energy/age_variances[channel,k]
            if glr>maximum:maximum,maximum_index=glr,k
            mix=_logadd(mix,glr)
            b025=_logadd(b025,table[channel,k,0]+table[channel,k,1]*energy/table[channel,k,2])
            b1=_logadd(b1,table[channel,k,3]+table[channel,k,4]*energy/table[channel,k,5])
        prefix[t%len(prefix),channel]=cumulative[channel]
        memory[channel,0]=max(memory[channel,0],maximum)
        memory[channel,1]=max(np.exp(-np.log(2.)/32.)*memory[channel,1],maximum)
        output[cursor+6]=maximum
        output[cursor+7]=mix-logs[count-1,0]
        output[cursor+8]=logs[maximum_index,2]
        output[cursor+9]=b025-logs[count-1,1]
        output[cursor+10]=b1-logs[count-1,1]
        output[cursor+11]=memory[channel,0]
        output[cursor+12]=memory[channel,1]
    counter[0]=t
    return output


@njit(cache=False)
def precomputed_step(point,filter_args,parameters,conditional,mode,calibration,work,args,age_variances,table,logs):
    return precomputed_channel_step(_values(point,filter_args,parameters,conditional,mode,calibration,work),args,age_variances,table,logs)


class PreparedPrecomputedSerialPredictor(PreparedUsedPredictor):
    def __init__(self,model,*,resident=True):
        super().__init__(model,resident=resident)
        if model.extension!={'kind':'serial_variance','settings':{'order':8,'normalization':'arch1','max_age':512}}:
            raise ValueError('Exact J102 serial variance contract required')
        self.runtime_source_sha256=source_hash()

    def new_state(self,historical):
        return PrecomputedSerialDetector(self,historical)


class PrecomputedSerialDetector:
    def __init__(self,prepared,historical):
        self.prepared=prepared
        self.old=UsedOldState(historical,prepared.old_columns)
        self.new=CompactPairState(historical,prepared.model.extension['settings'],prepared.mode)
        self.table,self.logs=precompute(self.new.evidence_args[0],self.new.evidence_args[1],self.new.age_variances)
        new_args=(*self.new.call_args,self.table,self.logs)
        self.output=np.empty(len(prepared.names),dtype=np.float32)
        self.call_args=(self.old.call_args,new_args,prepared.old_positions,prepared.new_positions,
            prepared.new_columns,self.output,prepared.tree_args)
        self.kernel=used_kernel(precomputed_step)
        self.resident=_resident_class(self.kernel,typeof(self.call_args))(self.call_args) if prepared.resident else None

    def predict_one(self,point):
        return float(self.resident.predict_one(float(point)) if self.resident is not None else self.kernel(float(point),self.call_args))

    @property
    def state_array_bytes(self):return unique_array_bytes(self.call_args[:-1])

    @property
    def immutable_tree_array_bytes(self):return unique_array_bytes(self.prepared.tree_args)
