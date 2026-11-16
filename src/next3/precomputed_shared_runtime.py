"""Prediction-exact shared runtime with H/fixed-age arithmetic precomputed."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
from numba import njit,typeof
from src.next.whitening import innovation_update
from src.next.channel_evidence import _logadd
from src.next2.pruned_runtime import pair_channel_update
from src.next2.complement_bank import raw_complement
from src.next2.precomputed_serial_runtime import precompute,precomputed_channel_step
from src.next2.used_feature_runtime import used_arch_step
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor,shared_model_step
from src.next.fused_runtime import _resident_class,unique_array_bytes
from src.next.runtime_components import resident_parts


def source_hash():
    from src.next2.shared_blend_runtime import source_hash as parent_hash
    from src.next2.precomputed_serial_runtime import source_hash as serial_hash
    return hashlib.sha256(Path(__file__).read_bytes()+parent_hash().encode()+serial_hash().encode()).hexdigest()


@njit(cache=False)
def precompute_dependence(ages,widths):
    table=np.empty((len(ages),4),dtype=np.float64);logs=np.empty(len(ages),dtype=np.float64)
    width=0.
    for k in range(len(ages)):
        age=ages[k];width+=widths[k]
        table[k,0]=np.log(widths[k])-.5*np.log1p(.25*age)
        table[k,1]=1.+.25*age
        table[k,2]=np.log(widths[k])-.5*np.log1p(age)
        table[k,3]=1.+age
        logs[k]=np.log(width)
    return table,logs


@njit(cache=False)
def precomputed_dependence(values,args,table,logs):
    ages,widths,ewma,cusum,cumulative,prefix,counter,output=args
    t=counter[0]+1
    for channel in range(4):
        value=min(max(values[channel],-12.),12.);cursor=channel*6
        for k,half in enumerate((8.,32.)):
            alpha=1.-np.exp(-np.log(2.)/half)
            ewma[channel,k]=(1.-alpha)*ewma[channel,k]+alpha*value
            variance=max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)),1e-12)
            output[cursor+k]=ewma[channel,k]/np.sqrt(variance)
        cusum[channel,0]=max(0.,cusum[channel,0]+value-.25)
        cusum[channel,1]=max(0.,cusum[channel,1]-value-.25)
        output[cursor+2]=np.log1p(cusum[channel,0]);output[cursor+3]=np.log1p(cusum[channel,1])
        cumulative[channel]+=value
        b025,b1=-1e300,-1e300;count=0
        for k in range(len(ages)):
            age=ages[k]
            if age>t:break
            count+=1
            total=cumulative[channel]-prefix[(t-age)%len(prefix),channel];energy=total*total
            b025=_logadd(b025,table[k,0]+.5*.25*energy/table[k,1])
            b1=_logadd(b1,table[k,2]+.5*energy/table[k,3])
        prefix[t%len(prefix),channel]=cumulative[channel]
        output[cursor+4]=b025-logs[count-1];output[cursor+5]=b1-logs[count-1]
    counter[0]=t
    return output


@njit(cache=False)
def feature_step(point,args):
    base,serial_table,serial_logs,dependence_table,dependence_logs=args
    old_args,shared_args,old_positions,new_positions,new_columns,output,models=base
    old=used_arch_step(point,old_args)
    (filter_args,parameters,conditional,pair_calibration,pair_work,pair_evidence,
        age_variances,ring,counter,center,calibration,work,empty,dependence_evidence)=shared_args
    residual=innovation_update(point,*filter_args)
    variance=max(parameters[2]+parameters[3]*conditional[0]+parameters[4]*conditional[1],parameters[6])
    centered=residual-parameters[0];z=min(max(centered/np.sqrt(variance),-4.),4.)
    pair_channel_update(residual,parameters,conditional,3,pair_work)
    for j in range(2):pair_work[j]=(pair_work[j]-pair_calibration[0,j])/pair_calibration[1,j]
    pair=precomputed_channel_step(pair_work,pair_evidence,age_variances,serial_table,serial_logs)
    raw_complement(z,empty,ring,counter,center,3,work)
    for j in range(4):work[j]=(work[j]-calibration[0,j])/calibration[1,j]
    precomputed_dependence(work,dependence_evidence,dependence_table,dependence_logs)
    for j in range(len(old_positions)):output[old_positions[j]]=old[j]
    for j in range(len(new_positions)):output[new_positions[j]]=pair[new_columns[j]]
    return output


@njit(cache=False)
def model_step(args):return shared_model_step(args[0])


@njit(cache=False)
def kernel(point,args):
    feature_step(point,args)
    return model_step(args)


class PreparedPrecomputedSharedPredictor(PreparedSharedBlendPredictor):
    def __init__(self,model,*,resident=True):
        super().__init__(model,resident=resident)
        ages=np.asarray([1,2,4,8,16,32,64,128],dtype=np.int64)
        self.dependence_table,self.dependence_logs=precompute_dependence(ages,np.diff(np.r_[0,ages]).astype(float))
        self.dependence_table.setflags(write=False);self.dependence_logs.setflags(write=False)
        self.runtime_source_sha256=source_hash()

    def new_state(self,historical):return PrecomputedSharedDetector(self,historical)


class PrecomputedSharedDetector:
    def __init__(self,prepared,historical):
        self.prepared=prepared
        # Reuse the exact frozen H initialization and tree mapping.
        from src.next2.shared_blend_runtime import SharedBlendDetector
        resident=prepared.resident
        # No mutation of a shared prepared object, including its resident flag.
        adapter=SimpleNamespace(primary=prepared.primary,complement=prepared.complement,model=prepared.model,resident=False)
        base=SharedBlendDetector(adapter,historical)
        self.base=base
        shared=base.call_args[1]
        serial_table,serial_logs=precompute(shared[5][0],shared[5][1],shared[6])
        self.call_args=(base.call_args,serial_table,serial_logs,prepared.dependence_table,prepared.dependence_logs)
        self.kernel=kernel
        self.resident=_resident_class(kernel,typeof(self.call_args))(self.call_args) if resident else None

    def predict_one(self,point):
        return float(self.resident.predict_one(float(point)) if self.resident is not None else kernel(float(point),self.call_args))

    @property
    def state_array_bytes(self):return unique_array_bytes((self.call_args[0][:-1],*self.call_args[1:]))

    @property
    def immutable_tree_array_bytes(self):return unique_array_bytes(self.call_args[0][-1])


class PrecomputedSharedParts:
    def __init__(self,state):
        self.resident=resident_parts(feature_step,model_step,typeof(state.call_args))(state.call_args)

    def feature_update(self,point):self.resident.feature_update(float(point))

    def model_predict(self):return float(self.resident.model_predict())
