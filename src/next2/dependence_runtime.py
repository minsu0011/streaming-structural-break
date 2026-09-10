"""Exact compact C205 state: two EWMAs, two CUSUMs and two Bayes statistics.

No CDF sorting/mapping, EWMA128, GLR, age-at-max or GLR memory is computed.
H initialization preserves the original normalization and calibration math.
"""
import numpy as np
from numba import njit,typeof
from src.next.whitening import fit_historical,innovation_update
from src.next.fast_variance_initialization import fast_variance_parameters
from src.next.channel_evidence import _logadd
from src.next2.complement_bank import normalized_one,historical_raw,raw_complement
from src.next.fused_runtime import PreparedNextPredictor,_kernel,_resident_class,unique_array_bytes

STATS=('ewma8','ewma32','cusum_positive','cusum_negative','bayes025','bayes1')
NAMES=tuple('next2_complement__dependence_'+str(k)+'_'+s for k in range(4) for s in STATS)


@njit(cache=False)
def historical_z(residuals,parameters,variance):
    state=np.full(3,variance)
    z=np.empty(len(residuals))
    for j in range(len(z)):
        z[j]=normalized_one(residuals[j],parameters,state)
    return z,state


@njit(cache=False)
def evidence(values,ages,widths,ewma,cusum,cumulative,prefix,counter,output):
    t=counter[0]+1
    for channel in range(4):
        value=min(max(values[channel],-12.),12.)
        cursor=channel*6
        for k,half in enumerate((8.,32.)):
            alpha=1.-np.exp(-np.log(2.)/half)
            ewma[channel,k]=(1.-alpha)*ewma[channel,k]+alpha*value
            variance=max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)),1e-12)
            output[cursor+k]=ewma[channel,k]/np.sqrt(variance)
        cusum[channel,0]=max(0.,cusum[channel,0]+value-.25)
        cusum[channel,1]=max(0.,cusum[channel,1]-value-.25)
        output[cursor+2]=np.log1p(cusum[channel,0])
        output[cursor+3]=np.log1p(cusum[channel,1])
        cumulative[channel]+=value
        b025,b1,width=-1e300,-1e300,0.
        for k in range(len(ages)):
            age=ages[k]
            if age>t:
                break
            width+=widths[k]
            total=cumulative[channel]-prefix[(t-age)%len(prefix),channel]
            energy=total*total
            b025=_logadd(b025,np.log(widths[k])-.5*np.log1p(.25*age)+.5*.25*energy/(1.+.25*age))
            b1=_logadd(b1,np.log(widths[k])-.5*np.log1p(age)+.5*energy/(1.+age))
        prefix[t%len(prefix),channel]=cumulative[channel]
        output[cursor+4]=b025-np.log(width)
        output[cursor+5]=b1-np.log(width)
    counter[0]=t
    return output


@njit(cache=False)
def step(point,filter_args,parameters,conditional,ring,counter,center,calibration,work,empty,evidence_args):
    residual=innovation_update(point,*filter_args)
    z=normalized_one(residual,parameters,conditional)
    raw_complement(z,empty,ring,counter,center,3,work)
    for j in range(4):
        work[j]=(work[j]-calibration[0,j])/calibration[1,j]
    return evidence(work,*evidence_args)


class CompactDependenceState:
    def __init__(self,historical):
        fit=fit_historical(historical,8)
        r=fit['historical_innovations']
        parameters,conditional=fast_variance_parameters(r,'arch1')
        # Use the original capped mean rather than invert omega/(1-alpha),
        # whose floating-point rounding need not reproduce H normalization.
        variance=max(float(np.minimum((r-parameters[0])**2,parameters[5]).mean()),1e-12)
        z,replayed=historical_z(r,parameters,variance)
        np.testing.assert_array_equal(replayed,conditional)
        center=np.asarray([z.mean(),(z*z).mean(),np.abs(z).mean()])
        empty=np.empty(0)
        raw,ring,counter=historical_raw(z,empty,center,3,4)
        raw=raw[min(160,len(raw)//4):]
        calibration=np.stack([raw.mean(axis=0),np.maximum(raw.std(axis=0),1e-6)])
        ages=np.asarray([1,2,4,8,16,32,64,128],dtype=np.int64)
        evidence_args=(ages,np.diff(np.r_[0,ages]).astype(float),np.zeros((4,2)),np.zeros((4,2)),
            np.zeros(4),np.zeros((129,4)),np.zeros(1,dtype=np.int64),np.empty(24,dtype=np.float32))
        self.args=(tuple(fit[k] for k in ('reference','coefficients','ring','counter')),
            parameters,conditional,ring,counter,center,calibration,np.empty(4),empty,evidence_args)

    def update(self,point):
        return step(float(point),*self.args)


class PreparedDependencePredictor(PreparedNextPredictor):
    def __init__(self,model,*,resident=True):
        from src.next2.io import register
        register(model.extension)
        super().__init__(model,resident=resident)
        if model.extension!={'kind':'next2_complement','settings':{'family':'dependence','order':8,'cdf_bins':8,'cap':4}} or tuple(model.names)!=NAMES:
            raise ValueError('Exact frozen C205 feature contract required')
        self.old_positions=np.empty(0,dtype=np.int32)
        self.old_columns=np.empty(0,dtype=np.int32)
        self.new_positions=np.arange(24,dtype=np.int32)
        self.new_columns=np.arange(24,dtype=np.int32)

    def new_state(self,historical):
        return DependenceDetector(self,historical)


class DependenceDetector:
    def __init__(self,prepared,historical):
        self.prepared=prepared
        self.new=CompactDependenceState(historical)
        self.output=np.empty(24,dtype=np.float32)
        self.call_args=(self.new.args,(),prepared.old_positions,prepared.old_columns,
            prepared.new_positions,prepared.new_columns,self.output,prepared.tree_args)
        self.kernel=_kernel(step,False)
        self.resident=_resident_class(self.kernel,typeof(self.call_args))(self.call_args) if prepared.resident else None

    def predict_one(self,point):
        return float(self.resident.predict_one(float(point)) if self.resident is not None else self.kernel(float(point),self.call_args))

    @property
    def state_array_bytes(self):
        return unique_array_bytes(self.call_args[:-1])

    @property
    def immutable_tree_array_bytes(self):
        return unique_array_bytes(self.prepared.tree_args)
