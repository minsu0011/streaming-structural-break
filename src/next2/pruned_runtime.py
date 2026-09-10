"""Exact two-channel runtime with actual removal of unused GLR/state work.

The scientific model remains the frozen NEXT binary. Modes are chosen from
the model's complete feature names, never guessed from a candidate label.
"""
from pathlib import Path
import hashlib
import numpy as np
from numba import njit, typeof
from src.next.whitening import fit_historical, innovation_update
from src.next.fast_variance_initialization import (
    fast_variance_parameters, FastARCHFeatureState, OLD_CONFIG)
from src.next.channel_evidence import make_channel_args, channel_step, _logadd
from src.next.serial_channel_evidence import (
    serial_channel_step, historical_correlations, sum_variances)
from src.next.fused_runtime import (
    PreparedNextPredictor, _kernel, _resident_class, _arch_args, unique_array_bytes)
from src.next.extensions import names_and_groups
from src.features.streaming import feature_names

LOCAL_BAYES_STATS = ('current','ewma8','ewma32','ewma128',
                     'cusum_positive','cusum_negative','bayes025','bayes1')


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@njit(cache=True)
def pair_channel_update(residual, parameters, conditional, mode, output):
    h_variance = parameters[2]/max(1.-parameters[3]-parameters[4],.05)
    q1,q2 = conditional[0],conditional[1]
    variance = conditional[2] if mode == 5 else parameters[2]+parameters[3]*q1+parameters[4]*q2
    variance = max(variance,parameters[6])
    centered = residual-parameters[0]
    squared = centered*centered
    output[0] = squared/h_variance-1.
    output[1] = np.log(variance/h_variance)
    capped = min(squared,parameters[5])
    conditional[1] = q1
    conditional[0] = capped
    conditional[2] = (1.-parameters[7])*conditional[2]+parameters[7]*capped
    return output


@njit(cache=True)
def historical_pair(residuals, parameters, mode):
    h_variance = parameters[2]/max(1.-parameters[3]-parameters[4],.05)
    conditional = np.full(3,h_variance)
    result = np.empty((len(residuals),2))
    for j in range(len(residuals)):
        pair_channel_update(residuals[j],parameters,conditional,mode,result[j])
    return result


def local_bayes_args(max_age):
    ages = np.asarray([2**k for k in range(max_age.bit_length())],dtype=np.int64)
    # No GLR maximum, age-at-max, log-mixture, or GLR-memory array exists.
    return (ages,np.diff(np.r_[0,ages]).astype(float),np.zeros((2,3)),
            np.zeros((2,2)),np.zeros(2),np.zeros((max_age+1,2)),
            np.zeros(1,dtype=np.int64),np.empty(16,dtype=np.float32))


@njit(cache=True)
def local_bayes_step(values,args):
    ages,widths,ewma,cusum,cumulative,prefix,counter,output = args
    t = counter[0]+1
    halves = (8.,32.,128.)
    for channel in range(2):
        value = min(max(values[channel],-12.),12.)
        cursor = channel*8
        output[cursor] = value
        for k in range(3):
            alpha = 1.-np.exp(-np.log(2.)/halves[k])
            ewma[channel,k] = (1.-alpha)*ewma[channel,k]+alpha*value
            variance = max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)),1e-12)
            output[cursor+1+k] = ewma[channel,k]/np.sqrt(variance)
        cusum[channel,0] = max(0.,cusum[channel,0]+value-.25)
        cusum[channel,1] = max(0.,cusum[channel,1]-value-.25)
        output[cursor+4] = np.log1p(cusum[channel,0])
        output[cursor+5] = np.log1p(cusum[channel,1])
        cumulative[channel] += value
        b025,b1,width = -1e300,-1e300,0.
        for k in range(len(ages)):
            age = ages[k]
            if age > t:
                break
            width += widths[k]
            total = cumulative[channel]-prefix[(t-age)%len(prefix),channel]
            energy = total*total
            b025 = _logadd(b025,np.log(widths[k])-.5*np.log1p(.25*age)+.5*.25*energy/(1.+.25*age))
            b1 = _logadd(b1,np.log(widths[k])-.5*np.log1p(age)+.5*energy/(1.+age))
        prefix[t%len(prefix),channel] = cumulative[channel]
        output[cursor+6] = b025-np.log(width)
        output[cursor+7] = b1-np.log(width)
    counter[0] = t
    return output


@njit(cache=True)
def _values(point,filter_args,parameters,conditional,mode,calibration,work):
    residual = innovation_update(point,*filter_args)
    pair_channel_update(residual,parameters,conditional,mode,work)
    for j in range(2):
        work[j] = (work[j]-calibration[0,j])/calibration[1,j]
    return work


@njit(cache=False)
def compact_full_step(point,filter_args,parameters,conditional,mode,calibration,work,args):
    return channel_step(_values(point,filter_args,parameters,conditional,mode,calibration,work),args)


@njit(cache=False)
def compact_serial_step(point,filter_args,parameters,conditional,mode,calibration,work,args,age_variances):
    return serial_channel_step(_values(point,filter_args,parameters,conditional,mode,calibration,work),args,age_variances)


@njit(cache=False)
def compact_bayes_step(point,filter_args,parameters,conditional,mode,calibration,work,args):
    return local_bayes_step(_values(point,filter_args,parameters,conditional,mode,calibration,work),args)


class CompactPairState:
    def __init__(self,historical,settings,mode):
        fit = fit_historical(historical,settings['order'])
        self.parameters,self.conditional = fast_variance_parameters(
            fit['historical_innovations'],settings['normalization'])
        self.mode = {'arch1':3,'arch2':4,'ewma':5}[settings['normalization']]
        channels = historical_pair(fit['historical_innovations'],self.parameters,self.mode)
        channels = channels[min(160,len(channels)//4):]
        self.calibration = np.stack([channels.mean(axis=0),np.maximum(channels.std(axis=0),1e-6)])
        self.filter_args = tuple(fit[k] for k in ('reference','coefficients','ring','counter'))
        self.work = np.empty(2)
        self.evidence_args = local_bayes_args(settings['max_age']) if mode == 'local_bayes' else make_channel_args(2,settings['max_age'])
        self.call_args = (self.filter_args,self.parameters,self.conditional,self.mode,
                          self.calibration,self.work,self.evidence_args)
        self.function = compact_bayes_step if mode == 'local_bayes' else compact_full_step
        if mode == 'serial':
            correlations = historical_correlations((channels-self.calibration[0])/self.calibration[1])
            self.age_variances = sum_variances(self.evidence_args[0],correlations)
            self.call_args += (self.age_variances,)
            self.function = compact_serial_step

    def update(self,point):
        return self.function(float(point),*self.call_args)


class PreparedCompactPredictor(PreparedNextPredictor):
    def __init__(self,model,*,resident=True):
        kind = getattr(model,'extension',{}).get('kind')
        if kind not in ('conditional_variance','serial_variance'):
            raise ValueError('Frozen pair variance or serial model required')
        super().__init__(model,resident=resident)
        self.old_positions = np.asarray([j for j,n in enumerate(model.names) if n.startswith('arch8__')],dtype=np.int32)
        self.new_positions = np.asarray([j for j,n in enumerate(model.names) if not n.startswith('arch8__')],dtype=np.int32)
        names = [model.names[j] for j in self.new_positions]
        allowed = (kind+'__raw_energy_',kind+'__log_variance_forecast_')
        if not names or not all(n.startswith(allowed) for n in names):
            raise ValueError('Model consumes a channel outside the exact two-channel specialization')
        bayes_names = [kind+'__'+c+'_'+s for c in ('raw_energy','log_variance_forecast') for s in LOCAL_BAYES_STATS]
        self.mode = 'serial' if kind == 'serial_variance' else 'local_bayes' if all(n in bayes_names for n in names) else 'full'
        old_names = ['arch8__'+name for name in feature_names(OLD_CONFIG)]
        new_names = bayes_names if self.mode == 'local_bayes' else names_and_groups(model.extension)[0][:26]
        self.old_columns = np.asarray([old_names.index(model.names[j]) for j in self.old_positions],dtype=np.int32)
        self.new_columns = np.asarray([new_names.index(model.names[j]) for j in self.new_positions],dtype=np.int32)
        self.runtime_source_sha256 = source_hash()

    def new_state(self,historical):
        return CompactDetector(self,historical)


class CompactDetector:
    def __init__(self,prepared,historical):
        self.prepared = prepared
        self.old = FastARCHFeatureState(historical) if len(prepared.old_positions) else None
        self.new = CompactPairState(historical,prepared.model.extension['settings'],prepared.mode)
        self.output = np.empty(len(prepared.model.names),dtype=np.float32)
        old_args = _arch_args(self.old) if self.old is not None else ()
        self.call_args = (self.new.call_args,old_args,prepared.old_positions,prepared.old_columns,
            prepared.new_positions,prepared.new_columns,self.output,prepared.tree_args)
        self.kernel = _kernel(self.new.function,self.old is not None)
        self.resident = _resident_class(self.kernel,typeof(self.call_args))(self.call_args) if prepared.resident else None

    def predict_one(self,point):
        return float(self.resident.predict_one(float(point)) if self.resident is not None else self.kernel(float(point),self.call_args))

    @property
    def state_array_bytes(self):
        return unique_array_bytes(self.call_args[:-1])

    @property
    def immutable_tree_array_bytes(self):
        return unique_array_bytes(self.prepared.tree_args)
