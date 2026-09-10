"""Exact H initialization without unused ECDF work or duplicate old H setup."""
from pathlib import Path
import hashlib
import numpy as np
from numba import njit,typeof
from src.next.whitening import fit_historical
from src.next.variance_evidence import VarianceState,VarianceConfig,_config,historical_variance_channels
from src.next.channel_evidence import make_channel_args
from src.next.pruned_variance_runtime import PreparedPrunedVariancePredictor,pruned_variance_step,source_hash as pruned_hash
from src.next.fused_runtime import _kernel,_resident_class,_arch_args,unique_array_bytes
from src.next.extensions import names_and_groups
from src.features.config import CONFIG
from src.features.streaming import StreamingFeatureState,feature_names,FEATURE_NAMES
from src.features.historical_calibration import CalibrationPolicy,_transform
from src.features.ar_order_input import historical_filter
from src.features.arch_input import ARCHCalibratedState,historical_variance_filter
from src.streaming.fast_initialization import shared_mad_reference,implementation_hash as old_initialization_hash


OLD_CONFIG = CONFIG.variant(normalization='median_mad',scales=(5,20,160))
OLD_POLICY = CalibrationPolicy(method='historical_median_mad')


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+pruned_hash().encode()+old_initialization_hash().encode()).hexdigest()


@njit(cache=False)
def conditional_replay_only(residuals,parameters,initial_variance):
    conditional = np.full(3,initial_variance)
    for residual in residuals:
        centered = residual-parameters[0]
        squared = min(centered*centered,parameters[5])
        conditional[1] = conditional[0]
        conditional[0] = squared
        conditional[2] = (1.-parameters[7])*conditional[2]+parameters[7]*squared
    return conditional


def fast_variance_parameters(residuals,mode):
    if mode not in ('arch1','arch2','ewma'):
        raise ValueError('Fast reference only supports the three variance modes')
    r = np.asarray(residuals,dtype=np.float64)
    mean = float(r.mean())
    sd = max(float(r.std()),1e-8)
    sq = (r-mean)**2
    cap = max(float(np.quantile(sq,.99)),1e-12)
    capped = np.minimum(sq,cap)
    variance = max(float(capped.mean()),1e-12)
    slopes = np.zeros(2)
    if mode in ('arch1','arch2'):
        order = 1 if mode=='arch1' else 2
        design = np.column_stack([capped[order-1-j:len(r)-1-j] for j in range(order)])
        target = capped[order:]
        center = design-design.mean(axis=0)
        gram = center.T@center
        ridge = max(float(np.trace(gram))/order*.001,1e-12)
        slopes[:order] = np.maximum(np.linalg.solve(gram+ridge*np.eye(order),center.T@(target-target.mean())),0.)
        if slopes.sum()>.95:
            slopes *= .95/slopes.sum()
    parameters = np.array([mean,sd,variance*(1.-slopes.sum()),slopes[0],slopes[1],cap,max(variance*.0001,1e-12),2./33.])
    conditional = conditional_replay_only(r,parameters,variance)
    return parameters,conditional


class FastVarianceState(VarianceState):
    def __init__(self,historical,config=VarianceConfig()):
        from dataclasses import asdict
        config = _config(asdict(config))
        self.config = config
        fit = fit_historical(historical,config.order)
        self.parameters,self.conditional = fast_variance_parameters(fit['historical_innovations'],config.normalization)
        self.mode = {'arch1':3,'arch2':4,'ewma':5}[config.normalization]
        h_variance = self.parameters[2]/max(1.-self.parameters[3]-self.parameters[4],.05)
        channels = historical_variance_channels(fit['historical_innovations'],self.parameters,np.full(3,h_variance),self.mode)
        channels = channels[min(160,len(channels)//4):]
        self.calibration = np.stack([channels.mean(axis=0),np.maximum(channels.std(axis=0),1e-6)])
        self.filter_args = tuple(fit[key] for key in ('reference','coefficients','ring','counter'))
        self.work = np.empty(6)
        self.evidence_args = make_channel_args(6,config.max_age)


class SharedHistoricalCalibration:
    def __init__(self,historical):
        self.state = StreamingFeatureState(historical,config=OLD_CONFIG)
        self.center,self.scale,self.mask = shared_mad_reference(historical,self.state,OLD_POLICY)
        self.policy = OLD_POLICY
        self.output = np.zeros(len(FEATURE_NAMES),dtype=np.float32)

    def update_and_get(self,point):
        _transform(self.state.update_and_get(point),self.center,self.scale,self.mask,self.policy.clip,self.output)
        return self.output

    @property
    def state_array_bytes(self):
        return self.state.state_array_bytes+self.center.nbytes+self.scale.nbytes+self.mask.nbytes+self.output.nbytes


class FastARCHFeatureState(ARCHCalibratedState):
    def __init__(self,historical):
        self.reference,self.coefficients,self.ring,residual = historical_filter(historical,OLD_CONFIG,8)
        self.counter = np.zeros(1,dtype=np.int64)
        self.variance_parameters,self.previous_squared,transformed = historical_variance_filter(residual,OLD_CONFIG)
        self.inner = SharedHistoricalCalibration(transformed)


class PreparedFastVariancePredictor(PreparedPrunedVariancePredictor):
    def __init__(self,model,*,resident=True):
        super().__init__(model,resident=resident)
        old_names = ['arch8__'+name for name in feature_names(OLD_CONFIG)]
        new_names,_ = names_and_groups(model.extension)
        self.old_positions = np.asarray([j for j,name in enumerate(model.names) if name.startswith('arch8__')],dtype=np.int32)
        self.new_positions = np.asarray([j for j,name in enumerate(model.names) if not name.startswith('arch8__')],dtype=np.int32)
        self.old_columns = np.asarray([old_names.index(model.names[j]) for j in self.old_positions],dtype=np.int32)
        self.new_columns = np.asarray([new_names.index(model.names[j]) for j in self.new_positions],dtype=np.int32)
        if np.any(self.new_columns>=26):
            raise RuntimeError('Fast two-channel reference cannot consume omitted channels')
        self.runtime_source_sha256 = source_hash()

    def new_state(self,historical):
        return FastVarianceDetector(self,historical)


class FastVarianceDetector:
    def __init__(self,prepared,historical):
        self.prepared = prepared
        self.old = FastARCHFeatureState(historical) if len(prepared.old_positions) else None
        self.new = FastVarianceState(historical,VarianceConfig(**prepared.model.extension['settings']))
        self.output = np.empty(len(prepared.model.names),dtype=np.float32)
        new = self.new
        extension_args = new.filter_args,new.parameters,new.conditional,new.mode,new.calibration,new.work,new.evidence_args
        old_args = _arch_args(self.old) if self.old is not None else ()
        self.call_args = (extension_args,old_args,prepared.old_positions,prepared.old_columns,prepared.new_positions,prepared.new_columns,self.output,prepared.tree_args)
        self.kernel = _kernel(pruned_variance_step,self.old is not None)
        self.resident = _resident_class(self.kernel,typeof(self.call_args))(self.call_args) if prepared.resident else None

    def predict_one(self,point):
        return float(self.resident.predict_one(float(point)) if self.resident is not None else self.kernel(float(point),self.call_args))

    @property
    def state_array_bytes(self):
        return unique_array_bytes(self.call_args[:-1])

    @property
    def immutable_tree_array_bytes(self):
        return unique_array_bytes(self.prepared.tree_args)
