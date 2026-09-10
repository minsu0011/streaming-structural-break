"""One compiled AR8 innovation, historical variance forecast, feature and tree call."""
import hashlib
from pathlib import Path
import numpy as np
from numba import njit
from src.features.ar_order_input import historical_filter
from src.features.ar_residual_input import innovation_one
from src.features.arch_input import arch_one,historical_variance_filter
from src.features.streaming import StreamingFeatureState,FEATURE_NAMES
from src.features.historical_calibration import historical_reference
from src.streaming.fused import PreparedFusedPredictor,fused_predict
from src.streaming.fast_initialization import shared_mad_reference
from src.streaming.fused_ar import implementation_hash as ar_hash


def implementation_hash():
    root=Path(__file__).resolve().parents[2]
    return hashlib.sha256(Path(__file__).read_bytes()+(root/'src/features/arch_input.py').read_bytes()+(root/'configs/arch_input.json').read_bytes()+ar_hash().encode()).hexdigest()


@njit(cache=False)
def fused_arch_predict(point,ar_args,variance_args,state_args,calibration_args,tree_args,is_xgboost):
    residual=innovation_one(point,*ar_args)
    standardized=arch_one(np.float64(residual),*variance_args)
    return fused_predict(np.float64(standardized),state_args,calibration_args,tree_args,True,is_xgboost,False)


class PreparedFusedARCHPredictor(PreparedFusedPredictor):
    def __init__(self,bundle,*,fast_initialization=False):
        from src.models.arch_calibrated import ARCHCalibratedBundle
        if not isinstance(bundle,ARCHCalibratedBundle):raise ValueError('Conditional-variance inference requires an explicit ARCH bundle')
        super().__init__(bundle);self.fast_initialization=fast_initialization
    def new_state(self,historical):return FusedARCHDetector(self,historical)


class FusedARCHDetector:
    def __init__(self,prepared,historical):
        self.prepared=prepared;bundle=prepared.bundle
        if prepared.fast_initialization:
            reference,coefficients,ring,residual=historical_filter(historical,bundle.feature_config,8)
            self.ar_args=(reference,coefficients,ring,np.zeros(1,dtype=np.int64))
            parameters,previous,transformed=historical_variance_filter(residual,bundle.feature_config);self.variance_args=(parameters,previous)
            self.state=StreamingFeatureState(transformed,config=bundle.feature_config)
            if bundle.policy.method=='historical_median_mad':center,scale,mask=shared_mad_reference(transformed,self.state,bundle.policy,selected_columns=np.unique(prepared.tree_args[1]))
            else:center,scale,mask=historical_reference(transformed,config=bundle.feature_config,policy=bundle.policy)
            output=np.zeros(len(FEATURE_NAMES),dtype=np.float32)
        else:
            original=bundle.make_state(historical);self.ar_args=(original.reference,original.coefficients,original.ring,original.counter)
            self.variance_args=(original.variance_parameters,original.previous_squared);inner=original.inner;self.state=inner.state
            center,scale,mask,output=inner.center,inner.scale,inner.mask,inner.output
        self.calibration_args=(center,scale,mask,float(bundle.policy.clip),output,np.zeros(1,dtype=np.float32))
        s=self.state;self.state_args=(s.reference,s.baseline,s.thresholds,s.ring,s.ew,s.counters,s.output,s.parameters,s.alphas,s.lags)
    def predict_one(self,point):
        return float(fused_arch_predict(float(point),self.ar_args,self.variance_args,self.state_args,self.calibration_args,self.prepared.tree_args,self.prepared.is_xgboost))
    @property
    def state_array_bytes(self):
        return self.state.state_array_bytes+sum(x.nbytes for x in (*self.ar_args,*self.variance_args))+sum(x.nbytes for x in self.calibration_args if isinstance(x,np.ndarray))
