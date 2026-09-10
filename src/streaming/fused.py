"""One JIT call for causal feature update, historical calibration and tree inference."""
import hashlib
from pathlib import Path
import numpy as np
from numba import njit
from src.features.streaming import StreamingFeatureState,_update
from src.features.historical_calibration import _transform
from src.features.startup_calibration import correct_startup
from src.models.compact_trees import tree_margin
from src.models.compact_xgboost import export_xgboost,probability_one

def implementation_hash():
    paths=[Path(__file__),Path(__file__).parents[1]/'models/compact_xgboost.py',Path(__file__).parents[1]/'features/startup_calibration.py']
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()

@njit(cache=False)
def fused_predict(point,state_args,calibration_args,tree_args,calibrated,is_xgboost,startup):
    _update(point,*state_args)
    if calibrated:
        source=state_args[6]
        if startup:
            correct_startup(source,state_args[8],state_args[5][0],calibration_args[5]);source=calibration_args[5]
        _transform(source,calibration_args[0],calibration_args[1],calibration_args[2],calibration_args[3],calibration_args[4])
        features=calibration_args[4]
    else:features=state_args[6]
    if is_xgboost:return float(probability_one(features,*tree_args))
    margin=tree_margin(features,*tree_args)
    return 1.0/(1.0+np.exp(-margin))

class PreparedFusedPredictor:
    """Immutable tree arrays are prepared once; every series gets fresh state."""
    def __init__(self,bundle):
        self.bundle=bundle
        base=bundle.base_model if hasattr(bundle,'base_model') else bundle
        self.calibrated=hasattr(bundle,'make_state')
        self.startup=self.calibrated and bundle.policy.method.endswith('_startup')
        self.is_xgboost=base.name=='M5_xgboost'
        tree=export_xgboost(base.estimator) if self.is_xgboost else base.compact
        if tree is None:raise ValueError('Fused path requires supported scalar trees')
        args=list(tree.args());mapped=tree.features.copy();nodes=mapped>=0;mapped[nodes]=base.columns[mapped[nodes]]
        args[1]=mapped;self.tree_args=tuple(args)
        self.source_hash=implementation_hash()

    def new_state(self,historical):return FusedTreeDetector(self,historical)

class FusedTreeDetector:
    def __init__(self,prepared,historical):
        self.prepared=prepared
        if prepared.calibrated:
            calibrated=prepared.bundle.make_state(historical);self.state=calibrated.state
            correction_buffer=calibrated.corrected if prepared.startup else np.zeros(1,dtype=np.float32)
            self.calibration_args=(calibrated.center,calibrated.scale,calibrated.mask,float(calibrated.policy.clip),calibrated.output,correction_buffer)
        else:
            self.state=StreamingFeatureState(historical,config=prepared.bundle.feature_config)
            self.calibration_args=(np.zeros(1),np.ones(1),np.zeros(1,dtype=np.bool_),0.,np.zeros(1,dtype=np.float32),np.zeros(1,dtype=np.float32))
        s=self.state
        self.state_args=(s.reference,s.baseline,s.thresholds,s.ring,s.ew,s.counters,s.output,s.parameters,s.alphas,s.lags)

    def predict_one(self,point):
        return float(fused_predict(float(point),self.state_args,self.calibration_args,self.prepared.tree_args,self.prepared.calibrated,self.prepared.is_xgboost,self.prepared.startup))

    @property
    def state_array_bytes(self):
        return self.state.state_array_bytes+sum(x.nbytes for x in self.calibration_args if isinstance(x,np.ndarray))
