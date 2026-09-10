"""Compose the already verified AR kernels with the frozen float32 mean rule."""
import hashlib
from pathlib import Path
import numpy as np
from numba import njit
from src.streaming.fused_ar import PreparedFusedARPredictor,fused_ar_predict,implementation_hash as ar_hash
from src.models.cross_order import implementation_hash as model_hash


def implementation_hash():
    resident=Path(__file__).with_name('resident_cross_order.py')
    return hashlib.sha256(Path(__file__).read_bytes()+resident.read_bytes()+ar_hash().encode()+model_hash().encode()).hexdigest()


@njit(cache=False)
def fused_equal_average(point,left_args,right_args):
    left=np.float64(np.float32(fused_ar_predict(point,*left_args)))
    right=np.float64(np.float32(fused_ar_predict(point,*right_args)))
    return np.float32(.5*left+.5*right)


class FusedCrossOrderState:
    def __init__(self,states):
        self.states=states
        self.call_args=tuple((s.ar_args,s.state_args,s.calibration_args,s.prepared.tree_args,s.prepared.is_xgboost) for s in states)
        from src.streaming.resident_cross_order import resident_state
        self.resident=resident_state(self.call_args)
        self.resident_predict_one=self.resident.predict_one
    def predict_one(self,point):
        return float(self.resident_predict_one(float(point)))
    def predict_one_tuple(self,point):
        """Numerically identical pre-resident dispatcher, retained for audits."""
        return float(fused_equal_average(float(point),*self.call_args))
    def predict_one_reference(self,point):
        """Original two-dispatch formula for numerical and paired timing audits."""
        left=float(np.float32(self.states[0].predict_one(point)))
        right=float(np.float32(self.states[1].predict_one(point)))
        return float(np.float32(.5*left+.5*right))
    @property
    def state_array_bytes(self):return sum(state.state_array_bytes for state in self.states)


class PreparedFusedCrossOrderPredictor:
    def __init__(self,model,*,fast_initialization=False):
        model.validate()
        self.children=tuple(PreparedFusedARPredictor(child,fast_initialization=fast_initialization) for child in model.components)
    def new_state(self,historical):return FusedCrossOrderState(tuple(child.new_state(historical) for child in self.children))
