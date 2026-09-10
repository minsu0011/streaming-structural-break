"""One variance feature update and two compact forests for a fixed blend."""
from functools import lru_cache
from pathlib import Path
import hashlib
import numpy as np
from numba import njit,typeof
from src.next.fused_runtime import _resident_class,unique_array_bytes
from src.next.fast_variance_initialization import PreparedFastVariancePredictor,source_hash as fast_hash
from src.next.paired_blend import source_hash as blend_hash
from src.models.compact_trees import tree_margin


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+fast_hash().encode()+blend_hash().encode()).hexdigest()


@lru_cache(maxsize=8)
def combined_kernel(left_kernel):
    @njit(cache=False)
    def predict(point,args):
        left_args,right_trees,weight = args
        left = np.float64(np.float32(left_kernel(point,left_args)))
        margin = tree_margin(left_args[6],*right_trees)
        right = np.float64(np.float32(1./(1.+np.exp(-margin))))
        return np.float32(weight*left+(1.-weight)*right)
    return predict


class PreparedSameBankVarianceBlend:
    def __init__(self,model):
        model.validate()
        self.model = model
        self.left = PreparedFastVariancePredictor(model.left)
        args = list(model.right.compact.args())
        features = model.right.compact.features.copy()
        nodes = features>=0
        features[nodes] = model.right_positions[model.right.columns[features[nodes]]]
        args[1] = features
        self.right_trees = tuple(args)

    def new_state(self,historical):
        return SameBankVarianceDetector(self,historical)


class SameBankVarianceDetector:
    def __init__(self,prepared,historical):
        self.left = prepared.left.new_state(historical)
        self.args = self.left.call_args,prepared.right_trees,float(prepared.model.left_weight)
        self.kernel = combined_kernel(self.left.kernel)
        self.resident = _resident_class(self.kernel,typeof(self.args))(self.args)

    def predict_one(self,point):
        return float(self.resident.predict_one(float(point)))

    @property
    def state_array_bytes(self):
        return self.left.state_array_bytes

    @property
    def immutable_tree_array_bytes(self):
        return self.left.immutable_tree_array_bytes+unique_array_bytes(self.args[1])
