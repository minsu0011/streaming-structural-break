"""Six causal gate inputs from the prediction-exact shared frozen pipeline."""
import numpy as np
from numba import njit
from src.next2.shared_blend_runtime import shared_feature_step
from src.models.compact_trees import tree_margin


@njit(cache=False)
def replay_gate_features(points,args):
    output=np.empty((len(points),6),dtype=np.float32)
    for t in range(len(points)):
        feature=shared_feature_step(float(points[t]),args)
        shared=args[1];pair=shared[5][-1];dependence=shared[-1][-1]
        jt,ct,weight=args[-1]
        output[t,0]=np.float32(1./(1.+np.exp(-tree_margin(feature,*jt))))
        output[t,1]=np.float32(1./(1.+np.exp(-tree_margin(dependence,*ct))))
        output[t,2]=max(abs(pair[1]),abs(pair[2]),abs(pair[14]),abs(pair[15]))
        strength=0.
        for lag in range(4):
            strength=max(strength,abs(dependence[6*lag]),abs(dependence[6*lag+1]))
        output[t,3]=strength;output[t,4]=pair[0];output[t,5]=pair[13]
    return output
