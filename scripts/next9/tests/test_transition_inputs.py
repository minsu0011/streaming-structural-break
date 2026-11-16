import src.next9
from types import SimpleNamespace
import numpy as np
from src.next9.states import describe_history
from src.next9.transitions import SelfLaw
from src.next9.transition_inputs import path_features


class Bank:
    def __init__(self,values):
        self.values=np.asarray(values,dtype=np.float32)
        self.times=np.arange(len(values));self.series_index=np.zeros(len(values),dtype=int)
        self.series=dict(context=np.zeros((1,6)),epsilon=np.full((1,3,1),.02))

    def pre_state(self,rows,j):
        history=np.stack([np.r_[np.zeros(16),self.values][r:r+16] for r in rows]) if len(rows) else np.empty((0,16))
        return describe_history(history,.02,True)


def test_transition_path_current_and_future_suffix_invariance():
    law=SelfLaw('T1_SELF_MARKOV',np.array([[.7,.1,.1,.1],[.1,.7,.1,.1],[.1,.1,.7,.1],[.25]*4]),None,{})
    original=np.sin(np.arange(50)*.2)
    modified=original.copy();modified[23:]=100
    a=path_features(Bank(original),np.arange(24),0,law)
    b=path_features(Bank(modified),np.arange(24),0,law)
    np.testing.assert_array_equal(a,b)
    assert a[0,2]==0
