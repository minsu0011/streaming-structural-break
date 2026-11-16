import src.next9
import pickle
from types import SimpleNamespace
import numpy as np
import pytest
from src.next9.streaming import PrefixBank,classifier_probabilities,IntegratedStream
from src.next9.states import describe_history
from src.next9.joint_design import assemble
from src.next9.transitions import SelfLaw
from src.next9.learning import fit_classifier
from src.next9.data import FitScope


class BatchBank:
    def __init__(self,seed,values,epsilon,context):
        self.full=np.vstack([seed,values]);self.times=np.arange(len(values));self.series_index=np.zeros(len(values),dtype=int)
        self.series=dict(epsilon=epsilon[None],context=context[None],H_length=np.array([512]))
    def history(self,rows,j):return np.stack([self.full[r:r+16,j] for r in rows]) if len(rows) else np.empty((0,16))
    def current(self,rows,columns):return self.full[np.asarray(rows)[:,None]+16,np.asarray(columns)[None,:]]
    def pre_state(self,rows,j):return describe_history(self.history(rows,j),self.series['epsilon'][0,0,j],True)


def test_stream_offline_parity_with_paths_and_conditional_columns():
    rng=np.random.default_rng(9101604);seed=rng.normal(size=(16,153)).astype(np.float32)
    values=rng.normal(size=(37,153)).astype(np.float32);epsilon=np.full((3,153),.02,dtype=np.float32);context=np.zeros(6,dtype=np.float32)
    batch=BatchBank(seed,values,epsilon,context);stream=PrefixBank(seed,epsilon,context,512)
    candidates=[1,72];rules={1:dict(pair=['S1_DIRECTION',2,0],low=-.5,high=.5)}
    maps={j:np.arange(10) for j in candidates}
    law=SelfLaw('T1_SELF_MARKOV',np.array([[.7,.1,.1,.1],[.1,.7,.1,.1],[.1,.1,.7,.1],[.25]*4]),None,{})
    expected,names=assemble(batch,np.arange(len(values)),candidates,rules,maps,{1:law},'M5')
    output=[]
    for t,v in enumerate(values):
        state_before=stream.pre_state(np.array([t]),1)
        stream.stage(v)
        for key,value in state_before.items():np.testing.assert_array_equal(value,stream.pre_state(np.array([t]),1)[key])
        x,nn=assemble(stream,np.array([t]),candidates,rules,maps,{1:law},'M5');assert nn==names;output.append(x[0])
        stream.commit()
        if t==16:stream=pickle.loads(pickle.dumps(stream,protocol=5))
    np.testing.assert_array_equal(output,expected)
    assert stream.past.shape==(17,153)
    with pytest.raises(ValueError,match='observable'):stream.history(np.array([stream.time+1]),1)


@pytest.mark.parametrize('classes,method',[(2,'LOGISTIC'),(4,'LOGISTIC'),(4,'DEPTH3')])
def test_fast_runtime_uses_same_frozen_classifier(classes,method):
    rng=np.random.default_rng(42);x=rng.normal(size=(256,7));y=np.arange(256)%classes;ids=np.arange(256)
    scope=FitScope(frozenset(ids),frozenset([256]),frozenset([256]),'toy')
    model=fit_classifier(x,y,ids,ids,scope,classes=classes,method=method)
    expected=model.predict_proba(x)
    np.testing.assert_allclose(classifier_probabilities(model,x),expected,rtol=0,atol=1e-14)
    scalar=np.concatenate([classifier_probabilities(model,row[None]) for row in x[:5]])
    np.testing.assert_allclose(scalar,expected[:5],rtol=0,atol=1e-14)


def test_failed_M7_gate_bypasses_new_features_and_preserves_bits():
    class State:
        def predict_one(self,p):return np.float64(p)
    class Baseline:
        def new_state(self,h):return State()
    state=IntegratedStream(SimpleNamespace(correction=None),Baseline(),[])
    assert state.research is None
    p=np.nextafter(np.float64(.5),np.float64(1))
    assert state.predict_one(p).tobytes()==p.tobytes()
