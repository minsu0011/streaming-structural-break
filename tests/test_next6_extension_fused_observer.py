import pickle
import numpy as np
import pytest
from src.next6_extension.factorized_coherent import CoherentGraph
from src.next6_extension.observer import ObservableModel
from src.next6_extension.runtime import Prepared

def model():
    rng=np.random.default_rng(66);g=CoherentGraph(np.full(8,-1,dtype=np.int64),np.full((2,8,3,3,3),1/3),np.full((2,8,3,3,3),1/3),np.zeros((2,8,3,3),dtype=np.int32),[],{});p=rng.uniform(.1,1,size=(2,7,7));p/=p.sum(axis=-1,keepdims=True);return ObservableModel(1.4,g,p,np.full((2,7,7),20,dtype=np.int32),{},{}),rng

def test_fused_reference_scalar_batch_and_resumed_prefix_parity():
    m,rng=model();h=rng.normal(size=2048);o=rng.normal(size=2048);expected=m.new_state(h).replay(o);prepared=Prepared(m);s=prepared.new_state(h);scalar=np.array([s.predict_one(x) for x in o]);batch=prepared.new_state(h).replay(o)
    np.testing.assert_array_equal(scalar,batch);np.testing.assert_allclose(expected,batch,rtol=0,atol=1e-12)
    s=prepared.new_state(h);prefix=s.replay(o[:119]);s=pickle.loads(pickle.dumps(s));np.testing.assert_array_equal(np.vstack((prefix,s.replay(o[119:]))),batch)

@pytest.mark.parametrize('point',[np.nan,np.inf,-np.inf,1e100])
def test_invalid_point_leaves_fused_state_untouched(point):
    m,rng=model();s=Prepared(m).new_state(rng.normal(size=1024));before=pickle.dumps(s)
    with pytest.raises(ValueError):s.predict_one(point)
    assert before==pickle.dumps(s)
