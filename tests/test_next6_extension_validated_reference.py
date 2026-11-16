import pickle
from pathlib import Path
import numpy as np
import pytest
from src.next6_extension.observer import ObservableModel
from src.next6_extension.factorized_coherent import CoherentGraph
from src.next6_extension.runtime import Prepared as Original
from src.next6_extension.validated_runtime import Prepared,InsufficientReferenceState

def model():
    p=np.full((2,8,3,3,3),1/3);g=CoherentGraph(np.full(8,-1),p,p.copy(),np.zeros((2,8,3,3),dtype=int),[],{})
    return ObservableModel(1.9975371345973232,g,np.full((2,7,7),1/7),np.zeros((2,7,7),dtype=int),{}, {})

@pytest.mark.parametrize('value',[0.,1.,-1.,1e30,-1e30,np.finfo(np.float32).max,-np.finfo(np.float32).max])
def test_invalid_reference_has_no_evidence_or_confirmation(value):
    s=Prepared(model()).new_state(np.full(1024,value));assert not s.reference_valid;o=np.full(256,20.);out=s.replay(o)
    assert np.all(out[:,0]==6);assert np.all(out[:,[1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,24,25]]==0);assert np.all(out[:,20:22]==1)
    assert out[-1,5]==16;assert np.sum(out[:,4])==16;assert out[0,26]==-1 and np.all(out[1:,26]==6)
    a=Prepared(model()).new_state(np.full(1024,value));np.testing.assert_array_equal([a.predict_one(x) for x in o],out)

def test_invalid_reference_resumes_and_rejects_bad_points_before_mutation():
    s=Prepared(model()).new_state(np.zeros(512));s.replay(np.ones(17));before=pickle.dumps(s)
    for values in ([0,np.nan],[np.inf],[1e100],[[1,2]]):
        with pytest.raises(ValueError):s.replay(values)
        assert pickle.dumps(s)==before
    restored=pickle.loads(before);np.testing.assert_array_equal(s.replay(np.ones(128)),restored.replay(np.ones(128)))
    assert s.replay([]).shape==(0,27) and s.bounded_mutable_state_bytes==8

def test_valid_reference_uses_identical_fused_class_and_outputs():
    rng=np.random.default_rng(987);h=rng.normal(size=2048);o=rng.normal(size=1024);a=Prepared(model()).new_state(h);b=Original(model()).new_state(h)
    assert a.reference_valid and type(a) is type(b);np.testing.assert_array_equal(a.replay(o),b.replay(o));assert a.bounded_mutable_state_bytes==b.bounded_mutable_state_bytes

def test_tiny_spread_reference_is_explicitly_insufficient():
    h=np.linspace(-1e-10,1e-10,1024);state=Prepared(model()).new_state(h);assert isinstance(state,InsufficientReferenceState);assert np.all(state.replay(np.ones(128))[:,1]==0)
