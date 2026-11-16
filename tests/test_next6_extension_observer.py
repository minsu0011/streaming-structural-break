import pickle
import numpy as np
from src.next6_extension.factorized_coherent import CoherentGraph
from src.next6_extension.observer import ObservableModel

def model():
    g=CoherentGraph(np.full(8,-1,dtype=np.int64),np.full((2,8,3,3,3),1/3),np.full((2,8,3,3,3),1/3),np.zeros((2,8,3,3),dtype=np.int32),[],{})
    rng=np.random.default_rng(623);p=rng.uniform(.1,1,size=(2,7,7));p/=p.sum(axis=-1,keepdims=True);return ObservableModel(1.5,g,p,np.full((2,7,7),20,dtype=np.int32),{},{}),rng

def test_both_macro_models_every_observed_transition_and_no_future_route():
    m,rng=model();h=rng.normal(size=1024);o=rng.normal(size=512);state=m.new_state(h);out=state.replay(o);assert out[0,24]==0 and out[0,26]==-1
    for t in range(1,len(o)):
        a,b=int(out[t-1,0]),int(out[t,0]);np.testing.assert_array_equal(out[t,20:22],m.macro_probabilities[:,a,b]);assert out[t,24]==1
    s=m.new_state(h);scalar=np.array([s.predict_one(x) for x in o]);np.testing.assert_array_equal(out,scalar)

def test_observer_chunk_serialization_and_empty_replay():
    m,rng=model();h=rng.normal(size=1024);o=rng.normal(size=512);expected=m.new_state(h).replay(o);s=m.new_state(h);a=s.replay(o[:63]);assert s.replay([]).shape==(0,27);s=pickle.loads(pickle.dumps(s));np.testing.assert_array_equal(np.vstack((a,s.replay(o[63:]))),expected)
