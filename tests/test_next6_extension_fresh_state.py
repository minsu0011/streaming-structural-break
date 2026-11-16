import pickle
import numpy as np
import pytest
from src.next6_extension.fresh_state import State

@pytest.fixture
def series():
    rng=np.random.default_rng(94613);return rng.normal(size=2048).astype(np.float32),rng.normal(size=1024).astype(np.float32)

def test_scalar_batch_prefix_and_serialization(series):
    h,o=series;batch=State(h,threshold=1.).replay(o);state=State(h,threshold=1.);scalar=np.array([state.predict_one(x) for x in o]);np.testing.assert_array_equal(batch,scalar)
    prefix=State(h,threshold=1.);a=prefix.replay(o[:117]);restored=pickle.loads(pickle.dumps(prefix));np.testing.assert_array_equal(a,batch[:117]);np.testing.assert_array_equal(restored.replay(o[117:]),batch[117:])
    changed=o.copy();changed[117:]=changed[117:]*100;np.testing.assert_array_equal(State(h,threshold=1.).replay(changed)[:117],batch[:117])

@pytest.mark.parametrize('start',[0,1,15,16,17,127,128,129,255])
@pytest.mark.parametrize('duration',[1,16])
def test_shock_has_finite_influence_on_all_outputs(series,start,duration):
    h,o=series;changed=o.copy();changed[start:start+duration]+=20*np.where(np.arange(duration)%2==0,1,-1)
    baseline=State(h,threshold=.8).replay(o);treatment=State(h,threshold=.8).replay(changed)
    np.testing.assert_array_equal(baseline[:start],treatment[:start])
    np.testing.assert_array_equal(baseline[start+duration+128:],treatment[start+duration+128:])

def test_only_distinct_completed_blocks_confirm(series):
    h,o=series;out=State(h,threshold=0.).replay(o)
    assert np.all(out[:63,0]!=4)
    assert np.flatnonzero(out[:,4]).tolist()==list(range(15,len(o),16))
    for j in range(16,len(o)):
        if out[j,4]==0:
            assert out[j,5]==out[j-1,5]
            assert out[j,1]==out[j-1,1]

def test_series_isolation_and_bounded_state(series):
    h,o=series;a=State(h,threshold=1.);b=State(h[::-1],threshold=1.);expected=a.replay(o);a=State(h,threshold=1.);size=a.bounded_mutable_state_bytes;actual=[]
    for x in o:actual.append(a.predict_one(x));b.predict_one(-x)
    np.testing.assert_array_equal(actual,expected);assert a.bounded_mutable_state_bytes==size and size<2048

def test_constant_reference_is_not_normal():
    out=State(np.ones(1024),threshold=1.).replay(np.r_[np.ones(128),np.ones(128)*20]);assert np.all(out[:,0]==6)

@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf,1e100,-1e100])
def test_invalid_scalar_and_batch_rejected_without_mutation(series,value):
    h,o=series;s=State(h);before=pickle.dumps(s)
    with pytest.raises(ValueError):s.predict_one(value)
    assert pickle.dumps(s)==before
    with pytest.raises(ValueError):s.replay([value])
    assert pickle.dumps(s)==before

def test_labels_cannot_enter_api(series):
    h,o=series;s=State(h)
    for name in ('tau','target','phase','final_length','dataset_id'):
        with pytest.raises(TypeError):s.predict_one(o[0],**{name:1})
