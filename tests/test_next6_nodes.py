import inspect
import numpy as np
import pytest
from src.next6.nodes import NodeState,NAMES

def stream(seed=0):
    r=np.random.default_rng(seed);return r.normal(size=320).astype(np.float32),r.normal(size=81).astype(np.float32)

def test_exact_16_fixed_names_and_api_forbidden_label_inputs():
    assert len(NAMES)==len(set(NAMES))==16
    assert tuple(inspect.signature(NodeState).parameters)==('historical',)
    assert tuple(inspect.signature(NodeState.update).parameters)==('self','point')
    h,o=stream()
    with pytest.raises(TypeError):NodeState(h,tau=20)

def test_batch_live_prefix_suffix_and_length():
    h,o=stream();state=NodeState(h);batch=state.replay(o)
    live=NodeState(h);actual=[live.update(x) for x in o]
    for k in range(3):np.testing.assert_array_equal(batch[k],np.stack([r[k] for r in actual]))
    mutated=np.r_[o[:31],np.full(140,200,dtype=np.float32)]
    short=NodeState(h).replay(mutated)
    for k in range(3):np.testing.assert_array_equal(batch[k][:31],short[k][:31])

def test_reset_deterministic_and_nonfinite_rejected():
    h,o=stream();a=NodeState(h).replay(o);h2,o2=stream(1);NodeState(h2).replay(o2);b=NodeState(h).replay(o)
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    with pytest.raises(ValueError):NodeState(h).update(np.nan)
    with pytest.raises(ValueError):NodeState(h).replay(np.array([1,np.inf]))

def test_lags_use_contiguous_actual_time():
    h,o=stream();state=NodeState(h);hx=state.pseudo_x.copy();x,s,t=state.replay(o)
    full=np.concatenate((hx,x));offset=len(hx)
    for k,lag in enumerate((1,2,4,8,16)):
        np.testing.assert_array_equal(t[:,16+16*k:32+16*k],full[offset-lag:offset+len(o)-lag])
    assert np.isin(s,[0,1,2,3]).all()
