import src.next9
import numpy as np
from src.next9.observability import observed_prefix_reference,finite_window_states


def test_full_prefix_reference_remains_causal_and_run_is_censored():
    seed=np.arange(16,dtype=np.float32)[:,None];values=np.arange(16,56,dtype=np.float32)[:,None];epsilon=np.array([.02],dtype=np.float32)
    runs,ages,s4=observed_prefix_reference(seed,values,epsilon)
    changed=values.copy();changed[20:]*=-10
    other=observed_prefix_reference(seed,changed,epsilon)
    for a,b in zip([runs,ages,s4],other):np.testing.assert_array_equal(a[:21],b[:21])
    rows,state=finite_window_states(seed,values,epsilon,stride=1)
    np.testing.assert_array_equal(state['direction_run_capped15'],np.minimum(runs,15))
    assert runs[-1,0]>15 and state['direction_run_capped15'][-1,0]==15


def test_long_flat_bridge_is_observability_limit_not_future_information():
    seed=np.arange(15,-1,-1,dtype=np.float32)[:,None]
    values=np.r_[np.zeros(20),1.,2.].astype(np.float32)[:,None];epsilon=np.array([.02],dtype=np.float32)
    full=observed_prefix_reference(seed,values,epsilon)[2]
    rows,finite=finite_window_states(seed,values,epsilon,stride=1)
    assert full[21,0]==3
    assert finite['S4_REVERSAL'][21,0]==0
    # Current1 at t20 is excluded from the state at t20.
    assert full[20,0]==finite['S4_REVERSAL'][20,0]==4
