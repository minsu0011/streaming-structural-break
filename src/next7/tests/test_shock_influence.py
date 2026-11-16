import numpy as np
import pytest
from src.next7.event import CombinedState
from src.next7.shock_influence import influence_bounds,compare_paths,LOCAL_COLUMNS,CURRENT_STATE_COLUMNS


@pytest.mark.parametrize('indices',[(15,),(16,),(15,16),(63,64,191)])
def test_block_boundary_perturbations_leave_local_evidence_and_current_state(indices):
    h=(np.sin(np.arange(2048)*.173)+.3*np.cos(np.arange(2048)*.611)).astype(np.float32)
    o=(np.sin(np.arange(512)*.267)+.6*np.cos(np.arange(512)*.431)).astype(np.float32)
    changed=o.copy()
    changed[list(indices)]+=100.
    states=[]
    for points in (changed,o):
        raw=CombinedState(h,core_cut=2.,scale_cut=2.).replay(points)
        # G1 supplies the first30 coordinates; unused G2 coordinates are equal
        # zeros in this unit test. Full G2 all1536 original pairs are audited.
        states.append(np.pad(raw,((0,0),(0,15))))
    result=compare_paths(states[0],states[1],np.array(indices))
    assert result['local_evidence_after_bound_bitwise_equal']
    assert result['current_state_after_bound_bitwise_equal']


def test_persistent_memory_is_permitted_but_local_leakage_is_rejected():
    a=np.zeros((128,45));b=a.copy()
    a[16:,30]=.7
    result=compare_paths(a,b,np.array([16]))
    assert result['qualified_intervention_only_latch_at_end']
    assert result['memory_max_difference_after_current_convergence']==.7
    a[100,1]=.01
    with pytest.raises(AssertionError):compare_paths(a,b,np.array([16]))
