import numpy as np
from src.next3.rank_dependence import make_state,rank_values,SETTINGS


def test_midrank_ties_extremes_and_frozen_h_mapping():
    h=np.array([-1.,0.,0.,1.])
    np.testing.assert_array_equal(rank_values(np.array([-9.,0.,9.]),h),np.array([-.8,0.,.8]))
    before=h.copy();rank_values(np.arange(100.),h)
    np.testing.assert_array_equal(h,before)


def test_rank_dependence_live_prefix_and_fixed_memory():
    rng=np.random.default_rng(821)
    h=rng.normal(size=512).astype(np.float32);o=rng.normal(size=700).astype(np.float32);o[50]=1000.
    state=make_state(h,SETTINGS);saved=state.args[3].copy()
    from src.next.fused_runtime import unique_array_bytes
    before=unique_array_bytes(state.args)
    live=np.array([state.update(v).copy() for v in o])
    np.testing.assert_array_equal(live,make_state(h,SETTINGS).replay(o))
    np.testing.assert_array_equal(state.args[3],saved)
    assert unique_array_bytes(state.args)==before and np.isfinite(live).all()
    changed=o.copy();changed[301:]*=20.
    np.testing.assert_array_equal(live[:301],make_state(h,SETTINGS).replay(changed)[:301])
    np.testing.assert_array_equal(live,make_state(h*2,SETTINGS).replay(o*2))
