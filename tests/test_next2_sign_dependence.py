import numpy as np
from src.next2.sign_dependence import make_state

SETTINGS={'version':'sign_dependence_v1','order':8}


def test_sign_dependence_streaming_prefix_and_units():
    rng=np.random.default_rng(210);h=rng.normal(size=512).astype(np.float32);o=rng.normal(size=600).astype(np.float32)
    expected=make_state(h,SETTINGS).replay(o)
    actual=np.asarray([make_state(h,SETTINGS).update(o[0])])
    np.testing.assert_array_equal(actual,expected[:1])
    state=make_state(h,SETTINGS);live=np.asarray([state.update(v).copy() for v in o])
    np.testing.assert_array_equal(live,expected)
    changed=o.copy();changed[128:]*=10
    np.testing.assert_array_equal(make_state(h,SETTINGS).replay(changed)[:128],expected[:128])
    np.testing.assert_array_equal(make_state(h*2,SETTINGS).replay(o*2),expected)
