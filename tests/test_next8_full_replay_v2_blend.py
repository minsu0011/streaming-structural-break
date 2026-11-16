"""Audit arithmetic matches scalar M6 despite float32 saved head arrays."""
import src.next8
import numpy as np
from src.next8.full_replay_audit_v2 import audit_m6_probability


def test_float32_head_blend_is_exact_actual_float64_scalar_composition():
    b=np.array([.112312398218,.834241332443,.624432139893],dtype=np.float64)
    p=np.array([.1234567,.9876543,.5367891],dtype=np.float32)
    expected=np.array([.95*float(x)+.05*float(y) for x,y in zip(b,p)])
    actual=audit_m6_probability(b,p,True)
    np.testing.assert_array_equal(actual.view(np.uint64),expected.view(np.uint64))
    assert not np.array_equal((.95*b+.05*p).view(np.uint64),expected.view(np.uint64))
    np.testing.assert_array_equal(audit_m6_probability(b,object(),False),b)
