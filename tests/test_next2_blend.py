import numpy as np
import pytest
from src.next2.blend import combine


def test_blend_uses_component_float32_before_float64_combination():
    left=np.array([.1,.4,.9],dtype=np.float64)
    right=np.array([.9,.3,.2],dtype=np.float64)
    for weight in (.9,.8,.67,.5):
        expected=(weight*left.astype(np.float32).astype(float)+(1-weight)*right.astype(np.float32).astype(float)).astype(np.float32)
        np.testing.assert_array_equal(combine(left,right,weight),expected)
        for j in range(3):
            assert combine(left[j],right[j],weight)==expected[j]


def test_blend_rejects_unregistered_weights_and_misalignment():
    with pytest.raises(ValueError):
        combine([.1],[.4],.731)
    with pytest.raises(ValueError):
        combine([.1,.2],[.4],.9)
