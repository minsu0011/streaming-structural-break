import numpy as np
import pytest
from src.next.late_blend import combine_scores


@pytest.mark.parametrize('weight',[.5,.67,.33])
def test_fixed_combination_matches_float32_scalar_contract(weight):
    rng = np.random.default_rng(179)
    left,right = rng.uniform(size=(2,10000))
    expected = np.asarray([np.float32(weight*float(np.float32(a))+(1.-weight)*float(np.float32(b))) for a,b in zip(left,right)])
    actual = combine_scores(left,right,weight)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(actual,np.asarray([combine_scores(a,b,weight) for a,b in zip(left,right)]))


def test_unregistered_weight_and_misaligned_scores_rejected():
    with pytest.raises(ValueError):
        combine_scores([.1],[.2],.501)
    with pytest.raises(ValueError):
        combine_scores([.1,.2],[.2],.5)
