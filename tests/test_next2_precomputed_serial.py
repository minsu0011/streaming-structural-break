import numpy as np
import pytest
from src.next.channel_evidence import make_channel_args
from src.next.serial_channel_evidence import serial_channel_step,sum_variances
from src.next2.precomputed_serial_runtime import precompute,precomputed_channel_step


@pytest.mark.parametrize('rho',[(0.,0.),(-.95,.95),(.7,-.4)])
def test_serial_precomputation_preserves_every_feature_across_age_boundaries_and_wraps(rho):
    original=make_channel_args(2,512);changed=make_channel_args(2,512)
    variance=sum_variances(original[0],rho);table,logs=precompute(original[0],original[1],variance)
    rng=np.random.default_rng(512)
    values=rng.standard_t(3,(4096,2))
    values[::129]=0.;values[::511]=1000.;values[::257]=-1000.
    for point in values:
        left=serial_channel_step(point,original,variance).copy()
        right=precomputed_channel_step(point,changed,variance,table,logs).copy()
        np.testing.assert_array_equal(left.view(np.uint32),right.view(np.uint32))
    for left,right in zip(original,changed):np.testing.assert_array_equal(left,right)
