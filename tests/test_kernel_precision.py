import numpy as np
from src.features.streaming import StreamingFeatureState,_update

def test_constant_reference_mixed_precision_regression():
    rng=np.random.default_rng(20260908)
    h=np.full(127,3,dtype=np.float32)
    a,b=StreamingFeatureState(h),StreamingFeatureState(h)
    for point in rng.normal(size=500):
        result=a.update_and_get(point).copy()
        _update.py_func(float(point),b.reference,b.baseline,b.thresholds,b.ring,b.ew,b.counters,b.output,b.parameters,b.alphas,b.lags)
        np.testing.assert_allclose(result,b.output,rtol=0,atol=1e-5)
