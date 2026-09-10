import inspect
import numpy as np
from src.models.sequential_detectors import SequentialDetectors,update

def test_complementary_detectors_prefix_reference_and_fixed_memory():
    rng=np.random.default_rng(60);h=rng.normal(size=1000);o=rng.normal(size=100)
    a,b,c=SequentialDetectors(h),SequentialDetectors(h),SequentialDetectors(h)
    actual=np.array([a.update_and_get(x).copy() for x in o])
    changed=np.r_[o[:40],np.full(20,1e3)]
    alternative=np.array([b.update_and_get(x).copy() for x in changed])
    np.testing.assert_array_equal(actual[:40],alternative[:40])
    reference=[]
    for x in o:
        update.py_func(float(x),c.reference,c.parameters,c.evidence,c.output);reference.append(c.output.copy())
    np.testing.assert_allclose(actual,reference,rtol=0,atol=1e-7)
    before=a.state_array_bytes
    for x in [np.nan,np.inf,-np.inf]*1000:a.update_and_get(x)
    assert a.state_array_bytes==before and np.isfinite(a.output).all()
    assert tuple(inspect.signature(SequentialDetectors.update_and_get).parameters)==('self','point')

def test_residual_detectors_respond_to_distinct_synthetic_changes():
    rng=np.random.default_rng(61);h=rng.normal(size=2000)
    normal=rng.normal(size=400);mean=normal+2;scale=normal*3
    outputs=[]
    for stream in (normal,mean,scale):
        state=SequentialDetectors(h);outputs.append(np.array([state.update_and_get(x).copy() for x in stream]))
    assert outputs[1][-100:,0].mean()>outputs[0][-100:,0].mean()+.2
    assert outputs[2][-100:,1].mean()>outputs[0][-100:,1].mean()+.2
    assert outputs[2][-100:,2].mean()>outputs[0][-100:,2].mean()+.2
