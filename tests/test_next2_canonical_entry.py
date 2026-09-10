from pathlib import Path
import importlib
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('role',['DISTILLED','PERFORMANCE'])
def test_canonical_entry_lazy_prefix_and_reentrant_series(role):
    runtime='tree_used_v1' if role=='DISTILLED' else 'compact_v1'
    folder=ROOT/'artifacts/next2/canonical_deployment'/(role+'_'+runtime)
    if not (folder/'model.json').exists():
        pytest.skip('Canonical research deployment absent')
    entry=importlib.import_module('next2_canonical_'+role.lower()+'_main')
    rng=np.random.default_rng(633891)
    h=rng.normal(size=512).astype(np.float32)
    o=rng.normal(size=241).astype(np.float32)
    consumed=[]
    def points():
        for i,p in enumerate(o):
            consumed.append(i)
            yield p
    def batches():
        consumed.append('batch')
        yield h,points()
    stream=entry.infer(batches(),folder)
    assert next(stream) is None and consumed==[]
    actual=[]
    for i in range(len(o)):
        actual.append(next(stream))
        assert consumed==['batch']+list(range(i+1))
    with pytest.raises(StopIteration):
        next(stream)
    prefix=list(entry.infer([(h,iter(o[:73]))],folder))
    np.testing.assert_array_equal(prefix[1:],actual[:73])
    repeated=list(entry.infer([(h,iter(o)),(h,iter(o))],folder))
    np.testing.assert_array_equal(repeated[1:len(o)+1],actual)
    np.testing.assert_array_equal(repeated[len(o)+1:],actual)
