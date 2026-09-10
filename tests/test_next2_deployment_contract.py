from pathlib import Path
from types import SimpleNamespace
import importlib
import numpy as np
import pytest
from src.next2.deployment_training import materialize_dev,OpaqueSeal

ROOT=Path(__file__).resolve().parents[1]


def tiny_guard():
    return SimpleNamespace(dev_ids={1,2},seal_ids={3},split=SimpleNamespace(mapping={1:None,2:None,3:None},digest='fixed'),check_state=lambda:None)


SPEC={'development_ids':[1,2],'known_training_ids':[1,2,3],'split_sha256':'fixed'}


def test_training_discards_seal_before_any_value_access_and_copies_buffers():
    buffer=np.arange(8,dtype=np.float32)
    def stream():
        yield 3,OpaqueSeal(),OpaqueSeal(),OpaqueSeal()
        yield 1,buffer,buffer,None
        buffer[:]=-7
        yield 2,buffer,buffer,3
    records,skipped=materialize_dev(stream(),SPEC,tiny_guard())
    assert skipped==1 and set(records)=={1,2}
    np.testing.assert_array_equal(records[1][0],np.arange(8,dtype=np.float32))
    assert records[1][2] is None and records[2][2]==3


@pytest.mark.parametrize('ids',[[1,2],[1,2,3,3],[1,2,4],[True,2,3],[1.,2,3]])
def test_training_rejects_bad_id_coverage_before_any_fit(ids):
    def stream():
        for sid in ids:
            yield sid,np.arange(8),np.arange(8),None
    with pytest.raises(RuntimeError):
        materialize_dev(stream(),SPEC,tiny_guard())


class OneShot:
    def __init__(self,values):
        self.values=iter(values)
        self.started=False
        self.consumed=0
    def __iter__(self):
        if self.started:
            raise AssertionError('Second iteration attempted')
        self.started=True
        return self
    def __next__(self):
        value=next(self.values)
        self.consumed+=1
        return value
    def __len__(self):
        raise AssertionError('Online horizon inspected')
    def __getitem__(self,key):
        raise AssertionError('Online indexing attempted')


@pytest.mark.parametrize('role',['DISTILLED','PERFORMANCE'])
def test_entry_first_none_lazy_one_shot_prefix_and_reset(role):
    folder=ROOT/'artifacts/next2/deployment_research'/role
    if not (folder/'model.json').exists():
        pytest.skip('Research deployment artifact absent from source-only checkout')
    entry=importlib.import_module('next2_'+role.lower()+'_main')
    rng=np.random.default_rng(31719)
    h=rng.normal(size=256).astype(np.float32)
    p=rng.normal(size=95).astype(np.float32)
    online=OneShot(p)
    batches=OneShot([(h,online)])
    stream=entry.infer(batches,folder)
    assert next(stream) is None
    assert batches.consumed==0 and online.consumed==0
    actual=[]
    for k in range(len(p)):
        actual.append(next(stream))
        assert online.consumed==k+1
        assert 0<=actual[-1]<=1 and np.isfinite(actual[-1])
    with pytest.raises(StopIteration):
        next(stream)
    prefix=list(entry.infer([(h,OneShot(p[:31]))],folder))
    np.testing.assert_array_equal(prefix[1:],actual[:31])
    repeated=list(entry.infer([(h,OneShot(p)),(h,OneShot(p))],folder))
    np.testing.assert_array_equal(repeated[1:1+len(p)],actual)
    np.testing.assert_array_equal(repeated[1+len(p):],actual)


@pytest.mark.parametrize('role',['DISTILLED','PERFORMANCE'])
def test_entry_rejects_nonfinite_current_point(role):
    folder=ROOT/'artifacts/next2/deployment_research'/role
    if not (folder/'model.json').exists():
        pytest.skip('Research deployment artifact absent')
    entry=importlib.import_module('next2_'+role.lower()+'_main')
    h=np.sin(np.arange(128)*.3).astype(np.float32)
    stream=entry.infer([(h,iter([0.,float('nan')]))],folder)
    assert next(stream) is None
    assert np.isfinite(next(stream))
    with pytest.raises(ValueError):
        next(stream)
