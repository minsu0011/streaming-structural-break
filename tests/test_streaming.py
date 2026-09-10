import builtins
import inspect
import io
import numpy as np
import pytest
from src.features.streaming import StreamingFeatureState,replay,FEATURE_NAMES
from src.models.baselines import BaselineDetector,BASELINES
from src.streaming.inference import write_engineering_baseline,infer
from src.validation.synthetic import fixture

def test_api_excludes_targets():
    assert tuple(inspect.signature(StreamingFeatureState).parameters)==('historical','config')
    assert tuple(inspect.signature(StreamingFeatureState.update_and_get).parameters)==('self','point')

def test_prefix_and_offline_live():
    h,o,_=fixture('mean_up')
    changed=np.r_[o[:170],np.full(83,1e6)]
    first=np.array(list(replay(h,o)))
    second=np.array(list(replay(h,changed)))
    np.testing.assert_array_equal(first[:170],second[:170])
    state=StreamingFeatureState(h)
    for expected,x in zip(first,o): np.testing.assert_array_equal(state.update_and_get(x),expected)
    for name in BASELINES:
        a,b=BaselineDetector(name,h),BaselineDetector(name,h)
        pa=[a.predict_one(x) for x in o]
        pb=[b.predict_one(x) for x in changed]
        np.testing.assert_array_equal(pa[:170],pb[:170])
        assert np.isfinite(pa).all() and min(pa)>=0 and max(pa)<=1

@pytest.mark.parametrize('h',[[],[0],[1,1,1],[np.nan,np.inf],[1e100,-1e100]])
def test_degenerate_nonfinite_inputs(h):
    state=StreamingFeatureState(h)
    for x in [0,1,np.nan,np.inf,-np.inf,1e308,-1e308]:
        f=state.update_and_get(x)
        assert f.dtype==np.float32 and f.shape==(len(FEATURE_NAMES),)
        assert np.isfinite(f).all()

def test_fixed_memory():
    h,o,_=fixture('no_break')
    state=StreamingFeatureState(h); before=state.state_array_bytes
    for _ in range(20):
        for x in o: state.update_and_get(x)
    assert before==state.state_array_bytes
    assert before<4096

class GuardedOnline:
    def __init__(self,points): self.points=iter(points); self.pending=False; self.started=False; self.count=0
    def __iter__(self):
        if self.started: raise AssertionError('Second iteration')
        self.started=True
        return self
    def __next__(self):
        if self.pending: raise AssertionError('Read ahead before yield')
        value=next(self.points); self.pending=True; self.count+=1
        return value
    def __len__(self): raise AssertionError('Future length access')
    def acknowledge(self):
        assert self.pending
        self.pending=False

def test_guarded_infer_and_determinism(tmp_path,monkeypatch):
    h,o,_=fixture('mean_up')
    write_engineering_baseline(tmp_path)
    original_open=builtins.open
    original_io_open=io.open
    def guarded(original):
        def opener(file,mode='r',*args,**kwargs):
            assert not any(x in mode for x in 'wax+'),'infer attempted write'
            return original(file,mode,*args,**kwargs)
        return opener
    # Warm JIT before the write guard to exclude third-party import-time setup.
    StreamingFeatureState(h).update_and_get(0)
    with monkeypatch.context() as mp:
        mp.setattr(builtins,'open',guarded(original_open)); mp.setattr(io,'open',guarded(original_io_open))
        outputs=[]
        for _ in range(2):
            guarded_online=GuardedOnline(o)
            def datasets():
                yield h,guarded_online
            generator=infer(datasets(),tmp_path)
            assert next(generator) is None and guarded_online.count==0
            result=[]
            for expected in range(len(o)):
                value=next(generator)
                assert guarded_online.count==expected+1
                assert isinstance(value,float) and np.isfinite(value) and 0<=value<=1
                result.append(value); guarded_online.acknowledge()
            with pytest.raises(StopIteration): next(generator)
            outputs.append(result)
    np.testing.assert_array_equal(*outputs)

def test_series_state_isolation(tmp_path):
    write_engineering_baseline(tmp_path)
    h,a,_=fixture('mean_up'); _,b,_=fixture('variance_up')
    combined=list(infer(iter([(h,a),(h,b)]),tmp_path))[1:]
    alone=list(infer(iter([(h,b)]),tmp_path))[1:]
    np.testing.assert_array_equal(combined[len(a):],alone)
