import numpy as np
from src.next4.b2_inference import iterate_current
from src.next4.negative_controls import permute_training_labels_within_time,detect_invalid_future_controls
from scripts.verify_next2_shared_public_api import Online


class FakePrepared:
    def new_state(self,historical):
        class State:
            def __init__(self):self.total=float(historical[-1])
            def predict_one(self,point):self.total+=float(point);return self.total
        return State()


def test_current_point_protocol_and_lazy_first_none():
    calls=[];online=Online(np.arange(6,dtype=np.float32))
    def factory():calls.append('load');return FakePrepared()
    def datasets():calls.append('data');yield np.array([2.]),online
    stream=iterate_current(datasets(),factory)
    assert calls==[] and next(stream) is None and calls==[]
    for k in range(6):
        assert next(stream)==2+sum(range(k+1))
        assert online.consumed==k+1
    assert list(stream)==[] and calls==['load','data']


def test_reordered_and_renamed_series_preserve_reset_outputs():
    records={11:(np.array([2.]),np.array([1.,2.])),72:(np.array([-1.]),np.array([3.,4.]))}
    def run(order,rename):
        output={}
        for sid in order:
            h,o=records[sid];stream=iterate_current(iter([(h,Online(o))]),FakePrepared)
            assert next(stream) is None;output[rename[sid]]=list(stream)
        return output
    a=run([11,72],{11:11,72:72});b=run([72,11],{11:98765,72:-3})
    assert a[11]==b[98765] and a[72]==b[-3]


def test_training_only_time_stratified_label_permutation():
    ids=np.repeat(np.arange(12),10);times=np.tile(np.arange(10),12)
    y=((ids%2==0)&(times>3)).astype(np.uint8)
    metadata={'dataset_id':ids,'time_online':times,'target':y}
    train=np.flatnonzero(ids<8);out=permute_training_labels_within_time(metadata,train,20260908)
    np.testing.assert_array_equal(out['target'][ids>=8],y[ids>=8])
    assert np.any(out['target'][train]!=y[train])
    for t in range(10):assert out['target'][(ids<8)&(times==t)].sum()==y[(ids<8)&(times==t)].sum()
    np.testing.assert_array_equal(out['target'],permute_training_labels_within_time(metadata,train,20260908)['target'])
    np.testing.assert_array_equal(metadata['target'],y)


def test_deliberately_invalid_future_controls_are_detected():
    assert detect_invalid_future_controls(Online,np.arange(8,dtype=np.float32))=={
        'future_length':True,'materialized_future_suffix':True,'one_step_future_shift':True}
