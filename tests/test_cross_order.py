import json
import numpy as np
import pytest
from src.streaming.training import train_frozen_candidate
from src.streaming.inference import infer
from src.models.cross_order import CrossOrderBlendBundle
from src.features.cross_order import feature_names


class Sealed:
    def __array__(self,*a,**kw):raise AssertionError('Sealed values inspected')
    def __iter__(self):raise AssertionError('Sealed values iterated')
    def __len__(self):raise AssertionError('Sealed length inspected')


def examples():
    rng=np.random.default_rng(8051)
    for sid in range(4):
        if sid==3:yield sid,Sealed(),Sealed(),Sealed();continue
        historical=rng.normal(size=300).astype(np.float32);online=rng.normal(size=120).astype(np.float32)
        if sid:online[60:]*=2
        yield sid,historical,online,60 if sid else None


@pytest.mark.parametrize('fused,fast',[(False,False),(True,False),(True,True)])
def test_cross_order_training_and_wire_average_are_exact(tmp_path,fused,fast):
    spec={'status':'FROZEN_PRIMARY','development_ids':[0,1,2],'known_training_ids':[0,1,2,3],
        'use_fused_inference':fused,'use_fast_historical_init':fast,
        'candidate':{'model':'M7_cross_order_equal','families':'ABCD','sampling':'S3','params':{},
            'feature_changes':{'normalization':'median_mad','scales':[5,20,160]},'representation':'cross_order_ar4_ar8_equal'}}
    descriptor=train_frozen_candidate(examples(),tmp_path,spec)
    assert descriptor['kind']=='cross_order_equal' and descriptor['seal_rows_fitted']==0
    assert descriptor['seal_series_skipped_before_array_or_label_access']==1 and descriptor['training_rows']==360
    model=CrossOrderBlendBundle.load(tmp_path/'model.joblib');_,historical,online,_=next(examples())
    stream=model.make_state(historical);before=stream.state_array_bytes
    matrix=np.stack([stream.update_and_get(p).copy() for p in online])
    assert matrix.shape==(120,114) and len(feature_names(model.feature_config))==114
    assert stream.state_array_bytes==before
    left=model.components[0].predict(matrix[:,:57]).astype(np.float32).astype(np.float64)
    right=model.components[1].predict(matrix[:,57:]).astype(np.float32).astype(np.float64)
    expected=(.5*left+.5*right).astype(np.float32)
    np.testing.assert_array_equal(model.predict(matrix),expected)
    observed=list(infer(iter([(historical,iter(online))]),tmp_path));assert observed[0] is None
    np.testing.assert_array_equal(np.asarray(observed[1:],dtype=np.float32),expected)
    altered=np.r_[online[:24],np.full(17,10000,dtype=np.float32)]
    suffix=list(infer(iter([(historical,iter(altered))]),tmp_path))
    np.testing.assert_array_equal(np.asarray(suffix[1:25],dtype=np.float32),expected[:24])
    if fused:
        from src.streaming.fused_cross_order import PreparedFusedCrossOrderPredictor
        prepared=PreparedFusedCrossOrderPredictor(model,fast_initialization=fast)
        resident=prepared.new_state(historical);reference=prepared.new_state(historical)
        initial_bytes=resident.state_array_bytes
        probe=np.r_[online,np.nan,np.inf,-np.inf,0.,1e30,-1e30].astype(np.float32)
        actual=np.asarray([resident.predict_one(p) for p in probe],dtype=np.float32)
        expected_tuple=np.asarray([reference.predict_one_tuple(p) for p in probe],dtype=np.float32)
        np.testing.assert_array_equal(actual,expected_tuple)
        assert resident.state_array_bytes==initial_bytes
        assert np.isfinite(actual).all()
        independent=prepared.new_state(historical)
        assert type(independent.resident) is type(resident.resident)
        assert not np.shares_memory(independent.states[0].ar_args[2],resident.states[0].ar_args[2])
        descriptor['cross_order_fused_source_hash']='corrupt'
        (tmp_path/'model.json').write_text(json.dumps(descriptor),encoding='utf-8')
        with pytest.raises(RuntimeError,match='cross_order_fused_source_hash'):next(infer(iter([(historical,iter(online))]),tmp_path))


def test_cross_order_training_rejects_unfrozen_formula():
    from src.models.cross_order import fit_cross_order
    with pytest.raises(ValueError,match='parameter tuning'):
        fit_cross_order(np.zeros((2,114),dtype=np.float32),np.array([0,1]),params={'weight':.6})
