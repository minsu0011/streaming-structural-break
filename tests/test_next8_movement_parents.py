"""Movement selection cannot use break labels or count duplicate controls."""
import numpy as np
import pytest
import src.next8
from src.next8.cache import FeatureCache
from src.next8.data import FitScope
from src.next8.features import N_FEATURES
from src.next8.movement import (own_design,augment,mechanical_design,history_value,
    movement_target,parent_selection_split,cross_eligible,forecast_delta)


def feature_fixture():
    rng=np.random.default_rng(8193);nseries=256;length=48;n=nseries*length
    c=FeatureCache.__new__(FeatureCache)
    c.values=rng.normal(size=(n,N_FEATURES)).astype(np.float32)
    for s in range(nseries):
        start=s*length
        c.values[start+2:start+length,0]=1.4*c.values[start:start+length-2,105]+rng.normal(0,.03,length-2)
    ids=np.repeat(np.arange(nseries),length)
    c.meta={'dataset_id':ids,'time_online':np.tile(np.arange(length),nseries),
            'target':rng.integers(0,2,n).astype(np.uint8)}
    c.series_index=ids.copy()
    c.series={'seed_values':rng.normal(size=(nseries,16,N_FEATURES)).astype(np.float32),
        'low':np.full((nseries,N_FEATURES),-.5),'high':np.full((nseries,N_FEATURES),.5),
        'valid':np.ones((nseries,N_FEATURES),dtype=bool),'epsilon':np.full((nseries,3,N_FEATURES),.02),
        'H_length':np.where(np.arange(nseries)%2,2048,512),
        'context':rng.normal(size=(nseries,6)).astype(np.float32)}
    scope=FitScope(frozenset(range(128)),frozenset(range(128,256)),frozenset(range(900,910)),'movement_oracle')
    return c,scope


def test_duplicate_parent_coordinates_do_not_change_regularization():
    c,_=feature_fixture();rows=np.arange(100)
    base=own_design(c,rows,0,1)
    again=augment(c,rows,base,[0],prefix='different_block')
    assert base.names==again.names
    np.testing.assert_array_equal(base.values,again.values)
    d=mechanical_design(c,rows,0,1)
    assert len(d.names)==len(set(d.names))


def test_primitive_already_in_mechanical_controls_is_not_new_factor():
    assert not cross_eligible(51,0)
    assert not cross_eligible(0,51)
    assert not cross_eligible(77,51)
    assert cross_eligible(105,0)


def test_duration_history_stops_at_own_H_boundary():
    c,_=feature_fixture();rows=np.array([48,49,63,64]);cols=[0,105]
    for lag in (2,3,5,7,15):
        before=history_value(c,rows,cols,lag)
        c.values[:48]+=100
        np.testing.assert_array_equal(before,history_value(c,rows,cols,lag))


def test_future_suffix_does_not_change_forecast_design():
    c,_=feature_fixture();rows=np.arange(32)
    before=mechanical_design(c,rows,0,16,parents=[105])
    c.values[32:48]*=-12
    after=mechanical_design(c,rows,0,16,parents=[105])
    np.testing.assert_array_equal(before.values,after.values)
    assert not any('future' in n for n in before.names)


def test_movement_target_rejects_cross_series_future():
    c,_=feature_fixture()
    with pytest.raises(ValueError):movement_target(c,np.array([47]),0,1)


def test_parent_selection_is_label_free_and_recovers_known_lag(tmp_path,monkeypatch):
    import src.next8.movement as m
    monkeypatch.setattr(m,'DETAILED',(0,105));monkeypatch.setattr(m,'HORIZONS',(1,))
    c,scope=feature_fixture()
    first,adjustment=parent_selection_split(c,scope,tmp_path/'first')
    edge=next(x for x in first[0]['candidates'] if x['parent']==105)
    assert edge['gain']>1 and edge['ci_low']>0
    # Break labels, including training labels, are wholly irrelevant to A.
    c.meta['target'][:]=1-c.meta['target']
    second,other=parent_selection_split(c,scope,tmp_path/'second')
    edge2=next(x for x in second[0]['candidates'] if x['parent']==105)
    assert edge['gain']==edge2['gain'] and edge['ci_low']==edge2['ci_low']
    assert adjustment==other
    assert first[0]['breakOccurred_labels_used'] is False


def test_forecast_known_bound_applies_equally_to_both_directions():
    class Extreme:
        def predict(self,d):return np.array([-100.,100.])
    current=np.array([-11.,11.])
    result=forecast_delta(Extreme(),None,current)
    np.testing.assert_array_equal(current+result,[-12.,12.])
