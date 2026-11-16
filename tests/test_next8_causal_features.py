"""Causal counterfactuals of the new feature engine, across every family."""
import copy
import inspect
import numpy as np
import pytest
import src.next8
from src.next8.features import (FeatureState, value_sign,
    movement_direction, categorize, relation_type, N_FEATURES, UNKNOWN)
from src.next8.data import FitScope
from src.next8.targets import future_targets


@pytest.fixture(scope='module')
def example():
    rng=np.random.default_rng(8101604)
    h=rng.normal(size=512).astype(np.float32)
    o=rng.normal(size=97).astype(np.float32)
    o[40]=8.;o[60:]*=.7
    return h,o,FeatureState(h)


def test_feature_value_sign_separate_from_movement():
    f=np.array([3.,2.,-1.,-2.])
    assert value_sign(f).tolist()==[1,1,-1,-1]
    assert movement_direction(np.diff(f),.02).tolist()==[0,0,0]


def test_movement_sign_separate_from_target_orientation():
    # Identical observed movement admits opposite population target mappings.
    f=np.array([-2.,-1.,0.,1.,2.])
    movement=movement_direction(np.diff(f),.02)
    y1=(f>0).astype(int);y2=1-y1
    assert np.all(movement==2)
    assert np.cov(f,y1)[0,1]>0 and np.cov(f,y2)[0,1]<0


def test_movement_future_target_not_in_inputs(example):
    h,o,initial=example
    a=FeatureState.restore(initial);b=FeatureState.restore(initial)
    changed=o.copy();changed[48:]+=10
    fa=a.replay(o)[0];fb=b.replay(changed)[0]
    np.testing.assert_array_equal(fa[:48],fb[:48])
    ids=np.zeros(len(o),dtype=int);t=np.arange(len(o))
    ta=future_targets(fa[:,0],ids,t,.02,-1,1,True,16)
    tb=future_targets(fb[:,0],ids,t,.02,-1,1,True,16)
    assert np.any(ta.value[32:48]!=tb.value[32:48])
    np.testing.assert_array_equal(fa[:48],fb[:48])


@pytest.mark.parametrize('point',[0.,3.,-8.])
def test_same_normalizer_calibration_and_online(example,point):
    h,_,initial=example
    state=FeatureState.restore(initial)
    params=(state.center.copy(),state.spread.copy(),state.low.copy(),state.high.copy(),state.epsilon.copy())
    assert state.calibration_reference_sha256==state.online_reference_sha256==state.normalizer_fit_sha256
    engine_ids=[id(e) for e in state.engines]
    state.update(point)
    assert [id(e) for e in state.engines]==engine_ids
    for old,new in zip(params,(state.center,state.spread,state.low,state.high,state.epsilon)):
        np.testing.assert_array_equal(old,new)


def test_H_only_parameters(example):
    _,o,initial=example
    a=FeatureState.restore(initial);b=FeatureState.restore(initial)
    a.replay(o);b.replay(o*-17)
    for attr in ('center','spread','epsilon','low','high','valid','context'):
        np.testing.assert_array_equal(getattr(a,attr),getattr(b,attr))


@pytest.mark.parametrize('forbidden',['tau','true_break_age','future_total_length','future_suffix','target','dataset_id'])
def test_feature_API_rejects_unobservable_arguments(example,forbidden):
    h,o,initial=example
    assert forbidden not in inspect.signature(FeatureState).parameters
    with pytest.raises(TypeError): FeatureState(h,**{forbidden:5})
    with pytest.raises(TypeError): initial.update(float(o[0]),**{forbidden:5})


def test_no_tau_in_state(example):
    assert 'tau' not in vars(example[2])


def test_no_true_break_age_in_state(example):
    assert 'true_break_age' not in vars(example[2])


def test_no_future_total_length(example):
    _,o,initial=example
    a=FeatureState.restore(initial);b=FeatureState.restore(initial)
    np.testing.assert_array_equal(a.replay(o[:21])[0],b.replay(np.r_[o,np.ones(200)])[0][:21])


def test_no_future_suffix(example):
    _,o,initial=example
    a=FeatureState.restore(initial);b=FeatureState.restore(initial)
    suffix=o.copy();suffix[31:]=np.linspace(-15,15,len(o)-31)
    np.testing.assert_array_equal(a.replay(o)[0][:31],b.replay(suffix)[0][:31])


@pytest.mark.parametrize('cut',[0,1,15,16,17,48,96,97])
def test_prefix_invariance(example,cut):
    _,o,initial=example
    a=FeatureState.restore(initial);b=FeatureState.restore(initial)
    np.testing.assert_array_equal(a.replay(o[:cut])[0],b.replay(o)[0][:cut])


def test_no_cross_series_lag():
    f=np.array([0.,1.,100.,101.],dtype=np.float32)
    target=future_targets(f,np.array([1,1,2,2]),np.array([0,1,0,1]),.02,-1,1,True,1)
    assert target.available.tolist()==[True,False,True,False]
    assert np.isnan(target.value[1])
    assert target.direction[1]==target.state[1]==target.reversal[1]==UNKNOWN


def test_early_forecast_target_does_not_cross_cutoff():
    target=future_targets(np.arange(40,dtype=np.float32),np.zeros(40,dtype=int),np.arange(40),.02,-1,1,True,16,cutoff=20)
    assert np.flatnonzero(target.available).tolist()==list(range(5))


def test_reset_between_series(example):
    h,o,initial=example
    a=FeatureState.restore(initial);a.replay(o)
    b=FeatureState(h);c=FeatureState.restore(initial)
    np.testing.assert_array_equal(b.replay(o)[0],c.replay(o)[0])
    assert b.counter==c.counter==len(o)


def test_state_save_restore(example):
    _,o,initial=example
    a=FeatureState.restore(initial);a.replay(o[:43]);snap=a.snapshot()
    first=a.replay(o[43:])[0]
    b=FeatureState.restore(snap);second=b.replay(o[43:])[0]
    np.testing.assert_array_equal(first,second)
    np.testing.assert_array_equal(a.history,b.history)
    # Restoring did not mutate the saved checkpoint.
    assert snap.counter==43


def test_stream_offline_parity(example):
    _,o,initial=example
    a=FeatureState.restore(initial);b=FeatureState.restore(initial)
    batch,states=a.replay(o)
    stream=[b.update(float(x)) for x in o]
    np.testing.assert_array_equal(batch,np.array([v[0] for v in stream]))
    np.testing.assert_array_equal(states,np.array([v[1] for v in stream]))
    for attr in ('history','last_raw','last_unclipped'):
        np.testing.assert_array_equal(getattr(a,attr),getattr(b,attr))
    assert batch.shape==(len(o),N_FEATURES)


def test_orientation_state_known_before_current_feature(example):
    _,o,initial=example
    a=FeatureState.restore(initial);b=FeatureState.restore(initial)
    va=a.update(-19.);vb=b.update(19.)
    np.testing.assert_array_equal(va[2],vb[2])
    assert np.any(va[0]!=vb[0])


@pytest.mark.parametrize('length',[0,1,31,96,159])
def test_unknown_state_fallback(length):
    s=FeatureState(np.zeros(length,dtype=np.float32))
    f,current,previous,_=s.update(99.)
    assert np.all(f==0) and np.all(current==UNKNOWN) and np.all(previous==UNKNOWN)


def test_mechanical_edges_tagged():
    assert relation_type(51,52)=='MECHANICAL_DERIVED'
    assert relation_type(77,81)=='MECHANICAL_DERIVED'
    assert relation_type(93,94)=='MECHANICAL_DERIVED'
    assert relation_type(0,0)=='SELF_HISTORY'
    assert 'SHARED_RAW' in relation_type(0,105)


def test_train_eval_provenance():
    s=FitScope(frozenset([1,2]),frozenset([3]),frozenset([4]),'fixture')
    s.admit_training([1]);s.admit_validation([3])
    for ids in ([3],[4],[1,3],[]):
        with pytest.raises(ValueError):s.admit_training(ids)
    with pytest.raises(ValueError):FitScope(frozenset([1]),frozenset([1]),frozenset(),'bad')
    assert s.provenance()['training_ids_sha256']!=s.provenance()['validation_ids_sha256']
