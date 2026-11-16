"""Counterfactual learned-rule and meta-selection leakage tests."""
import numpy as np
import pytest
import src.next8
from src.next8.data import FitScope
from src.next8.learning import (Design, fit_logistic, fit_ridge, orientation_design,
    estimate_target_orientation, feature_contribution, forecast_record,
    select_training_only, crossfit_join)
from src.next8.features import UNKNOWN


@pytest.fixture(scope='module')
def fitted():
    rng=np.random.default_rng(81016)
    ids=np.repeat(np.arange(40),32)
    f=rng.normal(size=len(ids)).astype(np.float32)
    state=rng.integers(0,3,len(ids))
    y=(rng.random(len(ids))<1/(1+np.exp(-np.where(state==0,2.,-2.)*f))).astype(int)
    training=ids<30;valid=~training
    scope=FitScope(frozenset(range(30)),frozenset(range(30,40)),frozenset(range(50,60)),'synthetic_orientation')
    design=orientation_design(f[training],state[training],np.zeros((training.sum(),1)),ids[training])
    model=fit_logistic(design,y[training],scope)
    validation=orientation_design(f[valid],state[valid],np.zeros((valid.sum(),1)),ids[valid])
    return model,validation,y[valid],scope


def test_orientation_not_constructed_from_validation_label(fitted):
    model,design,y,scope=fitted
    p=model.predict_proba(design)
    y[:]=1-y
    np.testing.assert_array_equal(p,model.predict_proba(design))
    assert estimate_target_orientation(model,.3,0)>0
    assert estimate_target_orientation(model,.3,2)<0


@pytest.mark.parametrize('kind',['state','feature_classification','graph'])
def test_selection_rejects_heldout_and_outer_membership(fitted,kind):
    _,_,_,scope=fitted
    for ids in ([31],[51],[1,31]):
        with pytest.raises(ValueError):select_training_only({'gain':1},scope,ids,kind)


def test_state_selection_training_only(fitted):
    scope=fitted[3]
    result=select_training_only({'threshold':1},scope,[1,2],'state')
    assert result['provenance']['training_ids_sha256']==scope.provenance()['training_ids_sha256']
    with pytest.raises(ValueError):select_training_only({},scope,[30],'state')


def test_feature_classification_training_only(fitted):
    scope=fitted[3]
    with pytest.raises(ValueError):select_training_only({'class':'INVARIANT_DIRECTION'},scope,[35],'feature_classification')
    assert select_training_only({},scope,[0],'feature_classification')['kind']=='feature_classification'


def test_graph_selection_training_only(fitted):
    scope=fitted[3]
    with pytest.raises(ValueError):select_training_only({'parent':1},scope,[50],'graph')
    assert select_training_only({},scope,[0],'graph')['kind']=='graph'


def test_meta_predictions_cross_fitted():
    ids=np.repeat(np.arange(4),2)
    parts=[{'positions':np.arange(4),'training_ids':[2,3],'selection_ids':[2,3],'prediction':np.full(4,.4)},
           {'positions':np.arange(4,8),'training_ids':[0,1],'selection_ids':[0,1],'prediction':np.full(4,.6)}]
    np.testing.assert_array_equal(crossfit_join(ids,parts),np.r_[np.full(4,.4),np.full(4,.6)])
    # Fits excluded the held group, but global feature selection did not.
    bad=[{**parts[0],'selection_ids':[0,1,2,3]},parts[1]]
    with pytest.raises(ValueError):crossfit_join(ids,bad)
    with pytest.raises(ValueError):crossfit_join(ids,parts[:1])


def test_abs_transform_not_claimed_as_dynamic_rule():
    f=np.array([-2.,-1.,0.,1.,2.]);s=np.array([0,0,1,2,2]);ids=np.arange(5)
    fixed=orientation_design(f,s,np.zeros((5,1)),ids,'O1')
    assert not any('state' in n for n in fixed.names)
    pos=fixed.values[:,fixed.names.index('F_positive')]
    neg=fixed.values[:,fixed.names.index('F_negative')]
    np.testing.assert_array_equal(pos+neg,np.abs(f)/4)


def test_prediction_of_future_not_logged_as_observed_event():
    record=forecast_record(3.,16,99)
    assert record['kind']=='FORECAST'
    assert record['observed_event'] is False and record['confirmed_break'] is False
    assert record['issued_at_observed_time']==99


@pytest.mark.parametrize('name',['tau','future_target_F','true_phase','true_break_age','total_online_length','validation_label'])
def test_learners_reject_unobservable_input_schema(name):
    s=FitScope(frozenset([1]),frozenset([2]),frozenset([3]),'fixture')
    design=Design(np.array([[1.],[2.]]),(name,),np.array([1,1]))
    with pytest.raises(ValueError):fit_ridge(design,np.array([0.,1.]),s)


def test_unknown_branch_preserves_other_score(fitted):
    model,design,_,_=fitted
    baseline=np.linspace(.2,.8,len(design.ids))
    contribution=feature_contribution(model,design,np.ones(len(design.ids),dtype=bool))
    np.testing.assert_array_equal(baseline+contribution,baseline)
    assert estimate_target_orientation(model,1.,UNKNOWN)==0


def test_rule_derivative_matches_fixed_contribution(fitted):
    model,_,_,_=fitted
    eps=.001
    a=orientation_design(np.array([.7-eps,.7+eps]),np.array([0,0]),np.zeros((2,1)),np.array([30,30]))
    c=feature_contribution(model,a,np.zeros(2,dtype=bool))
    assert np.isclose((c[1]-c[0])/(2*eps),estimate_target_orientation(model,.7,0),rtol=1e-4)
