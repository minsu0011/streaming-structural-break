"""Actual graph-state and final learner paths, with no outer outcome access."""
import src.next8
import numpy as np
import pytest
import joblib
from tests.test_next8_movement_parents import feature_fixture
from tests.test_next8_forecasts_transitions import graph_for
from src.next8.data import FitScope
from src.next8.movement import movement_target
from src.next8.transitions import FactorizedTransition,transition_inputs
from src.next8.graph_states import predict_graph_state,fit_graph_states
from src.next8.joint_models import (joint_design,fit_joint_models,load_state_partition,
    current_feature_contribution,preselected_linear_rule,raw_coefficients)
from src.next8.common import read_json
from src.next8.screening import sampled_rows
from src.next8.learning import fit_logistic


def selection_for(scope):
    return {'scope':scope.provenance(),'selection_ids':sorted(scope.training_ids),
        'stable_features':[6],'variable_features':[0],
        'classifications':[{'feature':6,'classification':'INVARIANT_DIRECTION'},
                           {'feature':0,'classification':'CONDITIONAL_VARIABLE_CANDIDATE'}],
        'orientation_state_specs':{'0':{'kind':'OWN_F'}}}


def test_M3_M5_have_identical_inputs_including_explicit_products():
    c,scope=feature_fixture();rows=np.arange(100);sel=selection_for(scope)
    cross=np.random.default_rng(44).integers(0,4,(len(rows),12),dtype=np.int8)
    a=joint_design(c,rows,sel,'M3',cross);b=joint_design(c,rows,sel,'M5',cross)
    assert a.names==b.names;np.testing.assert_array_equal(a.values,b.values)


def test_unknown_current_feature_contribution_is_exact_zero():
    c,scope=feature_fixture();sel=selection_for(scope);tr=sampled_rows(c,scope.training_ids)
    model=fit_logistic(joint_design(c,tr,sel,'M4'),c.meta['target'][tr],scope)
    c.series['valid'][128:,0]=False
    rows=np.arange(128*48,128*48+10);design=joint_design(c,rows,sel,'M4')
    np.testing.assert_array_equal(current_feature_contribution(model,design,0),0.)
    assert preselected_linear_rule(model,0,3)==0.
    coefficient,intercept=raw_coefficients(model)
    from scipy.special import expit
    np.testing.assert_allclose(expit(design.values@coefficient+intercept),model.predict_proba(design)[:,1],rtol=0,atol=1e-12)


def test_graph_forecast_is_issued_before_current_observation():
    c,scope=feature_fixture();tr=c.forecast_rows(np.arange(128*48),1)
    own,parent=transition_inputs(c,tr,0,[105])
    model=FactorizedTransition.fit(own,parent,movement_target(c,tr,0,1)['state'],c.meta['dataset_id'][tr],[105],scope)
    rows=np.arange(128*48,128*48+19);before=predict_graph_state(c,rows,0,model)
    c.values[rows[-1]:129*48]=12
    after=predict_graph_state(c,rows,0,model)
    for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
    assert before[0][0]==3 and before[1][0,3]==1.


def test_unconfident_graph_distribution_is_unknown():
    c,_=feature_fixture();rows=np.array([49,50,51])
    class Unconfident:
        parents=(105,)
        def predict(self,own,parent):
            return np.tile([.45,.40,.15,0.],(len(own),1)),{'active_parent_factors':np.ones(len(own),dtype=np.int8)}
    state,p,active=predict_graph_state(c,rows,0,Unconfident())
    assert (state==3).all() and (active==1).all()
    assert (p[:,3]==0).all()  # Abstaining decision differs from unknown true H state.


def test_complete_excluded_group_state_join_and_late_subset(tmp_path):
    c,parent=feature_fixture()
    a=FitScope(frozenset(range(64)),frozenset(range(64,128)),parent.outer_excluded_ids,'leaf_a')
    b=FitScope(frozenset(range(64,128)),frozenset(range(64)),parent.outer_excluded_ids,'leaf_b')
    parts=[]
    for scope in (a,b):
        p=tmp_path/scope.name;fit_graph_states(c,scope,graph_for(scope),p);parts.append((scope,p))
    rows=sampled_rows(c,parent.training_ids)
    state,receipt=load_state_partition(c,parent,rows,parts)
    assert (state==3).all() and len(receipt)==2
    # Scoring subsets can omit short series while forecast files still cover
    # the full heldout partition; that is not missing OOF coverage.
    subset=rows[c.meta['dataset_id'][rows]%2==0]
    sub,_=load_state_partition(c,parent,subset,parts);assert sub.shape==(len(subset),12)
    with pytest.raises(ValueError,match='coverage'):load_state_partition(c,parent,rows,parts[:1])


def test_actual_M_models_never_read_heldout_y(tmp_path):
    c,scope=feature_fixture();selection=selection_for(scope)
    original=c.meta['target'].copy()
    class TrainLabelsOnly:
        def __getitem__(self,rows):
            if np.any(c.meta['dataset_id'][rows]>=128):raise AssertionError('Heldout y read during model fit/prediction')
            return original[rows]
    c.meta['target']=TrainLabelsOnly()
    tr=sampled_rows(c,scope.training_ids);va=np.flatnonzero(c.meta['dataset_id']>=128)
    tx=np.full((len(tr),12),3,dtype=np.int8);vx=np.full((len(va),12),3,dtype=np.int8)
    result=fit_joint_models(c,scope,selection,tmp_path,tx,vx)
    assert result['validation_labels_used'] is False and result['model_aliases']['M5']=='M4'
    with np.load(tmp_path/'PREDICTIONS.npz') as p:
        for key in ('M1','M2','M3','M4','M5'):
            assert np.isfinite(p[key]).all() and ((p[key]>=0)&(p[key]<=1)).all()
        np.testing.assert_array_equal(p['M4'],p['M5'])
    assert fit_joint_models(c,scope,selection,tmp_path,tx,vx)==result


def test_empty_feature_selection_does_not_force_candidates(tmp_path):
    c,scope=feature_fixture();sel=selection_for(scope)
    sel['stable_features']=[];sel['variable_features']=[]
    tr=sampled_rows(c,scope.training_ids);va=np.flatnonzero(c.meta['dataset_id']>=128)
    done=fit_joint_models(c,scope,sel,tmp_path,np.full((len(tr),12),3,dtype=np.int8),np.full((len(va),12),3,dtype=np.int8))
    assert all(v=='M1' for v in done['model_aliases'].values())
    assert len(list(tmp_path.glob('*.joblib')))==1
