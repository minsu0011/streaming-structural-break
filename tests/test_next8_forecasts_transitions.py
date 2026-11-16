"""End-to-end auxiliary models: observable prefixes, support and fair controls."""
import src.next8
import numpy as np
import pytest
import joblib
from tests.test_next8_movement_parents import feature_fixture
from src.next8.learning import fit_ridge
from src.next8.movement import mechanical_design,movement_target
from src.next8.forecasting import (predict_cross_state,fit_cross_state_forecasts,
    fit_full_comparators,forecast_designs,validate_graph_scope,predict_bundle)
from src.next8.transitions import (FactorizedTransition,transition_inputs,
    transition_tabular_design,random_parent_map)
from src.next8.common import read_json


def graph_for(scope,edges=()):
    return {'scope':scope.provenance(),'edges':[{'source':p,'destination':j,
        'horizons':[{'horizon':h,'supported':True} for h in (1,4,16)]} for p,j in edges]}


def test_current_state_forecast_excludes_current_F_and_future_suffix():
    c,scope=feature_fixture();tr=np.arange(128*48);tr=c.forecast_rows(tr,1)
    design=mechanical_design(c,tr,0,1,[105])
    model=fit_ridge(design,movement_target(c,tr,0,1)['delta'],scope)
    rows=np.arange(128*48,128*48+20)
    before=predict_cross_state(c,rows,0,[105],model)
    # At t=19 its state must already have been forecast before F[19] is seen.
    c.values[rows[-1]:129*48]*=-7
    after=predict_cross_state(c,rows,0,[105],model)
    for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
    assert before[0][0]==3 and np.isnan(before[1][0])


def test_forecast_boundary_abstention_uses_H_only():
    c,_=feature_fixture();rows=np.array([48,49,50])
    class Boundary:
        def predict(self,d):return .5-d.values[:,0]
    state,pred=predict_cross_state(c,rows,0,[105],Boundary())
    np.testing.assert_array_equal(state,[3,3,3])
    np.testing.assert_allclose(pred[1:],.5,atol=1e-6)


def test_parent_graph_selection_scope_must_exclude_prediction_groups():
    c,scope=feature_fixture();g=graph_for(scope,[(105,0)])
    validate_graph_scope(g,scope)
    g['scope']={**g['scope'],'training_ids_sha256':'full_dataset'}
    with pytest.raises(ValueError,match='selection'):validate_graph_scope(g,scope)


def test_D4_only_adds_selected_variable_parent():
    c,_=feature_fixture();rows=np.arange(30)
    a=forecast_designs(c,rows,0,1,[],[105],[])
    assert a['D4'].names==a['D3'].names
    assert a['D5_CROSS'].names!=a['D5'].names
    b=forecast_designs(c,rows,0,1,[],[105],[105])
    assert b['D4'].names!=b['D3'].names


def test_empty_graph_returns_unknown_without_fit(tmp_path,monkeypatch):
    import src.next8.forecasting as f
    c,scope=feature_fixture()
    def fail(*args,**kwargs):raise AssertionError('No model should fit for an empty graph')
    monkeypatch.setattr(f,'fit_ridge',fail)
    done=fit_cross_state_forecasts(c,scope,graph_for(scope),tmp_path)
    with np.load(tmp_path/'STATE_PREDICTIONS.npz') as p:
        assert (p['state']==3).all() and np.isnan(p['predicted_feature']).all()
    assert all(r['status'].startswith('NO_H1') for r in done['models'])
    assert fit_cross_state_forecasts(c,scope,graph_for(scope),tmp_path)==done


def test_real_auxiliary_pipeline_does_not_read_break_labels(tmp_path,monkeypatch):
    import src.next8.forecasting as f
    monkeypatch.setattr(f,'DETAILED',(0,));monkeypatch.setattr(f,'HORIZONS',(1,))
    c,scope=feature_fixture()
    class ForbiddenLabels:
        def __getitem__(self,key):raise AssertionError('Auxiliary pipeline read break labels')
    c.meta['target']=ForbiddenLabels()
    selection={'scope':scope.provenance(),'stable_features':[],'variable_features':[]}
    fit_full_comparators(c,scope,selection,graph_for(scope),tmp_path)
    p=tmp_path/'F000_h1';done=read_json(p/'DONE.json')
    assert done['method_aliases']['D3']=='D2' and done['method_aliases']['D4']=='D2'
    assert done['method_aliases']['D5_CROSS']=='D5'
    with np.load(p/'PREDICTIONS.npz') as values:
        assert all(np.isfinite(values[k]).all() for k in values.files)
        for k in ('D0_state_majority','D0_state_persistence','D1_state','T2_state'):
            np.testing.assert_allclose(values[k].sum(axis=1),1.,atol=1e-6)
        assert set(np.unique(values['dataset_id']))==set(scope.validation_ids)
    tree=joblib.load(p/'T2_state.joblib')
    assert 'F000_current_H_state_3' in tree.names
    assert done['validation_future_targets_decoded_for_scoring'] is False


def test_transition_support_is_distinct_series_not_rows():
    c,scope=feature_fixture();ids=np.repeat(np.arange(20),100)
    own=np.zeros(len(ids),dtype=np.int8);other=np.zeros((len(ids),1),dtype=np.int8)
    target=np.zeros(len(ids),dtype=np.int8)
    model=FactorizedTransition.fit(own,other,target,ids,[105],scope)
    prediction,evidence=model.predict(np.array([0,3]),np.array([[0],[0]]))
    assert model.own_support[0]==20 and model.pair_support[0][0,0]==20
    assert not evidence['own_context_supported'].any()
    np.testing.assert_array_equal(prediction[1],[0,0,0,1])
    assert evidence['active_parent_factors'].sum()==0


def test_transition_unknown_parent_preserves_own_baseline():
    c,scope=feature_fixture();tr=np.arange(128*48);tr=c.forecast_rows(tr,1)
    own,other=transition_inputs(c,tr,0,[105]);target=movement_target(c,tr,0,1)['state']
    model=FactorizedTransition.fit(own,other,target,c.meta['dataset_id'][tr],[105],scope)
    parent=np.full((3,1),3,dtype=np.int8);state=np.arange(3)
    base,_=model.predict(state,parent,use_parents=False)
    full,evidence=model.predict(state,parent)
    np.testing.assert_array_equal(base,full)
    assert evidence['active_parent_factors'].sum()==0


def test_tree_receives_exact_transition_categories():
    c,_=feature_fixture();rows=np.arange(48,60)
    own,parent=transition_inputs(c,rows,0,[105])
    design=transition_tabular_design(c,rows,0,[105],mechanical_design(c,rows,0,1,[105]))
    for s in range(4):
        np.testing.assert_array_equal(design.values[:,design.names.index(f'F000_current_H_state_{s}')],own==s)
        np.testing.assert_array_equal(design.values[:,design.names.index(f'F105_previous_H_state_{s}')],parent[:,0]==s)


def test_random_graph_preserves_degree_and_is_deterministic():
    _,scope=feature_fixture()
    graph=graph_for(scope,[(105,0),(93,6),(81,11),(77,15)])
    first,receipt=random_parent_map(graph,1)
    assert (first,receipt)==random_parent_map(graph,1)
    assert receipt['same_edge_count_and_degrees']
    assert len(receipt['random_edges'])==4
    empty,none=random_parent_map(graph_for(scope),4)
    assert not any(empty.values()) and none['identical_graph_no_informative_random_contrast']
