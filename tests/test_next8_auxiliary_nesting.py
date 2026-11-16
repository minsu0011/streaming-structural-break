"""Original dependency groups remain intact through necessary auxiliary depth."""
import src.next8
import numpy as np
from src.next8.data import FitScope
from src.next8.nested_auxiliary import subdivide_scope,selected_cross_branch_needed,early_scope,fit_meta_leaf_states
from tests.test_next8_movement_parents import feature_fixture
from tests.test_next8_forecasts_transitions import graph_for


def test_auxiliary_leaf_excludes_complete_prediction_and_ancestor_groups():
    parent=FitScope(frozenset(range(60)),frozenset(range(60,80)),frozenset(range(80,100)),'meta')
    groups={s:s//2 for s in range(100)}
    parts=subdivide_scope(parent,groups)
    assert parts==subdivide_scope(parent,groups)
    assert set.union(*(set(s.validation_ids) for s in parts))==set(parent.training_ids)
    for scope in parts:
        assert scope.training_ids.isdisjoint(scope.validation_ids|parent.validation_ids|parent.outer_excluded_ids)
        assert {groups[s] for s in scope.training_ids}.isdisjoint({groups[s] for s in scope.validation_ids})
        for leaf in subdivide_scope(scope,groups):
            assert leaf.training_ids.isdisjoint(scope.validation_ids|parent.validation_ids|parent.outer_excluded_ids)


def test_unused_cross_branch_does_not_create_leaf_fits(tmp_path,monkeypatch):
    import src.next8.nested_auxiliary as module
    c,scope=feature_fixture()
    monkeypatch.setattr(module,'procedure_scope',lambda *args:scope)
    def fail(*args,**kwargs):raise AssertionError('No unnecessary leaf fits')
    monkeypatch.setattr(module,'selection_splits',fail)
    selection={'scope':scope.provenance(),'variable_features':[]}
    done=fit_meta_leaf_states(c,0,0,selection,graph_for(scope,[(105,0)]),tmp_path)
    assert done['required'] is False and done['parts']==[]


def test_early_scope_and_selected_h1_parent_requirement():
    _,scope=feature_fixture();early=early_scope(scope,True)
    assert early.training_ids==scope.training_ids and early.validation_ids==scope.validation_ids
    assert early.name.endswith('_EARLY_T199')
    graph=graph_for(scope,[(105,0)])
    assert selected_cross_branch_needed({'variable_features':[0]},graph)
    assert not selected_cross_branch_needed({'variable_features':[6]},graph)
    graph['edges'][0]['horizons'][0]['supported']=False
    assert not selected_cross_branch_needed({'variable_features':[0]},graph)
