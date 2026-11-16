"""Additional frozen origins remain prior to the current observation."""
import src.next8
import numpy as np
from tests.test_next8_movement_parents import feature_fixture
from src.next8.transitions import FactorizedTransition,transition_inputs
from src.next8.movement import movement_target
from src.next8.state_observability import replay_current_comparators


def test_current_state_comparator_does_not_read_current_F_or_break_y():
    c,scope=feature_fixture();train=c.forecast_rows(np.arange(128*48),1)
    own,parent=transition_inputs(c,train,0,[])
    model=FactorizedTransition.fit(own,parent,movement_target(c,train,0,1)['state'],c.meta['dataset_id'][train],[],scope)
    class NoLabels:
        def __getitem__(self,key):raise AssertionError('State observable replay read break labels')
    c.meta['target']=NoLabels();rows=np.array([128*48+1,128*48+2])
    first=replay_current_comparators(c,rows,0,model,None,[],.8,np.full(3,1/3))
    c.values[rows[-1]:129*48]*=-12
    second=replay_current_comparators(c,rows,0,model,None,[],.8,np.full(3,1/3))
    for name in first:np.testing.assert_array_equal(first[name],second[name])
    assert np.all(first['T2_SAME_INPUT_TREE'][:,3]==1)


def test_actual_saved_state_pipeline_distinguishes_abstention(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tests.test_next8_forecasts_transitions import graph_for
    from src.next8.common import digest
    import src.next8.forecasting as f
    import src.next8.transitions as t
    import src.next8.state_observability as e
    from src.next8.graph_states import fit_graph_states
    for module in (f,t):monkeypatch.setattr(module,'DETAILED',(0,));monkeypatch.setattr(module,'HORIZONS',(1,))
    monkeypatch.setattr(e,'BASE',tmp_path)
    c,scope=feature_fixture();graph=graph_for(scope)
    c.data=SimpleNamespace(guard=SimpleNamespace(split=SimpleNamespace(groups={s:s for s in range(256)})))
    class NoLabels:
        def __getitem__(self,key):raise AssertionError('Observability pipeline read break labels')
    c.meta['target']=NoLabels();selection={'scope':scope.provenance(),'stable_features':[],'variable_features':[]}
    f.fit_full_comparators(c,scope,selection,graph,tmp_path/'forecasts'/scope.name)
    t.fit_transition_comparators(c,scope,graph,tmp_path/'transitions'/scope.name)
    fit_graph_states(c,scope,graph,tmp_path/'graph_states'/scope.name)
    from src.next8.features import DETAILED
    monkeypatch.setattr(f,'DETAILED',DETAILED)
    f.fit_cross_state_forecasts(c,scope,graph,tmp_path/'state_forecasts'/scope.name)
    release={'files_sha256':{p.relative_to(tmp_path).as_posix():digest(p) for p in tmp_path.rglob('*') if p.is_file()}}
    result,arrays=e.evaluate_state_feature(c,scope,0,release)
    records=[r for r in result['metrics'] if r['domain']=='ALL' and r['method']=='T3_CONFIDENT_CROSS_CONDITION']
    assert records[0]['issued_coverage']==0 and records[0]['selective_accuracy_when_issued'] is None
    assert result['ridge_current_F_error']['status']=='NO_ISSUED_RIDGE_FORECAST'
    assert result['break_labels_used'] is False and 'T1_OWN_TABLE_GAIN' in arrays
