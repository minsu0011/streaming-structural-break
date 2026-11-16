import src.next8
import numpy as np
import pytest
from src.next8.state_observability_v2 import known_prediction_issued, paired_issued_loss


def test_unissued_unknown_is_missing_not_zero_accuracy_or_infinite_loss():
    groups=np.arange(128)
    target=np.zeros(128,dtype=np.int8)
    reference=np.tile([.8,.1,.1,0.],(128,1))
    absent=np.tile([0.,0.,0.,1.],(128,1))
    r,keys,gain=paired_issued_loss(groups,target,reference,absent,np.ones(128,dtype=bool))
    assert r['status']=='INSUFFICIENT_SUPPORT' and r['gain'] is None
    assert r['paired_issued_rows']==0 and r['row_paired_issued_coverage']==0.
    assert len(keys)==len(gain)==0
    # A valid soft physical-state forecast is still issued even if a stricter
    # hard decision rule abstains at max probability < .60 or margin < .15.
    assert known_prediction_issued(np.asarray([[.4,.3,.3,0.],[0.,0.,0.,1.]])).tolist()==[True,False]


def test_paired_loss_requires_matched_issued_rows_and_reports_conditional_coverage():
    groups=np.arange(256);target=np.zeros(256,dtype=np.int8)
    reference=np.tile([.8,.1,.1,0.],(256,1))
    candidate=np.tile([.9,.05,.05,0.],(256,1));candidate[:128]=[0.,0.,0.,1.]
    r,keys,gain=paired_issued_loss(groups,target,reference,candidate,np.ones(256,dtype=bool))
    assert r['status']=='SUPPORTED' and r['paired_issued_rows']==128
    assert r['row_paired_issued_coverage']==.5
    np.testing.assert_array_equal(keys,np.arange(128,256))
    np.testing.assert_allclose(gain,np.log(.9/.8))
    with pytest.raises(ValueError):known_prediction_issued(np.asarray([[0.,0.,0.,0.]]))


def test_actual_frozen_no_parent_pipeline_does_not_report_a_T3_error(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tests.test_next8_movement_parents import feature_fixture
    from tests.test_next8_forecasts_transitions import graph_for
    from src.next8.common import digest
    import src.next8.forecasting as f
    import src.next8.transitions as t
    import src.next8.state_observability_v2 as e
    from src.next8.graph_states import fit_graph_states
    for module in (f,t):
        monkeypatch.setattr(module,'DETAILED',(0,));monkeypatch.setattr(module,'HORIZONS',(1,))
    monkeypatch.setattr(e,'BASE',tmp_path)
    cache,scope=feature_fixture();graph=graph_for(scope)
    cache.data=SimpleNamespace(guard=SimpleNamespace(split=SimpleNamespace(groups={s:s for s in range(256)})))
    class NoLabels:
        def __getitem__(self,key):raise AssertionError('State audit read break labels')
    cache.meta['target']=NoLabels()
    selection={'scope':scope.provenance(),'stable_features':[],'variable_features':[]}
    f.fit_full_comparators(cache,scope,selection,graph,tmp_path/'forecasts'/scope.name)
    t.fit_transition_comparators(cache,scope,graph,tmp_path/'transitions'/scope.name)
    fit_graph_states(cache,scope,graph,tmp_path/'graph_states'/scope.name)
    from src.next8.features import DETAILED
    monkeypatch.setattr(f,'DETAILED',DETAILED)
    f.fit_cross_state_forecasts(cache,scope,graph,tmp_path/'state_forecasts'/scope.name)
    release={'files_sha256':{p.relative_to(tmp_path).as_posix():digest(p) for p in tmp_path.rglob('*') if p.is_file()}}
    result,arrays=e.evaluate_state_feature(cache,scope,0,release)
    records=[r for r in result['metrics'] if r['domain']=='ALL' and r['method']=='T3_ACTUAL_CROSS_DISTRIBUTION']
    assert records and all(r['status']=='INSUFFICIENT_SUPPORT' and r['logloss'] is None for r in records)
    assert all(r['status']=='INSUFFICIENT_SUPPORT' and r['gain'] is None for r in result['comparisons'])
    assert all(len(v)==0 for v in arrays.values())
    assert result['break_labels_used'] is False
