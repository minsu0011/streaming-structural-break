"""Frozen forecast readers align aliases and real future targets without y."""
import src.next8
from types import SimpleNamespace
import numpy as np
from tests.test_next8_movement_parents import feature_fixture
from tests.test_next8_forecasts_transitions import graph_for
from src.next8.common import digest


def test_actual_forecast_to_evaluation_pipeline_without_break_labels(tmp_path,monkeypatch):
    import src.next8.forecasting as f
    import src.next8.transitions as t
    import src.next8.forecast_evaluation as e
    for module in (f,t):monkeypatch.setattr(module,'DETAILED',(0,));monkeypatch.setattr(module,'HORIZONS',(1,))
    monkeypatch.setattr(e,'BASE',tmp_path)
    c,scope=feature_fixture()
    c.data=SimpleNamespace(guard=SimpleNamespace(split=SimpleNamespace(groups={s:s for s in range(256)})))
    class NoLabels:
        def __getitem__(self,key):raise AssertionError('Movement evaluation read break labels')
    c.meta['target']=NoLabels();graph=graph_for(scope)
    selection={'scope':scope.provenance(),'stable_features':[],'variable_features':[]}
    f.fit_full_comparators(c,scope,selection,graph,tmp_path/'forecasts'/scope.name)
    t.fit_transition_comparators(c,scope,graph,tmp_path/'transitions'/scope.name)
    release={'files_sha256':{p.relative_to(tmp_path).as_posix():digest(p) for p in tmp_path.rglob('PREDICTIONS.npz')}}
    result,arrays=e.evaluate_forecast(c,scope,0,1,release)
    assert result['no_break_labels_used'] and result['T4_identical_parent_contrast']
    match=next(r for r in result['comparisons'] if r['domain']=='ALL' and r['reference']=='D5_delta')
    assert match['gain']==match['ci_low']==match['ci_high']==0
    assert np.all(arrays[match['group_array_key']+'_gain']==0)
    random=next(r for r in result['comparisons'] if r['reference']=='T4_state')
    assert random['informative_random_contrast'] is False
