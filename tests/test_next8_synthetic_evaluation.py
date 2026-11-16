"""Synthetic response summaries preserve matched variants and fixed windows."""
import src.next8
from types import SimpleNamespace
import numpy as np
from tests.test_next8_streaming_runtime import actual_cache,fitted_fixture
from src.next8.synthetic_evaluation import summarize_scores,predict_block_heads,SCORE_METHODS,WINDOWS
from src.next8.streaming_runtime import PreparedJoint


def test_score_windows_are_fixed_and_do_not_use_labels():
    values=np.repeat(np.arange(5)/5,1024)[:,None]
    result=summarize_scores(values)
    assert result.shape==(5,len(WINDOWS),1,4)
    np.testing.assert_allclose(result[:,0,0,0],np.arange(5)/5,atol=1e-12)
    assert result[4,0,0,2]==1 and result[4,0,0,3]==0


def test_synthetic_head_application_uses_frozen_models_and_M6_fallback(actual_cache):
    cache,h,online=actual_cache;heads={}
    for method in ('M1','M2','M3','M4','M5'):
        model,selection,graph,_=fitted_fixture(cache,method)
        heads[method]=PreparedJoint(model,selection,method,{0:graph},resident=False)
    class FixedBaseline:
        def predict(self,x):return np.full(len(x),.3)
    class NoLabels:
        def __getitem__(self,key):raise AssertionError('Synthetic prediction used labels')
    old=cache.meta['target'];cache.meta['target']=NoLabels()
    try:
        matrices=[np.zeros((len(cache.values),1))]*2
        p,cross,aliases=predict_block_heads(cache,heads,[FixedBaseline(),FixedBaseline()],matrices,False)
        np.testing.assert_array_equal(p[:,0],p[:,SCORE_METHODS.index('M6')])
        assert np.isfinite(p).all() and cross.shape==(len(cache.values),12)
    finally:cache.meta['target']=old


def test_synthetic_auxiliary_application_respects_aliases_and_known_H(actual_cache,monkeypatch):
    import src.next8.synthetic_evaluation as se
    from src.next8.data import FitScope
    from src.next8.learning import fit_ridge
    from src.next8.movement import own_design,mechanical_design,movement_target
    cache,_,_=actual_cache
    monkeypatch.setattr(se,'DETAILED',(0,));monkeypatch.setattr(se,'HORIZONS',(1,))
    scope=FitScope(frozenset(map(int,cache.series['dataset_id'])),frozenset({-1}),frozenset({-1}),'synthetic_test')
    rows=cache.forecast_rows(np.flatnonzero(cache.meta['time_online']%8==0),1)
    target=movement_target(cache,rows,0,1)['delta']
    a=fit_ridge(own_design(cache,rows,0,1),target,scope)
    b=fit_ridge(mechanical_design(cache,rows,0,1),target,scope)
    spec={(0,1):{'models':{'D1':a,'D5':b,'D5_CROSS':b},'parents':[],
        'aliases':{'D1':'D1','D5':'D5','D5_CROSS':'D5'}}}
    result,count=se.auxiliary_errors(cache,spec)
    np.testing.assert_array_equal(result[:,:,2],result[:,:,3])
    assert result.shape==(1,1,4,5,len(WINDOWS),2) and count.shape==(1,1,5,len(WINDOWS))
