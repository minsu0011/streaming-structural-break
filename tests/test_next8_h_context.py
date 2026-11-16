"""H state thresholds are fitted on training series only, before current F."""
import numpy as np
import pytest
import src.next8
from src.next8.cache import FeatureCache
from src.next8.data import FitScope
from src.next8.features import N_FEATURES,UNKNOWN
from src.next8.h_context import HStateSpec,HStateCache
from src.next8.screening import conditional_models


def fixture():
    rng=np.random.default_rng(81124);nseries=512;length=32;n=nseries*length
    c=FeatureCache.__new__(FeatureCache)
    c.values=rng.normal(size=(n,N_FEATURES)).astype(np.float32)
    ids=np.repeat(np.arange(nseries),length);c.series_index=ids.copy()
    context=rng.normal(size=(nseries,6)).astype(np.float32)
    context[:,2]=np.tile(np.linspace(-.8,.8,256),2)
    context[:,4]=np.tile(np.linspace(0,.1,256),2)[::-1]
    c.series={'dataset_id':np.arange(nseries),'context':context,
        'valid':np.ones((nseries,N_FEATURES),dtype=bool),'low':np.full((nseries,N_FEATURES),-.5),
        'high':np.full((nseries,N_FEATURES),.5),'seed_values':rng.normal(size=(nseries,16,N_FEATURES)).astype(np.float32),
        'H_length':np.where(np.arange(nseries)%2,512,2048)}
    c.meta={'dataset_id':ids,'time_online':np.tile(np.arange(length),nseries)}
    scope=FitScope(frozenset(range(256)),frozenset(range(256,512)),frozenset(range(900,910)),'H_context_fixture')
    spec=HStateSpec.fit(c,scope,'H_LAG1');proxy=HStateCache(c,spec)
    states=proxy.states(np.arange(n),[0],previous=True)[:,0]
    eta=np.where(states==0,3.,-3.)*c.values[:,0]
    c.meta['target']=(rng.random(n)<1/(1+np.exp(-eta))).astype(np.uint8)
    return c,scope,spec,proxy


def test_H_population_state_boundaries_training_only():
    c,scope,spec,_=fixture()
    c.series['context'][256:,2]=999
    other=HStateSpec.fit(c,scope,'H_LAG1')
    assert vars(spec)==vars(other)
    assert np.all(other.states_from_context(c.series['context'][256:])==UNKNOWN)
    assert other.training['H_series_count']==256


def test_H_condition_known_before_current_F_and_constant_within_series():
    c,scope,spec,proxy=fixture();rows=np.arange(100)
    before=proxy.states(rows,[0],previous=True)
    c.values[:100]=np.linspace(-12,12,100)[:,None]
    after=proxy.states(rows,[0],previous=True)
    np.testing.assert_array_equal(before,after)
    assert len(np.unique(after[:32]))==1


def test_degenerate_H_context_abstains_instead_of_quantile_jitter():
    c,scope,_,_=fixture();c.series['context'][:,4]=0
    spec=HStateSpec.fit(c,scope,'H_TAIL3')
    assert not spec.valid
    assert np.all(spec.states_from_context(c.series['context'])==UNKNOWN)


def test_complete_H_rule_fit_does_not_use_validation_labels(tmp_path):
    c,scope,spec,proxy=fixture()
    ids=c.meta['dataset_id'];tr=np.flatnonzero(ids<256);va=np.flatnonzero(ids>=256)
    first=conditional_models(proxy,0,scope,tr,va,tmp_path/'first')
    assert first['status']=='COMPLETE' and first['opposing_train_fixed_derivatives_on_common_range']
    c.meta['target'][va]=1-c.meta['target'][va]
    second=conditional_models(proxy,0,scope,tr,va,tmp_path/'second')
    assert second['status']=='COMPLETE'
    for method in first['models']:assert first['models'][method]['sha256']==second['models'][method]['sha256']
    assert first['comparisons']['O1_STATE_MAIN']['gain']>0
    assert second['comparisons']['O1_STATE_MAIN']['gain']<0
