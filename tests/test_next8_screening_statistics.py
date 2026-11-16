"""Independent numerical checks for screening uncertainty and cache causality."""
import numpy as np
import pytest
import src.next8
from src.next8.statistics import rank_influence,holm,cluster_difference
from src.scoring.ts_auc import ts_auc
from src.next8.cache import FeatureCache
from src.next8.features import N_FEATURES
from src.next8.screening import common_range


def brute_auc(y,p,t,ids,weights):
    wins=pairs=0.
    for a in range(len(y)):
        for b in range(len(y)):
            if y[a]==1 and y[b]==0 and t[a]==t[b]:
                w=weights[ids[a]]*weights[ids[b]]
                pairs+=w;wins+=w*(float(p[a]>p[b])+.5*float(p[a]==p[b]))
    return wins/pairs


@pytest.mark.parametrize('seed',[0,1,7])
def test_rank_influence_matches_pair_metric_and_series_weight_derivative(seed):
    rng=np.random.default_rng(seed)
    ids=np.repeat(np.arange(8),5);t=np.tile(np.arange(5),8)
    y=rng.integers(0,2,len(ids));p=rng.integers(-2,3,len(ids)).astype(float)
    r,keys,influence=rank_influence(y,p,t,ids)
    assert np.isclose(r['TS_AUC'],ts_auc(y,p,t),atol=1e-14)
    assert abs(influence.sum())<1e-14
    for k,i in enumerate(keys):
        a=np.ones(8);b=np.ones(8);a[i]+=.0001;b[i]-=.0001
        derivative=(brute_auc(y,p,t,ids,a)-brute_auc(y,p,t,ids,b))/.0002
        assert np.isclose(derivative,influence[k],atol=1e-8)


def test_time_pooling_negative_control():
    # Time perfectly predicts y marginally, but a time-only score cannot rank
    # positive versus negative series AT THE SAME observation time.
    ids=np.repeat(np.arange(100),2);t=np.tile([0,1],100)
    y=((ids%10)<np.where(t==0,2,8)).astype(int)
    result,_,_=rank_influence(y,t.astype(float),t,ids)
    assert result['TS_AUC']==.5
    assert np.corrcoef(t,y)[0,1]>.5


def test_shared_series_duplicate_rows_not_independent_precision():
    ids=np.repeat(np.arange(100),4);t=np.tile(np.arange(4),100)
    rng=np.random.default_rng(2);y=rng.integers(0,2,len(ids));p=rng.normal(size=len(ids))
    r,_,_=rank_influence(y,p,t,ids)
    duplicate,_,_=rank_influence(np.r_[y,y],np.r_[p,p],np.r_[t,t+4],np.r_[ids,ids])
    assert np.isclose(r['TS_AUC'],duplicate['TS_AUC'])
    assert np.isclose(r['cluster_se'],duplicate['cluster_se'])


def test_paired_series_bootstrap_zero_difference_exact():
    ids=np.repeat(np.arange(80),3);loss=np.linspace(0,1,len(ids))
    result=cluster_difference(ids,loss,loss)
    assert result['gain']==result['ci_low']==result['ci_high']==0
    assert result['positive_resample_fraction']==0


def test_holm_monotonic_and_input_order_preserved():
    result=holm([.04,.001,.02])
    np.testing.assert_allclose(result,[.04,.003,.04])


def fake_cache():
    c=FeatureCache.__new__(FeatureCache)
    c.values=np.arange(66*N_FEATURES,dtype=np.float32).reshape(66,N_FEATURES)
    c.meta={'dataset_id':np.repeat([101,202],33),'time_online':np.tile(np.arange(33),2)}
    c.series_index=np.repeat(np.arange(2),33)
    c.series={'seed_values':-np.arange(2*16*N_FEATURES,dtype=np.float32).reshape(2,16,N_FEATURES)-1000}
    return c


@pytest.mark.parametrize('lag',[1,4,16])
def test_cache_lag_uses_only_same_series_or_its_own_H(lag):
    c=fake_cache();rows=np.array([0,1,15,16,32,33,34,48,49,65]);cols=np.array([0,77,105,152])
    values=c.lag(rows,cols,lag)
    for k,row in enumerate(rows):
        sid=c.series_index[row];t=c.meta['time_online'][row]
        expected=c.values[row-lag,cols] if t>=lag else c.series['seed_values'][sid,16+t-lag,cols]
        np.testing.assert_array_equal(values[k],expected)
    # A different preceding series must not change second-series H fallback.
    old=c.lag(np.array([33]),cols,lag);c.values[:33]+=9999
    np.testing.assert_array_equal(old,c.lag(np.array([33]),cols,lag))


def test_forecast_availability_excludes_series_end_and_cutoff():
    c=fake_cache();rows=np.arange(66)
    allowed=c.forecast_rows(rows,16,cutoff=20)
    assert allowed.tolist()==list(range(5))+list(range(33,38))


def test_common_range_never_reestimated_on_validation():
    rng=np.random.default_rng(4);state=np.tile(np.arange(3),100)
    f=rng.normal(size=300)
    domain=common_range(f,state)
    assert domain is not None
    validation=np.full(100,999.)
    assert not np.any((validation>=domain[0])&(validation<=domain[1]))
    assert domain==common_range(f,state)


def test_disjoint_feature_ranges_insufficient_instead_of_reversal():
    state=np.repeat(np.arange(3),100)
    f=np.concatenate([np.linspace(-12,-8,100),np.linspace(-1,1,100),np.linspace(8,12,100)])
    assert common_range(f,state) is None


def test_end_to_end_conditional_rule_fit_excludes_validation_labels(tmp_path):
    from src.next8.screening import conditional_models
    from src.next8.data import FitScope
    from src.next8.common import read_json,digest
    rng=np.random.default_rng(8101604)
    c=FeatureCache.__new__(FeatureCache)
    nseries=240;length=32;n=nseries*length
    c.values=rng.normal(size=(n,N_FEATURES)).astype(np.float32)
    ids=np.repeat(np.arange(nseries),length);t=np.tile(np.arange(length),nseries)
    c.series_index=ids.copy()
    c.series={'seed_values':rng.normal(size=(nseries,16,N_FEATURES)).astype(np.float32),
              'low':np.full((nseries,N_FEATURES),-.5),'high':np.full((nseries,N_FEATURES),.5),
              'valid':np.ones((nseries,N_FEATURES),dtype=bool),'H_length':np.full(nseries,2048),
              'context':rng.normal(size=(nseries,6)).astype(np.float32)}
    c.meta={'dataset_id':ids,'time_online':t}
    previous=c.states(np.arange(n),[0],previous=True)[:,0]
    logit=np.where(previous==0,3.,-3.)*c.values[:,0]
    c.meta['target']=(rng.random(n)<1/(1+np.exp(-logit))).astype(np.uint8)
    scope=FitScope(frozenset(range(160)),frozenset(range(160,240)),frozenset(range(900,910)),'conditional_fixture')
    tr=np.flatnonzero(ids<160);va=np.flatnonzero(ids>=160)
    first=conditional_models(c,0,scope,tr,va,tmp_path/'first')
    assert first['status']=='COMPLETE'
    c.meta['target'][va]=1-c.meta['target'][va]
    second=conditional_models(c,0,scope,tr,va,tmp_path/'second')
    assert second['status']=='COMPLETE'
    assert first['domain']==second['domain']
    for method in first['models']:
        assert first['models'][method]['sha256']==second['models'][method]['sha256']
    assert first['opposing_train_fixed_derivatives_on_common_range']
    assert first['comparisons']['O1_STATE_MAIN']['gain']>0
    assert second['comparisons']['O1_STATE_MAIN']['gain']<0


def test_fold_sign_changes_do_not_certify_inconsistent_state_rule(monkeypatch):
    import copy
    import src.next8.screening as screening
    from src.next8.data import FitScope
    monkeypatch.setattr(screening,'NAMES',('fixture',))
    parent=FitScope(frozenset(range(10)),frozenset([10]),frozenset([11]),'fixture')
    conditional={'status':'COMPLETE','converged':True,'opposing_train_fixed_derivatives_on_common_range':True,
        'comparisons':{k:{'gain':.01,'ci_low':.005} for k in ('O1_STATE_MAIN','STATE_ONLY')},
        'domain':[-1,1],'derivative_weights_raw_design':{'F_linear':0.,'own_state_0_times_F':1.,'own_state_2_times_F':-1.},
        'heldout_domains':[{'domain_type':kind,'domain':str(k),'gain':.01} for kind in ('time','H_length') for k in range(2)]}
    row={'direction':1,'invariant_direction_supported':False,'fixed_nonlinear_supported':False,'all_converged':True,
         'scores':{'O1':{'TS_AUC':.6}},'conditional':conditional}
    panels=[[copy.deepcopy(row)] for _ in range(3)]
    panels[2][0]['conditional']={'status':'INSUFFICIENT_SUPPORT'}
    correction=[[{'feature':0,'contrast_index':k,'Holm_p':.001} for k in range(3)] for _ in range(3)]
    consistent=screening.summarize_screen(panels,correction,parent)
    assert consistent['classifications'][0]['classification']=='CONDITIONAL_VARIABLE_CANDIDATE'
    panels[1][0]['conditional']['derivative_weights_raw_design']={'F_linear':0.,'own_state_0_times_F':-1.,'own_state_2_times_F':1.}
    inconsistent=screening.summarize_screen(panels,correction,parent)
    assert inconsistent['classifications'][0]['classification']=='UNRESOLVED'
    assert inconsistent['variable_features']==[]
