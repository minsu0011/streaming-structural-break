"""Exact bootstrap pair counts, ties and dependency-group resampling."""
import src.next8
import numpy as np
from src.next8.paired_bootstrap import (PreparedRank,group_resample_weights,
    evaluate_partition,summarize_paired)
from src.scoring.ts_auc import ts_auc


def test_weighted_auc_equals_literal_whole_group_duplication():
    rng=np.random.default_rng(883);ids=np.repeat(np.arange(8),7);t=np.tile(np.arange(7),8)
    y=rng.integers(0,2,len(ids));p=rng.integers(0,4,len(ids))/4.
    prepared=PreparedRank.prepare(y,p,t,ids)
    w,receipt=group_resample_weights(prepared.groups,draws=32,partition='test')
    actual,pairs=prepared.evaluate(w)
    for k,row in enumerate(w):
        at=np.repeat(np.arange(len(ids)),row[ids])
        np.testing.assert_allclose(actual[k],ts_auc(y[at],p[at],t[at]),rtol=0,atol=1e-15)
        expected=sum(np.sum(y[at][t[at]==j])*np.sum(1-y[at][t[at]==j]) for j in range(7))
        assert pairs[k]==expected
    assert receipt['unit'].startswith('Whole original dependency group')


def test_bootstrap_same_weights_for_all_models_and_direction():
    rng=np.random.default_rng(81);ids=np.repeat(np.arange(80),10);t=np.tile(np.arange(10),80)
    y=rng.integers(0,2,len(ids));p=np.clip(.6*y+rng.uniform(0,.4,len(ids)),0,1)
    points,draws,receipt=evaluate_partition(y,{'reference':1-p,'candidate':p},t,ids,partition='shared',draws=100)
    np.testing.assert_allclose(draws['reference']+draws['candidate'],1.,rtol=0,atol=1e-15)
    result=summarize_paired([points],[draws],'reference','candidate')
    assert result['ci_low']>0 and result['positive_folds']==1
    assert receipt['same_weights_for_every_method'] and result['positive_resample_fraction_is_posterior'] is False


def test_dependency_groups_move_together():
    # Two different series share each original dependency group.
    group=np.repeat(np.arange(4),6);t=np.tile([0,1,2,0,1,2],4)
    y=np.tile([0,0,1,1,1,1],4);p=np.linspace(0,1,len(y))
    prepared=PreparedRank.prepare(y,p,t,group);w,_=group_resample_weights(prepared.groups,draws=10)
    assert w.shape==(10,4) and np.all(w.sum(axis=1)==4)
    observed,_=prepared.evaluate(w)
    for k,row in enumerate(w):
        at=np.repeat(np.arange(len(group)),row[group])
        assert abs(observed[k]-ts_auc(y[at],p[at],t[at]))<1e-12


def test_zero_pair_resamples_are_explicitly_unsupported():
    ids=np.repeat(np.arange(4),3);t=np.tile(np.arange(3),4)
    points,draws,receipt=evaluate_partition(np.zeros(len(ids)),{'a':np.zeros(len(ids)),'b':np.ones(len(ids))},t,ids,partition='none',draws=10)
    assert points['a']['TS_AUC'] is None and receipt['zero_pair_draws']==10
    assert np.isnan(draws['a']).all()
    assert summarize_paired([points],[draws],'a','b')['status']=='INSUFFICIENT_SUPPORT'
