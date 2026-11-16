import numpy as np
import pandas as pd
import pytest
from src.next4 import diagnostics as d


def test_fixed_scopes_keep_age_controls_and_direction_support_separate():
    meta=pd.DataFrame({'tau':[-1,2,2,2],'historical_regime':[0,0,1,1],
        'signature':['NO_BREAK_CONTROL','SCALE_DOMINANT','SCALE_DOMINANT','TAIL_DOMINANT']},index=[1,2,2,3])
    y=np.array([0,0,1,1]);t=np.array([10,1,8,8])
    masks=d.evaluation_masks(meta,y,t,{2:'decrease'})
    assert len(masks)==26
    np.testing.assert_array_equal(masks['age_0-4'],[True,True,False,False])
    np.testing.assert_array_equal(masks['age_5-9'],[True,True,True,True])
    np.testing.assert_array_equal(masks['mechanism_observed_scale_decrease'],[True,True,True,False])
    np.testing.assert_array_equal(masks['mechanism_observed_scale_increase'],[True,False,False,False])


def test_identical_tied_scores_have_zero_disagreement():
    p=np.array([.1,.1,.9,.4,.5,.6]);t=np.array([0,0,0,1,1,1]);y=np.array([0,0,1,0,1,1])
    out=d.diversity(p,p,t,y)
    for name in ('pearson','spearman_pooled','within_time_spearman'):assert out[name]==pytest.approx(1.)
    for name in ('mean_absolute_rank_disagreement','mean_absolute_score_disagreement','top1_disagreement','top5_negative_point_disagreement'):
        assert out[name]==0


def test_constant_correlations_are_explicitly_undefined():
    assert d.correlation(np.ones(10),np.arange(10)) is None
    assert d.correlation(np.ones(1),np.ones(1)) is None


def test_direction_reconstructs_only_frozen_scale_magnitude(monkeypatch):
    monkeypatch.setattr(d,'read_json',lambda _: {'mechanism':{'window':16,'minimum_each_side':8}})
    class Data:
        def get(self,sid):
            assert sid==2
            return np.tile([-1.,1.],64),np.r_[np.tile([-1.,1.],8),np.tile([-.5,.5],8)],16
    series=pd.DataFrame({'tau':[16,16],'signature':['SCALE_DOMINANT','TAIL_DOMINANT'],
        'scale_effect':[abs(np.log(.5)),100.]},index=[2,3])
    direction,report=d.scale_directions(Data(),series)
    assert direction=={2:'decrease'} and report['direction_counts']['decrease']==1
    series.loc[2,'scale_effect']=.1
    with pytest.raises(ValueError,match='magnitude'):d.scale_directions(Data(),series)
