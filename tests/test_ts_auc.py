import numpy as np
import pandas as pd
import pytest
from src.scoring.ts_auc import ts_auc, score_frames

def test_hard_reference_scores():
    y = np.array([0,1,0,1,0,1])
    t = np.array([0,0,1,1,2,2])
    assert ts_auc(y,np.full(6,.5),t) == .5
    assert ts_auc(y,y,t) == 1
    assert ts_auc(y,1-y,t) == 0
    assert ts_auc(np.zeros(3),np.arange(3)/3,np.arange(3)) == .5
    assert ts_auc([],[],[]) == .5

def test_pair_weighting_and_ties():
    assert ts_auc([0,1,0,0,1],[0,1,1,1,0],[0,0,1,1,1]) == pytest.approx(1/3)
    assert ts_auc([0,1,1],[.1,.1,.2],[0,0,0]) == .75

@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_nonfinite_rejected(bad):
    with pytest.raises(ValueError):
        ts_auc([0,1],[0,bad],[0,0])

def test_frame_alignment_and_shuffle():
    index = pd.MultiIndex.from_tuples([(1,10),(1,11),(2,20),(2,21),(2,22)],names=['id','time'])
    target = pd.DataFrame({'target':[0,1,0,0,1]},index=index)
    pred = pd.DataFrame({'prediction':[.1,.9,.2,.3,.7]},index=index)
    expected = score_frames(pred,target)
    assert score_frames(pred.sample(frac=1,random_state=3),target.sample(frac=1,random_state=4)) == expected
    with pytest.raises(ValueError):
        score_frames(pred.iloc[:-1],target)
    with pytest.raises(ValueError):
        score_frames(pd.concat([pred,pred.iloc[:1]]),target)

def test_invalid_targets_and_times():
    for y,p,t in [([0,2],[0,1],[0,0]),([0,1],[0,1],[0,-1]),([0,1],[0,1],[0,.5]),([0],[0,1],[0])]:
        with pytest.raises(ValueError):
            ts_auc(y,p,t)
