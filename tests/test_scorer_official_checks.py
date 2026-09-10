import contextlib
import io
import numpy as np
import pandas as pd
import pytest
from scripts.verify_scorer import load_official,official_score
from src.scoring.ts_auc import score_frames

@pytest.mark.parametrize('case',['nan','inf','duplicate','missing','integer_prediction','wrong_column'])
def test_official_check_and_local_both_reject(case):
    official=load_official()
    index=pd.MultiIndex.from_tuples([(0,10),(1,20)],names=['id','time'])
    p=pd.DataFrame({'prediction':[.1,.9]},index=index);y=pd.DataFrame({'target':[0,1]},index=index)
    if case=='nan':p.iloc[0,0]=np.nan
    elif case=='inf':p.iloc[0,0]=np.inf
    elif case=='duplicate':p=pd.concat([p,p.iloc[:1]])
    elif case=='missing':p=p.iloc[:1]
    elif case=='integer_prediction':p=p.astype(int)
    elif case=='wrong_column':p=p.rename(columns={'prediction':'bad'})
    official._load_prediction=lambda _:p
    official._load_y_test=lambda _:y
    with contextlib.redirect_stdout(io.StringIO()),pytest.raises(official.ParticipantVisibleError):official.check('unused','unused')
    with pytest.raises(ValueError):score_frames(p,y)

def test_no_pairs_official_parity():
    index=pd.MultiIndex.from_tuples([(0,10),(1,20)],names=['id','time'])
    p=pd.DataFrame({'prediction':[.1,.9]},index=index);y=pd.DataFrame({'target':[1,1]},index=index)
    assert official_score(load_official(),p,y)==score_frames(p,y)==.5
