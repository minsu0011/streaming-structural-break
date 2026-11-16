import src.next9
import numpy as np
from src.next9.inference import compare as original
from src.next9.inference_numeric_recovery import compare

def test_saturated_float32_logloss_finite_and_rank_unchanged():
    y=np.tile([0,1],12);groups=np.repeat(np.arange(12),2);times=np.tile([0,0],12)
    p=np.tile(np.array([0.,1.],np.float32),12)
    result=compare(y,{'O1':p,'STATE_ONLY':np.full(24,.5,np.float32)},times,groups,'saturation',draws=40)
    assert np.isfinite(result['logloss']['O1'])
    assert result['logloss']['O1']<1.1e-8
    assert result['TS_AUC']['O1']==1.

def test_previously_finite_results_exactly_preserved():
    y=np.tile([0,1],12);groups=np.repeat(np.arange(12),2);times=np.zeros(24,int)
    predictions={'O1':np.tile(np.array([.1,.8],np.float32),12),'STATE_ONLY':np.full(24,.5,np.float32)}
    assert compare(y,predictions,times,groups,'finite',draws=40)==original(y,predictions,times,groups,'finite',draws=40)
