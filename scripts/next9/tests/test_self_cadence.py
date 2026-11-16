import src.next9
import numpy as np
import pytest
from src.next9.transitions import SelfLaw
from src.next9.self_cadence import replay_series,VARIANTS


@pytest.mark.parametrize('h_length',[159,512])
def test_precurrent_full_cadence_and_short_history(h_length):
    seed=np.zeros((16,153),dtype=np.float32)
    values=np.zeros((40,153),dtype=np.float32)
    values[:,1]=np.sin(np.arange(40)*.8).astype(np.float32)
    epsilon=np.full((1,153),.02,dtype=np.float32);context=np.zeros(6,dtype=np.float32)
    table=np.tile(np.array([.2,.4,.3,.1]),(4,1))
    laws={1:{v:SelfLaw(v,table.copy(),None,{}) for v in VARIANTS}}
    loss,r=replay_series(seed,values,epsilon,context,h_length,laws,{1:'T0_PRIOR'},'NORMAL')
    assert r['points']==40 and r['save_restore'] and r['forecast_issued_before_current_F']
    assert r['first_surprise_zero'] and r['state_and_momentum_bitwise']
    assert r['maximum_probability_error']==0 and r['maximum_path_error']==0
    assert np.isfinite(loss).all()
    if h_length<160:np.testing.assert_allclose(loss,-np.log(.1))
