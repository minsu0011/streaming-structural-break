import numpy as np
import pytest
from src.next2.counterexamples import policy
from src.next2.cross_background_scale import series,FACTORS,BACKGROUNDS


@pytest.mark.parametrize('background',BACKGROUNDS)
def test_factor_does_not_change_history_prebreak_or_control(background):
    lock=policy();h0,o0,t0=series(lock,background,1.5,2,control=True)
    assert t0==-1
    for factor in FACTORS:
        h,o,t=series(lock,background,factor,2)
        assert t==512
        np.testing.assert_array_equal(h,h0)
        np.testing.assert_array_equal(o[:t],o0[:t])
        hc,oc,tc=series(lock,background,factor,2,control=True)
        np.testing.assert_array_equal(hc,h0);np.testing.assert_array_equal(oc,o0);assert tc==-1
        hs,os,ts=series(lock,background,factor,2,horizon=700)
        np.testing.assert_array_equal(hs,h);np.testing.assert_array_equal(os,o[:700]);assert ts==t


def test_gaussian_variance_factor_is_not_standard_deviation_factor():
    lock=policy()
    _,control,_=series(lock,'N0',2.,0,control=True)
    _,changed,t=series(lock,'N0',2.,0)
    np.testing.assert_allclose(changed[t:],control[t:]*np.sqrt(2.),rtol=2e-7,atol=1e-7)
