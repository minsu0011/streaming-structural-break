import numpy as np
from src.next.synthetic_variance_stress import REGIMES,SCENARIOS,generate_series


def policy():
    return {'seed':202609099,'regimes':list(REGIMES),'scenarios':list(SCENARIOS),
        'burn_points':512,'historical_points':2048,'online_points':1024,'break_times':[128,256,512]}


def test_scenarios_have_same_h_and_identical_prebreak_prefix():
    for regime in range(5):
        h,online,tau,polarity = generate_series(policy(),1,regime,3)
        assert len(h)==2048 and polarity==-1 and tau==128
        for name,values in online.items():
            np.testing.assert_array_equal(values[:tau],online['NO_CHANGE'][:tau])
            if name!='NO_CHANGE':
                assert not np.array_equal(values[tau:],online['NO_CHANGE'][tau:])
        h2,o2,tau2,p2 = generate_series(policy(),1,regime,3)
        np.testing.assert_array_equal(h,h2)
        for name in online:
            np.testing.assert_array_equal(online[name],o2[name])


def test_calibration_randomness_is_separate_and_mean_scale_are_exact():
    h,o,tau,polarity = generate_series(policy(),1,0,0)
    hc,oc,_,_ = generate_series(policy(),0,0,0)
    assert not np.array_equal(h,hc)
    assert set(oc)=={'NO_CHANGE'}
    np.testing.assert_allclose(o['MEAN_SHIFT'][tau:],o['NO_CHANGE'][tau:]+.5,rtol=1e-6,atol=3e-7)
    np.testing.assert_allclose(o['SCALE_SHIFT'][tau:],1.5*o['NO_CHANGE'][tau:],rtol=1e-6,atol=3e-7)
