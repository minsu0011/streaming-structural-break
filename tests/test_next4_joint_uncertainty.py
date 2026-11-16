import numpy as np
from src.next4.joint_uncertainty import describe_joint


def test_duplicated_candidates_do_not_double_the_joint_threshold():
    rng=np.random.default_rng(19);x=np.r_[.2,.2+rng.normal(0,.1,2000)]
    one=describe_joint(x[:,None]);two=describe_joint(np.c_[x,x])
    np.testing.assert_allclose(one['centered_max_95_critical'],two['centered_max_95_critical'],rtol=0,atol=1e-14)
    np.testing.assert_allclose(two['one_sided_95_joint_lower'],one['one_sided_95_joint_lower'][0])
    assert abs(two['correlation_participation_ratio']-1)<1e-12


def test_joint_order_and_scale_equivariance_without_resampling():
    rng=np.random.default_rng(20);x=rng.normal(size=(2001,4))*.01;x[0]=[.03,.02,.01,0]
    a=describe_joint(x);b=describe_joint(x[:,::-1]*7)
    np.testing.assert_allclose(a['one_sided_95_joint_lower'][::-1]*7,b['one_sided_95_joint_lower'])
    np.testing.assert_allclose(a['empirical_centered_max_tail_fraction'][::-1],b['empirical_centered_max_tail_fraction'])
    np.testing.assert_allclose(a['bootstrap_best_frequency'].sum(),1.)
