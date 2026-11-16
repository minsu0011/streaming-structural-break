import numpy as np
from scipy.special import expit
from src.next4.outlier_attribution import allocate_probability_change,group_name


def test_probability_allocation_closes_margin_to_sigmoid_difference():
    a=np.array([[1,-1,.4],[2,.5,-.7],[-4,2,-.1],[12,8,0.]])
    b=a+np.array([[.2,.3,0],[-.7,.1,0],[.5,1.5,0],[.1,-.2,0]])
    allocation,delta=allocate_probability_change(a,b,expert_weight=.8)
    expected=.8*(expit(b.sum(axis=1))-expit(a.sum(axis=1)))
    np.testing.assert_allclose(delta,expected,rtol=0,atol=1e-15)
    np.testing.assert_allclose(allocation.sum(axis=1),expected,rtol=0,atol=1e-15)


def test_cancelling_and_zero_margin_changes_remain_finite_and_additive():
    a=np.zeros((2,3));b=np.array([[1,-1,0],[0,0,0.]])
    allocation,delta=allocate_probability_change(a,b,expert_weight=.1)
    np.testing.assert_allclose(allocation,[[.025,-.025,0],[0,0,0]],rtol=0,atol=1e-15)
    np.testing.assert_array_equal(delta,[0,0]);np.testing.assert_array_equal(allocation.sum(axis=1),[0,0])


def test_frozen_feature_groups_are_exhaustive_and_do_not_use_observation_labels():
    from src.next4.scale_asymmetry import feature_names as b_names
    from src.next4.sequential import feature_names as d_names
    for candidate,settings in [('B1_SPLIT_ENERGY_BRANCHES',{'version':'separate_energy_branches_v1','kind':'split_energy'}),
            ('B2_LOG_ENERGY_SEMITAIL',{'version':'separate_energy_branches_v1','kind':'log_semitail'})]:
        # Use each module's actual fixed menu, not guessed feature positions.
        names=b_names(settings)
        groups=[group_name(candidate,name) for name in names]
        assert len(groups)==93 and len(set(groups))==7 and groups.count('old_ARCH')==51
    for kind,candidate,expected in [('scale_mixture','D2_DISCRETE_SCALE_MIXTURE',5),('hazard_mixture','D3_HAZARD_MIXTURE',4)]:
        names=d_names({'version':'discrete_likelihood_v1','kind':kind})
        assert len({group_name(candidate,name) for name in names})==expected
