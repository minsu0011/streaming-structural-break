"""Fixed nonlinear kinks and own-state mixture cannot create certification."""
import src.next8
import numpy as np
from src.next8.direction_domains import directional_derivatives,training_cases


def test_U_kink_has_no_single_signed_direction():
    w={'F_positive':2.,'F_negative':2.,'own_state_0_times_F':1.,'own_state_2_times_F':-1.}
    for state in (0,2):
        result=directional_derivatives(w,0.,state)
        assert result['direction']==0 and result['left_derivative']<0<result['right_derivative']


def test_graph_direction_holds_own_condition_fixed():
    w={'F_linear':.5,'own_state_0_times_F':3.,'cross_state_0_times_F':1.,'cross_state_2_times_F':-2.}
    assert directional_derivatives(w,1.,0,cross=True,own_stratum=0)['direction']==1
    assert directional_derivatives(w,1.,2,cross=True,own_stratum=0)['direction']==1
    assert directional_derivatives(w,1.,2,cross=True,own_stratum=1)['direction']==-1


def test_disjoint_F_ranges_do_not_create_a_common_direction_test():
    state=np.repeat(np.arange(3),128);f=state*10+np.tile(np.linspace(0,1,128),3)
    refit={'status':'COMPLETE','domain':[0.,21.],'derivative_weights_raw_design':{}}
    result=training_cases(f,state,state,np.arange(len(f))%2,np.arange(len(f)),refit)
    assert result[0]['status']=='INSUFFICIENT_SUPPORT'
