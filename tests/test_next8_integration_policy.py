import src.next8
import numpy as np
from src.next8.integration import fixed_blend,integration_eligible


def test_fixed_integration_is_not_a_weight_search():
    b=np.array([.1,.9],dtype=np.float32);m=np.array([.9,.1],dtype=np.float32)
    np.testing.assert_array_equal(fixed_blend(b,m),.95*b.astype(float)+.05*m.astype(float))


def test_integration_threshold_does_not_round_up_near_miss():
    r={'status':'SUPPORTED','fold_deltas':[.0002,.0002,-.0001],'mean_delta':.00009999999999,
       'positive_folds':2,'ci_low':.00001}
    assert not integration_eligible(r)
    r['mean_delta']=.0001;assert integration_eligible(r)
    r['ci_low']=0.;assert not integration_eligible(r)


def test_constant_state_interaction_cannot_duplicate_regularized_F():
    from src.next8.joint_models import prune_training_design
    from src.next8.learning import Design
    values=np.column_stack([np.arange(50),np.arange(50),np.ones(50)])
    full=Design(values,('F_linear','constant_state_times_F','state_constant'),np.zeros(50,dtype=int))
    reduced,columns,active=prune_training_design(full)
    assert columns==active==[0] and reduced.names==('F_linear',)
