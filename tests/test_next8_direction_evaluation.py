"""Shared original-group uncertainty tests both opposing directions directly."""
import src.next8
import numpy as np
from src.next8.direction_evaluation import state_rank_bootstrap,pair_rank_evidence


def test_opposite_conditional_directions_and_paired_difference():
    group=np.repeat(np.arange(128),6);t=np.tile(np.arange(6),128);state=t%3
    y=(group%2).astype(np.uint8);f=np.where(state==0,y,np.where(state==1,1-y,.5)).astype(float)
    records,samples,receipt=state_rank_bootstrap(y,f,t,group,state,partition='direction_fixture')
    pair={'state_pair':[0,1],'directions':[1,-1],'eligible_for_orientation_replication':True}
    result=pair_rank_evidence(records,samples,pair)
    assert result['both_claimed_directions_reproduced'] and result['difference_ci_low']==1
    assert receipt['same_draws_across_states']
    pair['directions']=[1,1];pair['eligible_for_orientation_replication']=False
    assert pair_rank_evidence(records,samples,pair)['both_claimed_directions_reproduced'] is False


def test_sparse_state_does_not_receive_a_certificate():
    group=np.arange(80);state=np.r_[np.zeros(78),[1,2]];y=group%2;f=y.astype(float)
    records,samples,_=state_rank_bootstrap(y,f,np.zeros(80),group,state,partition='sparse_fixture')
    result=pair_rank_evidence(records,samples,{'state_pair':[0,1],'directions':[1,-1],'eligible_for_orientation_replication':True})
    assert result['status']=='INSUFFICIENT_SUPPORT' and not result['both_claimed_directions_reproduced']
