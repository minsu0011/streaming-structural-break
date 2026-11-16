"""A, B and final score improvement are logically distinct claims."""
import src.next8
from src.next8.evidence_synthesis import joint_certificate_check,science_status


def test_good_direction_and_state_accuracy_do_not_certify_different_head():
    orientation={'conditional_orientation_supported':True,'state_estimator_id':'T3_H1','direction_head_sha256':'O3_REFIT'}
    state={'supported':True,'state_estimator_id':'T3_H1'}
    result=joint_certificate_check(orientation,movement_state_path=state,execution_direction_head='DISTINCT_M5',execution_pass=True)
    assert not result['JOINT_VARIABLE_FEATURE_SUPPORTED'] and 'DIRECTION_HEAD_NOT_THE_EXECUTED_CERTIFIED_HEAD' in result['reasons']
    assert joint_certificate_check(orientation,movement_state_path=state,execution_direction_head='O3_REFIT',execution_pass=True)['JOINT_VARIABLE_FEATURE_SUPPORTED']


def test_model_gain_without_B_is_not_a_variable_feature_discovery():
    comparison={'reference':'M0','candidate':'M5','status':'SUPPORTED','mean_delta':.002,'ci_low':.001}
    assert science_status(0,0,[comparison])=='NEXT8_NO_CERTIFIED_VARIABLE_FEATURE'
    assert science_status(1,0,[comparison])=='NEXT8_VARIABLE_FEATURES_AND_MODEL_GAIN'
    assert science_status(1,0,[])=='NEXT8_VARIABLE_FEATURES_SUPPORTED_NO_PROMOTION'
