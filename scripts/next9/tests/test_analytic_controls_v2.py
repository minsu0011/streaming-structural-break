from src.next9.analytic_controls_v2 import fixed_u_shape


def test_common_range_raw_sign_flip_can_be_fixed_U_shape():
    result=fixed_u_shape();down,up=result['records']
    assert down['raw_F_signed_effect']<0<up['raw_F_signed_effect']
    assert down['common_low']==up['common_low'] and down['common_high']==up['common_high']
    assert all(r['O3_increment_over_O1']==0 for r in result['records'])
    assert all(r['O1_expected_logloss']<r['STATE_ONLY_expected_logloss'] for r in result['records'])
