"""Common rules, domains and explicit direction evidence are all required."""
import src.next8
import copy
from src.next8.feature_certification import common_rule_repetition,atlas_support


def entries():
    domains=[{'domain_family':kind,'domain':kind+str(k),'direction_and_gain_reproduced':True} for kind in ('TIME','H_LENGTH') for k in range(2)]
    return [{'fold':f,'case':{'training_case':{'domain':[-1.,1.],'F_anchor':0.}},'pair':{
        'both_claimed_directions_reproduced':True,'matched_model_comparisons':[{'reference':m,'gain':.001} for m in ('FIXED_CONTROL','STATE_ONLY')],
        'heldout_domains':copy.deepcopy(domains)}} for f in range(5)]


def test_four_of_five_rule_repetition_keeps_failures_visible():
    values=entries();values[4]['pair']['both_claimed_directions_reproduced']=False
    result=common_rule_repetition(values)
    assert result['passed'] and result['folds']==[0,1,2,3]
    values[3]['pair']['heldout_domains'][0]['direction_and_gain_reproduced']=False
    assert not common_rule_repetition(values)['passed']


def test_different_anchor_or_non_significance_is_not_invariance():
    values=entries()
    for k,e in enumerate(values):e['case']['training_case']['F_anchor']=k*.01
    assert not common_rule_repetition(values)['passed']
    rows=[{'mode':'RAW_F','domain_family':'ALL','status':'SUPPORTED','oriented_ci_low':.45,'UNKNOWN_domain':False}]
    assert not atlas_support(rows,'RAW_F')[0]


def test_opposing_fifth_fold_stays_in_paired_certification(tmp_path):
    import numpy as np
    from src.next8.feature_certification import aggregate_conditional_unit
    values=entries()
    for fold,e in enumerate(values):
        e['case']['array_prefix']='C_'
        difference=.04 if fold<4 else -.30
        e['pair'].update({'status':'SUPPORTED','state_raw_AUC_difference':difference,
            'training_rule':{'state_pair':[0,1],'directions':[1,-1],'eligible_for_orientation_replication':True}})
        if fold==4:e['pair']['both_claimed_directions_reproduced']=False
        arrays={'C_STATE_0_BOOTSTRAP':np.full(2000,.5+difference),'C_STATE_1_BOOTSTRAP':np.full(2000,.5)}
        e['pair']['matched_model_comparisons']=[{'reference':name,'gain':.001,'group_array_key':name} for name in ('FIXED_CONTROL','STATE_ONLY','TABULAR')]
        for name in ('FIXED_CONTROL','STATE_ONLY','TABULAR'):arrays['C_'+name+'_GAIN']=np.full(128,.001)
        path=tmp_path/f'fold{fold}.npz';np.savez(path,**arrays);e['group_path']=path
    result=aggregate_conditional_unit(values)
    assert result['rule_repetition']['passed'] and result['direct_state_difference']<0
    assert not result['certification_before_Holm'] and result['all_supported_matching_rule_folds_included_in_uncertainty']==list(range(5))
