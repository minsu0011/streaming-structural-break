"""Separate A dynamics, B orientation, conditional information and C models."""
from collections import Counter
import numpy as np
from src.next8.common import BASE,read_json,digest
from src.next8.features import NAMES,NORMALIZATION,N_FEATURES,DETAILED
from src.next8.feature_certification import load_feature_records
from src.next8.diagnostics import equal_fold_loss_summary
from src.next8.statistics import holm

BL_PAIRS=(('BL_CONTEXT','BL_FIXED_F'),('BL_STATE_ONLY','BL_OWN_DIRECTION'),('BL_FIXED_STATE','BL_OWN_DIRECTION'),
          ('BL_OWN_DIRECTION','BL_CROSS_DIRECTION'),('BL_SAME_INPUT_TREE','BL_CROSS_DIRECTION'))


def paired_information(panels,j,reference,candidate):
    parts=[];included=[];point=[];missing=[]
    for fold,panel in enumerate(panels):
        record,path=panel[j];comparison=next((r for r in record['BL80_conditional_information'].get('comparisons',[])
            if r['reference']==reference and r['candidate']==candidate),None)
        if comparison is None:missing.append(fold);continue
        with np.load(path,allow_pickle=False) as a:values=a[comparison['group_array_key']+'_GAIN'].copy()
        if len(values)<64:missing.append(fold);continue
        parts.append(values);included.append(fold);point.append(comparison['TS_AUC_delta'])
    r=equal_fold_loss_summary(parts)
    good=bool(r['status']=='SUPPORTED' and len(included)>=4 and r['mean_gain']>0 and r['ci_low']>0 and r['positive_folds']>=4)
    return {'feature':j,'name':NAMES[j],'reference':reference,'candidate':candidate,**r,'included_folds':included,
        'missing_or_not_run_folds':missing,'mean_TS_AUC_delta_descriptive':float(np.mean(point)) if point and all(v is not None for v in point) else None,
        'information_support_before_Holm':good,'p_value':r['p_normal_cluster'] if good else 1.,
        'uncertainty_metric':'Reference binary log loss minus candidate binary log loss, per original group',
        'not_a_new_standalone_model_vs_original_BL80':True,'not_used_to_choose_features_or_M_models':True}


def summarize_conditional_information():
    panels=load_feature_records(False)
    rows=[paired_information(panels,j,ref,cand) for j in range(N_FEATURES) for ref,cand in BL_PAIRS]
    for r,p in zip(rows,holm([r['p_value'] for r in rows])):
        r['Holm_p']=float(p);r['conditional_information_supported']=bool(r['information_support_before_Holm'] and p<.05)
    return {'status':'COMPLETE','rows':rows,'family_size':len(rows),
        'multiplicity':'Holm across fixed153 x5 conditional-information contrast slots; not fitted/unsupported slots p=1',
        'fixed_feature_information_count':sum(r['conditional_information_supported'] for r in rows if r['candidate']=='BL_FIXED_F'),
        'conditional_direction_information_count':len({r['feature'] for r in rows if r['conditional_information_supported'] and r['candidate']!='BL_FIXED_F'}),
        'all_supported_matching_folds_in_intervals_including_adverse_fifth':True,
        'lack_of_detected_increment_is_not_equivalence_to_no_information':True}


def joint_certificate_check(orientation,*,movement_state_path,execution_direction_head,execution_pass):
    """Separate refitted direction heads are not interchangeable certificates."""
    reasons=[]
    if not orientation.get('conditional_orientation_supported'):reasons.append('B_DIRECTION_NOT_CERTIFIED')
    if not movement_state_path.get('supported'):reasons.append('A_SAME_STATE_PATH_NOT_SUPPORTED')
    if movement_state_path.get('state_estimator_id')!=orientation.get('state_estimator_id'):reasons.append('STATE_ESTIMATOR_PATH_DIFFERS')
    if not orientation.get('direction_head_sha256') or orientation.get('direction_head_sha256')!=execution_direction_head:reasons.append('DIRECTION_HEAD_NOT_THE_EXECUTED_CERTIFIED_HEAD')
    if not execution_pass:reasons.append('CAUSAL_EXECUTION_NOT_VERIFIED')
    return {'JOINT_VARIABLE_FEATURE_SUPPORTED':not reasons,'reasons':reasons}


def science_status(conditional_count,nonlinear_count,comparisons):
    gains=[r for r in comparisons if r['reference']=='M0' and r['candidate'] in ('M1','M2','M3','M4','M5','M6') and
           r.get('status')=='SUPPORTED' and r['mean_delta']>0 and r['ci_low']>0]
    if conditional_count:return 'NEXT8_VARIABLE_FEATURES_AND_MODEL_GAIN' if gains else 'NEXT8_VARIABLE_FEATURES_SUPPORTED_NO_PROMOTION'
    if nonlinear_count and any(r['candidate'] in ('M1','M2') for r in gains):return 'NEXT8_NONLINEAR_FEATURE_GAIN_ONLY'
    # A supported own-history forecast is not a formal absence-of-information
    # test for B/C. Keep the stronger DYNAMICS_ONLY label unassigned here.
    return 'NEXT8_NO_CERTIFIED_VARIABLE_FEATURE'


def verified(directory,name):
    directory=BASE/directory;done=read_json(directory/'DONE.json')
    assert digest(directory/name)==done['files_sha256'][name]
    return read_json(directory/name)


def synthesize_science(information):
    features=verified('feature_certification','NORMAL.json');early=verified('feature_certification','EARLY.json')
    movement=verified('movement_summary','NORMAL.json');early_movement=verified('movement_summary','EARLY.json')
    models=verified('model_summary','NORMAL.json');early_models=verified('model_summary','EARLY.json')
    feature_rows=[]
    for r in features['features']:
        j=r['feature'];own=[a for a in movement['own_dynamics'] if a['feature']==j]
        cross=[e for e in movement['cross_edges'] if e['destination']==j and e['cross_factor_supported']]
        info=[v for v in information['rows'] if v['feature']==j and v['candidate']=='BL_FIXED_F'][0]
        feature_rows.append({**r,'normalization':NORMALIZATION,'in_fixed_detailed_movement_menu':j in DETAILED,
            'self_movement_supported_horizons':[v['horizon'] for v in own if v['self_dynamics_supported']],
            'cross_movement_supported_parent_horizons':[{'parent':e['source'],'horizon':e['horizon']} for e in cross],
            'BL_conditioned_fixed_F_information_supported':info['conditional_information_supported'],
            'BL_conditioned_fixed_F_mean_loss_gain':info.get('mean_gain'),'BL_conditioned_fixed_F_gain_ci_low':info.get('ci_low'),
            'BL_conditioned_fixed_F_gain_ci_high':info.get('ci_high'),
            'dynamics_only_not_assigned_from_nonsignificant_discrimination':True,
            'JOINT_VARIABLE_FEATURE_SUPPORTED':False,
            'joint_reason':'Conditional orientation refit evidence is not automatically a certificate for the distinct executed M5 direction head',
            'observed_movement_does_not_imply_target_orientation':True})
    certificate=[]
    for r in features['conditional_units']:
        certificate.append({**r,'JOINT_VARIABLE_FEATURE_SUPPORTED':False,
            'joint_execution_status':'No automatic transfer of O2/O3 refit orientation rule to separately fitted M4/M5 head',
            'B_only_certificate_preserved':r['conditional_orientation_supported'],
            'runtime_validation_required_for_each_distinct_certified_direction_head':True})
    counts={**features['counts'],**movement['counts'],'BL_conditioned_fixed_F_information_count':information['fixed_feature_information_count'],
        'joint_variable_feature_count':0,'taxonomy_counts':dict(Counter(r['classification'] for r in feature_rows))}
    status=science_status(counts['conditional_features_supported'],counts['nonlinear_supported'],models['comparisons'])
    return {'status':'COMPLETE','timestamp_scope':'Scientific synthesis before separate final audit approval',
        'scientific_status_subject_to_final_integrity_audit':status,'counts':counts,'features':feature_rows,'conditional_certificates':certificate,
        'normal_model_comparisons':models['comparisons'],'normal_mean_TS_AUC':models['mean_TS_AUC'],
        'early_counts':{**early['counts'],**early_movement['counts']},'early_mean_TS_AUC':early_models['mean_TS_AUC'],
        'cross_graph_summary':movement['summary_graph_edges'],
        'movement_factors_and_orientation_conditions_same_by_definition':False,
        'factorized_state_value_supported_pairs':[{'feature':r['feature'],'horizon':r['horizon']} for r in movement['transition_tests'] if r['factorized_transition_value_supported']],
        'A_B_C_separate':True,'C_single_feature_information_is_not_M_model_gain':True,
        'B_certificate_does_not_imply_JOINT_or_model_promotion':True,'promotion_final':False,
        'promotion_pending':'Causal/runtime/determinism/reproduction/final integrity checks and measured time contract',
        'negative_result_limit':'No certified relation under this fixed data/procedure/support menu; not proof that every feature or richer model lacks predictive information',
        'empty_feature_model_limit':'If TRAIN-selected sets are empty, M heads reduce to intercepts; those comparisons do not reject the general expressive power of nonlinear or graph models',
        'historically_adaptive_RP_development_not_independent_hidden_test_confirmation':True}
