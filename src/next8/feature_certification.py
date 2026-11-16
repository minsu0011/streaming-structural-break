"""Cross-fold feature evidence; certification never changes fitted models."""
from collections import Counter,defaultdict
from itertools import combinations
import numpy as np
from scipy.stats import norm
from src.next8.common import BASE,read_json,digest
from src.next8.features import N_FEATURES,NAMES,NORMALIZATION
from src.next8.statistics import holm
from src.next8.diagnostics import equal_fold_loss_summary


def atlas_support(rows,mode):
    chosen=[r for r in rows if r['mode']==mode]
    overall=next((r for r in chosen if r['domain_family']=='ALL'),None)
    supported=[r for r in chosen if r['domain_family']!='ALL' and r['status']=='SUPPORTED' and not r['UNKNOWN_domain']]
    count=Counter(r['domain_family'] for r in supported)
    coverage=all(count[k]>=2 for k in ('TIME','H_LENGTH','OWN_PRE_STATE')) and max(count['H_LAG1'],count['H_TAIL3'])>=2
    passed=bool(overall and overall['status']=='SUPPORTED' and overall['oriented_ci_low'] is not None and overall['oriented_ci_low']>.505 and
        coverage and all(r['oriented_ci_low'] is not None and r['oriented_ci_low']>.47 for r in supported))
    return passed,dict(count)


def get_loss(panel,reference,candidate):
    return next((r for r in panel.get('comparisons',[]) if r['reference']==reference and r['candidate']==candidate),None)


def stable_fold(record):
    label=record['training_classification']['classification'];mode='RAW_F' if label=='INVARIANT_DIRECTION' else 'O1_FIXED_NONLINEAR'
    atlas,coverage=atlas_support(record['atlas'],mode)
    if label=='INVARIANT_DIRECTION':
        direction=record['training_classification']['signed_direction']
        return {'passed':bool(atlas and len(set(direction))==1),'classification':label,'direction':direction[0] if len(set(direction))==1 else 0,'coverage':coverage}
    if label=='STABLE_NONLINEAR':
        panel=record['fixed_refit'];shape=get_loss(panel,'O0','O1');gain=get_loss(panel,'CONTEXT_ONLY','O1')
        passed=bool(atlas and shape and gain and shape['TS_AUC_delta'] is not None and shape['TS_AUC_delta']>.002 and
            gain['status']=='SUPPORTED' and gain['gain']>.0001 and gain['ci_low']>0)
        return {'passed':passed,'classification':label,'direction':None,'coverage':coverage}
    return {'passed':False,'classification':label,'direction':None,'coverage':coverage}


def common_rule_repetition(entries):
    """Four matching folds, keeping every available fifth fold in uncertainty."""
    passed=[]
    for entry in entries:
        pair=entry['pair'];comparison={r['reference']:r for r in pair.get('matched_model_comparisons',[])}
        a=comparison.get('FIXED_CONTROL');b=comparison.get('STATE_ONLY')
        if pair.get('both_claimed_directions_reproduced') and a and b and a['gain'] is not None and b['gain'] is not None and a['gain']>.0001 and b['gain']>0:
            passed.append(entry)
    for members in combinations(passed,4):
        low=max(e['case']['training_case']['domain'][0] for e in members);high=min(e['case']['training_case']['domain'][1] for e in members)
        anchors=[e['case']['training_case']['F_anchor'] for e in members]
        if high-low<=1e-3 or max(anchors)-min(anchors)>1e-12:continue
        sets=[{(r['domain_family'],r['domain']) for r in e['pair']['heldout_domains'] if r['direction_and_gain_reproduced']} for e in members]
        domains=set.intersection(*sets)
        if any(sum(k==family for k,_ in domains)<2 for family in ('TIME','H_LENGTH')):continue
        return {'passed':True,'folds':[e['fold'] for e in members],'common_F_domain':[low,high],
            'shared_training_F_anchor':anchors[0],'common_domains':[list(r) for r in sorted(domains)],
            'same_anchor_was_frozen_within_each_training_scope':True,
            'heldout_effects_evaluated_on_each_scope_pre_frozen_domain_not_refiltered_using_other_folds':True}
    return {'passed':False,'passing_direction_and_loss_folds':[e['fold'] for e in passed],
        'reason':'No four matching frozen rules with two common supported time and H domains and a common pre-frozen anchor'}


def aggregate_conditional_unit(entries):
    repetition=common_rule_repetition(entries);rank_entries=[e for e in entries if e['pair'].get('status')=='SUPPORTED']
    result={'fold_count':len(entries),'folds':[e['fold'] for e in entries],'rule_repetition':repetition,
        'training_eligible_folds':[e['fold'] for e in entries if e['pair']['training_rule']['eligible_for_orientation_replication']],
        'heldout_direction_pass_folds':[e['fold'] for e in entries if e['pair'].get('both_claimed_directions_reproduced')],
        'p_value':1.,'certification_before_Holm':False,'combined_loss':{}}
    if len(rank_entries)<4:return {**result,'status':'INSUFFICIENT_REPEATED_SUPPORT'}
    bootstrap=[];points=[];loss_parts={name:[] for name in ('FIXED_CONTROL','STATE_ONLY','TABULAR')}
    for entry in rank_entries:
        case=entry['case'];pair=entry['pair'];prefix=case['array_prefix'];a,b=pair['training_rule']['state_pair']
        with np.load(entry['group_path'],allow_pickle=False) as data:
            bootstrap.append(data[prefix+f'STATE_{a}_BOOTSTRAP']-data[prefix+f'STATE_{b}_BOOTSTRAP'])
            points.append(pair['state_raw_AUC_difference'])
            for comparison in pair['matched_model_comparisons']:
                loss_parts[comparison['reference']].append(data[prefix+comparison['group_array_key']+'_GAIN'].copy())
    array=np.stack(bootstrap);valid=np.isfinite(array).all(axis=0)
    if valid.sum()<1900:return {**result,'status':'INSUFFICIENT_BOOTSTRAP_SUPPORT'}
    mean=array[:,valid].mean(axis=0);point=float(np.mean(points));sd=float(np.std(mean,ddof=1))
    p=float(2*norm.sf(abs(point)/sd)) if sd else (0. if point else 1.)
    combined={name:equal_fold_loss_summary(parts) for name,parts in loss_parts.items()}
    gain=combined['FIXED_CONTROL'];state=combined['STATE_ONLY']
    difference_low=float(np.quantile(mean,.025));difference_high=float(np.quantile(mean,.975))
    claimed_direction=entries[0]['pair']['training_rule']['directions'][0]
    direct_supported=difference_low>0 if claimed_direction>0 else difference_high<0 if claimed_direction<0 else False
    passed=bool(repetition['passed'] and gain['status']==state['status']=='SUPPORTED' and gain['mean_gain']>.0001 and
        gain['ci_low']>0 and gain['positive_folds']>=4 and state['mean_gain']>0 and state['ci_low']>0 and state['positive_folds']>=4 and direct_supported)
    return {**result,'status':'SUPPORTED','p_value':p,'direct_state_difference':point,
        'direct_difference_ci_low':difference_low,'direct_difference_ci_high':difference_high,'combined_difference_supports_training_direction':direct_supported,
        'p_method':'Normal approximation from paired original-group bootstrap SD of equal-fold state contrast',
        'combined_loss':combined,'all_supported_matching_rule_folds_included_in_uncertainty':[e['fold'] for e in rank_entries],
        'failed_fifth_matching_fold_not_dropped':True,'certification_before_Holm':passed}


def load_feature_records(early=False):
    result=[]
    for fold in range(5):
        scope=f'outer_{fold}'+('_EARLY_T199' if early else '');directory=BASE/'feature_evaluation'/scope
        done=read_json(directory/'DONE.json');assert done['features']==N_FEATURES
        records=[]
        for j in range(N_FEATURES):
            p=directory/f'F{j:03d}';assert digest(p/'DONE.json')==done['files_sha256'][f'F{j:03d}/DONE.json']
            record=read_json(p/'DONE.json');assert digest(p/'GROUP_EVIDENCE.npz')==record['group_evidence_sha256']
            records.append((record,p/'GROUP_EVIDENCE.npz'))
        result.append(records)
    return result


def certify_features(early=False):
    panels=load_feature_records(early);units=defaultdict(list);features=[]
    for j in range(N_FEATURES):
        records=[p[j][0] for p in panels];checks=[stable_fold(r) for r in records]
        inv=Counter(r['direction'] for r in checks if r['passed'] and r['classification']=='INVARIANT_DIRECTION')
        nonlinear=sum(r['passed'] and r['classification']=='STABLE_NONLINEAR' for r in checks)
        invariant=max(inv.values(),default=0)>=4;nl=nonlinear>=4
        selected_stable=sum(r['selected_stable'] for r in records);selected_variable=sum(r['selected_variable'] for r in records)
        gains=[r['training_classification'].get('mean_conditional_gain') for r in records]
        gains=[g for g in gains if g is not None]
        features.append({'feature':j,'name':NAMES[j],'early_only':early,'invariant_supported':invariant,'nonlinear_supported':nl,
            'stable_fold_checks':checks,'invariant_direction':inv.most_common(1)[0][0] if invariant else None,
            'selected_stable_TRAIN_folds':selected_stable,'selected_variable_TRAIN_folds':selected_variable,
            'mean_inner_conditional_gain':float(np.mean(gains)) if gains else None,
            'mean_inner_fixed_AUC':float(np.mean([r['training_classification']['mean_fixed_nonlinear_AUC'] for r in records])),
            'candidate_entry_folds':[f for f,r in enumerate(records) if r['selected_variable']],
            'conditional_certified':False,'retained_stable_summary':False,'retained_variable_summary':False})
        for fold,(record,path) in enumerate(p[j] for p in panels):
            for branch in record['conditional']:
                definition=branch['definition']
                for ci,case in enumerate(branch['cases']):
                    for pair in case.get('pairs',[]):
                        rule=pair['training_rule'];own=case['training_case']['own_stratum']
                        key=(j,branch['model'],definition['selection_state_family'],definition['orientation_estimator_family'],own,
                            tuple(rule['state_pair']),tuple(rule['directions']))
                        units[key].append({'fold':fold,'case_index':ci,'case':case,'pair':pair,'group_path':path})
    conditional=[]
    for key,entries in sorted(units.items(),key=lambda item:str(item[0])):
        j,model,own_family,estimator,own,pair,directions=key
        evidence=aggregate_conditional_unit(entries)
        conditional.append({'feature':j,'name':NAMES[j],'orientation_model':model,'own_or_H_condition_family':own_family,
            'state_estimator_family':estimator,'own_stratum':own,'state_pair':list(pair),'directions':list(directions),
            'normalization':NORMALIZATION,'early_only':early,**evidence})
    adjusted=holm([r['p_value'] for r in conditional])
    for r,p in zip(conditional,adjusted):
        r['Holm_p']=float(p);r['conditional_orientation_supported']=bool(r['certification_before_Holm'] and p<.05)
        r['JOINT_VARIABLE_FEATURE_SUPPORTED']=False;r['joint_status']='REQUIRES_SAME_STATE_PATH_MOVEMENT_AND_EXECUTION_EVIDENCE'
        r['certificate_scope']='Conditional orientation development evidence; no causal mechanism or independent hidden-test confirmation'
        if r['conditional_orientation_supported']:features[r['feature']]['conditional_certified']=True
    stable=[r for r in features if (r['invariant_supported'] or r['nonlinear_supported']) and r['selected_stable_TRAIN_folds']>=4]
    stable.sort(key=lambda r:(-r['selected_stable_TRAIN_folds'],-r['mean_inner_fixed_AUC'],r['feature']))
    for r in stable[:6]:r['retained_stable_summary']=True
    variable=[r for r in features if r['conditional_certified']]
    variable.sort(key=lambda r:(-r['selected_variable_TRAIN_folds'],-(r['mean_inner_conditional_gain'] if r['mean_inner_conditional_gain'] is not None else -float('inf')),r['feature']))
    for r in variable[:6]:r['retained_variable_summary']=True
    for r in features:
        r['classification']=('CONDITIONAL_VARIABLE_CERTIFIED' if r['conditional_certified'] else 'STABLE_NONLINEAR' if r['nonlinear_supported'] else
            'INVARIANT_DIRECTION' if r['invariant_supported'] else 'CONDITIONAL_VARIABLE_CANDIDATE' if r['selected_variable_TRAIN_folds'] else 'UNRESOLVED')
    return {'status':'COMPLETE','early_only':early,'features':features,'conditional_units':conditional,
        'counts':{'invariant_supported':sum(r['invariant_supported'] for r in features),'nonlinear_supported':sum(r['nonlinear_supported'] for r in features),
            'variable_candidates_ever_selected':sum(r['selected_variable_TRAIN_folds']>0 for r in features),
            'conditional_features_supported':sum(r['conditional_certified'] for r in features),
            'retained_stable':sum(r['retained_stable_summary'] for r in features),'retained_variable':sum(r['retained_variable_summary'] for r in features)},
        'conditional_family_size':len(conditional),'conditional_multiplicity':'Holm across every observed frozen feature/state-family/own-stratum/pair/direction unit, including unsupported units with p=1',
        'retention_uses_TRAIN_selection_frequency_and_INNER_gain_not_outer_effect_size':True,
        'A_and_C_not_inferred_from_B':True,'new_models_refitted_from_certificates':False,
        'common_domain_caution':'Per-fold heldout effects use that fold own TRAIN-frozen range. Intersections describe shared support; no pooled-validation range refiltering or claim of uniform effect at every F.'}
