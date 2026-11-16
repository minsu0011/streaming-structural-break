"""Explain fixed TRAIN discovery gate failures without relaxing any gate."""
from collections import Counter
from src.next8.common import BASE,read_json,digest
from src.next8.screening import direction_supported


def atlas_gate_reasons(atlas):
    overall=next(r for r in atlas if r['domain_type']=='all')
    supported=[r for r in atlas if r['support_status']=='SUPPORTED' and r['domain_type']!='all' and
               not (r['domain_type']=='own_pre_state' and r['domain']=='3')]
    passed,coverage=direction_supported(atlas);failures=[]
    if overall['support_status']!='SUPPORTED':failures.append('OVERALL_SPARSE_SAME_TIME_SUPPORT')
    if overall['ci_low']<=.505:failures.append('OVERALL_USEFUL_DIRECTION_LOWER_BOUND_NOT_ABOVE_0505')
    if any(n<2 for n in coverage.values()):failures.append('FEWER_THAN_TWO_SUPPORTED_DOMAINS_IN_REQUIRED_FAMILY')
    domains=[]
    for r in supported:
        if r['ci_low']>.47:continue
        kind=('SUPPORTED_MATERIAL_REVERSAL' if r['ci_high']<.47 else
              'POINT_REVERSAL_WITH_UNCERTAIN_INTERVAL' if r['TS_AUC']<.47 else
              'UNCERTAINTY_DOES_NOT_EXCLUDE_MATERIAL_REVERSAL')
        domains.append({'domain_type':r['domain_type'],'domain':r['domain'],'failure_type':kind,
            'oriented_TS_AUC':r['TS_AUC'],'ci_low':r['ci_low'],'ci_high':r['ci_high'],
            'positive_series':r['positive_series'],'negative_series':r['negative_series'],'series':r['series']})
    if domains:failures.append('AT_LEAST_ONE_SUPPORTED_DOMAIN_LOWER_BOUND_NOT_ABOVE_047')
    return {'passed':passed,'reasons':failures,'coverage':coverage,'failed_domains':domains,
        'overall_oriented_TS_AUC':overall['TS_AUC'],'overall_ci_low':overall['ci_low'],
        'point_reversal_is_not_itself_statistically_supported_reversal':True}


def diagnose_discovery():
    release=read_json(BASE/'PREDICTION_FREEZE_NEXT8.json');rows=[];domain_rows=[];scopes=[]
    for p in sorted((BASE/'selection').glob('*/SELECTION.json')):
        name=p.parent.name
        assert digest(p)==release['files_sha256'][p.relative_to(BASE).as_posix()]
        selected=read_json(p);final=read_json(BASE/'h_context_selection'/name/'SELECTION.json')
        counts=Counter();domains=Counter()
        for inner in range(3):
            for j in range(153):
                path=p.parent/f'split_{inner}'/f'F{j:03d}.json'
                assert digest(path)==release['files_sha256'][path.relative_to(BASE).as_posix()]
                r=read_json(path)
                raw=atlas_gate_reasons([a for a in r['atlas'] if a['mode']=='SIGNED_F'])
                nonlinear=atlas_gate_reasons([a for a in r['atlas'] if a['mode']=='FIXED_NONLINEAR_MODEL'])
                assert raw['passed']==r['invariant_direction_supported']
                gain=r['O1_increment_beyond_context'];delta=r['scores']['O1']['TS_AUC']-r['scores']['O0']['TS_AUC']
                expected=nonlinear['passed'] and delta>.002 and gain['ci_low'] is not None and gain['ci_low']>0 and gain['gain']>.0001
                assert expected==r['fixed_nonlinear_supported']
                conditional=r['conditional'];comparisons=conditional.get('comparisons',{})
                record={'scope':name,'inner':inner,'feature':j,'name':r['name'],'training_final_classification':final['classifications'][j]['classification'],
                    'initial_stable_selected':j in selected['stable_features'],'final_stable_selected':j in final['stable_features'],
                    'initial_variable_selected':j in selected['variable_features'],'final_variable_selected':j in final['variable_features'],
                    'all_optimizers_converged':r['all_converged'],'invariant_gate_passed':raw['passed'],
                    'invariant_failure_reasons':'|'.join(raw['reasons']),'nonlinear_atlas_passed':nonlinear['passed'],
                    'nonlinear_atlas_failure_reasons':'|'.join(nonlinear['reasons']),
                    'nonlinear_minus_linear_TS_AUC':delta,'nonlinear_shape_improvement_passed':delta>.002,
                    'nonlinear_context_loss_gain':gain['gain'],'nonlinear_context_gain_ci_low':gain['ci_low'],
                    'fixed_nonlinear_all_gates_passed':r['fixed_nonlinear_supported'],'conditional_status':conditional['status'],
                    'conditional_gain_vs_fixed_state':comparisons.get('O1_STATE_MAIN',{}).get('gain'),
                    'conditional_gain_ci_low':comparisons.get('O1_STATE_MAIN',{}).get('ci_low'),
                    'conditional_repeated_inner_folds':final['classifications'][j]['conditional_supported_folds'],
                    'no_new_selection_or_fit':True}
                rows.append(record)
                for mode,evidence in [('SIGNED_F',raw),('FIXED_NONLINEAR_MODEL',nonlinear)]:
                    for failure in evidence['failed_domains']:
                        domain_rows.append({'scope':name,'inner':inner,'feature':j,'mode':mode,**failure})
                        domains[(mode,failure['domain_type'],failure['domain'],failure['failure_type'])]+=1
                counts['optimizer_nonconvergence']+=not r['all_converged']
                counts['inner_invariant_passes']+=raw['passed'];counts['inner_nonlinear_passes']+=r['fixed_nonlinear_supported']
        scopes.append({'scope':name,'inner_gate_counts':dict(counts),'initial_classification_counts':dict(Counter(r['classification'] for r in selected['classifications'])),
            'final_classification_counts':dict(Counter(r['classification'] for r in final['classifications'])),
            'domain_failures':[{'mode':k[0],'domain_type':k[1],'domain':k[2],'failure_type':k[3],'count':v} for k,v in sorted(domains.items())],
            'initial_stable':selected['stable_features'],'final_stable':final['stable_features'],
            'initial_variable':selected['variable_features'],'final_variable':final['variable_features']})
    return rows,domain_rows,{'status':'COMPLETE','scopes':scopes,'scope_count':len(scopes),
        'all_fixed_gate_replay_exact':True,'TRAIN_results_only':True,'outer_scores_used':False,
        'uncertainty_failure_is_not_proof_of_opposite_direction_or_uninformative_feature':True,
        'threshold_or_candidate_change_after_diagnosis':False,'historical_adaptive_development_scope_retained':True}
