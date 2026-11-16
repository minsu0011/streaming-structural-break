"""Read-only scientific aggregation of all once-frozen original outer folds."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,csv_write,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary,target_free_npz


def main():
    violations=install_workspace_boundary()
    lock=read_json(BASE/'SCIENCE_AGGREGATION_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    with target_free_npz(),activity('Aggregate once-frozen original outer A B C and same-horizon cross evidence','SCIENTIFIC_AGGREGATION','B2_NO_REPLICATED_OPPOSITE_ORIENTATION'):
        import numpy as np
        from collections import defaultdict
        from src.next9.science_summary import aggregate_contrast,score_promotion,CONTRASTS
        from src.next9.certification import orientation_pass,tabular_comparison,domain_replication
        # M quality: all original folds and every predefined domain, no new fit.
        groups=defaultdict(list);bootstraps=defaultdict(list);cv=[];sources={}
        global_freeze=read_json(BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json')
        same_M5_input={}
        for mode in ['NORMAL','EARLY']:
            same_M5_input[mode]=all(read_json(BASE/r['path'])['M6_alias']=='M5' for r in global_freeze['records'] if r['mode']==mode)
        for fold in range(5):
            root=BASE/'outer_model_evaluation'/f'outer_{fold}'
            assert read_json(root/'DONE.json')['status']=='COMPLETE';sources[str(root.relative_to(BASE))+'/DONE.json']=digest(root/'DONE.json')
            for mode in ['NORMAL','EARLY']:
                receipt=read_json(root/mode/'DONE.json')
                for domain in receipt['domains']:
                    out=root/mode/domain;assert digest(out/'DONE.json')==receipt['domain_receipts_sha256'][domain]
                    r=read_json(out/'DONE.json');assert digest(out/'BOOTSTRAP.npz')==r['bootstrap_sha256']
                    with np.load(out/'BOOTSTRAP.npz') as z:samples={k:z[k] for k in z.files}
                    groups[(mode,domain)].append(r);bootstraps[(mode,domain)].append(samples)
                    for method,point in r['points'].items():
                        cv.append(dict(fold=fold,mode=mode,family=r['family'],domain=domain,method=method,TS_AUC=point['TS_AUC'],
                            **r['support'],major_domain_supported=r['major_domain_supported']))
        comparisons=[];uncertainty=[];stresses=[];model_gate={}
        for (mode,domain),records in sorted(groups.items()):
            samples=bootstraps[(mode,domain)]
            for candidate,reference in CONTRASTS:
                r=aggregate_contrast(records,samples,candidate,reference)
                flat={k:v for k,v in r.items() if not isinstance(v,(dict,list))}
                uncertainty.append(dict(mode=mode,domain=domain,**flat,fold_deltas='|'.join(map(str,r.get('fold_deltas',[]))),folds='|'.join(map(str,r['folds'])),
                    exact_same_input_tabular=(candidate=='M6' or (candidate=='M5' and same_M5_input[mode])) if reference=='TABULAR_SAME_INPUT' else None))
            if domain=='ALL':
                model_gate[mode]=aggregate_contrast(records,samples,'M7','M0')
                for method in records[0]['points']:
                    values=[r['points'][method]['TS_AUC'] for r in records]
                    assert all(v is not None for v in values) and len(values)==5
                    contrast=aggregate_contrast(records,samples,method,'M0')
                    comparisons.append(dict(mode=mode,method=method,mean_TS_AUC=float(np.mean(values)),mean_delta_vs_BL80=contrast['point'],
                        ci_low=contrast['ci_low'],ci_high=contrast['ci_high'],positive_folds=contrast['positive_folds'],worst_fold_delta=contrast['worst_fold'],
                        **{f'fold_{k}_TS_AUC':v for k,v in enumerate(values)},**{f'fold_{k}_delta':v for k,v in enumerate(contrast['fold_deltas'])}))
            else:
                r=aggregate_contrast(records,samples,'M7','M0',major_only=True)
                stresses.append(dict(mode=mode,domain=domain,family=records[0]['family'],contrast=r))
        normal_bl=next(r['mean_TS_AUC'] for r in comparisons if r['mode']=='NORMAL' and r['method']=='M0')
        early_bl=next(r['mean_TS_AUC'] for r in comparisons if r['mode']=='EARLY' and r['method']=='M0')
        assert abs(normal_bl-.6203652100698587)<1e-12,normal_bl
        assert abs(early_bl-.6287908798814562)<1e-12,early_bl
        promotion=score_promotion(model_gate['NORMAL'],model_gate['EARLY'],stresses)
        csv_write('MODEL_COMPARISON_NEXT9.csv',comparisons);csv_write('CV_RESULTS_NEXT9.csv',cv);csv_write('PAIRED_UNCERTAINTY_NEXT9.csv',uncertainty)
        for family,name in [('TIME','TIME_GENERALIZATION_NEXT9.csv'),('H','H_GENERALIZATION_NEXT9.csv')]:
            csv_write(name,[r for r in cv if (r['family']=='TIME' if family=='TIME' else r['family'].startswith('H_'))])
        # B certificates are replications of the TRAIN-chosen pair, not newly
        # selected outer pair patterns. All other outer tests remain diagnostics.
        orientation=[];candidate_ranks=defaultdict(list);keys=set()
        for fold in range(5):
            scope=f'outer_{fold}';candidate=read_json(BASE/'candidates'/scope/'FROZEN.json')
            for rank,j in enumerate(candidate['normal_candidates']):candidate_ranks[j].append(rank)
            root=BASE/'outer_orientation_evaluation'/scope;done=read_json(root/'DONE.json');assert done['status']=='COMPLETE'
            sources[str(root.relative_to(BASE))+'/DONE.json']=digest(root/'DONE.json')
            for member in done['records']:
                path=root/member['mode']/'RESULTS.json';assert digest(path)==member['sha256']
                rows=read_json(path)['records'];orientation.extend(rows)
                for r in rows:
                    rule=r['provisional_rule']
                    if rule is not None:keys.add((r['feature'],rule['pair_index'],rule['pattern']))
        rules=[]
        for feature,pair,pattern in sorted(keys):
            selected=[r for r in orientation if r['feature']==feature and r['pair_index']==pair
                and r['provisional_rule'] is not None and r['provisional_rule']['pair_index']==pair and r['provisional_rule']['pattern']==pattern]
            normal=[r for r in selected if r['mode']=='NORMAL' and r['domain_name']=='ALL']
            early=[r for r in selected if r['mode']=='EARLY' and r['domain_name']=='ALL']
            good_normal=[r for r in normal if orientation_pass(r,pattern)];good_early=[r for r in early if orientation_pass(r,pattern)]
            domains={}
            for name in sorted(set(r['domain_name'] for r in selected if r['mode']=='NORMAL' and r['domain_name']!='ALL')):
                rr=[r for r in selected if r['mode']=='NORMAL' and r['domain_name']==name]
                family=rr[0]['family']
                if family in ['H_LAG1','H_TAIL3'] and name.endswith('_3'):continue
                domains[name]=dict(family=family,**domain_replication(rr,pattern))
            time=[n for n,r in domains.items() if r['family']=='TIME' and r['supported']]
            h=[n for n,r in domains.items() if r['family'].startswith('H_') and r['supported']]
            certified=len(good_normal)>=4 and len(good_early)>=4 and len(time)>=2 and len(h)>=2
            ni=[r['fold'] for r in good_normal if tabular_comparison(r)['noninferior']]
            superior=[r['fold'] for r in good_normal if tabular_comparison(r)['superior']]
            rules.append(dict(feature=feature,pair_index=pair,pattern=pattern,normal_passing_folds=[r['fold'] for r in good_normal],
                early_passing_folds=[r['fold'] for r in good_early],time_domains=time,H_domains=h,domain_replication=domains,
                certified_before_budget=certified,tabular_noninferior_folds=ni,tabular_superior_folds=superior,
                research_signal_science_gate=certified and len(ni)>=4))
        accepted=sorted([r for r in rules if r['certified_before_budget']],key=lambda r:(float(np.mean(candidate_ranks[r['feature']])),r['feature']))[:6]
        certified_features={r['feature'] for r in accepted}
        classifications=[dict(feature=j,outer_A_selected_count=len(candidate_ranks.get(j,[])),
            classification='MOVEMENT_CONDITIONAL_VARIABLE_CERTIFIED' if j in certified_features else
                'SELF_DYNAMICS_SUPPORTED_B_UNRESOLVED' if len(candidate_ranks.get(j,[]))>=4 else 'NOT_MOVEMENT_ELIGIBLE_OR_UNRESOLVED') for j in range(153)]
        cert=dict(status='COMPLETE',timestamp=utc_now(),certified_variable_count=len(accepted),certified_rules=accepted,all_rule_checks=rules,
            feature_classification=classifications,A_candidates_in_atleast4outerTRAIN=sum(len(v)>=4 for v in candidate_ranks.values()),
            domain_multiplicity='Separate full candidate x8pair Holm family per prespecified domain, not simultaneous acrossdomains.',
            no_outer_pair_selection=True,unresolved_not_proof_of_absence=True,adaptive_RP_not_independent_hidden_confirmation=True,
            new_research_role_pending_leakage_audit=True,canonical_role_changed=False)
        atomic(BASE/'VARIABLE_CERTIFICATION_NEXT9.json',cert)
        flat_orientation=[];ranges=[]
        for r in orientation:
            row={k:r[k] for k in ['fold','mode','feature','pair_index','method','family','domain_name','status','difference_Holm_p','orientation_pattern','orientation_pass']}
            row.update(common_low=r['domain']['low'],common_high=r['domain']['high'],TRAIN_range_status=r['domain']['status'])
            ranges.append({**row,**{f'state_{state}_{k}':v for state,s in r['validation_support'].items() for k,v in s.items()}})
            if r['status']=='COMPLETE':
                for side,value in r['evidence']['orientation'].items():
                    row.update({side+'_'+k:value.get(k) for k in ['point','ci_low','ci_high']})
                for reference in ['O1','STATE_ONLY','O6_TABULAR']:
                    c=r['evidence']['contrasts'][r['method']+'__vs__'+reference]
                    for metric,stats in c.items():row.update({f'vs_{reference}_{metric}_{k}':stats.get(k) for k in ['point','ci_low','ci_high']})
            flat_orientation.append(row)
        csv_write('CONDITIONAL_ORIENTATION_RESULTS.csv',flat_orientation);csv_write('COMMON_RANGE_RESULTS.csv',ranges)
        # Self-law quality stays separate from whether its branch was admitted
        # to a break model; an absent B rule does not erase auxiliary results.
        transition_rows=[];transition_models=[];outer_a=[]
        for fold in range(5):
            for mode in ['NORMAL','EARLY']:
                scope=f'outer_{fold}'
                root=BASE/'outer_self_transition_evaluation'/scope/mode
                manifest=read_json(root/'DONE.json');assert manifest['status']=='COMPLETE'
                for feature in manifest['features']:
                    directory=root/f"F{feature['feature']:03d}";assert digest(directory/'DONE.json')==feature['sha256']
                    done=read_json(directory/'DONE.json');j=done['feature']
                    assert digest(directory/'TRANSITION_MATRICES.json')==done['matrices_sha256']
                    transition_models.append(dict(fold=fold,mode=mode,feature=j,chosen_variant=done['chosen_variant'],
                        matrices=read_json(directory/'TRANSITION_MATRICES.json')))
                    for member in done['domains']:
                        path=directory/member['domain']/'DONE.json';assert digest(path)==member['sha256'];r=read_json(path)
                        for contrast,stats in r['contrasts'].items():
                            transition_rows.append(dict(fold=fold,mode=mode,feature=j,family=r['family'],domain=r['domain'],contrast=contrast,
                                groups=r['groups'],rows=r['rows'],chosen_variant=done['chosen_variant'],**stats))
                root=BASE/'outer_movement_evaluation'/scope/mode;manifest=read_json(root/'DONE.json')
                for feature in manifest['features']:
                    directory=root/f"F{feature['feature']:03d}";assert digest(directory/'DONE.json')==feature['sha256'];done=read_json(directory/'DONE.json')
                    for member in done['domains']:
                        path=directory/member['domain']/'DONE.json';assert digest(path)==member['sha256'];r=read_json(path)
                        for contrast,stats in r['contrasts'].items():
                            outer_a.append(dict(fold=fold,mode=mode,feature=feature['feature'],family=r['family'],domain=r['domain'],contrast=contrast,
                                groups=r['groups'],rows=r['rows'],**stats))
        csv_write('SELF_TRANSITION_RESULTS.csv',transition_rows)
        atomic(BASE/'SELF_TRANSITION_MODELS.json',dict(status='COMPLETE',timestamp=utc_now(),records=transition_models,
            chosen_laws_fixed_in_TRAIN=True,outer_comparator_diagnostics_never_change_M5_branch=True))
        csv_write('OUTER_MOVEMENT_RESULTS.csv',outer_a)
        csv_write('OUTER_H_CENTER_COMPARISON.csv',[r for r in outer_a if '__vs__D0B_MSE' in r['contrast']])
        csv_write('OUTER_EARLY_LATE_MOVEMENT_RESULTS.csv',[r for r in outer_a if r['mode']=='EARLY'])
        # Cross topology stays provisional unless independently repeated at the
        # very same TRAIN-chosen horizon and its destination has B support.
        cross=[]
        for fold in range(5):
            root=BASE/'outer_cross_evaluation'/f'outer_{fold}';done=read_json(root/'DONE.json');assert done['status']=='COMPLETE'
            for member in done['modes']:
                path=root/member['mode']/'DONE.json';assert digest(path)==member['sha256'];cross.extend(read_json(path)['records'])
        edge_checks=[]
        for destination,parent,horizon in sorted(set((r['edge']['destination'],r['edge']['parent'],r['TRAIN_selected_horizon']) for r in cross)):
            members=[r for r in cross if r['edge']['destination']==destination and r['edge']['parent']==parent and r['TRAIN_selected_horizon']==horizon]
            good={mode:[r['fold'] for r in members if r['mode']==mode and r['passes_frozen_horizon']] for mode in ['NORMAL','EARLY']}
            edge_checks.append(dict(destination=destination,parent=parent,horizon=horizon,passing_folds=good,
                certified=destination in certified_features and all(len(v)>=4 for v in good.values()),causal_claim=False))
        edges=[r for r in edge_checks if r['certified']];assert len(edges)<=8
        graph=dict(status='COMPLETE',timestamp=utc_now(),cross_edge_count=len(edges),certified_edges=edges,edge_checks=edge_checks,
            provisional_outer_checks=len(cross),self_transition_models_separate=True,no_invented_edges=True,causal_parent_claim=False)
        atomic(BASE/'PREDICTIVE_FACTOR_GRAPH_NEXT9.json',graph)
        cross_rows=[]
        for r in cross:
            for name,c in r['contrasts'].items():cross_rows.append(dict(fold=r['fold'],mode=r['mode'],**r['edge'],contrast=name,**c,passes_frozen_horizon=r['passes_frozen_horizon']))
        if not cross_rows:cross_rows=[dict(status='ZERO_TRAIN_SELECTED_CROSS_EDGES',edge_count=0,criterion_relaxed=False)]
        csv_write('CROSS_FACTOR_RESULTS.csv',cross_rows)
        assert not violations
        atomic(BASE/'SCIENTIFIC_RESULTS_NEXT9.json',dict(status='COMPLETE',timestamp=utc_now(),
            A=dict(eligible_outerTRAIN_candidates=cert['A_candidates_in_atleast4outerTRAIN'],claim='Auxiliary predictability, not cause or target gain'),
            B=dict(certified_variable_count=len(accepted),status='MOVEMENT_CONDITIONAL_VARIABLE_CERTIFIED' if accepted else 'NO_MOVEMENT_VARIABLE_FEATURE',absence_not_proven=True),
            C=dict(M7_vs_BL80=model_gate,promotion_score_gates=promotion),cross_edge_count=len(edges),
            canonical_primary='BL80',canonical_robustness='D2',canonical_distilled='J102',canonical_roles_changed=False,
            sources_sha256=sources,implementation_sha256=digest(BASE/'SCIENCE_AGGREGATION_LOCK_NEXT9.json'),
            global_freeze_sha256=digest(BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json'),no_model_or_candidate_modification=True))


if __name__=='__main__':main()
