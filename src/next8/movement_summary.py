"""Cross-fold A evidence, with mechanical controls and matched edge removal."""
from collections import defaultdict
import numpy as np
from src.next8.common import BASE,read_json,digest
from src.next8.features import DETAILED,HORIZONS,NAMES
from src.next8.movement import cross_eligible
from src.next8.statistics import holm
from src.next8.diagnostics import equal_fold_loss_summary


def collect_comparison(panels,reference,candidate,*,domain='ALL',target='delta',metric='MSE',informative_only=False,memo=None):
    """Keep every supported matching fold, including a contrary fifth fold."""
    cache_key=(reference,candidate,domain,target,metric,informative_only)
    if memo is not None and cache_key in memo:return memo[cache_key]
    parts=[];folds=[];missing=[];uninformative=[]
    for fold,record,arrays in panels:
        matches=[r for r in record['comparisons'] if r['reference']==reference and r['candidate']==candidate and
                 r['domain']==domain and r['target']==target and r['metric']==metric]
        assert len(matches)<=1
        if not matches:missing.append(fold);continue
        item=matches[0]
        if informative_only and not item['informative_random_contrast']:
            uninformative.append(fold);continue
        values=arrays[item['group_array_key']+'_gain']
        keys=arrays[item['group_array_key']+'_group']
        assert len(values)==len(keys)==len(np.unique(keys)) and np.isfinite(values).all()
        if len(values)<64:missing.append(fold);continue
        folds.append(fold);parts.append(values)
    result=equal_fold_loss_summary(parts)
    result={**result,'reference':reference,'candidate':candidate,'domain':domain,'target':target,'metric':metric,
        'included_folds':folds,'missing_or_sparse_folds':missing,'uninformative_random_folds':uninformative,
        'all_matching_supported_folds_included':True,'opposite_fifth_fold_not_dropped':True}
    if memo is not None:memo[cache_key]=result
    return result


def supported_improvement(result,*,minimum_folds=4):
    return bool(result['status']=='SUPPORTED' and result['folds']>=minimum_folds and
                result['mean_gain']>0 and result['ci_low']>0 and result['positive_folds']>=4)


def component_p(result):
    return result['p_normal_cluster'] if supported_improvement(result) else 1.


def shared_edge_domains(panels,parent):
    """Replicated positive matched gains in fixed time/H bins; no new cutpoints."""
    by_domain=defaultdict(list)
    for fold,record,arrays in panels:
        refs=defaultdict(dict)
        for r in record['comparisons']:
            if r['domain_family']=='ALL' or r['target']!='delta' or r['metric']!='MSE':continue
            if r['candidate']!='D5_CROSS_delta':continue
            if r['reference'] not in ('D5_delta',f'D5_REMOVE_F{parent:03d}_delta'):continue
            refs[(r['domain_family'],r['domain'])][r['reference']]=r
        for key,pair in refs.items():
            if len(pair)==2 and all(r['status']=='SUPPORTED' and r['gain'] is not None and r['gain']>0 for r in pair.values()):
                by_domain[key].append(fold)
    rows=[{'family':k[0],'domain':k[1],'positive_matched_folds':v,'replicated':len(v)>=4} for k,v in sorted(by_domain.items())]
    counts={family:sum(r['family']==family and r['replicated'] for r in rows) for family in ('TIME','H_LENGTH')}
    return {'passed':all(n>=2 for n in counts.values()),'replicated_domain_counts':counts,'domains':rows,
        'scope':'Positive supported-domain replication; global intervals use all supported matching folds'}


def edge_evidence(panels,parent,train_edges,*,memo=None):
    own=collect_comparison(panels,'D1_delta','D5_CROSS_delta',memo=memo)
    mechanical=collect_comparison(panels,'D5_delta','D5_CROSS_delta',memo=memo)
    removed=collect_comparison(panels,f'D5_REMOVE_F{parent:03d}_delta','D5_CROSS_delta',memo=memo)
    domains=shared_edge_domains(panels,parent)
    folds=sorted(train_edges)
    passed=len(folds)>=4 and domains['passed'] and all(supported_improvement(r) for r in (own,mechanical,removed))
    return {'TRAIN_selected_folds':folds,'TRAIN_selection_count':len(folds),
        'mean_inner_selection_score':float(np.mean(list(train_edges.values()))) if train_edges else None,
        'own_control':own,'mechanical_control':mechanical,'matched_parent_removal':removed,
        'time_H_replication':domains,'cross_support_before_Holm':bool(passed),
        'p_value':max(component_p(r) for r in (own,mechanical,removed)) if passed else 1.,
        'causal':False,'common_raw':True,'all_possible_features_exhausted':False}


def transition_evidence(panels,*,memo=None):
    controls=[]
    for name in ('D0_state_persistence','D0_state_majority','T1_state','T2_state','T4_state'):
        controls.append(collect_comparison(panels,name,'T3_state',target='state',metric='LOGLOSS',informative_only=name=='T4_state',memo=memo))
    passed=all(supported_improvement(r) for r in controls)
    random=controls[-1]
    return {'controls':controls,'graph_value_before_Holm':passed,
        'p_value':max(component_p(r) for r in controls) if passed else 1.,
        'random_control_assessable':len(random['included_folds'])>=4,
        'T2_comparison_passed':supported_improvement(controls[-2]),
        'target':'Future physical F H-state, proper log loss, known H calibration only',
        'full_graph_value_requires_both_persistence_and_majority_and_own_and_tabular_and_nonidentical_random':True,
        'point_state_accuracy_not_substituted_for_proper_loss':True}


def load_forecast_panels(early=False):
    result=defaultdict(list);graphs=[]
    release=read_json(BASE/'PREDICTION_FREEZE_NEXT8.json');assert release['status']=='FROZEN'
    for fold in range(5):
        scope=f'outer_{fold}'+('_EARLY_T199' if early else '')
        directory=BASE/'forecast_evaluation'/scope;done=read_json(directory/'DONE.json')
        assert done['feature_horizon_pairs']==len(DETAILED)*len(HORIZONS)
        path=BASE/'parent_selection'/scope/'GRAPH.json'
        assert digest(path)==release['files_sha256'][path.relative_to(BASE).as_posix()]
        graphs.append(read_json(path))
        for j in DETAILED:
            for h in HORIZONS:
                p=directory/f'F{j:03d}_h{h}'
                assert digest(p/'DONE.json')==done['files_sha256'][p.name+'/DONE.json']
                record=read_json(p/'DONE.json');assert record['scope']==scope
                assert digest(p/'GROUP_GAINS.npz')==record['group_gain_sha256']
                with np.load(p/'GROUP_GAINS.npz',allow_pickle=False) as a:arrays={k:a[k].copy() for k in a.files}
                result[(j,h)].append((fold,record,arrays))
    return result,graphs


def summarize_movement(early=False):
    all_panels,graphs=load_forecast_panels(early);own=[];edges=[];transitions=[];ablations=[];metrics=[]
    for (j,h),panels in sorted(all_panels.items()):
        base={'feature':j,'name':NAMES[j],'horizon':h,'early_only':early}
        memo={};primary=collect_comparison(panels,'D0_delta','D1_delta',memo=memo)
        own.append({**base,**primary,'self_support_before_Holm':supported_improvement(primary,minimum_folds=5),
            'p_value':component_p(primary),'mechanical_self_predictability_is_not_a_new_cause':True})
        transition=transition_evidence(panels,memo=memo);transitions.append({**base,**transition})
        comparisons=set()
        for fold,record,arrays in panels:
            metrics.extend({**base,'outer_fold':fold,**r} for r in record['metrics'])
            comparisons.update((r['domain'],r['reference'],r['candidate'],r['target'],r['metric']) for r in record['comparisons'])
        for domain,reference,candidate,target,metric in sorted(comparisons):
            evidence=collect_comparison(panels,reference,candidate,domain=domain,target=target,metric=metric,
                informative_only=reference=='T4_state',memo=memo)
            ablations.append({**base,**evidence,'descriptive_family_not_used_to_choose_horizon':True})
        for parent in DETAILED:
            if not cross_eligible(parent,j):continue
            training={}
            for fold,graph in enumerate(graphs):
                matching=[e for e in graph['edges'] if e['source']==parent and e['destination']==j and
                          any(r['horizon']==h and r['supported'] for r in e['horizons'])]
                assert len(matching)<=1
                if matching:training[fold]=matching[0]['selection_score']
            edges.append({**base,'source':parent,'destination':j,**edge_evidence(panels,parent,training,memo=memo)})
    for rows,before,after in ((own,'self_support_before_Holm','self_dynamics_supported'),
                              (edges,'cross_support_before_Holm','cross_factor_supported'),
                              (transitions,'graph_value_before_Holm','factorized_transition_value_supported')):
        for r,p in zip(rows,holm([r['p_value'] for r in rows])):
            r['Holm_p']=float(p);r[after]=bool(r[before] and p<.05)
    # Summary graph is a TRAIN-ranked description, never a refitted pooled graph.
    edge_units=defaultdict(list)
    for e in edges:edge_units[(e['source'],e['destination'])].append(e)
    eligible=[]
    for (source,destination),rows in edge_units.items():
        good=[r for r in rows if r['cross_factor_supported']]
        if not good:continue
        eligible.append({'source':source,'destination':destination,'horizons':[r['horizon'] for r in good],
            'TRAIN_selection_frequency':max(r['TRAIN_selection_count'] for r in good),
            'mean_INNER_priority':max(r['mean_inner_selection_score'] for r in good),'common_raw':True,'causal':False,
            'edge_type':'CROSS_PREDICTIVE_VALIDATED_WITHIN_FIXED_MENU','is_deployable_pooled_model':False})
    eligible.sort(key=lambda r:(-r['TRAIN_selection_frequency'],-r['mean_INNER_priority'],r['destination'],r['source']))
    retained=[];counts=defaultdict(int)
    for edge in eligible:
        if len(retained)>=12:break
        if counts[edge['destination']]>=2:continue
        retained.append(edge);counts[edge['destination']]+=1
    return {'status':'COMPLETE','early_only':early,'own_dynamics':own,'cross_edges':edges,
        'transition_tests':transitions,'factor_ablations':ablations,'fold_metrics':metrics,'summary_graph_edges':retained,
        'counts':{'self_supported_feature_horizons':sum(r['self_dynamics_supported'] for r in own),
            'self_supported_features':len({r['feature'] for r in own if r['self_dynamics_supported']}),
            'cross_supported_edge_horizons':sum(r['cross_factor_supported'] for r in edges),
            'cross_supported_features':len({r['destination'] for r in edges if r['cross_factor_supported']}),
            'retained_cross_edges':len(retained),'factorized_transition_supported_feature_horizons':sum(r['factorized_transition_value_supported'] for r in transitions)},
        'multiplicity':{'own_family':len(own),'all_eligible_edge_horizon_family':len(edges),'transition_family':len(transitions),
            'method':'Separate Holm families; edge and graph claims use maximum component p (intersection-union), unsupported/unselected=1'},
        'A_does_not_certify_B_or_C':True,'frozen_per_fold_graphs_not_modified':True,
        'mechanical_controls_limit':'Named shared-primitive bank, not exhaustive latent filter sufficient statistics',
        'early_replication_kept_separate':True,'all_horizons_retained':list(HORIZONS)}


def summarize_observability(early=False):
    result=[]
    for j in DETAILED:
        records=[];parts=defaultdict(list);fold_metrics=[]
        for fold in range(5):
            scope=f'outer_{fold}'+('_EARLY_T199' if early else '')
            directory=BASE/'state_observability'/scope;done=read_json(directory/'DONE.json');p=directory/f'F{j:03d}'
            assert digest(p/'DONE.json')==done['files_sha256'][p.name+'/DONE.json']
            record=read_json(p/'DONE.json');assert digest(p/'GROUP_GAINS.npz')==record['group_gain_sha256']
            with np.load(p/'GROUP_GAINS.npz',allow_pickle=False) as a:
                for r in record['comparisons']:parts[r['reference']].append(a[r['group_array_key']+'_GAIN'].copy())
            records.append(record);fold_metrics.extend({'outer_fold':fold,**r} for r in record['metrics'])
        evidence={name:equal_fold_loss_summary(values) for name,values in parts.items()}
        result.append({'feature':j,'name':NAMES[j],'early_only':early,'comparisons':evidence,'fold_metrics':fold_metrics,
            'ridge_current_F_error_by_fold':[r['ridge_current_F_error'] for r in records],
            'same_actual_M5_state_path':True,'additional_origins_diagnostic_not_a_new_graph_selection_test':True,
            'B_direction_reproduction_not_implied_by_state_accuracy':True})
    return {'status':'COMPLETE','early_only':early,'features':result,'state_estimator':'Actual frozen T3 h1 M5 distribution/abstention',
        'H_constant_context_not_mislabelled_as_dynamic_F_state':True,'fit_or_selection_after_outer_results':False}
