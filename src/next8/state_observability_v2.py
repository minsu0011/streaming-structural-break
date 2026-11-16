"""State observability v2: unissued UNKNOWN is missing, not a wrong forecast.

Preserves every frozen predictor and the separate D/T certification metrics.
Proper-loss comparisons condition on both methods issuing a known-state
distribution. Their observable coverage is reported and is not full coverage.
"""
import numpy as np
import joblib
from src.next8.common import BASE,read_json,digest
from src.next8.features import DETAILED,UNKNOWN
from src.next8.forecasting import _persistence_distribution
from src.next8.movement import mechanical_design
from src.next8.transitions import transition_inputs,transition_tabular_design
from src.next8.statistics import classification_metrics,multiclass_losses,regression_metrics,series_weights
from src.next8.diagnostics import original_groups,time_H_domains,loss_evidence


def known_prediction_issued(probability):
    probability=np.asarray(probability)
    if probability.ndim!=2 or probability.shape[1]!=4 or not np.isfinite(probability).all():
        raise ValueError('Finite four-state probability matrix required')
    if np.any(probability<0) or not np.allclose(probability.sum(axis=1),1.,rtol=0.,atol=1e-6):
        raise ValueError('Valid probability distribution required')
    # Producers use [0,0,0,1] for unavailable/abstaining known-state forecasts.
    # A physical-state distribution can still be issued when its hard
    # confidence decision abstains; that distribution remains evaluable.
    return probability[:,:3].sum(axis=1)>0


def paired_issued_loss(groups,target,reference,candidate,domain):
    mask=np.asarray(domain)&known_prediction_issued(reference)&known_prediction_issued(candidate)
    evidence,keys,delta=loss_evidence(np.asarray(groups)[mask],
        multiclass_losses(np.asarray(target)[mask],np.asarray(reference)[mask]),
        multiclass_losses(np.asarray(target)[mask],np.asarray(candidate)[mask]))
    return {**evidence,'available_domain_rows':int(np.sum(domain)),
        'paired_issued_rows':int(mask.sum()),
        'row_paired_issued_coverage':float(mask.sum()/np.sum(domain)) if np.any(domain) else None,
        'proper_loss_conditional_on_both_observable_forecasts_issued':True,
        'not_an_unconditional_population_or_abstention_penalty_metric':True},keys,delta


def replay_current_comparators(cache,rows,j,model,tree,parents,stay,prior):
    rows=np.asarray(rows);own=cache.states(rows,[j],previous=True)[:,0];known=own<UNKNOWN
    output={'T0_PERSISTENCE':np.zeros((len(rows),4),dtype=np.float32),'T0_MAJORITY':np.zeros((len(rows),4),dtype=np.float32)}
    output['T0_PERSISTENCE'][known,:3]=_persistence_distribution(own[known],stay,3)
    output['T0_MAJORITY'][known,:3]=prior
    for name in output:output[name][~known,3]=1.
    for name in ('T1_OWN_TABLE','T2_SAME_INPUT_TREE'):output[name]=np.tile([0.,0.,0.,1.],(len(rows),1)).astype(np.float32)
    issued=np.flatnonzero(cache.meta['time_online'][rows]>=1)
    for start in range(0,len(issued),50000):
        at=issued[start:start+50000];origin=rows[at]-1
        np.testing.assert_array_equal(cache.meta['dataset_id'][origin],cache.meta['dataset_id'][rows[at]])
        current,parent=transition_inputs(cache,origin,j,parents)
        output['T1_OWN_TABLE'][at]=model.predict(current,parent,use_parents=False)[0]
        if tree is not None:
            design=transition_tabular_design(cache,origin,j,parents,mechanical_design(cache,origin,j,1,parents))
            p=tree.predict_proba(design);valid=cache.series['valid'][cache.series_index[origin],j]
            full=np.zeros((len(origin),4),dtype=np.float32);full[valid,:p.shape[1]]=p[valid];full[~valid,3]=1.
            output['T2_SAME_INPUT_TREE'][at]=full
    return output


def load_state_archives(cache,scope,release):
    gp=BASE/'graph_states'/scope.name;rp=BASE/'state_forecasts'/scope.name
    archives=[]
    for p in (gp,rp):
        assert read_json(p/'DONE.json')['scope']==scope.provenance()
        source=p/'STATE_PREDICTIONS.npz';assert digest(source)==release['files_sha256'][source.relative_to(BASE).as_posix()]
        with np.load(source,allow_pickle=False) as a:archives.append({k:a[k].copy() for k in a.files})
        current=archives[-1]
        np.testing.assert_array_equal(current['feature'],DETAILED)
        np.testing.assert_array_equal(cache.meta['dataset_id'][current['global_row']],current['dataset_id'])
        assert set(map(int,np.unique(current['dataset_id'])))==scope.validation_ids
        assert np.all(np.diff(current['global_row'])>0)
    return archives


def evaluate_state_feature(cache,scope,j,release,*,early=False,state_archives=None):
    archives=load_state_archives(cache,scope,release) if state_archives is None else state_archives
    graph,ridge=archives;np.testing.assert_array_equal(graph['global_row'],ridge['global_row'])
    k=DETAILED.index(j);rows=graph['global_row'];keep=np.ones(len(rows),dtype=bool)
    if early:keep &= cache.meta['time_online'][rows]>=400
    rows=rows[keep];scope.admit_validation(np.unique(cache.meta['dataset_id'][rows]))
    fp=BASE/'forecasts'/scope.name/f'F{j:03d}_h1';fd=read_json(fp/'DONE.json')
    tp=BASE/'transitions'/scope.name/f'F{j:03d}_h1';td=read_json(tp/'DONE.json')
    model_path=tp/'T3.joblib';assert digest(model_path)==release['files_sha256'][model_path.relative_to(BASE).as_posix()]
    model=joblib.load(model_path)
    tree_path=fp/'T2_state.joblib';tree=None
    if tree_path.exists():
        assert digest(tree_path)==release['files_sha256'][tree_path.relative_to(BASE).as_posix()];tree=joblib.load(tree_path)
    pp=fp/'PREDICTIONS.npz';assert digest(pp)==release['files_sha256'][pp.relative_to(BASE).as_posix()]
    with np.load(pp,allow_pickle=False) as a:
        majority=a['D0_state_majority'];valid=np.flatnonzero(majority[:,3]==0)
        prior=majority[valid[0],:3].copy() if len(valid) else np.full(3,1/3)
        if len(valid):assert np.all(majority[valid,:3]==prior)
    predictions=replay_current_comparators(cache,rows,j,model,tree,fd['parents'],fd['D0_policy']['state_persistence_train_probability'],prior)
    predictions['T3_ACTUAL_CROSS_DISTRIBUTION']=graph['probability'][keep,k]
    state=cache.states(rows,[j])[:,0];groups=original_groups(cache,rows);t=cache.meta['time_online'][rows]
    decisions={'T3_CONFIDENT_CROSS_CONDITION':graph['state'][keep,k],'RIDGE_H1_CONDITION':ridge['state'][keep,k]}
    metrics=[];comparisons=[];arrays={}
    for family,domain,mask in time_H_domains(cache,rows):
        for target_domain,sub in (('KNOWN_H_T_GE_1',mask&(state<UNKNOWN)&(t>=1)),('ALL_FOUR_STATES',mask)):
            for name,p in predictions.items():
                common={'feature':j,'scope':scope.name,'domain_family':family,'domain':domain,'target_domain':target_domain,'method':name}
                evaluated=sub&known_prediction_issued(p)
                coverage={'available_domain_rows':int(sub.sum()),'issued_rows':int(evaluated.sum()),
                    'row_issued_probability_coverage':float(evaluated.sum()/sub.sum()) if sub.any() else None,
                    'probability_metrics_conditional_on_known_state_forecast_issued':True,
                    'unissued_UNKNOWN_not_scored_as_a_confident_wrong_physical_state':True}
                if evaluated.any():
                    metric=classification_metrics(state[evaluated],p[evaluated],groups[evaluated]);metrics.append({**common,**coverage,**metric,
                        'status':'SUPPORTED' if metric['independent_series']>=64 else 'INSUFFICIENT_SUPPORT'})
                else:metrics.append({**common,**coverage,'status':'INSUFFICIENT_SUPPORT','rows':0,
                    'reason':'NO_ISSUED_KNOWN_STATE_FORECAST_OR_NO_DOMAIN_ROWS',
                    'balanced_accuracy':None,'logloss':None,'Brier':None})
        known=mask&(state<UNKNOWN)&(t>=1)
        weights=np.zeros(len(rows))
        if known.any():weights[known]=series_weights(groups[known])
        for name,decision in decisions.items():
            emitted=known&(decision<UNKNOWN);recalls=[]
            for s in range(3):
                target=known&(state==s);recalls.append(float(np.average(decision[target]==s,weights=weights[target])) if target.any() else None)
            metrics.append({'feature':j,'scope':scope.name,'domain_family':family,'domain':domain,'method':name,
                'target_domain':'KNOWN_H_T_GE_1','rows':int(known.sum()),'issued_rows':int(emitted.sum()),
                'issued_coverage':float(weights[emitted].sum()/weights[known].sum()) if known.any() else None,
                'row_issued_coverage':float(emitted.sum()/known.sum()) if known.any() else None,
                'selective_accuracy_when_issued':float(np.average(decision[emitted]==state[emitted],weights=weights[emitted])) if emitted.any() else None,
                'selective_metric_weights':'Each original group has equal weight over its known-domain rows before selective conditioning',
                'recall_with_abstentions_not_counted_correct':recalls,
                'unknown_abstention_not_an_observed_state_confirmation':True,
                'status':'SUPPORTED' if len(np.unique(groups[emitted]))>=64 else 'INSUFFICIENT_SUPPORT'})
        if family=='ALL':
            for name in ('T0_PERSISTENCE','T0_MAJORITY','T1_OWN_TABLE','T2_SAME_INPUT_TREE'):
                evidence,keys,delta=paired_issued_loss(groups,state,predictions[name],
                    predictions['T3_ACTUAL_CROSS_DISTRIBUTION'],known)
                arrays[name+'_GROUP']=keys;arrays[name+'_GAIN']=delta
                comparisons.append({'reference':name,'candidate':'T3_ACTUAL_CROSS_DISTRIBUTION',**evidence,'group_array_key':name})
    estimate=ridge['predicted_feature'][keep,k];available=np.isfinite(estimate)&(state<UNKNOWN)&(t>=1)
    continuous=regression_metrics(cache.current(rows,[j])[:,0][available],estimate[available],groups[available]) if available.any() else {'status':'NO_ISSUED_RIDGE_FORECAST'}
    return {'status':'COMPLETE','scope':scope.provenance(),'feature':j,'metrics':metrics,'comparisons':comparisons,
        'ridge_current_F_error':continuous,'same_physical_H_target':True,'break_labels_used':False,
        'T3_probability_can_exist_while_discrete_condition_abstains':True,
        'T3_no_parent_is_UNKNOWN_in_actual_M5_even_if_own_table_can_forecast':True,
        'comparators_replayed_from_frozen_models_no_refit':True,
        'additional_origin_diagnostic_not_used_for_candidate_or_graph_selection':True,
        'diagnostic_version':'STATE_OBSERVABILITY_V2_UNISSUED_FORECAST',
        'unissued_known_state_probability_is_missing_not_false_confident_error':True,
        'probability_losses_condition_on_issued_distribution_and_report_coverage':True,
        'D0_D5_T0_T4_primary_forecast_certification_and_M0_M6_quality_unchanged':True,
        'H_background_condition_not_compared_as_if_it_were_physical_F_state':True},arrays
