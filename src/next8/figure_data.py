"""Actual heldout timeline, transition tables and canonical same-time curves."""
import hashlib
import numpy as np
import joblib
from src.next8.common import BASE,read_json,digest
from src.next8.features import DETAILED,NAMES,NORMALIZATION,movement_direction
from src.next8.data import outer_scope,rows_for
from src.next8.nested_auxiliary import early_scope
from src.next8.evaluation import frozen_predictions
from src.next8.joint_models import orientation_states,preselected_linear_rule,raw_coefficients
from src.scoring.ts_auc import ts_auc


def same_time_curves(cache):
    result=[];checks=[]
    for early in (False,True):
        for fold in range(5):
            scope=early_scope(outer_scope(fold,cache.data),early)
            rows,predictions,provenance=frozen_predictions(cache,scope,fold,early=early)
            authority=read_json(BASE/'model_evaluation'/scope.name/'ALL/DONE.json')['points']
            y=cache.meta['target'][rows];times=cache.meta['time_online'][rows];seen={}
            for method,values in predictions.items():
                array=np.asarray(values,dtype=np.float64);sha=hashlib.sha256(array.tobytes()).hexdigest()
                if sha in seen:score,details=seen[sha]
                else:
                    score,details=ts_auc(y,array,times,return_details=True);seen[sha]=(score,details)
                pairs=sum(r['weight'] for r in details);expected=authority[method]['TS_AUC']
                assert (expected is None and pairs==0) or (expected is not None and abs(score-expected)<1e-12)
                checks.append({'scope':scope.name,'method':method,'canonical_mean':score if pairs else None,
                    'frozen_score_authority':expected,'parity_tolerance':1e-12,'pairs':pairs})
                result.extend({'early_only':early,'fold':fold,'method':method,**r,
                    'pointwise_descriptive_no_independent_time_samples':True} for r in details)
    return result,checks


def heldout_timeline(cache):
    scope=outer_scope(0,cache.data);sid=min(scope.validation_ids)
    rows=rows_for(cache.meta,[sid]);rows=rows[:512]
    selection=read_json(BASE/'h_context_selection'/scope.name/'SELECTION.json')
    j=min(selection['variable_features']) if selection['variable_features'] else 0
    current=cache.current(rows,[j])[:,0];previous=cache.lag(rows,[j],1)[:,0]
    s=int(cache.series_index[rows[0]]);graph_path=BASE/'graph_states'/scope.name/'STATE_PREDICTIONS.npz'
    release=read_json(BASE/'PREDICTION_FREEZE_NEXT8.json')
    assert digest(graph_path)==release['files_sha256'][graph_path.relative_to(BASE).as_posix()]
    with np.load(graph_path,allow_pickle=False) as a:
        at=np.searchsorted(a['global_row'],rows);np.testing.assert_array_equal(a['global_row'][at],rows)
        graph=a['state'][at,DETAILED.index(j)].copy()
    beta=[None]*len(rows);fixed={};actual=None;condition=cache.states(rows,[j],previous=True)[:,0]
    if j in selection['variable_features']:
        md=read_json(BASE/'models'/scope.name/'DONE.json');actual=md['model_aliases']['M5']
        mp=BASE/'models'/scope.name/(actual+'.joblib');assert digest(mp)==release['files_sha256'][mp.relative_to(BASE).as_posix()]
        model=joblib.load(mp);condition=orientation_states(cache,rows,j,selection)
        weights=dict(zip(model.names,raw_coefficients(model)[0]));prefix=f'F{j:03d}_'
        fixed={key:float(weights.get(prefix+'variable_'+key,0.)) for key in ('positive','negative','square')}
        if actual in ('M4','M5'):beta=[preselected_linear_rule(model,j,int(a),int(b)) for a,b in zip(condition,graph)]
    return {'status':'COMPLETE','scope':scope.name,'series_id':sid,'feature':j,'name':NAMES[j],
        'TRAIN_selected_variable':j in selection['variable_features'],'selection_rule':'First numeric outer0 heldout ID; lowest TRAIN variable or preregistered F000 fallback',
        'time':cache.meta['time_online'][rows].tolist(),'F':current.tolist(),'value_sign':np.sign(current).astype(int).tolist(),
        'observed_movement_direction':movement_direction(current-previous,cache.epsilon(rows,[j],1)[:,0]).tolist(),
        'physical_F_H_state':cache.states(rows,[j])[:,0].tolist(),'own_or_H_condition_known_before_point':condition.tolist(),
        'predicted_graph_state_known_before_point':graph.tolist(),'linear_beta_chosen_before_point':beta,
        'fixed_nonlinear_weights':fixed,'actual_model':actual,'normalization':NORMALIZATION,
        'H_length':int(cache.series['H_length'][s]),'physical_F_H_low':float(cache.series['low'][s,j]),
        'physical_F_H_high':float(cache.series['high'][s,j]),'physical_H_state_valid':bool(cache.series['valid'][s,j]),
        'full_derivative_not_claimed_when_fixed_nonlinear_terms_exist':True,'observed_movement_uses_past_h1_not_future_target':True,
        'all_feature_families_share_raw_series':True,'raw_H_Online_values_exported':False}


def transition_illustration():
    path=BASE/'transitions/outer_0/F000_h1/T3.joblib';release=read_json(BASE/'PREDICTION_FREEZE_NEXT8.json')
    assert digest(path)==release['files_sha256'][path.relative_to(BASE).as_posix()];model=joblib.load(path)
    return {'status':'COMPLETE','scope':'outer_0','feature':0,'horizon':1,'source_model_sha256':digest(path),
        'own_transition_probability':model.own_table.tolist(),'own_training_group_support':model.own_support.tolist(),
        'training_prior':model.prior.tolist(),'parents':list(model.parents),'parent_conditional_probability':[p.tolist() for p in model.pair_tables],
        'parent_context_training_group_support':[p.tolist() for p in model.pair_support],
        'training':model.training,'table_is_TRAIN_estimate_not_heldout_accuracy':True,
        'uncertainty_source':'movement_summary/NORMAL.json transition_tests F000 h1; original-group heldout proper-loss intervals',
        'empty_parent_set_is_not_an_observed_causal_graph':True}
