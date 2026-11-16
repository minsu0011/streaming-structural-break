"""D0-D5 forecast comparators. Outer predictions freeze before their scoring."""
from pathlib import Path
import numpy as np
import joblib
from src.next8.common import BASE,atomic,read_json,digest,utc_now,activity,ids_hash
from src.next8.cache import immutable_npz
from src.next8.features import DETAILED,HORIZONS,UNKNOWN,movement_direction,categorize
from src.next8.learning import Design,fit_ridge,fit_logistic,fit_tree
from src.next8.movement import (own_design,augment,mechanical_design,movement_target,forecast_delta)
from src.next8.statistics import series_weights
from src.next8.screening import sampled_rows


def subset(design,keep):
    return Design(design.values[keep],design.names,design.ids[keep])


def fit_bundle(design,target,scope):
    weights=series_weights(design.ids)
    bundle={'delta':fit_ridge(design,target['delta'],scope,weights),
            'movement':fit_logistic(design,target['movement'],scope,weights,classes=3),
            'reversal':fit_logistic(design,target['reversal'],scope,weights,classes=2)}
    known=target['state']<UNKNOWN
    bundle['state']=fit_logistic(subset(design,known),target['state'][known],scope,
                    series_weights(design.ids[known]),classes=3) if np.any(known) else None
    for name,model in bundle.items():
        if model is not None:
            actual=design.ids[known] if name=='state' else design.ids
            model.training.update({'actual_training_ids_sha256':ids_hash(np.unique(actual)),
                'actual_training_series':len(np.unique(actual)),'actual_training_rows':len(actual),
                'target_kind':'AUXILIARY_FUTURE_FEATURE_'+name,'breakOccurred_used':False,
                'future_target_never_input':True})
    return bundle


def predict_bundle(bundle,design,current,H_valid):
    out={'delta':forecast_delta(bundle['delta'],design,current).astype(np.float32),
         'movement':bundle['movement'].predict_proba(design).astype(np.float32),
         'reversal':bundle['reversal'].predict_proba(design).astype(np.float32)}
    state=np.zeros((len(design.ids),4),dtype=np.float32)
    if bundle['state'] is not None:
        p=bundle['state'].predict_proba(design)
        state[np.asarray(H_valid),:3]=p[np.asarray(H_valid)]
    else:H_valid=np.zeros(len(design.ids),dtype=bool)
    state[~np.asarray(H_valid),3]=1.
    out['state']=state
    return out


def _prior(y,ids,classes):
    w=series_weights(ids);counts=np.bincount(y,weights=w,minlength=classes)
    return (counts+.5)/(counts.sum()+.5*classes)


def _persistent_probability(prediction,target,ids,classes):
    w=series_weights(ids)
    # Fixed train-only calibration keeps persistence the argmax. It is only
    # one D0 control; the empirical majority-probability control is also kept.
    stay=float(np.clip(np.average(prediction==target,weights=w),1/classes+.001,.99))
    return stay


def _persistence_distribution(prediction,stay,classes):
    p=np.full((len(prediction),classes),(1-stay)/(classes-1),dtype=np.float32)
    p[np.arange(len(prediction)),prediction]=stay
    return p


def d0_predictions(cache,tr,va,j,horizon,targets):
    tids=cache.meta['dataset_id'][tr];n=len(va)
    now=cache.current(tr,[j])[:,0];old=cache.lag(tr,[j],1)[:,0]
    training_direction=movement_direction(now-old,cache.epsilon(tr,[j],1)[:,0])
    vnow=cache.current(va,[j])[:,0];vold=cache.lag(va,[j],1)[:,0]
    validation_direction=movement_direction(vnow-vold,cache.epsilon(va,[j],1)[:,0])
    stay=_persistent_probability(training_direction,targets['movement'],tids,3)
    current_train=cache.states(tr,[j])[:,0];current_valid=cache.states(va,[j])[:,0]
    known=targets['state']<UNKNOWN
    state_prior=_prior(targets['state'][known],tids[known],3) if np.any(known) else np.full(3,1/3)
    state_stay=_persistent_probability(current_train[known],targets['state'][known],tids[known],3) if np.any(known) else 1/3+.001
    state_majority=np.zeros((n,4),dtype=np.float32);state_persist=state_majority.copy()
    vk=current_valid<UNKNOWN
    state_majority[vk,:3]=state_prior
    state_persist[vk,:3]=_persistence_distribution(current_valid[vk],state_stay,3)
    state_majority[~vk,3]=1.;state_persist[~vk,3]=1.
    return ({'D0_delta':np.zeros(n,dtype=np.float32),
        'D0_movement_majority':np.tile(_prior(targets['movement'],tids,3),(n,1)).astype(np.float32),
        'D0_movement_persistence':_persistence_distribution(validation_direction,stay,3),
        'D0_state_majority':state_majority,'D0_state_persistence':state_persist,
        'D0_reversal':np.tile(_prior(targets['reversal'],tids,2),(n,1)).astype(np.float32)},
        {'movement_persistence_train_probability':stay,'state_persistence_train_probability':state_stay,
         'probability_calibration_training_only':True,'reversal_value_sign_persistence_point_prediction':0})


def parents_at(graph,j,horizon):
    return [int(e['source']) for e in graph['edges'] if int(e['destination'])==j and
            any(h['horizon']==horizon and h['supported'] for h in e['horizons'])]


def validate_graph_scope(graph,scope):
    if graph['scope']!=scope.provenance():
        raise ValueError('Parent selection must exclude the complete prediction scope')
    from src.next8.movement import cross_eligible
    edges=graph['edges']
    if len(edges)>12:raise ValueError('Predictive edge budget exceeded')
    pairs=[(int(e['source']),int(e['destination'])) for e in edges]
    if len(set(pairs))!=len(pairs):raise ValueError('Duplicate predictive edge')
    for p,j in pairs:
        if p not in DETAILED or j not in DETAILED or not cross_eligible(p,j):
            raise ValueError('Unsupported or mechanical edge in predictive graph')
        if sum(target==j for _,target in pairs)>2:raise ValueError('Parent budget exceeded')


def predict_cross_state(cache,rows,j,parents,model):
    """Issue F[t] state forecast exclusively from the observed prefix at t-1."""
    rows=np.asarray(rows,dtype=np.int64)
    output=np.full(len(rows),UNKNOWN,dtype=np.int8)
    estimates=np.full(len(rows),np.nan,dtype=np.float32)
    usable=np.flatnonzero(cache.meta['time_online'][rows]>=1)
    for start in range(0,len(usable),50000):
        at=usable[start:start+50000];origin=rows[at]-1
        if not np.array_equal(cache.meta['dataset_id'][origin],cache.meta['dataset_id'][rows[at]]):
            raise ValueError('Cross-series forecast origin')
        x=mechanical_design(cache,origin,j,1,parents)
        now=cache.current(origin,[j])[:,0]
        prediction=now+forecast_delta(model,x,now)
        s=cache.series_index[rows[at]]
        state=categorize(prediction,cache.series['low'][s,j],cache.series['high'][s,j],cache.series['valid'][s,j])
        margin=5.*cache.epsilon(rows[at],[j],1)[:,0]
        ambiguous=(np.abs(prediction-cache.series['low'][s,j])<=margin)|(np.abs(prediction-cache.series['high'][s,j])<=margin)
        state[ambiguous]=UNKNOWN
        output[at]=state;estimates[at]=prediction
    return output,estimates


def forecast_designs(cache,rows,j,horizon,stable,parents,variables):
    d1=own_design(cache,rows,j,horizon)
    d2=own_design(cache,rows,j,horizon,context=True)
    d3=augment(cache,rows,d2,[k for k in stable if k!=j],prefix='stable')
    d4=augment(cache,rows,d3,[p for p in parents if p in variables],prefix='variable')
    d5=mechanical_design(cache,rows,j,horizon)
    result={'D1':d1,'D2':d2,'D3':d3,'D4':d4,'D5':d5,
            'D5_CROSS':mechanical_design(cache,rows,j,horizon,parents=parents)}
    for p in parents:result[f'D5_REMOVE_F{p:03d}']=mechanical_design(cache,rows,j,horizon,parents=[q for q in parents if q!=p])
    return result


def fit_full_comparators(cache,scope,selection,graph,directory,*,early=False):
    """Train on this scope's groups; retain unscored sampled heldout forecasts."""
    validate_graph_scope(graph,scope)
    if selection['scope']!=scope.provenance():raise ValueError('Feature selection scope mismatch')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    cutoff=199 if early else None
    tr_all=sampled_rows(cache,scope.training_ids,cutoff)
    va=sampled_rows(cache,scope.validation_ids)
    if early:va=va[cache.meta['time_online'][va]>=400]
    scope.admit_training(np.unique(cache.meta['dataset_id'][tr_all]))
    scope.admit_validation(np.unique(cache.meta['dataset_id'][va]))
    stable=selection['stable_features']
    records=[]
    for j in DETAILED:
        for horizon in HORIZONS:
            p=directory/f'F{j:03d}_h{horizon}'
            if (p/'DONE.json').exists():
                done=read_json(p/'DONE.json')
                assert done['scope']==scope.provenance()
                for name,sha in done['files_sha256'].items():assert digest(p/name)==sha
                records.append(done);continue
            if p.exists() and any(p.iterdir()):raise RuntimeError('Partial forecast comparator preserved for explicit repair')
            with activity(f'Fit D0-D5 {scope.name} F{j:03d} h{horizon}','MOVEMENT_COMPARATORS','B3_SELF_HISTORY_SUFFICIENCY'):
                p.mkdir(parents=True,exist_ok=False)
                tr=cache.forecast_rows(tr_all,horizon,cutoff)
                target=movement_target(cache,tr,j,horizon)
                parents=parents_at(graph,j,horizon)
                training=forecast_designs(cache,tr,j,horizon,stable,parents,selection['variable_features'])
                validation=forecast_designs(cache,va,j,horizon,stable,parents,selection['variable_features'])
                outputs,d0=d0_predictions(cache,tr,va,j,horizon,target)
                outputs.update({'global_row':va,'dataset_id':cache.meta['dataset_id'][va]})
                bundles={};aliases={};schemas={};files={};fitting=[]
                current=cache.current(va,[j])[:,0]
                valid_H=cache.series['valid'][cache.series_index[va],j]
                for method,design in training.items():
                    key=design.names
                    if key in schemas:
                        aliases[method]=schemas[key];continue
                    schemas[key]=method;aliases[method]=method
                    bundle=fit_bundle(design,target,scope)
                    predictions=predict_bundle(bundle,validation[method],current,valid_H)
                    for name,a in predictions.items():outputs[method+'_'+name]=a
                    model_path=p/(method+'.joblib');joblib.dump(bundle,model_path)
                    files[model_path.name]=digest(model_path);bundles[method]=bundle
                    fitting.append({'method':method,'input_columns':len(design.names),'names':list(design.names),
                        'targets':{k:(m.training if m is not None else {'status':'NO_SUPPORTED_H_STATE'}) for k,m in bundle.items()}})
                    print(utc_now(),'FORECAST FIT',scope.name,j,horizon,method,flush=True)
                # T2 also receives the identical H-calibrated state categories
                # used by the factorized T1/T3 tables, in addition to D5 input.
                from src.next8.transitions import transition_tabular_design
                tree_train=transition_tabular_design(cache,tr,j,parents,training['D5_CROSS'])
                tree_valid=transition_tabular_design(cache,va,j,parents,validation['D5_CROSS'])
                state_known=target['state']<UNKNOWN
                tree=None
                if np.any(state_known):
                    td=subset(tree_train,state_known)
                    labels=target['state'][state_known]
                    tree=fit_tree(td,labels,scope,series_weights(td.ids)) if len(np.unique(labels))>=2 else fit_logistic(td,labels,scope,series_weights(td.ids),classes=3)
                    probability=tree.predict_proba(tree_valid)
                    full=np.zeros((len(va),4),dtype=np.float32)
                    full[valid_H,:probability.shape[1]]=probability[valid_H]
                    full[~valid_H,3]=1.;outputs['T2_state']=full
                    joblib.dump(tree,p/'T2_state.joblib');files['T2_state.joblib']=digest(p/'T2_state.joblib')
                else:outputs['T2_state']=np.tile([0.,0.,0.,1.],(len(va),1)).astype(np.float32)
                immutable_npz(p/'PREDICTIONS.npz',**outputs);files['PREDICTIONS.npz']=digest(p/'PREDICTIONS.npz')
                done={'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),'feature':j,'horizon':horizon,
                    'training_rows':len(tr),'prediction_rows':len(va),'stable_features':stable,'parents':parents,
                    'D4_variable_parents':[p for p in parents if p in selection['variable_features']],
                    'method_aliases':aliases,'D0_policy':d0,'fitting':fitting,'files_sha256':files,
                    'future_predictions_issued_even_without_future_observation':True,
                    'validation_future_targets_decoded_for_scoring':False,'breakOccurred_labels_used':False,
                    'early_training_cutoff':cutoff,'early_auxiliary_targets_cannot_cross_cutoff':True,
                    'T2_input_parity':'D5_CROSS continuous coordinates plus exact own/current and parent/pre-current H-calibrated categories used by T1/T3',
                    'point_predictions_are_forecasts_not_observed_events':True}
                atomic(p/'DONE.json',done);records.append(done)
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),
        'feature_horizon_experiments':len(records),'all_prediction_sha256':[r['files_sha256']['PREDICTIONS.npz'] for r in records],
        'outer_scores_released':False,'breakOccurred_labels_used':False})
    return records


def fit_cross_state_forecasts(cache,scope,graph,directory,*,early=False,reuse_directory=None):
    """h1 ridge for current-state forecasting. No label-fitted state coefficients.

    At current time t, forecast uses origin t-1. The first online point and a
    feature with no supported h1 cross parent explicitly return UNKNOWN.
    """
    validate_graph_scope(graph,scope)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'DONE.json').exists():
        done=read_json(directory/'DONE.json')
        assert done['scope']==scope.provenance()
        assert digest(directory/'STATE_PREDICTIONS.npz')==done['prediction_sha256']
        for r in done['models']:
            if r['status']=='COMPLETE':assert digest(directory/f"F{r['feature']:03d}_h1.joblib")==r['model_sha256']
        return done
    if any(p.name!='START.json' for p in directory.iterdir()):
        raise RuntimeError('Partial cross-state output preserved for explicit repair')
    cutoff=199 if early else None
    tr=cache.forecast_rows(sampled_rows(cache,scope.training_ids,cutoff),1,cutoff)
    from src.next8.data import rows_for
    va=rows_for(cache.meta,scope.validation_ids)
    scope.admit_training(np.unique(cache.meta['dataset_id'][tr]));scope.admit_validation(np.unique(cache.meta['dataset_id'][va]))
    output=np.full((len(va),len(DETAILED)),UNKNOWN,dtype=np.int8)
    estimates=np.full((len(va),len(DETAILED)),np.nan,dtype=np.float32)
    records=[]
    for k,j in enumerate(DETAILED):
        parents=parents_at(graph,j,1)
        if not parents:
            records.append({'feature':j,'status':'NO_H1_CROSS_PARENT_UNKNOWN_FALLBACK'});continue
        with activity(f'Cross-fitted observable state forecast {scope.name} F{j:03d}', 'STATE_FORECAST','B2_STATE_OBSERVABILITY'):
            design=mechanical_design(cache,tr,j,1,parents)
            target=movement_target(cache,tr,j,1)['delta']
            reused=None
            donor=Path(reuse_directory)/f'F{j:03d}_h1' if reuse_directory is not None else None
            if donor is not None and (donor/'DONE.json').exists():
                receipt=read_json(donor/'DONE.json')
                assert receipt['scope']==scope.provenance() and receipt['parents']==parents
                assert receipt['training_rows']==len(tr)
                method=receipt['method_aliases']['D5_CROSS']
                path=donor/(method+'.joblib')
                assert digest(path)==receipt['files_sha256'][path.name]
                model=joblib.load(path)['delta']
                assert model.names==design.names and model.training['training_ids_sha256']==ids_hash(scope.training_ids)
                reused={'path':path.relative_to(BASE).as_posix(),'sha256':digest(path),'new_fit':False}
            else:model=fit_ridge(design,target,scope,series_weights(design.ids))
            model.training.update({'feature':j,'horizon':1,'parents':parents,'parent_selection_scope':scope.provenance(),
                'all_state_prediction_training_and_selection_groups_exclude_validation':True})
            output[:,k],estimates[:,k]=predict_cross_state(cache,va,j,parents,model)
            path=directory/f'F{j:03d}_h1.joblib'
            if path.exists():raise FileExistsError(path)
            joblib.dump(model,path)
            records.append({'feature':j,'status':'COMPLETE','parents':parents,'model_sha256':digest(path),
                'reused_comparator_model':reused,'new_fit':reused is None,
                'training_rows':len(tr),'prediction_rows':len(va),'prediction_origin':'t-1 for current-state forecast at t; parent raw-feature lags are therefore t-2,t-5,t-17',
                'first_online_point':'UNKNOWN','training':model.training})
    immutable_npz(directory/'STATE_PREDICTIONS.npz',global_row=va,dataset_id=cache.meta['dataset_id'][va],
        feature=np.asarray(DETAILED,dtype=np.int32),state=output,predicted_feature=estimates)
    done={'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),
        'models':records,'prediction_sha256':digest(directory/'STATE_PREDICTIONS.npz'),
        'state_prediction_not_observed_event':True,'pre_current_F_only':True,
        'uncertainty_fallback':'UNKNOWN within5x H-only h1 flat epsilon of either state threshold; fixed half-H-median-change band with0.1 floor, not calibrated prediction coverage',
        'all_prediction_groups_excluded_from_fits_and_parent_selection':True,'outer_scores_released':False}
    atomic(directory/'DONE.json',done)
    return done
