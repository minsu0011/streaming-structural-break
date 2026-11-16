"""Training-only single-feature direction refits; heldout predictions unscored."""
from pathlib import Path
import numpy as np
import joblib
from src.next8.common import atomic,read_json,digest,utc_now,activity
from src.next8.features import DETAILED,UNKNOWN
from src.next8.cache import immutable_npz
from src.next8.learning import orientation_design,fit_logistic,fit_tree
from src.next8.joint_models import orientation_states,model_design
from src.next8.screening import sampled_rows,context_for,common_range,select_columns,_state_support
from src.next8.statistics import logistic_contrast
from src.next.fitting import weights_from_training


def designs(cache,rows,j,state,cross=None):
    ids=cache.meta['dataset_id'][rows];f=cache.current(rows,[j])[:,0]
    c=context_for(cache,rows,j)
    kind='O2' if cross is None else 'O3'
    conditional=orientation_design(f,state,c,ids,kind,cross_state=cross)
    control=select_columns(conditional,lambda n:not n.endswith('_times_F') if cross is None else not(n.startswith('cross_state_') and n.endswith('_times_F')))
    state_only=select_columns(conditional,lambda n:not(n.startswith('F_') or n.endswith('_times_F')))
    return {'DIRECTION':conditional,'FIXED_CONTROL':control,'STATE_ONLY':state_only,'TABULAR':conditional}


def _fit(design,y,scope,weights,tree=False):
    # Refit the same O-family parameterization used by the original nested
    # feature selector. The separately frozen M-family prunes duplicates in
    # every inner and outer model; do not retrofit that new recipe onto an
    # already selected O-family rule without repeating its inner selection.
    model=fit_tree(design,y,scope,weights) if tree else fit_logistic(design,y,scope,weights)
    model.training.update({'O_parameterization_matches_original_inner_selector':True,
        'M_family_duplicate_pruning_not_retrofitted_to_O_selection':True})
    return model


def fit_conditional_refit(cache,scope,j,selection,tr,va,train_cross,valid_cross,directory,*,cross=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    own=orientation_states(cache,tr,j,selection);vown=orientation_states(cache,va,j,selection)
    cs=train_cross[:,DETAILED.index(j)] if cross else None
    vs=valid_cross[:,DETAILED.index(j)] if cross else None
    defining=cs if cross else own
    f=cache.current(tr,[j])[:,0]
    domain=common_range(f,defining)
    if domain is None:return {'status':'INSUFFICIENT_SUPPORT','reason':'No common training F range across three observable conditions','cross':cross}
    keep=(f>=domain[0])&(f<=domain[1])&(own<UNKNOWN)&(defining<UNKNOWN)
    rows=tr[keep];state=own[keep];cstate=cs[keep] if cross else None
    y=cache.meta['target'][rows];ids=cache.meta['dataset_id'][rows]
    support=_state_support(ids,y,(cstate if cross else state))
    if any(r['series']<64 or min(r['positive_series'],r['negative_series'])<32 for r in support):
        return {'status':'INSUFFICIENT_SUPPORT','reason':'Common-domain training state class support','domain':domain,'support':support,'cross':cross}
    x=designs(cache,rows,j,state,cstate)
    weights=weights_from_training(y,cache.meta['time_online'][rows],ids,'W3')
    models={name:_fit(d,y,scope,weights,tree=name=='TABULAR') for name,d in x.items()}
    # Predictions are issued for every sampled heldout row. Domain/UNKNOWN
    # support flags are separately stored and never inferred from heldout y.
    v=designs(cache,va,j,vown,vs)
    outputs={name:m.predict_proba(model_design(m,v[name]))[:,1].astype(np.float32) for name,m in models.items()}
    vf=cache.current(va,[j])[:,0]
    known=(vf>=domain[0])&(vf<=domain[1])&(vown<UNKNOWN)
    if cross:known &= vs<UNKNOWN
    outputs.update({'global_row':va,'dataset_id':cache.meta['dataset_id'][va],'within_training_domain':known,
                    'own_condition':vown,'cross_condition':np.full(len(va),UNKNOWN,dtype=np.int8) if vs is None else vs})
    immutable_npz(directory/'PREDICTIONS.npz',**outputs)
    files={'PREDICTIONS.npz':digest(directory/'PREDICTIONS.npz')}
    for name,m in models.items():
        joblib.dump(m,directory/(name+'.joblib'));files[name+'.joblib']=digest(directory/(name+'.joblib'))
    prefix='cross_state_' if cross else 'own_state_'
    model=models['DIRECTION'];md=model_design(model,x['DIRECTION'])
    contrasts=[{'state_a':a,'state_b':b,**logistic_contrast(model,md,y,ids,weights,
        prefix+f'{a}_times_F',prefix+f'{b}_times_F')} for a,b in ((0,1),(2,1),(0,2))]
    if model.estimator is not None and hasattr(model.estimator,'coef_'):
        coefficients=dict(zip(model.names,(model.estimator.coef_[0]/model.scale).tolist()))
    else:coefficients={}
    return {'status':'COMPLETE','scope':scope.provenance(),'feature':j,'cross':cross,'domain':domain,
        'training_support':support,'training_rows':len(rows),'prediction_rows':len(va),
        'training_contrasts':contrasts,'derivative_weights_raw_design':coefficients,
        'models_training':{name:m.training for name,m in models.items()},'files_sha256':files,
        'selected_orientation_state':selection['orientation_state_specs'][str(j)]['kind'],
        'cross_state_estimator':'FACTORISED_T3_PRE_CURRENT_V1' if cross else None,
        'fixed_control':'Same rows, state main effects and own-direction terms; only cross F interactions removed' if cross else 'Same rows and state main effects, F-state interactions removed',
        'validation_labels_used':False,'outer_scores_released':False,
        'prediction_arrays_for_diagnostic_comparison_only':'Out-of-training-domain and UNKNOWN predictions remain flagged, never certified directional evidence or used as live confirmations'}


def fit_orientation_refits(cache,scope,selection,train_cross,valid_cross,directory,*,early=False):
    assert selection['scope']==scope.provenance()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    tr=sampled_rows(cache,scope.training_ids,199 if early else None)
    va=sampled_rows(cache,scope.validation_ids)
    if early:va=va[cache.meta['time_online'][va]>=400]
    if train_cross.shape!=(len(tr),len(DETAILED)) or valid_cross.shape!=(len(va),len(DETAILED)):
        raise ValueError('Matched sampled causal state rows required')
    scope.admit_training(np.unique(cache.meta['dataset_id'][tr]));scope.admit_validation(np.unique(cache.meta['dataset_id'][va]))
    y=cache.meta['target'][tr];weights=weights_from_training(y,cache.meta['time_online'][tr],cache.meta['dataset_id'][tr],'W3')
    targets=sorted(set(DETAILED)|set(selection['stable_features']))
    records=[]
    for j in targets:
        p=directory/f'F{j:03d}';p.mkdir(parents=True,exist_ok=True)
        if (p/'DONE.json').exists():
            done=read_json(p/'DONE.json');assert done['scope']==scope.provenance()
            for name,sha in done['files_sha256'].items():assert digest(p/name)==sha
            records.append(done);continue
        if any(p.iterdir()):raise RuntimeError('Partial orientation refit preserved for explicit repair')
        with activity(f'Fit heldout direction refits {scope.name} F{j:03d}','DIRECTION_REFIT','B1_CONDITIONAL_DIRECTION'):
            state=cache.states(tr,[j],previous=True)[:,0];vstate=cache.states(va,[j],previous=True)[:,0]
            models={};outputs={'global_row':va,'dataset_id':cache.meta['dataset_id'][va]};files={}
            for method in ('O0','O1','CONTEXT_ONLY'):
                kind='O0' if method in ('O0','CONTEXT_ONLY') else 'O1'
                td=orientation_design(cache.current(tr,[j])[:,0],state,context_for(cache,tr,j),cache.meta['dataset_id'][tr],kind)
                vd=orientation_design(cache.current(va,[j])[:,0],vstate,context_for(cache,va,j),cache.meta['dataset_id'][va],kind)
                if method=='CONTEXT_ONLY':
                    td=select_columns(td,lambda n:not n.startswith('F_'));vd=select_columns(vd,lambda n:not n.startswith('F_'))
                model=_fit(td,y,scope,weights);models[method]=model
                outputs[method]=model.predict_proba(model_design(model,vd))[:,1].astype(np.float32)
                joblib.dump(model,p/(method+'.joblib'));files[method+'.joblib']=digest(p/(method+'.joblib'))
            immutable_npz(p/'PREDICTIONS.npz',**outputs);files['PREDICTIONS.npz']=digest(p/'PREDICTIONS.npz')
            conditional={}
            if j in selection['variable_features']:
                for cross,name in ((False,'O2_OWN'),(True,'O3_GRAPH')):
                    result=fit_conditional_refit(cache,scope,j,selection,tr,va,train_cross,valid_cross,p/name,cross=cross)
                    atomic(p/name/'DONE.json',result);conditional[name]=result
                    files[name+'/DONE.json']=digest(p/name/'DONE.json')
                    for f,sha in result.get('files_sha256',{}).items():files[name+'/'+f]=sha
            done={'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),'feature':j,
                'model_training':{name:m.training for name,m in models.items()},'conditional':conditional,
                'files_sha256':files,'detailed_variable_eligible':j in DETAILED,
                'stable_screening_refit_only':j not in DETAILED,'validation_labels_used':False,'outer_scores_released':False}
            atomic(p/'DONE.json',done);records.append(done)
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),
        'feature_refits':targets,'detailed_variable_menu':list(DETAILED),'selected_conditional_candidates':selection['variable_features'],
        'records_sha256':{f"F{r['feature']:03d}/DONE.json":digest(directory/f"F{r['feature']:03d}/DONE.json") for r in records},
        'outer_scores_released':False,'no_extra_variable_candidate_beyond_fixed12':True})
