"""Bounded per-feature incremental-information audit against strict BL80 OOF.

This screening audit does not add candidates or alter any selected M model.
"""
from pathlib import Path
import numpy as np
from scipy.special import logit
import joblib
from src.next8.features import N_FEATURES,DETAILED,UNKNOWN
from src.next8.common import atomic,read_json,digest,utc_now,activity
from src.next8.cache import immutable_npz
from src.next8.learning import Design,fit_logistic,fit_tree,orientation_design
from src.next8.joint_models import orientation_states,prune_training_design,model_design
from src.next8.screening import context_for,select_columns
from src.next.fitting import weights_from_training


def baseline_designs(cache,rows,j,baseline,selection,cross):
    ids=cache.meta['dataset_id'][rows];f=cache.current(rows,[j])[:,0]
    b=logit(np.clip(np.asarray(baseline,dtype=np.float64),1e-6,1-1e-6))
    if b.shape!=(len(rows),):raise ValueError('Exact aligned excluded-group BL80 probabilities')
    context=np.column_stack([b,context_for(cache,rows,j)]).astype(np.float32)
    state=cache.states(rows,[j],previous=True)[:,0]
    fixed=orientation_design(f,state,context,ids,'O1')
    null=select_columns(fixed,lambda n:not n.startswith('F_'))
    output={'BL_CONTEXT':null,'BL_FIXED_F':fixed}
    if j in selection['variable_features']:
        own=orientation_states(cache,rows,j,selection)
        graph=cross[:,DETAILED.index(j)]
        full=orientation_design(np.where(own<UNKNOWN,f,0.),own,context,ids,'O3',cross_state=graph)
        output.update({'BL_STATE_ONLY':select_columns(full,lambda n:not(n.startswith('F_') or n.endswith('_times_F'))),
            'BL_FIXED_STATE':select_columns(full,lambda n:not n.endswith('_times_F')),
            'BL_OWN_DIRECTION':select_columns(full,lambda n:not(n.startswith('cross_state_') and n.endswith('_times_F'))),
            'BL_CROSS_DIRECTION':full,'BL_SAME_INPUT_TREE':full})
    return output


def fit_baseline_conditioned(cache,scope,selection,tr,va,train_baseline,valid_baseline,train_cross,valid_cross,directory):
    if selection['scope']!=scope.provenance():raise ValueError('Training-only fixed feature selection required')
    scope.admit_training(np.unique(cache.meta['dataset_id'][tr]));scope.admit_validation(np.unique(cache.meta['dataset_id'][va]))
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    target=cache.meta['target'][tr]
    weights=weights_from_training(target,cache.meta['time_online'][tr],cache.meta['dataset_id'][tr],'W3')
    records=[]
    for j in range(N_FEATURES):
        p=directory/f'F{j:03d}';p.mkdir(parents=True,exist_ok=True)
        if (p/'DONE.json').exists():
            done=read_json(p/'DONE.json');assert done['scope']==scope.provenance()
            for name,sha in done['files_sha256'].items():assert digest(p/name)==sha
            records.append(done);continue
        if any(p.iterdir()):raise RuntimeError('Partial baseline-conditioned feature fit preserved')
        with activity(f'BL80 conditional feature audit {scope.name} F{j:03d}','BASELINE_CONDITIONAL_SCREEN','B6_BL80_REDUNDANCY'):
            train=baseline_designs(cache,tr,j,train_baseline,selection,train_cross)
            valid=baseline_designs(cache,va,j,valid_baseline,selection,valid_cross)
            outputs={'global_row':va,'dataset_id':cache.meta['dataset_id'][va]};files={};fitted={};aliases={};models={}
            for name,design in train.items():
                reduced,columns,active=prune_training_design(design)
                key=('tree' if name=='BL_SAME_INPUT_TREE' and active else 'logistic',tuple(design.names[k] for k in active))
                if key in fitted:
                    aliases[name]=fitted[key];outputs[name]=outputs[fitted[key]];continue
                model=fit_tree(reduced,target,scope,weights) if key[0]=='tree' else fit_logistic(reduced,target,scope,weights)
                model.training.update({'original_input_names':list(design.names),'retained_input_indices':columns,
                    'baseline_training_input':'Verified inner excluded-group BL80 OOF, no full-DEV deployment predictions',
                    'state_training_input':'Complete excluded-group T3 forecasts where used; otherwise UNKNOWN',
                    'hypothesis':'Incremental current-F information conditional on BL80, H/time/own lags and matched state controls',
                    'not_an_additional_final_M_candidate':True})
                outputs[name]=model.predict_proba(model_design(model,valid[name]))[:,1].astype(np.float32)
                path=p/(name+'.joblib');joblib.dump(model,path);files[path.name]=digest(path)
                models[name]=model.training;fitted[key]=name;aliases[name]=name
            immutable_npz(p/'PREDICTIONS.npz',**outputs);files['PREDICTIONS.npz']=digest(p/'PREDICTIONS.npz')
            done={'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),'feature':j,
                'model_aliases':aliases,'models_training':models,'files_sha256':files,
                'conditional_models_only_for_selected_variables':j in selection['variable_features'],
                'validation_labels_used':False,'outer_scores_released':False,
                'no_feedback_to_existing_feature_selection_or_final_models':True}
            atomic(p/'DONE.json',done);records.append(done)
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),
        'screened_features':N_FEATURES,'detailed_dynamic_candidates_unchanged':list(DETAILED),
        'selected_variable_features':selection['variable_features'],
        'records_sha256':{f"F{r['feature']:03d}/DONE.json":digest(directory/f"F{r['feature']:03d}/DONE.json") for r in records},
        'outer_scores_released':False,'not_used_for_new_candidate_selection':True})
