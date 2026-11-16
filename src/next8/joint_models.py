"""Frozen small M1-M5 designs with distinct value, history and orientation."""
from pathlib import Path
import hashlib
import numpy as np
import joblib
from scipy.special import expit
from src.next8.common import atomic,read_json,digest,utc_now,activity,ids_hash
from src.next8.features import UNKNOWN,DETAILED
from src.next8.learning import Design,one_hot,fit_logistic,fit_tree
from src.next8.h_context import HStateSpec
from src.next8.cache import immutable_npz
from src.next8.data import rows_for
from src.next8.screening import sampled_rows
from src.next.fitting import weights_from_training


def orientation_states(cache,rows,j,selection):
    spec=selection['orientation_state_specs'][str(j)]
    if spec['kind']=='OWN_F':return cache.states(rows,[j],previous=True)[:,0]
    estimator=HStateSpec(**spec)
    s=cache.series_index[rows]
    return estimator.states_from_context(cache.series['context'][s],cache.series['valid'][s,j])


def _features(selection):
    stable=list(map(int,selection['stable_features']));variable=list(map(int,selection['variable_features']))
    if len(stable)>6 or len(variable)>6 or len(set(stable+variable))!=len(stable+variable):
        raise ValueError('Six stable/six disjoint variable feature budget required')
    if not set(variable)<=set(DETAILED):raise ValueError('Variable outside detailed fixed menu')
    return stable,variable


def joint_design(cache,rows,selection,method,cross_state=None):
    if method not in ('M1','M2','M3','M4','M5'):raise ValueError('Frozen M menu only')
    rows=np.asarray(rows,dtype=np.int64);n=len(rows);stable,variable=_features(selection)
    columns=[np.ones(n,dtype=np.float32)];names=['constant_context']
    def add(name,value):names.append(name);columns.append(np.asarray(value,dtype=np.float32))
    classes={int(r['feature']):r['classification'] for r in selection['classifications']}
    for j in stable:
        f=cache.current(rows,[j])[:,0]/4.
        add(f'F{j:03d}_stable_linear',f)
        if classes[j]=='STABLE_NONLINEAR':
            add(f'F{j:03d}_stable_positive',np.maximum(f,0));add(f'F{j:03d}_stable_negative',np.maximum(-f,0))
            add(f'F{j:03d}_stable_square',f*f)
        elif classes[j]!='INVARIANT_DIRECTION':raise ValueError('Uncertified stable input')
    if method!='M1' and variable:
        # Same raw H/time/history context in fixed and conditional models.
        for k,v in enumerate(cache.context(rows).T):add(f'known_H_time_context_{k}',v)
        if cross_state is None:cross_state=np.full((n,len(DETAILED)),UNKNOWN,dtype=np.int8)
        if cross_state.shape!=(n,len(DETAILED)) or np.any((cross_state<0)|(cross_state>UNKNOWN)):
            raise ValueError('Exact cross-fitted graph-state schema required')
        for j in variable:
            f=cache.current(rows,[j])[:,0]/4.
            state=orientation_states(cache,rows,j,selection)
            for lag in (1,4,16):add(f'F{j:03d}_own_lag{lag}',cache.lag(rows,[j],lag)[:,0]/4.)
            # H/time/own lags are the observable nuisance baseline. UNKNOWN
            # removes current-F evidence; it cannot invent a direction.
            current=f if method=='M2' else np.where(state<UNKNOWN,f,0.)
            add(f'F{j:03d}_variable_linear',current)
            add(f'F{j:03d}_variable_positive',np.maximum(current,0))
            add(f'F{j:03d}_variable_negative',np.maximum(-current,0))
            add(f'F{j:03d}_variable_square',current*current)
            if method in ('M3','M4','M5'):
                own=one_hot(state)
                for s in (0,2,3):add(f'F{j:03d}_own_condition_{s}',own[:,s])
                for s in (0,2):add(f'F{j:03d}_own_condition_{s}_times_F',own[:,s]*current)
            if method in ('M3','M5'):
                cross=cross_state[:,DETAILED.index(j)]
                for s in (0,1,2):
                    # No cross-state intercept-only predictor in M5: this
                    # branch is explicitly F evidence, not a hidden state head.
                    flag=(cross==s).astype(np.float32)
                    add(f'F{j:03d}_graph_condition_{s}_times_F',flag*current)
                # M3 gets exactly the same explicit products as M5. It can
                # combine them freely; no inaccessible state map advantage.
    result=Design(np.column_stack(columns).astype(np.float32),tuple(names),cache.meta['dataset_id'][rows])
    result.validate();return result


def raw_coefficients(model):
    if model.constant_probabilities is not None:
        p=float(model.constant_probabilities[1]);return np.zeros(len(model.names)),float(np.log(p/(1-p)))
    if model.estimator is None or not hasattr(model.estimator,'coef_'):raise ValueError('Linear direction model required')
    coefficient=model.estimator.coef_[0]/model.scale
    intercept=float(model.estimator.intercept_[0]-model.center@coefficient)
    return coefficient,intercept


def prune_training_design(design):
    """Drop exact duplicate/constant coordinates before regularization.

    A constant state times F must not improve a logistic model merely by
    presenting the same F coordinate twice. The selection is label-free and
    uses training rows only; the same retained schema is used at inference.
    """
    selected=[];seen={}
    for k,name in enumerate(design.names):
        column=design.values[:,k]
        if np.ptp(column)<=1e-12:continue
        key=hashlib.sha256(column.tobytes()).hexdigest()
        if key in seen:
            assert np.array_equal(column,design.values[:,seen[key]])
            continue
        seen[key]=k;selected.append(k)
    active=list(selected)
    if not selected:selected=[0]  # Deterministic intercept/prior model.
    return Design(design.values[:,selected],tuple(design.names[k] for k in selected),design.ids),selected,active


def model_design(model,design):
    original=model.training.get('original_input_names')
    if original is None:return design
    if tuple(original)!=design.names:raise ValueError('Changed original model input schema')
    columns=model.training['retained_input_indices']
    return Design(design.values[:,columns],tuple(design.names[k] for k in columns),design.ids)


def current_feature_contribution(model,design,j):
    design=model_design(model,design)
    model.transform(design);coefficient,_=raw_coefficients(model)
    prefix=f'F{j:03d}_'
    take=np.asarray([name.startswith(prefix) and ('_variable_' in name or name.endswith('_times_F')) for name in model.names])
    return design.values[:,take]@coefficient[take]


def preselected_linear_rule(model,j,own_state,cross_state=UNKNOWN):
    """Training-fixed linear contribution only, chosen without current F.

    Fixed abs/positive/negative/quadratic terms are separately identified and
    may alter the complete local derivative. This function never hides that
    fixed nonlinearity inside a claim of dynamic directional selection.
    """
    if own_state==UNKNOWN:return 0.
    coef,_=raw_coefficients(model);b=dict(zip(model.names,coef));p=f'F{j:03d}_'
    return float((b.get(p+'variable_linear',0.)+b.get(p+f'own_condition_{own_state}_times_F',0.)+
        b.get(p+f'graph_condition_{cross_state}_times_F',0.))/4.)


def load_state_partition(cache,parent_scope,rows,parts):
    """Join state forecasts after checking actual fit AND graph selection IDs."""
    rows=np.asarray(rows,dtype=np.int64);ids=cache.meta['dataset_id'][rows]
    expected=set(map(int,np.unique(ids)))
    out=np.full((len(rows),len(DETAILED)),UNKNOWN,dtype=np.int8);seen=np.zeros(len(rows),dtype=bool)
    receipts=[]
    for scope,directory in parts:
        directory=Path(directory);done=read_json(directory/'DONE.json')
        if done['scope']!=scope.provenance():raise ValueError('State fit/selection provenance mismatch')
        if scope.validation_ids & scope.training_ids:raise ValueError('State selection leakage')
        if not scope.training_ids<=parent_scope.training_ids:raise ValueError('State model trained outside parent TRAIN')
        if not scope.validation_ids<=(parent_scope.training_ids|parent_scope.validation_ids):
            raise ValueError('State prediction group outside complete parent partition')
        p=directory/'STATE_PREDICTIONS.npz'
        if digest(p)!=done['files_sha256']['STATE_PREDICTIONS.npz']:raise ValueError('State prediction hash mismatch')
        with np.load(p,allow_pickle=False) as a:
            if not np.array_equal(a['feature'],DETAILED):raise ValueError('Changed state channel order')
            source_rows=a['global_row'];source_ids=a['dataset_id']
            if np.any(np.diff(source_rows)<=0):raise ValueError('State rows must be sorted and unique')
            if not np.array_equal(cache.meta['dataset_id'][source_rows],source_ids):raise ValueError('State row mapping mismatch')
            if set(map(int,np.unique(source_ids)))!=scope.validation_ids:raise ValueError('Incomplete excluded-group state output')
            at=np.flatnonzero(np.isin(ids,list(scope.validation_ids)))
            loc=np.searchsorted(source_rows,rows[at])
            if np.any(loc>=len(source_rows)) or not np.array_equal(source_rows[loc],rows[at]):raise ValueError('Missing state rows')
            if seen[at].any():raise ValueError('Duplicate state predictions')
            out[at]=a['state'][loc];seen[at]=True
        receipts.append({'scope':scope.provenance(),'prediction_sha256':digest(p),
                         'graph_selection_ids_sha256':ids_hash(scope.training_ids)})
    if not seen.all():raise ValueError('Complete excluded-group state prediction coverage required')
    return out,receipts


def load_heldout_states(cache,scope,rows,directory):
    # Inference is from the model fit on this exact parent TRAIN.
    return load_state_partition(cache,scope,rows,[(scope,directory)])


def fit_joint_models(cache,scope,selection,directory,train_cross,valid_cross,*,early=False,methods=None):
    if selection['scope']!=scope.provenance():raise ValueError('Complete feature/state selection scope required')
    if set(map(int,selection['selection_ids']))!=scope.training_ids:raise ValueError('Feature selection IDs mismatch')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'DONE.json').exists():
        done=read_json(directory/'DONE.json');assert done['scope']==scope.provenance()
        for name,sha in done['files_sha256'].items():assert digest(directory/name)==sha
        return done
    if any(p.name not in ('START.json','STATE_INPUT_PROVENANCE.json') for p in directory.iterdir()):
        raise RuntimeError('Partial final-model outputs preserved for explicit repair')
    tr=sampled_rows(cache,scope.training_ids,199 if early else None)
    va=rows_for(cache.meta,scope.validation_ids)
    if early:va=va[cache.meta['time_online'][va]>=400]
    scope.admit_training(np.unique(cache.meta['dataset_id'][tr]));scope.admit_validation(np.unique(cache.meta['dataset_id'][va]))
    if train_cross.shape!=(len(tr),len(DETAILED)) or valid_cross.shape!=(len(va),len(DETAILED)):
        raise ValueError('Training and heldout state rows must exactly match model rows')
    y=cache.meta['target'][tr]
    weights=weights_from_training(y,cache.meta['time_online'][tr],cache.meta['dataset_id'][tr],'W3')
    methods=['M1','M2','M4','M5','M3'] if methods is None else list(methods)
    outputs={'global_row':va,'dataset_id':cache.meta['dataset_id'][va]};aliases={};fitted={};files={};records=[]
    for method in methods:
        with activity(f'Fit {method} {scope.name}','MODEL_FIT','B4_MOVEMENT_VS_DISCRIMINATION'):
            td=joint_design(cache,tr,selection,method,train_cross)
            reduced,columns,active=prune_training_design(td)
            key=('tree' if method=='M3' and len(active) else 'logistic',tuple(td.names[k] for k in active))
            if key in fitted:
                donor=fitted[key];aliases[method]=donor;outputs[method]=outputs[donor]
                records.append({'method':method,'alias':donor,'reason':'Same learner and exactly identical active observable coordinates'})
                continue
            model=fit_tree(reduced,y,scope,weights) if key[0]=='tree' else fit_logistic(reduced,y,scope,weights)
            model.training.update({'feature_selection_ids_sha256':ids_hash(scope.training_ids),
                'classification_and_state_selection_excluded_validation':True,'weighting':'W3_TRAIN_ONLY',
                'early_training_cutoff':199 if early else None,'state_meta_inputs_cross_fitted':True,
                'original_input_names':list(td.names),'retained_input_indices':columns,
                'exact_duplicate_and_constant_columns_removed_using_training_values_only':True})
            prediction=np.empty(len(va),dtype=np.float32)
            for start in range(0,len(va),50000):
                sl=slice(start,start+50000)
                design=joint_design(cache,va[sl],selection,method,valid_cross[sl])
                prediction[sl]=model.predict_proba(model_design(model,design))[:,1]
            path=directory/(method+'.joblib');joblib.dump(model,path);files[path.name]=digest(path)
            outputs[method]=prediction;aliases[method]=method;fitted[key]=method
            records.append({'method':method,'alias':None,'input_columns':len(td.names),'active_columns':len(active),
                'input_names':list(td.names),'training':model.training})
    immutable_npz(directory/'PREDICTIONS.npz',**outputs);files['PREDICTIONS.npz']=digest(directory/'PREDICTIONS.npz')
    done={'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),'methods':methods,
        'model_aliases':aliases,'fitting':records,'files_sha256':files,'training_rows':len(tr),'prediction_rows':len(va),
        'stable_features':selection['stable_features'],'variable_features':selection['variable_features'],
        'ordinary_model_input_parity':'M3 and M5 have identical raw named columns and values, including explicit interaction products',
        'UNKNOWN_fallback':'Variable current-F contribution zero; known H/time/history nuisance baseline preserved; graph UNKNOWN removes only graph F terms',
        'direction_rule':'Training-fixed state-specific linear beta available before F[t]; fixed nonlinear terms are separate and not claimed as advance direction',
        'outer_or_meta_scores_released':False,'validation_labels_used':False}
    atomic(directory/'DONE.json',done)
    return done
