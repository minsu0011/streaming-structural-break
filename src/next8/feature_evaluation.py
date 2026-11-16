"""All-bank heldout atlas and single-feature information, with no retuning."""
import numpy as np
from src.next8.common import BASE,read_json,digest
from src.next8.features import NAMES,UNKNOWN
from src.next8.h_context import HStateSpec
from src.next8.statistics import binary_losses
from src.next8.diagnostics import original_groups,safe_rank,loss_evidence,time_H_domains
from src.next8.direction_evaluation import conditional_case


def load_predictions(path,cache,scope,release):
    assert digest(path)==release['files_sha256'][path.relative_to(BASE).as_posix()]
    with np.load(path,allow_pickle=False) as a:values={k:a[k].copy() for k in a.files}
    rows=values['global_row'];scope.admit_validation(np.unique(values['dataset_id']))
    np.testing.assert_array_equal(cache.meta['dataset_id'][rows],values['dataset_id'])
    assert np.all(np.diff(rows)>0)
    return rows,values


def atlas(cache,rows,j,prediction,direction,mode,specs):
    y=cache.meta['target'][rows];t=cache.meta['time_online'][rows];groups=original_groups(cache,rows)
    condition=cache.states(rows,[j],previous=True)[:,0];definitions=list(time_H_domains(cache,rows))
    for s in range(4):definitions.append(('OWN_PRE_STATE',f'OWN_PRE_STATE_{s}',condition==s))
    for kind,value in specs.items():
        spec=HStateSpec(**value);state=spec.states_from_context(cache.series['context'][cache.series_index[rows]],cache.series['valid'][cache.series_index[rows],j])
        for s in range(4):definitions.append((kind,f'{kind}_{s}',state==s))
    result=[]
    for family,domain,keep in definitions:
        raw=safe_rank(y[keep],prediction[keep],t[keep],groups[keep])
        oriented=None;lower=None;upper=None
        if direction and raw['TS_AUC'] is not None:
            oriented=raw['TS_AUC'] if direction>0 else 1-raw['TS_AUC']
            lower=raw['ci_low'] if direction>0 else 1-raw['ci_high'];upper=raw['ci_high'] if direction>0 else 1-raw['ci_low']
        result.append({'feature':j,'name':NAMES[j],'mode':mode,'domain_family':family,'domain':domain,
            'direction_from_TRAIN':direction,'oriented_TS_AUC':oriented,'oriented_ci_low':lower,'oriented_ci_high':upper,**raw,
            'H_condition_constant_within_series':family in specs,'UNKNOWN_domain':domain.endswith('_3') and family in ('OWN_PRE_STATE',*specs)})
    return result


def fixed_panel(cache,rows,values,pairs):
    y=cache.meta['target'][rows];t=cache.meta['time_online'][rows];groups=original_groups(cache,rows)
    predictions={k:v for k,v in values.items() if k not in ('global_row','dataset_id')};ranks={};comparisons=[];arrays={}
    for name,p in predictions.items():ranks[name]=safe_rank(y,p,t,groups)
    for i,(reference,candidate) in enumerate(pairs):
        if reference not in predictions or candidate not in predictions:continue
        result,keys,gain=loss_evidence(groups,binary_losses(y,predictions[reference]),binary_losses(y,predictions[candidate]))
        key=f'C{i:03d}';arrays[key+'_GROUP']=keys;arrays[key+'_GAIN']=gain
        delta=(ranks[candidate]['TS_AUC']-ranks[reference]['TS_AUC']) if ranks[candidate]['TS_AUC'] is not None and ranks[reference]['TS_AUC'] is not None else None
        comparisons.append({'reference':reference,'candidate':candidate,'group_array_key':key,'TS_AUC_delta':delta,**result})
    return {'ranks':ranks,'comparisons':comparisons},arrays


def evaluate_feature(cache,scope,j,rows,selection,specs,direction_definitions,release,*,early=False):
    classification=selection['classifications'][j];signs=classification['signed_direction']
    direction=int(signs[0]) if len(set(signs))==1 else 0
    result={'status':'COMPLETE','scope':scope.provenance(),'feature':j,'name':NAMES[j],
        'training_classification':classification,'selected_stable':j in selection['stable_features'],
        'selected_variable':j in selection['variable_features'],'atlas_stride':8,'certification_not_assigned_by_this_individual_fold':True}
    result['atlas']=atlas(cache,rows,j,cache.current(rows,[j])[:,0],direction,'RAW_F',specs)
    arrays={};p=BASE/'orientation_refits'/scope.name/f'F{j:03d}'
    if (p/'DONE.json').exists():
        orows,values=load_predictions(p/'PREDICTIONS.npz',cache,scope,release);np.testing.assert_array_equal(orows,rows)
        panel,extra=fixed_panel(cache,rows,values,[('O0','O1'),('CONTEXT_ONLY','O1')]);result['fixed_refit']=panel
        arrays.update({'FIXED_'+k:v for k,v in extra.items()})
        for record in panel['comparisons']:record['group_array_key']='FIXED_'+record['group_array_key']
        result['atlas']+=atlas(cache,rows,j,values['O1'],1,'O1_FIXED_NONLINEAR',specs)
    else:result['fixed_refit']={'status':'NOT_REFIT_OUTSIDE_FIXED12_AND_RETAINED_STABLE_SHORTLIST'}
    result['conditional']=[]
    for definition in direction_definitions:
        if definition['feature']!=j:continue
        model=definition['orientation_model'];cp=p/model
        complete=read_json(cp/'DONE.json')['status']=='COMPLETE'
        branch={'model':model,'definition':definition,'cases':[]}
        if complete:
            crows,values=load_predictions(cp/'PREDICTIONS.npz',cache,scope,release);np.testing.assert_array_equal(crows,rows)
            for i,case in enumerate(definition['training_cases']):
                record,extra=conditional_case(cache,rows,j,values,case,model=model,scope_name=scope.name,case_index=i)
                prefix=f'{model}_CASE{i}_';arrays.update({prefix+k:v for k,v in extra.items()})
                record['array_prefix']=prefix;branch['cases'].append(record)
        else:branch['status']='INSUFFICIENT_TRAIN_SUPPORT_NO_REFIT'
        result['conditional'].append(branch)
    if not early:
        bp=BASE/'baseline_conditioned'/scope.name/f'F{j:03d}'/'PREDICTIONS.npz'
        brows,values=load_predictions(bp,cache,scope,release);np.testing.assert_array_equal(brows,rows)
        pairs=[('BL_CONTEXT','BL_FIXED_F'),('BL_STATE_ONLY','BL_OWN_DIRECTION'),('BL_FIXED_STATE','BL_OWN_DIRECTION'),
            ('BL_OWN_DIRECTION','BL_CROSS_DIRECTION'),('BL_SAME_INPUT_TREE','BL_CROSS_DIRECTION')]
        panel,extra=fixed_panel(cache,rows,values,pairs);result['BL80_conditional_information']=panel
        arrays.update({'BL_'+k:v for k,v in extra.items()})
        for record in panel['comparisons']:record['group_array_key']='BL_'+record['group_array_key']
    else:result['BL80_conditional_information']={'status':'PRIMARY_NORMAL_SCOPES_ONLY','early_M_models_and_references_evaluated_separately':True}
    return result,arrays
