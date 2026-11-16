"""Direct heldout state-direction contrasts with a shared group bootstrap."""
import numpy as np
from scipy.stats import norm
from src.next8.paired_bootstrap import PreparedRank,group_resample_weights
from src.next8.diagnostics import safe_rank,loss_evidence,time_H_domains
from src.next8.statistics import binary_losses


def state_rank_bootstrap(y,f,t,groups,state,*,partition):
    groups=np.asarray(groups);keys=np.unique(groups);records={};samples={}
    if not len(keys):return {},{}, {'status':'INSUFFICIENT_SUPPORT','groups':0}
    weights,receipt=group_resample_weights(keys,2000,partition=partition)
    for s in range(3):
        keep=state==s;record=safe_rank(y[keep],f[keep],t[keep],groups[keep])
        if record['status']=='SUPPORTED':
            prepared=PreparedRank.prepare(y[keep],f[keep],t[keep],groups[keep])
            columns=np.searchsorted(keys,prepared.groups);draws,pairs=prepared.evaluate(weights[:,columns])
            valid=np.isfinite(draws)
            if valid.sum()>=1900:record.update({'ci_low':float(np.quantile(draws[valid],.025)),
                'ci_high':float(np.quantile(draws[valid],.975)),'interval_method':'Exact whole-group same-time pair-weighted bootstrap; shared group draws across states'})
            else:record.update({'status':'INSUFFICIENT_SUPPORT','ci_low':None,'ci_high':None})
            record['valid_draws']=int(valid.sum());samples[str(s)]=draws
        else:samples[str(s)]=np.full(2000,np.nan)
        records[str(s)]=record
    return records,samples,{**receipt,'same_draws_across_states':True,'missing_state_group_columns_keep_original_multiplicities':True}


def pair_rank_evidence(state_records,samples,pair):
    a,b=map(str,pair['state_pair']);sa,sb=pair['directions'];result={'training_rule':pair}
    if a not in state_records or b not in state_records or any(state_records[s]['status']!='SUPPORTED' for s in (a,b)):
        return {**result,'status':'INSUFFICIENT_SUPPORT','p_value':1.,'both_claimed_directions_reproduced':False}
    point=state_records[a]['TS_AUC']-state_records[b]['TS_AUC'];draw=samples[a]-samples[b];valid=np.isfinite(draw)
    if valid.sum()<1900:return {**result,'status':'INSUFFICIENT_SUPPORT','p_value':1.,'both_claimed_directions_reproduced':False}
    sd=float(np.std(draw[valid],ddof=1));p=float(2*norm.sf(abs(point)/sd)) if sd else (0. if point else 1.)
    lower=[];oriented=[]
    for s,direction in ((a,sa),(b,sb)):
        record=state_records[s]
        lower.append(record['ci_low'] if direction>0 else 1-record['ci_high'] if direction<0 else None)
        oriented.append(record['TS_AUC'] if direction>0 else 1-record['TS_AUC'] if direction<0 else None)
    difference_low=float(np.quantile(draw[valid],.025));difference_high=float(np.quantile(draw[valid],.975))
    difference_support=(difference_low>0 if sa>0 else difference_high<0 if sa<0 else False)
    return {**result,'status':'SUPPORTED','state_raw_AUC_difference':point,'difference_ci_low':difference_low,
        'difference_ci_high':difference_high,'p_value':p,'p_method':'Normal approximation using paired whole-group bootstrap SD; no permutation claim',
        'oriented_state_AUC':oriented,'oriented_state_CI_lower':lower,
        'both_claimed_directions_reproduced':bool(pair['eligible_for_orientation_replication'] and all(x is not None and x>.505 for x in lower) and difference_support),
        'draws':2000,'valid_draws':int(valid.sum())}


def conditional_case(cache,rows,feature,outputs,case,*,model,scope_name,case_index):
    from src.next8.diagnostics import original_groups
    if case['status']!='FROZEN':return {'training_case':case,'status':'INSUFFICIENT_SUPPORT'},{}
    f=cache.current(rows,[feature])[:,0];own=outputs['own_condition'];cross=outputs['cross_condition']
    defining=cross if model=='O3_GRAPH' else own
    keep=outputs['within_training_domain']&(f>=case['domain'][0])&(f<=case['domain'][1])&(defining<3)
    if model=='O3_GRAPH':keep &= own==case['own_stratum']
    ids=original_groups(cache,rows);y=cache.meta['target'][rows];t=cache.meta['time_online'][rows]
    ranks,boot,resampling=state_rank_bootstrap(y[keep],f[keep],t[keep],ids[keep],defining[keep],partition=f'{scope_name}_F{feature}_{model}_C{case_index}')
    arrays={f'STATE_{s}_BOOTSTRAP':a for s,a in boot.items()};pairs=[]
    for pi,pair in enumerate(case['pairs']):
        evidence=pair_rank_evidence(ranks,boot,pair);mask=keep&np.isin(defining,pair['state_pair']);comparisons=[]
        for control in ('FIXED_CONTROL','STATE_ONLY','TABULAR'):
            ref=binary_losses(y[mask],outputs[control][mask]);candidate=binary_losses(y[mask],outputs['DIRECTION'][mask])
            gain,keys,means=loss_evidence(ids[mask],ref,candidate);key=f'PAIR{pi}_{control}'
            arrays[key+'_GROUP']=keys;arrays[key+'_GAIN']=means
            comparisons.append({'reference':control,'candidate':'DIRECTION','group_array_key':key,**gain})
        domain_records=[]
        for family,domain,domain_keep in time_H_domains(cache,rows):
            if family=='ALL':continue
            sub=mask&domain_keep;states=[]
            for s,sign in zip(pair['state_pair'],pair['directions']):
                at=sub&(defining==s)
                rank=safe_rank(y[at],f[at]*(sign if sign else 1),t[at],ids[at])
                states.append({'state':s,'training_direction':sign,**rank})
            supported=all(r['status']=='SUPPORTED' for r in states)
            if supported:
                loss,_,_=loss_evidence(ids[sub],binary_losses(y[sub],outputs['FIXED_CONTROL'][sub]),binary_losses(y[sub],outputs['DIRECTION'][sub]))
            else:loss={'status':'INSUFFICIENT_SUPPORT','gain':None,'ci_low':None,'ci_high':None}
            domain_records.append({'domain_family':family,'domain':domain,'states':states,'increment':loss,
                'supported':supported,'direction_and_gain_reproduced':bool(supported and all(r['training_direction']!=0 and r['TS_AUC']>.5 and r['ci_low']>.47 for r in states) and loss['gain']>0)})
        pairs.append({'pair_index':pi,**evidence,'matched_model_comparisons':comparisons,'heldout_domains':domain_records})
    return {'status':'COMPLETE','training_case':case,'rows_in_common_domain':int(keep.sum()),'state_ranks':ranks,
        'resampling':resampling,'pairs':pairs,'O_family_refit_not_automatically_joint_M5_certificate':True},arrays
