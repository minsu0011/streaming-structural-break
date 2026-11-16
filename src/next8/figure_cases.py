"""Predeclared descriptive movement examples, including errors and misses."""
import numpy as np
from src.next8.common import BASE,read_json,digest
from src.next8.features import DETAILED,HORIZONS,NAMES,NORMALIZATION
from src.next8.movement import movement_target

CATEGORIES=('success','failure','delay_or_miss','false_alarm')


def choose_case_indices(ids,t,target,delta,p_reversal):
    """First by numeric ID/time; keep all eligible counts, never best effect."""
    ids=np.asarray(ids);t=np.asarray(t);actual=np.asarray(target['delta']);reversal=np.asarray(target['reversal'])
    error=(actual-delta)**2;baseline=actual**2;selected={};counts={k:0 for k in CATEGORIES}
    for sid in np.unique(ids):
        ix=np.flatnonzero(ids==sid);ix=ix[np.argsort(t[ix],kind='stable')]
        if len(ix)>=32:
            reference=float(baseline[ix].mean());model=float(error[ix].mean())
            for category,eligible in (('success',reference>0 and model<=.9*reference),('failure',reference>0 and model>=1.1*reference)):
                if eligible:
                    counts[category]+=1
                    if category not in selected:selected[category]={'index':int(ix[0]),'series_id':int(sid),'series_origins':len(ix),'D0_MSE':reference,'D1_MSE':model}
        miss=ix[(reversal[ix]==1)&(p_reversal[ix]<=.5)];false=ix[(reversal[ix]==0)&(p_reversal[ix]>.5)]
        counts['delay_or_miss']+=len(miss);counts['false_alarm']+=len(false)
        if len(miss) and 'delay_or_miss' not in selected:
            first=int(miss[0]);later=ix[(t[ix]>t[first])&(t[ix]<=t[first]+32)&(p_reversal[ix]>.5)]
            selected['delay_or_miss']={'index':first,'series_id':int(sid),'series_origins':len(ix),
                'miss_time':int(t[first]),'first_later_forecast_time':int(t[later[0]]) if len(later) else None,
                'case_label':'MISSED_REVERSAL_WITH_LATER_FORECAST' if len(later) else 'MISSED_REVERSAL',
                'later_forecast_not_assumed_to_confirm_same_event':True}
        if len(false) and 'false_alarm' not in selected:selected['false_alarm']={'index':int(false[0]),'series_id':int(sid),'series_origins':len(ix),'false_forecast_time':int(t[false[0]])}
    return selected,counts


def trace_case(cache,rows,target,delta,p_reversal,j,h,category,case):
    index=case['index'];sid=case['series_id'];t=cache.meta['time_online'][rows];event=int(t[index])
    location=cache.series_index[rows[index]];offsets=cache.data.series['online_offsets'];start=int(offsets[location]);stop=int(offsets[location+1])
    low=0 if category in ('success','failure') else max(event-64,0)
    high=min(stop-start-1,511 if category in ('success','failure') else event+64+h)
    dense=np.arange(start+low,start+high+1);keep=(cache.meta['dataset_id'][rows]==sid)&(t>=low)&(t+h<=high)
    current=cache.current(rows[keep],[j])[:,0]
    return {**case,'category':category,'feature':j,'name':NAMES[j],'horizon':h,'method':'D1',
        'trace_time':cache.meta['time_online'][dense].tolist(),'trace_F':cache.current(dense,[j])[:,0].tolist(),
        'forecast_origin_time':t[keep].tolist(),'forecast_target_time':(t[keep]+h).tolist(),
        'origin_F':current.tolist(),'actual_future_F':target['future_F'][keep].tolist(),
        'predicted_future_F':(current+delta[keep]).tolist(),'reversal_probability_at_origin':p_reversal[keep].tolist(),
        'actual_available_reversal':target['reversal'][keep].tolist(),'H_length':int(cache.series['H_length'][location]),
        'normalization':NORMALIZATION,'shared_raw':True,'self_predictability_can_be_mechanical':True,
        'raw_competition_observations_exported':False,'is_independent_confirmation':False,
        'case_selection':'First eligible in preregistered feature/horizon/fold/ID/time order; descriptive after errors, never model selection'}


def collect_real_cases(cache):
    release=read_json(BASE/'PREDICTION_FREEZE_NEXT8.json');selected={};count_rows=[];scanned=0
    for j in sorted(DETAILED):
        for h in HORIZONS:
            for fold in range(5):
                scope=f'outer_{fold}';directory=BASE/'forecasts'/scope/f'F{j:03d}_h{h}'
                done=read_json(directory/'DONE.json');path=directory/'PREDICTIONS.npz'
                assert digest(path)==done['files_sha256']['PREDICTIONS.npz']==release['files_sha256'][path.relative_to(BASE).as_posix()]
                donor=done['method_aliases']['D1']
                with np.load(path,allow_pickle=False) as a:
                    rows=a['global_row'].copy();delta=a[donor+'_delta'].copy();p=a[donor+'_reversal'][:,1].copy()
                usable=cache.forecast_rows(rows,h);at=np.searchsorted(rows,usable)
                known=cache.series['valid'][cache.series_index[usable],j];usable=usable[known];at=at[known]
                target=movement_target(cache,usable,j,h);delta=delta[at];p=p[at]
                ids=cache.meta['dataset_id'][usable];times=cache.meta['time_online'][usable]
                candidates,counts=choose_case_indices(ids,times,target,delta,p);scanned+=1
                count_rows.append({'feature':j,'horizon':h,'fold':fold,'known_available_origins':len(usable),
                    'series':len(np.unique(ids)),**counts,'prediction_sha256':digest(path),
                    'count_unit':'Success/failure count eligible series; miss/false-alarm count eligible origin events'})
                for category,case in candidates.items():
                    if category not in selected:selected[category]={'scope':scope,**trace_case(cache,usable,target,delta,p,j,h,category,case)}
    for category in CATEGORIES:
        if category not in selected:selected[category]={'category':category,'status':'NO_QUALIFYING_CASE_UNDER_FROZEN_RULE'}
    return {'status':'COMPLETE','cases':selected,'eligibility_counts':count_rows,'all_fixed_feature_horizon_fold_panels_scanned':scanned,
        'case_policy_sha256':digest(BASE/'FIGURE_CASE_POLICY_NEXT8.json'),'case_selection_changes_predictions':False,
        'selected_excerpts_are_descriptive_not_out_of_sample_discovery_evidence':True}
