"""Paired score and feature impulse responses, with explicit recovery censoring."""
import numpy as np
from src.next5.synthetic_metrics import paired_mean_difference


def paired_curves(arrays):
    changed=np.flatnonzero(~arrays['control']);clean=np.flatnonzero(arrays['control'])
    np.testing.assert_array_equal(arrays['synthetic_seed_index'][changed],arrays['synthetic_seed_index'][clean])
    return changed,clean


def recovery_times(excess,onsets,burst_length,*,tolerance=.01,consecutive=16):
    times=[];censored=[]
    for values,onset in zip(excess,onsets):
        first=int(onset)+burst_length;good=np.abs(values)<=tolerance;found=None
        for position in range(first,len(values)-consecutive+1):
            if np.all(good[position:position+consecutive]):found=position-first;break
        times.append(len(values)-first if found is None else found);censored.append(found is None)
    return np.asarray(times),np.asarray(censored)


def impulse_tables(candidate,reference,*,candidate_id,scope,fold,scenario,menu,lock,counts,feature_names):
    changed,clean=paired_curves(candidate);index=candidate['synthetic_seed_index'][changed]
    onsets=np.asarray([lock['tau_cycle'][int(i)%len(lock['tau_cycle'])] for i in index],dtype=np.int64)
    p={'candidate':candidate['prediction'],'expert':candidate['expert']}
    p.update({name:reference[name] for name in ('BL80','B2','D2') if name in reference})
    excess={name:values[changed]-values[clean] for name,values in p.items()};lags=menu['impulse_lags'];lag_rows=[];summary=[];feature_rows=[]
    for name,values in excess.items():
        for lag in lags:
            at_lag=values[np.arange(len(index)),onsets+lag]
            lag_rows.append({'candidate':candidate_id,'scope':scope,'fold':fold,'scenario':scenario,'model':name,'lag_from_first_impulse':lag,
                'mean_score_excess':float(at_lag.mean()),'q05_score_excess':float(np.quantile(at_lag,.05)),
                'q95_score_excess':float(np.quantile(at_lag,.95)),'positive_excess_fraction':float(np.mean(at_lag>0))})
        metrics={'mean_excess_at_horizon1024':values[:,:1024].mean(axis=1),'last64_excess_at_horizon1024':values[:,960:1024].mean(axis=1),
            'last64_excess_at_horizon2048':values[:,-64:].mean(axis=1),
            'post_impulse_mean_excess_at_horizon2048':np.asarray([v[o:].mean() for v,o in zip(values,onsets)])}
        for metric,vector in metrics.items():summary.append({'candidate':candidate_id,'scope':scope,'fold':fold,'scenario':scenario,'model':name,
            'metric':metric,'estimate':float(vector.mean()),'censored_fraction':None,'recovered_only_median':None})
        recovery,censored=recovery_times(values,onsets,16 if scenario=='OB16' else 1)
        summary.append({'candidate':candidate_id,'scope':scope,'fold':fold,'scenario':scenario,'model':name,
            'metric':'recovery_after_shock_end_abs_excess_le_0_01_for_16_points','estimate':float(np.median(recovery)),
            'censored_fraction':float(censored.mean()),'recovered_only_median':float(np.median(recovery[~censored])) if (~censored).any() else None})
    comparisons=[]
    for name in ('BL80','B2','D2'):
        if name not in excess:continue
        for lag in lags:
            left=excess['candidate'][np.arange(len(index)),onsets+lag];right=excess[name][np.arange(len(index)),onsets+lag]
            comparisons.append({'candidate':candidate_id,'scope':scope,'fold':fold,'scenario':scenario,'reference':name,
                'metric':'score_excess_at_lag','lag_from_first_impulse':lag,'negative_delta_means_less_score_contamination':True,**paired_mean_difference(left,right,counts)})
        for metric,sl in (('mean_excess_at_horizon1024',slice(0,1024)),('last64_excess_at_horizon1024',slice(960,1024)),('last64_excess_at_horizon2048',slice(-64,None))):
            comparisons.append({'candidate':candidate_id,'scope':scope,'fold':fold,'scenario':scenario,'reference':name,'metric':metric,
                'lag_from_first_impulse':None,'negative_delta_means_less_score_contamination':True,
                **paired_mean_difference(excess['candidate'][:,sl].mean(axis=1),excess[name][:,sl].mean(axis=1),counts)})
    features=candidate['features_at_impulse_lags'];difference=features[changed].astype(float)-features[clean].astype(float)
    for j,lag in enumerate(lags):
        for k,name in enumerate(feature_names):
            values=difference[:,j,k];feature_rows.append({'candidate':candidate_id,'scope':scope,'fold':fold,'scenario':scenario,
                'feature':name,'lag_from_first_impulse':lag,'mean_feature_excess':float(values.mean()),
                'q05_feature_excess':float(np.quantile(values,.05)),'q95_feature_excess':float(np.quantile(values,.95))})
    return lag_rows,summary,comparisons,feature_rows
