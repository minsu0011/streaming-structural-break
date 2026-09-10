import numpy as np
from src.scoring.ts_auc import ts_auc

def profile(target,prediction,time_online,ids,metadata):
    y=np.asarray(target);p=np.asarray(prediction);t=np.asarray(time_online);ids=np.asarray(ids)
    rows=[]
    def add(kind,label,mask):
        ym,pm,tm=y[mask],p[mask],t[mask]
        score,steps=ts_auc(ym,pm,tm,return_details=True)
        weight=sum(step['weight'] for step in steps)
        rows.append({'kind':kind,'slice':label,'rows':len(ym),'positive':int(ym.sum()),'negative':int(len(ym)-ym.sum()),
            'pair_weight':weight,'ts_auc':score if weight else None,'score_mean':float(pm.mean()) if len(pm) else None,
            'negative_score_p95':float(np.quantile(pm[ym==0],.95)) if np.any(ym==0) else None,
            'interpretation':'within-time discrimination' if weight else 'no comparable positive-negative pairs; AUC undefined'})
    for low,high in [(0,10),(10,25),(25,50),(50,100),(100,200),(200,400),(400,None)]:
        add('online_time',f'{low}-{high-1}' if high else f'{low}+', (t>=low)&(True if high is None else t<high))
    metadata=metadata.set_index('dataset_id') if 'dataset_id' in metadata.columns else metadata
    tau=metadata.tau_index.reindex(ids).to_numpy(dtype=float)
    age=t-tau
    for low,high in [(0,5),(5,10),(10,20),(20,50),(50,None)]:
        positives=(y==1)&(age>=low)&(True if high is None else age<high)
        add('evidence_age_with_negative_controls',f'{low}-{high-1}' if high else f'{low}+',positives|(y==0))
    for label,chosen in [
        ('early_break',metadata.index[metadata.tau_fraction<1/3]),
        ('middle_break',metadata.index[(metadata.tau_fraction>=1/3)&(metadata.tau_fraction<2/3)]),
        ('late_break',metadata.index[metadata.tau_fraction>=2/3]),
        ('no_break',metadata.index[~metadata.has_break]),
        ('short_historical',metadata.index[metadata.historical_length<2000]),
        ('long_historical',metadata.index[metadata.historical_length>=2000]),
        ('short_online',metadata.index[metadata.online_length<100]),
        ('long_online',metadata.index[metadata.online_length>=100])]:
        add('series_metadata',label,np.isin(ids,chosen))
    if {'historical_lag1','historical_excess_kurtosis'}.issubset(metadata.columns):
        for label,chosen in [
            ('negative_acf_below_minus_0p3',metadata.index[metadata.historical_lag1<-.3]),
            ('weak_acf_abs_below_0p1',metadata.index[metadata.historical_lag1.abs()<.1]),
            ('strong_positive_acf_above_0p5',metadata.index[metadata.historical_lag1>.5]),
            ('near_unit_acf_above_0p9',metadata.index[metadata.historical_lag1>.9]),
            ('heavy_tail_kurtosis_above_10',metadata.index[metadata.historical_excess_kurtosis>10]),
            ('extreme_tail_kurtosis_above_100',metadata.index[metadata.historical_excess_kurtosis>100]),
            ('mild_tail_kurtosis_below_1',metadata.index[metadata.historical_excess_kurtosis<1])]:
            add('historical_structure',label,np.isin(ids,chosen))
    return rows
