"""Check whether ARCH(1) leaves a material historical lag-two magnitude signal.

This is an outcome-blind mechanism audit. No candidate is fitted or scored.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.features.config import CONFIG
from src.features.ar_order_input import historical_filter
from src.features.arch_input import historical_variance_filter
from src.data.loader import series_frames
from src.validation.splits import FrozenSplit
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    directory=ROOT/'artifacts/real/arch_historical_residual_audit'
    if (directory/'POLICY.json').exists():raise RuntimeError('Historical audit policy already exists')
    split=FrozenSplit(ROOT/'artifacts/validation');ids=set(split.frame.loc[split.frame.role=='DEVELOPMENT_POOL','dataset_id'])
    policy={'timestamp':utc_now(),'scope':'All DEV historical segments only; no outcome table or online values enter the calculation.',
        'hypothesis':'A second conditional-variance lag is worth considering only if the already fixed ARCH(1) transform leaves historical lag-two magnitude correlation in a material part of DEV.',
        'transforms':['squared_winsor99','absolute'],'lags':[1,2,4,8,16],
        'followup_trigger':'At least 10% of DEV histories have post-ARCH(1) winsorized squared-residual ACF2 > 0.1. Trigger is checked before any ARCH(2) model score and does not change frozen model-promotion gates.',
        'split_sha256':split.digest,'script_sha256':sha256(__file__),'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    write_json(directory/'POLICY.json',policy);rows=[];started=time.perf_counter();covered=set()
    config=CONFIG.variant(normalization='median_mad',scales=(5,20,160))
    for sid,frame in series_frames(ROOT/'data/raw/X_train.parquet'):
        if sid not in ids:continue
        historical=frame.loc[frame.period==1,'value'].to_numpy(dtype=np.float32)
        _,_,_,residual=historical_filter(historical,config,8)
        _,_,standardized=historical_variance_filter(residual,config)
        for stage,x in [('AR8',residual[1:].astype(np.float64)),('AR8_ARCH1',standardized.astype(np.float64))]:
            x=x-x.mean();q=x*x
            for label,values in [('squared_winsor99',np.minimum(q,np.quantile(q,.99))),('absolute',abs(x))]:
                values=values-values.mean();variance=float(np.dot(values,values)/len(values))
                row={'dataset_id':sid,'fold':split.assignment(sid)[1],'stage':stage,'transform':label,'historical_points_used':len(x)}
                for lag in policy['lags']:row[f'acf_{lag}']=float(np.dot(values[lag:],values[:-lag])/(len(values)-lag)/variance) if variance>0 else 0.
                rows.append(row)
        covered.add(sid)
        if len(covered)%2000==0:print('ARCH HISTORY AUDIT',len(covered),'DEV histories',flush=True)
    if covered!=ids:raise AssertionError('Historical DEV coverage differs')
    table=pd.DataFrame(rows);summaries=[]
    for (stage,transform),frame in table.groupby(['stage','transform']):
        for lag in policy['lags']:
            values=frame[f'acf_{lag}'];summaries.append({'stage':stage,'transform':transform,'lag':lag,'mean_acf':float(values.mean()),'median_acf':float(values.median()),
                'p90_acf':float(values.quantile(.9)),'fraction_above_0p1':float((values>.1).mean()),'fraction_above_0p2':float((values>.2).mean())})
    trigger=next(r for r in summaries if r['stage']=='AR8_ARCH1' and r['transform']=='squared_winsor99' and r['lag']==2)['fraction_above_0p1']
    table.to_csv(directory/'SERIES_DIAGNOSTICS.csv',index=False);pd.DataFrame(summaries).to_csv(directory/'SUMMARY.csv',index=False)
    report={'status':'PASS','timestamp':utc_now(),'policy':policy,'summaries':summaries,'series':len(covered),'elapsed_seconds':time.perf_counter()-started,
        'ARCH2_followup_trigger_value':trigger,'ARCH2_followup_trigger_met':trigger>=.1,'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False,
        'interpretation':'Historical fit residuals are in-sample diagnostics of remaining dependence, not outcome performance or proof of forecast calibration.'}
    write_json(directory/'RESULTS.json',report);print('ARCH2 FOLLOWUP TRIGGER',trigger,trigger>=.1,flush=True)


if __name__=='__main__':main()
