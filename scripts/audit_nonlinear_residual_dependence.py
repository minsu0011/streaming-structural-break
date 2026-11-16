"""Look for predictable historical residual magnitude, without outcome tables."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.loader import series_frames
from src.features.config import CONFIG
from src.features.ar_order_input import historical_filter
from src.validation.splits import FrozenSplit
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    split=FrozenSplit(ROOT/'artifacts/validation');selected=set(split.frame.loc[split.frame.role=='DEVELOPMENT_POOL','dataset_id'])
    config=CONFIG.variant(normalization='median_mad',scales=(5,20,160));rows=[];covered=set()
    for sid,frame in series_frames(ROOT/'data/raw/X_train.parquet'):
        if sid not in selected:continue
        h=frame.loc[frame.period==1,'value'].to_numpy(dtype=np.float32)
        _,_,_,residual=historical_filter(h,config,8);x=residual.astype(np.float64);x-=x.mean();q=x*x
        for label,values in [('absolute',abs(x)),('squared',q),('squared_winsor99',np.minimum(q,np.quantile(q,.99)))]:
            values=values-values.mean();variance=float(np.dot(values,values)/len(values))
            row={'dataset_id':sid,'fold':split.assignment(sid)[1],'transform':label,'historical_points_used':len(x)}
            for lag in (1,2,4,8,16):row[f'acf_{lag}']=float(np.dot(values[lag:],values[:-lag])/(len(values)-lag)/variance) if variance>0 else 0.
            rows.append(row)
        covered.add(sid)
        if len(covered)%2000==0:print('NONLINEAR HISTORY',len(covered),'DEV series',flush=True)
    if covered!=selected:raise RuntimeError('Historical DEV coverage differs from the frozen split')
    result=pd.DataFrame(rows);summaries=[]
    for label,frame in result.groupby('transform'):
        for lag in (1,2,4,8,16):
            values=frame[f'acf_{lag}'];summaries.append({'transform':label,'lag':lag,'mean_acf':float(values.mean()),'median_acf':float(values.median()),
                'p90_acf':float(values.quantile(.9)),'fraction_above_0p05':float((values>.05).mean()),'fraction_above_0p1':float((values>.1).mean()),'fraction_above_0p2':float((values>.2).mean())})
    directory=ROOT/'artifacts/real/nonlinear_residual_audit';directory.mkdir(parents=True,exist_ok=True)
    result.to_csv(directory/'SERIES_DIAGNOSTICS.csv',index=False);pd.DataFrame(summaries).to_csv(directory/'SUMMARY.csv',index=False)
    report={'status':'PASS','timestamp':utc_now(),'series':len(covered),'selection':'All 8000 frozen DEV historical segments; no outcome table opened',
        'input_scope':'Parquet batches also contain online and sealed raw rows; only selected DEV historical values enter the calculation.',
        'AR_order':8,'config':config.__dict__,'split_sha256':split.digest,'source_sha256':sha256(__file__),'AR_source_sha256':sha256(ROOT/'src/features/ar_order_input.py'),
        'summary':summaries,'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    write_json(directory/'AUDIT.json',report);print(pd.DataFrame(summaries).to_string(index=False),flush=True)


if __name__=='__main__':main()
