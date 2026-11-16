"""Historical-only residual dependence probe, without online values or outcomes."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.loader import series_frames
from src.features.config import CONFIG
from src.features.ar_order_input import historical_filter
from src.validation.splits import FrozenSplit
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    split=FrozenSplit(ROOT/'artifacts/validation');selected=set()
    for fold in range(5):
        selected.update(split.frame.loc[(split.frame.role=='DEVELOPMENT_POOL')&(split.frame.fold==fold),'dataset_id'].sort_values().head(128))
    config=CONFIG.variant(normalization='median_mad',scales=(5,20,160));rows=[]
    for sid,frame in series_frames(ROOT/'data/raw/X_train.parquet'):
        if sid not in selected:continue
        h=frame.loc[frame.period==1,'value'].to_numpy(dtype=np.float32)
        for order in (2,4,8):
            _,coef,_,residual=historical_filter(h,config,order);x=residual.astype(np.float64);x-=x.mean();variance=float(np.dot(x,x)/len(x))
            row={'dataset_id':sid,'fold':split.assignment(sid)[1],'order':order,'historical_length':len(h),
                'coefficient_l2':float(np.linalg.norm(coef)),'residual_variance':variance}
            for lag in (1,2,4,8,16,32):row[f'acf_{lag}']=float(np.dot(x[lag:],x[:-lag])/(len(x)-lag)/variance) if variance>0 else 0.
            rows.append(row)
        if len(rows)==len(selected)*3:break
    result=pd.DataFrame(rows);summaries=[]
    for order,frame in result.groupby('order'):
        for lag in (1,2,4,8,16,32):
            values=frame[f'acf_{lag}'].abs()
            summaries.append({'order':int(order),'lag':lag,'mean_absolute_acf':float(values.mean()),'median_absolute_acf':float(values.median()),
                'p90_absolute_acf':float(values.quantile(.9)),'fraction_abs_above_0p05':float((values>.05).mean())})
    destination=ROOT/'artifacts/real/historical_residual_audit';destination.mkdir(parents=True,exist_ok=True)
    result.to_csv(destination/'SERIES_DIAGNOSTICS.csv',index=False);pd.DataFrame(summaries).to_csv(destination/'ACF_SUMMARY.csv',index=False)
    report={'status':'PASS','timestamp':utc_now(),'series':len(selected),'selection':'first 128 ascending DEV IDs per frozen fold; no outcome-based selection',
        'input_scope':'Only historical rows enter the residual filter. Parquet batches also contain online rows, which are not used in this analysis. No label or tau table is opened.','orders':[2,4,8],'config':config.__dict__,
        'source_sha256':sha256(ROOT/'src/features/ar_order_input.py'),'split_sha256':split.digest,'summary':summaries,
        'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    write_json(destination/'AUDIT.json',report);print(pd.DataFrame(summaries).to_string(index=False),flush=True)


if __name__=='__main__':main()
