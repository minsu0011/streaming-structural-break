import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.data.loader import load_training
from src.utils.artifacts import sha256,utc_now,write_json
from src.validation.splits import freeze_group_split
from src.data.grouping import freeze_policy,historical_signature,group_histories

def raw_manifest(directory,output):
    records=[]
    for path in sorted(Path(directory).glob('*.parquet')):
        parquet=pq.ParquetFile(path)
        metadata=json.loads((parquet.schema_arrow.metadata or {}).get(b'pandas',b'{}'))
        records.append({'name':path.name,'bytes':path.stat().st_size,'rows':parquet.metadata.num_rows,
            'row_groups':parquet.metadata.num_row_groups,'columns':parquet.schema_arrow.names,
            'dtypes':{field.name:str(field.type) for field in parquet.schema_arrow},
            'index_metadata':metadata.get('index_columns',[]),'sha256':sha256(path),'mtime_ns':path.stat().st_mtime_ns})
    result={'timestamp':utc_now(),'status':'ACQUIRED' if records else 'BLOCKED_AUTH','files':records,'raw_modified':False}
    if Path(output).exists():
        previous=json.loads(Path(output).read_text(encoding='utf-8'))
        identity = lambda items: sorted((r['name'], r['bytes'], r['sha256']) for r in items)
        if previous.get('files') and identity(previous['files']) != identity(records):
            raise RuntimeError('Immutable raw manifest mismatch')
    result['immutable_identity'] = ['name', 'bytes', 'sha256']
    result['mtime_authority'] = 'informational_only'
    write_json(output,result)
    return result

def _finite_stats(x):
    x=np.asarray(x,dtype=float)
    finite=x[np.isfinite(x)]
    if not len(finite):
        return {'mean':None,'std':None,'median':None,'mad':None,'min':None,'max':None,'nan_count':int(np.isnan(x).sum()),'inf_count':int(np.isinf(x).sum())}
    median=np.median(finite)
    return {'mean':float(finite.mean()),'std':float(finite.std()),'median':float(median),'mad':float(np.median(abs(finite-median))),
        'min':float(finite.min()),'max':float(finite.max()),'nan_count':int(np.isnan(x).sum()),'inf_count':int(np.isinf(x).sum())}

def audit_training(directory,output_directory,split_directory):
    output_directory=Path(output_directory); output_directory.mkdir(parents=True,exist_ok=True)
    policy_hash=freeze_policy(output_directory)
    rows=[]; histories=[]; fingerprints=[]; identities=[]
    for series in load_training(directory,development_only=False):
        h,o,tau=series.historical,series.online,series.tau
        stats=_finite_stats(h); pre=o if tau is None else o[:tau]
        z=(h-(stats['mean'] or 0))/max(stats['std'] or 0,1e-8)
        finite_z=z[np.isfinite(z)]
        identity,fingerprint=historical_signature(h)
        identities.append({'dataset_id':series.dataset_id,**identity})
        histories.append(np.asarray(h,dtype=np.float32).copy()); fingerprints.append(fingerprint)
        rows.append({'dataset_id':series.dataset_id,'historical_length':len(h),'online_length':len(o),'has_break':tau is not None,
            'tau_index':tau,'tau_fraction':None if tau is None or not len(o) else tau/len(o),
            'pre_break_online_length':len(pre),'post_break_length':0 if tau is None else len(o)-tau,
            **{f'historical_{k}':v for k,v in stats.items()},
            'online_nan_count':int(np.isnan(o).sum()),'online_inf_count':int(np.isinf(o).sum()),
            'prebreak_mean_departure':None if not len(pre) else float((np.nanmean(pre)-(stats['mean'] or 0))/max(stats['std'] or 0,1e-8)),
            'historical_lag1':float(np.mean(finite_z[1:]*finite_z[:-1])) if len(finite_z)>1 else None,
            'historical_excess_kurtosis':float(np.mean(finite_z**4)-3) if len(finite_z) else None,
            'full_series_sha256':hashlib.sha256(np.concatenate([h,o]).astype('<f8').tobytes()).hexdigest(),
            **identity})
    frame=pd.DataFrame(rows)
    frame.to_parquet(output_directory/'SERIES_METADATA.parquet',index=False)
    groups,duplicates=group_histories(pd.DataFrame(identities),histories,fingerprints,output_directory)
    split=freeze_group_split(groups,split_directory,group_policy_sha256=policy_hash,raw_sha256=sha256(Path(directory)/'X_train.parquet'))
    report={'status':'PASS','series_count':len(frame),'break_count':int(frame.has_break.sum()),'no_break_count':int((~frame.has_break).sum()),
        'break_fraction':float(frame.has_break.mean()),'duplicate_full_series':int(frame.full_series_sha256.duplicated().sum()),
        'duplicate_historical':int(frame.historical_sha256.duplicated().sum()),'identical_prefix64':int(frame.prefix64_sha256.duplicated().sum()),
        'near_constant_count':int((frame.historical_std<1e-8).sum()),'short_online_lt25':int((frame.online_length<25).sum()),
        'metadata_only_for_diagnostics':['online_length','tau_index','tau_fraction','has_break','post_break_length'],
        'distributions':frame.select_dtypes(include='number').describe(percentiles=[.01,.1,.5,.9,.99]).replace([np.nan,np.inf,-np.inf],None).to_dict(),
        'seal_evaluated':False,'LOCAL_FINAL_SEAL_OPENED':False,
        'group_count':duplicates['group_count'],'non_singleton_groups':duplicates['non_singleton_groups'],
        'historical_length_range':[int(frame.historical_length.min()),int(frame.historical_length.max())],
        'online_length_range':[int(frame.online_length.min()),int(frame.online_length.max())],
        'split_counts':split.lock['counts'],'audit_order':['raw_metadata','exact_duplicates','outcome_blind_near_groups','group_split_freeze'],
        'duplicate_review_required':False,'limitations':['Post-hoc signatures are not true DGP labels.',*__import__('src.data.grouping',fromlist=['POLICY']).POLICY['limitations']]}
    write_json(output_directory/'DATA_AUDIT.json',report)
    (output_directory/'DATA_AUDIT.md').write_text('# Training audit\n\n'+json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report
