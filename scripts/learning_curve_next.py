"""Fit fixed nested training fractions and reuse unchanged full-size OOF."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
from pathlib import Path
import gc
import json
import sys
import time
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next.diagnostic_training import DiagnosticTraining,ordered_rows,implementation_hash
from src.next.references import verify_reference_sources
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    policy_path = ROOT/'configs/next_learning_curve_policy.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    if policy['training_adapter_sha256']!=implementation_hash():
        raise RuntimeError('Frozen diagnostic training adapter changed')
    verify_reference_sources(ROOT)
    data = prepare_dev(ROOT)
    if policy['split_sha256']!=data.guard.split.digest or policy['seal_lock_sha256']!=data.guard.lock_sha:
        raise RuntimeError('Learning-curve split identity changed')
    outroot = ROOT/'artifacts/next/learning_curve'
    outroot.mkdir(parents=True,exist_ok=True)
    records = []
    for eid in policy['candidates']:
        study = DiagnosticTraining(data,eid)
        out = outroot/eid
        out.mkdir(exist_ok=True)
        identity = {'candidate':eid,'policy_sha256':sha256(policy_path),'script_sha256':sha256(Path(__file__)),'sources':study.sources}
        if (out/'IDENTITY.json').exists():
            if json.loads((out/'IDENTITY.json').read_text(encoding='utf-8'))!=identity:
                raise RuntimeError('Learning-curve resume identity changed')
        else:
            write_json(out/'IDENTITY.json',identity)
        for item in policy['folds']:
            fold = item['outer_fold']
            valid = ordered_rows(study.metadata,item['validation_ids'])
            np.testing.assert_array_equal(valid,np.flatnonzero(study.metadata['fold']==fold))
            for subset in item['subsets']:
                fraction = subset['fraction']
                name = f'fold_{fold}_fraction_{int(100*fraction)}'
                path = out/(name+'.json')
                if path.exists():
                    record = json.loads(path.read_text(encoding='utf-8'))
                    if record['status']!='PASS' or any(sha256(out/n)!=digest for n,digest in record['files_sha256'].items()):
                        raise RuntimeError('Completed learning-curve artifact changed')
                else:
                    tick = time.perf_counter()
                    train = ordered_rows(study.metadata,subset['training_ids'])
                    model,prediction,detail = study.fit(train,valid)
                    parity = study.validate_saved_predictions(model,out/(name+'.joblib'),valid,prediction)
                    np.save(out/(name+'.npy'),prediction,allow_pickle=False)
                    record = {'status':'PASS','candidate':eid,'fold':fold,'training_fraction':fraction,
                        'ts_auc':ts_auc(study.metadata['target'][valid],prediction,study.metadata['time_online'][valid]),
                        'training_series':len(subset['training_ids']),'validation_series':len(item['validation_ids']),
                        'elapsed_seconds':time.perf_counter()-tick,'created_utc':utc_now(),'engineering':detail,**parity,
                        'seal_rows':0,'files_sha256':{n:sha256(out/n) for n in [name+'.joblib',name+'.npy']}}
                    write_json(path,record)
                    print('NEXT LEARNING',eid,fold,fraction,record['ts_auc'],round(record['elapsed_seconds'],1),flush=True)
                    del model,prediction
                    gc.collect()
                records.append(record)
            records.append({'status':'REUSED_FROZEN_PRIMARY_OOF','candidate':eid,'fold':fold,'training_fraction':1.,
                'ts_auc':ts_auc(study.metadata['target'][valid],study.prediction[valid],study.metadata['time_online'][valid]),
                'training_series':8000-len(item['validation_ids']),'validation_series':len(item['validation_ids']),'elapsed_seconds':0.,'seal_rows':0})
        del study
        gc.collect()
        pd.DataFrame([{k:v for k,v in r.items() if k not in ('files_sha256','engineering')} for r in records]).to_csv(outroot/'FOLDS.csv',index=False)
    frame = pd.read_csv(outroot/'FOLDS.csv')
    summary = frame.groupby(['candidate','training_fraction']).ts_auc.agg(['mean','median','min','max']).reset_index()
    summary.to_csv(outroot/'SUMMARY.csv',index=False)
    write_json(outroot/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'new_partial_training_fits':sum(r['status']=='PASS' for r in records),
        'reused_primary_oof_points':sum(r['status']=='REUSED_FROZEN_PRIMARY_OOF' for r in records),
        'policy_sha256':sha256(policy_path),'summary':summary.to_dict('records'),'seal_rows':0,'reduced_new_usage':0,
        'authority':'Training-size sensitivity diagnostic; no changes to primary selection gates.'})
    print('NEXT LEARNING COMPLETE',flush=True)


if __name__=='__main__':
    main()
