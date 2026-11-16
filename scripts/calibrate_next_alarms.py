"""Fit smaller train-only models, freeze null thresholds, then score outer data."""
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
from src.next.alarm_calibration import null_maximum_threshold,series_maxima,first_alarm_records,summarize_alarms
from src.next.references import verify_reference_sources
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    policy_path = ROOT/'configs/next_alarm_calibration_policy.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    if policy['training_adapter_sha256']!=implementation_hash() or policy['alarm_code_sha256']!=sha256(ROOT/'src/next/alarm_calibration.py'):
        raise RuntimeError('Frozen alarm diagnostic source changed')
    verify_reference_sources(ROOT)
    data = prepare_dev(ROOT)
    if policy['split_sha256']!=data.guard.split.digest or policy['seal_lock_sha256']!=data.guard.lock_sha:
        raise RuntimeError('Alarm diagnostic split identity changed')
    tau = dict(zip(map(int,data.ids),map(int,data.series['tau'])))
    outroot = ROOT/'artifacts/next/alarm_calibration'
    outroot.mkdir(parents=True,exist_ok=True)
    folds,series_records = [],[]
    for eid in policy['candidates']:
        study = DiagnosticTraining(data,eid)
        out = outroot/eid
        out.mkdir(exist_ok=True)
        identity = {'candidate':eid,'policy_sha256':sha256(policy_path),'script_sha256':sha256(Path(__file__)),'sources':study.sources}
        if (out/'IDENTITY.json').exists():
            if json.loads((out/'IDENTITY.json').read_text(encoding='utf-8'))!=identity:
                raise RuntimeError('Nested calibration identity changed')
        else:
            write_json(out/'IDENTITY.json',identity)
        for item in policy['folds']:
            fold = item['outer_fold']
            folder = out/f'fold_{fold}'
            folder.mkdir(exist_ok=True)
            data.guard.partition(item['training_ids'],item['calibration_ids'])
            data.guard.partition(item['training_ids']+item['calibration_ids'],item['validation_ids'])
            if set(item['training_ids']+item['calibration_ids']+item['validation_ids'])!=set(data.guard.dev_ids):
                raise RuntimeError('Nested partition does not cover all DEV series exactly')
            complete = folder/'RESULTS.json'
            if complete.exists():
                record = json.loads(complete.read_text(encoding='utf-8'))
                if record['status']!='PASS' or any(sha256(folder/name)!=digest for name,digest in record['files_sha256'].items()):
                    raise RuntimeError('Completed alarm diagnostic changed')
            else:
                tick = time.perf_counter()
                train = ordered_rows(study.metadata,item['training_ids'])
                calibration = ordered_rows(study.metadata,item['calibration_ids'])
                valid = ordered_rows(study.metadata,item['validation_ids'])
                model,cal_prediction,detail = study.fit(train,calibration)
                parity = study.validate_saved_predictions(model,folder/'model.joblib',calibration,cal_prediction)
                cal_ids,maxima = series_maxima(study.metadata['dataset_id'][calibration],cal_prediction)
                null = np.asarray([tau[int(sid)]<0 for sid in cal_ids])
                thresholds = [null_maximum_threshold(maxima[null],alpha) for alpha in policy['alphas']]
                write_json(folder/'THRESHOLDS.json',{'candidate':eid,'outer_fold':fold,'thresholds':thresholds,
                    'model_sha256':sha256(folder/'model.joblib'),'created_utc':utc_now(),
                    'outer_predictions_computed_yet':False,'calibration_null_ids':cal_ids[null].astype(int).tolist()})
                np.save(folder/'calibration_prediction.npy',cal_prediction,allow_pickle=False)
                # The calibration artifact exists before the first outer prediction.
                prediction = np.asarray(model.predict(study.design[valid]),dtype=np.float32)
                np.save(folder/'outer_prediction.npy',prediction,allow_pickle=False)
                per_series,summaries = [],[]
                for threshold in thresholds:
                    frame = first_alarm_records(study.metadata['dataset_id'][valid],study.metadata['time_online'][valid],prediction,tau,threshold['threshold'])
                    frame['candidate'],frame['fold'],frame['nominal_alpha'] = eid,fold,threshold['nominal_alpha']
                    per_series.append(frame)
                    summaries.append({'candidate':eid,'fold':fold,**threshold,**summarize_alarms(frame)})
                pd.concat(per_series,ignore_index=True).to_csv(folder/'SERIES.csv',index=False)
                names = ['model.joblib','THRESHOLDS.json','calibration_prediction.npy','outer_prediction.npy','SERIES.csv']
                record = {'status':'PASS','candidate':eid,'fold':fold,'summaries':summaries,
                    'secondary_smaller_model_outer_ts_auc':ts_auc(study.metadata['target'][valid],prediction,study.metadata['time_online'][valid]),
                    'training_series':len(item['training_ids']),'calibration_series':len(item['calibration_ids']),
                    'validation_series':len(item['validation_ids']),'engineering':detail,**parity,
                    'elapsed_seconds':time.perf_counter()-tick,'created_utc':utc_now(),'seal_rows':0,
                    'files_sha256':{name:sha256(folder/name) for name in names}}
                write_json(complete,record)
                print('NEXT ALARM',eid,fold,record['secondary_smaller_model_outer_ts_auc'],round(record['elapsed_seconds'],1),flush=True)
                del model,prediction,cal_prediction
                gc.collect()
            folds.extend(record['summaries'])
            series_records.append(pd.read_csv(folder/'SERIES.csv'))
        del study
        gc.collect()
    all_series = pd.concat(series_records,ignore_index=True)
    pd.DataFrame(folds).to_csv(outroot/'FOLDS.csv',index=False)
    all_series.to_csv(outroot/'SERIES.csv',index=False)
    aggregate = [{'candidate':eid,'nominal_alpha':alpha,**summarize_alarms(frame)} for (eid,alpha),frame in all_series.groupby(['candidate','nominal_alpha'])]
    pd.DataFrame(aggregate).to_csv(outroot/'SUMMARY.csv',index=False)
    write_json(outroot/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'refits':10,'summary':aggregate,
        'policy_sha256':sha256(policy_path),'seal_rows':0,'reduced_new_usage':0,
        'authority':'Secondary first-alarm diagnostic on fixed outer DEV folds; no production threshold selected and no refit after calibration.'})
    print('NEXT ALARM COMPLETE',flush=True)


if __name__=='__main__':
    main()
