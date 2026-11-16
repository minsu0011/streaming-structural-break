"""Run the prospectively fixed controlled null/shift diagnostic, with resumes."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import gc
import json
import sys
import time
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.synthetic_variance_stress import generate_series,HistoricalTemplate,source_hash
from src.next.alarm_calibration import null_maximum_threshold,first_alarm_records,summarize_alarms
from src.next.candidate_io import load_full_oof,model_class
from src.next.data import prepare_dev
from src.next.references import verify_reference_sources
from src.next.fast_variance_initialization import PreparedFastVariancePredictor,source_hash as fast_hash
from src.models.cross_order import CrossOrderBlendBundle
from src.models.arch_calibrated import ARCHCalibratedBundle
from src.streaming.prepared import prepare_model,source_hashes
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    started = time.perf_counter()
    policy_path = ROOT/'configs/next_synthetic_stress_policy.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    if policy['generator_sha256']!=source_hash() or policy['alarm_code_sha256']!=sha256(ROOT/'src/next/alarm_calibration.py'):
        raise RuntimeError('Frozen controlled-stress source changed')
    verify_reference_sources(ROOT)
    data = prepare_dev(ROOT)
    candidate,_,_,_,oof_sources = load_full_oof(data,'V007_VARIANCE_FORECAST')
    prepared,receipts = {},{}
    for role,path in policy['models'].items():
        if sha256(ROOT/path)!=policy['model_sha256'][role]:
            raise RuntimeError('Controlled-stress fixed model changed')
        if role=='PRIMARY':
            model = CrossOrderBlendBundle.load(ROOT/path)
        elif role=='HIGHEST_MEAN':
            model = ARCHCalibratedBundle.load(ROOT/path)
        else:
            model = model_class(candidate).load(ROOT/path)
        prepared[role] = prepare_model(model,fast_initialization=True) if role in ('PRIMARY','HIGHEST_MEAN') else PreparedFastVariancePredictor(model)
        receipts[role] = source_hashes(model,fast_initialization=True) if role in ('PRIMARY','HIGHEST_MEAN') else {'fast_runtime_sha256':fast_hash(),'oof_sources':oof_sources}
    # Raw synthetic values/predictions stay outside the small handoff reports.
    cache = ROOT/'data/next/synthetic_variance_stress_v1'
    cache.mkdir(parents=True,exist_ok=True)
    out = ROOT/'artifacts/next/synthetic_variance_stress'
    out.mkdir(parents=True,exist_ok=True)
    identity = {'policy_sha256':sha256(policy_path),'script_sha256':sha256(Path(__file__)),'runtime_receipts':receipts,
        'cache_scope':'Entirely generated synthetic sequences, distinct from official DEV/seal rows.'}
    manifest = out/'IDENTITY.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8'))!=identity:
            raise RuntimeError('Controlled-stress resume identity changed')
    else:
        write_json(manifest,identity)
    roles = list(prepared)
    blocks = []
    for partition in (0,1):
        scenarios = ['NO_CHANGE'] if partition==0 else policy['scenarios']
        count = policy['calibration_series_per_regime'] if partition==0 else policy['evaluation_series_per_regime']
        if partition==1 and not (out/'THRESHOLDS.json').exists():
            raise RuntimeError('Synthetic null thresholds must be frozen before changed-series evaluation')
        for regime in range(len(policy['regimes'])):
            for first in range(0,count,100):
                last = min(first+100,count)
                name = f'partition_{partition}_regime_{regime}_block_{first:04d}'
                path = cache/(name+'.npy')
                record_path = out/(name+'.json')
                if record_path.exists():
                    record = json.loads(record_path.read_text(encoding='utf-8'))
                    if record['status']!='PASS' or record['prediction_sha256']!=sha256(path):
                        raise RuntimeError('Completed controlled-stress block changed')
                else:
                    tick = time.perf_counter()
                    values = np.empty((len(roles),len(scenarios),last-first,policy['online_points']),dtype=np.float32)
                    clone_checks = 0
                    for j,index in enumerate(range(first,last)):
                        historical,online,tau,polarity = generate_series(policy,partition,regime,index)
                        for r,role in enumerate(roles):
                            template = HistoricalTemplate(prepared[role],historical,role)
                            for s,scenario in enumerate(scenarios):
                                points = online[scenario]
                                prediction = template.predict(points)
                                if index<2:
                                    fresh = prepared[role].new_state(historical)
                                    reference = np.asarray([fresh.predict_one(point) for point in points],dtype=np.float32)
                                    np.testing.assert_array_equal(prediction,reference)
                                    clone_checks += 1
                                values[r,s,j] = prediction
                    if not np.isfinite(values).all() or np.any((values<0)|(values>1)):
                        raise RuntimeError('Controlled-stress model output became invalid')
                    np.save(path,values,allow_pickle=False)
                    record = {'status':'PASS','partition':partition,'regime':regime,'first_index':first,'last_index_exclusive':last,
                        'shape':list(values.shape),'prediction_sha256':sha256(path),'clone_fresh_constructor_equal_checks':clone_checks,
                        'elapsed_seconds':time.perf_counter()-tick,'created_utc':utc_now(),'official_data_rows':0}
                    write_json(record_path,record)
                    print('NEXT CONTROLLED STRESS',partition,regime,last,count,round(record['elapsed_seconds'],1),flush=True)
                    del values
                    gc.collect()
                blocks.append(record)
        if partition==0:
            maxima = {role:[] for role in roles}
            for record in blocks:
                name = f"partition_0_regime_{record['regime']}_block_{record['first_index']:04d}.npy"
                values = np.load(cache/name,mmap_mode='r',allow_pickle=False)
                for r,role in enumerate(roles):
                    maxima[role].extend(values[r,0].max(axis=1).tolist())
            thresholds = {role:[null_maximum_threshold(maxima[role],alpha) for alpha in policy['alphas']] for role in roles}
            threshold_path = out/'THRESHOLDS.json'
            if threshold_path.exists():
                prior = json.loads(threshold_path.read_text(encoding='utf-8'))
                if prior['thresholds']!=thresholds:
                    raise RuntimeError('Reconstructed synthetic null thresholds changed')
            else:
                write_json(threshold_path,{'status':'LOCKED','thresholds':thresholds,'created_utc':utc_now(),
                    'null_series_per_model':5000,'changed_series_evaluated_yet':False,'policy_sha256':sha256(policy_path)})
    thresholds = json.loads((out/'THRESHOLDS.json').read_text(encoding='utf-8'))['thresholds']
    summaries,auc_rows = [],[]
    for regime,regime_policy in enumerate(policy['regimes']):
        relevant = [b for b in blocks if b['partition']==1 and b['regime']==regime]
        arrays = [np.load(cache/f"partition_1_regime_{regime}_block_{b['first_index']:04d}.npy",allow_pickle=False) for b in relevant]
        values = np.concatenate(arrays,axis=2)
        count,length = values.shape[2:]
        synthetic_ids = 1000000+regime*10000+np.arange(count)
        ids = np.repeat(synthetic_ids,length)
        times = np.tile(np.arange(length,dtype=np.int32),count)
        taus = np.asarray([policy['break_times'][j%len(policy['break_times'])] for j in range(count)])
        target = (times>=np.repeat(taus,length)).astype(np.uint8)
        for r,role in enumerate(roles):
            for s,scenario in enumerate(policy['scenarios']):
                prediction = values[r,s].ravel()
                tau_map = dict(zip(map(int,synthetic_ids),map(int,taus if s else np.full(count,-1))))
                for threshold in thresholds[role]:
                    records = first_alarm_records(ids,times,prediction,tau_map,threshold['threshold'])
                    summaries.append({'model':role,'regime':regime_policy['name'],'scenario':scenario,
                        'nominal_alpha':threshold['nominal_alpha'],'threshold':threshold['threshold'],**summarize_alarms(records)})
                if s:
                    # Compare each changed cell to its paired stationary null
                    # futures. These are synthetic time-aligned AUCs only.
                    auc = ts_auc(np.r_[target,np.zeros_like(target)],np.r_[prediction,values[r,0].ravel()],np.r_[times,times])
                    auc_rows.append({'model':role,'regime':regime_policy['name'],'scenario':scenario,'synthetic_ts_auc':auc,
                        'changed_series':count,'paired_null_series':count})
        pd.DataFrame(summaries).to_csv(out/'ALARM_SUMMARY.csv',index=False)
        pd.DataFrame(auc_rows).to_csv(out/'SYNTHETIC_AUC.csv',index=False)
        del arrays,values
        gc.collect()
    write_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'policy_sha256':sha256(policy_path),
        'identity_sha256':sha256(manifest),'blocks':len(blocks),'unique_calibration_sequences':5000,
        'unique_evaluation_futures':35000,'model_sequence_evaluations':120000,'scalar_predictions':122880000,
        'clone_fresh_constructor_equal_checks':sum(b['clone_fresh_constructor_equal_checks'] for b in blocks),
        'elapsed_this_invocation_seconds':time.perf_counter()-started,'official_rows_in_controlled_diagnostic':0,
        'official_access':'Frozen DEV model and original OOF integrity restoration only; every controlled diagnostic input and target is generated independently.',
        'seal_rows':0,'reduced_new_usage':0,'model_refits':0,
        'files_sha256':{name:sha256(out/name) for name in ['ALARM_SUMMARY.csv','SYNTHETIC_AUC.csv','THRESHOLDS.json']},
        'interpretation':'Controlled synthetic mechanism/null diagnostic only. No synthetic score or calibrated threshold enters primary DEV promotion or deployment. Numerical/model failures would require investigation, not exclusion of an inconvenient cell.'})
    print('NEXT CONTROLLED STRESS COMPLETE',flush=True)


if __name__=='__main__':
    main()
