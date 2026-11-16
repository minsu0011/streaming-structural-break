"""Fixed train-null thresholds, evaluated once on frozen held-out DEV folds."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import gc,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.validation.alarm_diagnostics import training_threshold,first_alarm_outcome,summarize_outcomes
from src.validation.splits import FrozenSplit,assert_development_ids
from src.validation.adaptive import ResearchBudget
from src.models.representations import bundle_class,representation_feature_names
from src.models.baselines import BaselineDetector
from src.data.loader import load_training
from src.utils.artifacts import write_json,sha256,utc_now

CANDIDATES=('B1_official_ewma','M7_cross_order_AR4_LGB_AR8_HGB_equal')
ALPHAS=(.01,.05)


def main():
    if not ResearchBudget(ROOT/'artifacts/validation/REAL_SESSION.json').can_start(1800):raise RuntimeError('Final buffer protected')
    directory=ROOT/'artifacts/real/first_alarm_diagnostics'
    if (directory/'RESULTS.json').exists():raise RuntimeError('The fixed alarm diagnostic already completed')
    split=FrozenSplit(ROOT/'artifacts/validation');registry=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text())
    lookup={r['experiment_id']:r for r in registry['results']}
    policy={'timestamp':utc_now(),'candidates':CANDIDATES,'training_null_series_tail_fractions':ALPHAS,
        'threshold':'higher quantile (1-alpha) of per-series maximum prediction on no-break TRAIN groups, predicted by that same outer-fold model; strict > alarm',
        'validation':'Untouched held-out fold OOF probability, first threshold crossing only; no reset after a premature alarm',
        'training_predictions':'In-sample train-null predictions for the fitted learner. This may underestimate held-out false alarms; it is deliberately measured, not claimed as guaranteed calibration.',
        'no_crossfold_oof_for_threshold_calibration':True,'no_validation_labels_or_predictions_used_to_choose_threshold':True,
        'horizons':'Any alarm over the observed online prefix. Fixed-delay rates include only break series observed for that full post-break interval; premature alarms count as failures.',
        'not_a_new_model_or_selection_rule':True,'thresholds_never_change_continuous_competition_predictions':True,
        'split_sha256':split.digest,'source_sha256':sha256(__file__),'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    if (directory/'POLICY.json').exists():raise RuntimeError('Existing policy requires an explicit reproduction run')
    write_json(directory/'POLICY.json',policy);started=time.perf_counter()
    ids=split.frame.loc[split.frame.role=='DEVELOPMENT_POOL','dataset_id'].tolist()
    metadata=pd.read_parquet(ROOT/'artifacts/audits/SERIES_METADATA.parquet',filters=[('dataset_id','in',ids)]).set_index('dataset_id')
    baseline={}
    for series in load_training(ROOT/'data/raw',split=split):
        detector=BaselineDetector(CANDIDATES[0],series.historical)
        baseline[series.dataset_id]=np.array([detector.predict_one(p) for p in series.online],dtype=np.float32)
    if set(baseline)!=set(ids):raise RuntimeError('Baseline diagnostic DEV coverage differs')
    aggregate=[];fold_results=[];threshold_records=[]
    for eid in CANDIDATES:
        result=lookup[eid];outcomes={alpha:[] for alpha in ALPHAS}
        if eid!=CANDIDATES[0]:
            manifest=json.loads((ROOT/'data/processed/oof'/eid/'MANIFEST.json').read_text());cache=ROOT/manifest['cache_directory'];cache_manifest=json.loads((cache/'MANIFEST.json').read_text())
            if manifest['split_sha256']!=split.digest:raise RuntimeError('Candidate OOF split differs')
            if not all(sha256(cache/f'fold_{k}.parquet')==cache_manifest['fold_sha256'][str(k)] for k in range(5)):raise RuntimeError('Feature cache changed')
        for fold in range(5):
            val_ids=set(split.frame.loc[(split.frame.role=='DEVELOPMENT_POOL')&(split.frame.fold==fold),'dataset_id'])
            train_ids=set(ids)-val_ids;null_ids=set(metadata.index[~metadata.has_break])&train_ids
            if {split.groups[i] for i in null_ids}&{split.groups[i] for i in val_ids}:raise RuntimeError('Threshold groups cross validation boundary')
            if eid==CANDIDATES[0]:maxima=[float(baseline[sid].max()) for sid in sorted(null_ids)];validation_predictions={sid:baseline[sid] for sid in val_ids}
            else:
                model=bundle_class(result['representation']).load(ROOT/'artifacts/models/real'/eid/f'fold_{fold}.joblib')
                names=representation_feature_names(result['representation'],model.feature_config);maxima=[];covered=set()
                for k in range(5):
                    if k==fold:continue
                    frame=pd.read_parquet(cache/f'fold_{k}.parquet',columns=['dataset_id',*names]);frame=frame.loc[frame.dataset_id.isin(null_ids)].copy()
                    assert_development_ids(frame.dataset_id.unique(),split)
                    prediction=model.predict(frame.loc[:,names].to_numpy(dtype=np.float32)).astype(np.float32)
                    grouped=pd.Series(prediction,index=frame.dataset_id.to_numpy()).groupby(level=0).max();maxima.extend(grouped.to_list());covered.update(grouped.index)
                    del frame,prediction;gc.collect()
                if covered!=null_ids:raise RuntimeError('Training null calibration ID coverage differs')
                frame=pd.read_parquet(cache/f'fold_{fold}.parquet',columns=['dataset_id','time_online'])
                prediction=np.load(ROOT/'data/processed/oof'/eid/f'fold_{fold}.npy',allow_pickle=False)
                validation_predictions={int(sid):prediction[positions] for sid,positions in frame.groupby('dataset_id',sort=False).indices.items()}
                for sid,positions in frame.groupby('dataset_id',sort=False).indices.items():
                    if not np.array_equal(frame.time_online.to_numpy()[positions],np.arange(len(positions))):raise RuntimeError('Validation temporal coordinates differ')
                if set(validation_predictions)!=val_ids:raise RuntimeError('Validation diagnostic coverage differs')
                del model,frame;gc.collect()
            for alpha in ALPHAS:
                threshold=training_threshold(maxima,alpha);threshold_records.append({'candidate':eid,'fold':fold,'nominal_training_tail_fraction':alpha,'threshold':threshold,
                    'training_null_series':len(maxima),'training_null_any_alarm_rate':float(np.mean(np.asarray(maxima)>threshold)),
                    'training_null_ids_sha256':__import__('hashlib').sha256(','.join(map(str,sorted(null_ids))).encode()).hexdigest()})
                rows=[]
                for sid in sorted(val_ids):
                    tau=int(metadata.loc[sid,'tau_index']) if bool(metadata.loc[sid,'has_break']) else None
                    rows.append(first_alarm_outcome(validation_predictions[sid],tau,threshold))
                outcomes[alpha].extend(rows);fold_results.append({'candidate':eid,'fold':fold,'nominal_training_tail_fraction':alpha,'threshold':threshold,**summarize_outcomes(rows)})
            print('FIRST ALARM',eid,'fold',fold,'complete',flush=True)
            del validation_predictions;gc.collect()
        for alpha,rows in outcomes.items():aggregate.append({'candidate':eid,'nominal_training_tail_fraction':alpha,**summarize_outcomes(rows)})
    pd.DataFrame(aggregate).to_csv(directory/'AGGREGATE.csv',index=False);pd.DataFrame(fold_results).to_csv(directory/'FOLDS.csv',index=False)
    pd.DataFrame(threshold_records).to_csv(directory/'TRAIN_THRESHOLDS.csv',index=False)
    write_json(directory/'RESULTS.json',{'status':'PASS','timestamp':utc_now(),'policy':policy,'aggregate':aggregate,'folds':fold_results,'training_thresholds':threshold_records,
        'elapsed_seconds':time.perf_counter()-started,'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False,
        'interpretation':'Post-selection DEV diagnostic, no independent test claim. Threshold fitting uses outer-training labels only. Conditional detected-only median delays omit undetected/censored series; report detection rates and premature alarms beside delays.'})
    print('FIRST ALARM DIAGNOSTICS COMPLETE',flush=True)


if __name__=='__main__':main()
