"""Nested DEV calibration for fixed primary/dependence models and age maps."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
from pathlib import Path
import sys
import json
import time
import gc
import hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next2.evaluation import load_full_oof
from src.next2.io import design,fit,model_class
from src.next.diagnostic_training import ordered_rows
from src.next.fitting import fitting_hash
from src.next2.blend import combine
from src.next2.synthetic_analysis import age_normalize,threshold
from src.next2.alarm_diagnostics import series_maxima64,first_alarm_records64,verify_within_time_ranks
from src.next.alarm_calibration import summarize_alarms
from src.scoring.ts_auc import ts_auc
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now

J='J102_SERIAL_VARIANCE_512'
C='C205_DEPENDENCE'
B='BL2_C205_DEPENDENCE_80'
BETAS=(0.,.5,1.)
ALPHAS=(.01,.05)


def main():
    source=ROOT/'configs/next_alarm_calibration_policy.json'
    old=json.loads(source.read_text())
    data=prepare_dev(ROOT)
    if old['split_sha256']!=data.guard.split.digest or old['seal_lock_sha256']!=data.guard.lock_sha:
        raise RuntimeError('Old nested partition provenance changed')
    tau=dict(zip(map(int,data.ids),map(int,data.series['tau'])))
    out=ROOT/'artifacts/next2/alarm_calibration'
    out.mkdir(parents=True,exist_ok=True)
    studies={}
    for eid in (J,C):
        candidate,prior,_,_,sources=load_full_oof(data,eid)
        x,meta,names,features,rows=design(data,candidate)
        np.testing.assert_array_equal(rows,np.arange(len(rows)))
        studies[eid]=(candidate,x,meta,names,{'oof':sources,'features':features})
    identity={'study':'NEXT2_NESTED_ALARM_V1','partition_file_sha256':sha256(source),
        'hypothesis_sha256':sha256(ROOT/'docs/NEXT2_AGE_ALARM_HYPOTHESIS.md'),
        'script_sha256':sha256(Path(__file__)),'fitting_sha256':fitting_hash(),
        'diagnostics_sha256':sha256(ROOT/'src/next2/alarm_diagnostics.py'),
        'normalization_sha256':sha256(ROOT/'src/next2/synthetic_analysis.py'),
        'candidate_sources':{eid:study[-1] for eid,study in studies.items()},
        'betas':list(BETAS),'alphas':list(ALPHAS),'blend_weight':.8,
        'training_policy':'Same frozen nested training IDs; thresholds on null calibration IDs; never refit after calibration',
        'quality_authority':'Secondary smaller-model diagnostic; no deployment threshold selected'}
    if (out/'POLICY.json').exists():
        if json.loads((out/'POLICY.json').read_text())!=identity:
            raise RuntimeError('Nested alarm policy changed')
    else:
        write_json(out/'POLICY.json',identity)
    all_summaries,all_series=[],[]
    for item in old['folds']:
        fold=item['outer_fold']
        folder=out/f'fold_{fold}'
        folder.mkdir(exist_ok=True)
        data.guard.partition(item['training_ids'],item['calibration_ids'])
        data.guard.partition(item['training_ids']+item['calibration_ids'],item['validation_ids'])
        if set(item['training_ids']+item['calibration_ids']+item['validation_ids'])!=set(data.guard.dev_ids):
            raise RuntimeError('Nested partitions do not cover exact DEV')
        if (folder/'RESULTS.json').exists():
            r=json.loads((folder/'RESULTS.json').read_text())
            if r['status']!='PASS' or any(sha256(folder/n)!=v for n,v in r['files_sha256'].items()):
                raise RuntimeError('Nested alarm fold changed')
        else:
            started=time.perf_counter()
            models,cal={},{}
            for eid,(candidate,x,meta,names,_) in studies.items():
                train=ordered_rows(meta,item['training_ids'])
                calibration=ordered_rows(meta,item['calibration_ids'])
                valid=ordered_rows(meta,item['validation_ids'])
                path=folder/(eid+'.joblib')
                if path.exists():
                    raise RuntimeError('Partial nested model requires reviewed resume, never silently refit')
                model,prediction,detail=fit(data,candidate,x,meta,names,train,calibration)
                model.save(path)
                loaded=model_class(candidate).load(path)
                np.testing.assert_array_equal(loaded.predict(x[calibration[:4096]]).astype(np.float32),prediction[:4096])
                sid=int(meta['dataset_id'][calibration[0]])
                h,o,_=data.get(sid)
                state=loaded.make_state(h)
                live=np.asarray([loaded.predict_one(state.update(p)) for p in o],dtype=np.float32)
                np.testing.assert_array_equal(live,prediction[meta['dataset_id'][calibration]==sid])
                models[eid]=loaded
                cal[eid]=prediction
                np.save(folder/(eid+'_calibration.npy'),prediction,allow_pickle=False)
                write_json(folder/(eid+'_FIT.json'),{'status':'PASS','detail':detail,'save_load_parity':True,
                    'complete_calibration_series_live_parity':True,'model_sha256':sha256(path)})
            cal[B]=combine(cal[J],cal[C],.8)
            levels=[]
            for eid in (J,C,B):
                for beta in BETAS:
                    q=age_normalize(cal[eid],meta['time_online'][calibration],beta)
                    ids,maxima=series_maxima64(meta['dataset_id'][calibration],q)
                    null=np.asarray([tau[int(sid)]<0 for sid in ids])
                    for alpha in ALPHAS:
                        levels.append({'candidate':eid,'beta':beta,'alpha':alpha,
                            'threshold':threshold(maxima[null],alpha),'null_calibration_series':int(null.sum()),
                            'null_calibration_ids_sha256':hashlib.sha256(ids[null].astype(np.int32).tobytes()).hexdigest()})
            write_json(folder/'THRESHOLDS.json',{'created_utc':utc_now(),'thresholds':levels,
                'outer_predictions_computed_yet':False,'refit_after_calibration':False,
                'model_sha256':{eid:sha256(folder/(eid+'.joblib')) for eid in (J,C)}})
            # No model sees the outer design in predict() until every threshold
            # for this fold has been materialized above.
            outer={eid:np.asarray(model.predict(studies[eid][1][valid]),dtype=np.float32) for eid,model in models.items()}
            outer[B]=combine(outer[J],outer[C],.8)
            for eid,prediction in outer.items():
                np.save(folder/(eid+'_outer.npy'),prediction,allow_pickle=False)
            frames,summaries,rank_checks=[],[],[]
            for eid in (J,C,B):
                original_score=ts_auc(meta['target'][valid],outer[eid],meta['time_online'][valid])
                for beta in BETAS:
                    q=age_normalize(outer[eid],meta['time_online'][valid],beta)
                    ranks=verify_within_time_ranks(outer[eid],q,meta['time_online'][valid])
                    transformed_score=ts_auc(meta['target'][valid],q,meta['time_online'][valid])
                    if original_score!=transformed_score:
                        raise AssertionError('Official nested score changed under monotone age map')
                    rank_checks.append({'candidate':eid,'beta':beta,'secondary_outer_ts_auc':original_score,**ranks})
                    for level in [v for v in levels if v['candidate']==eid and v['beta']==beta]:
                        frame=first_alarm_records64(meta['dataset_id'][valid],meta['time_online'][valid],q,tau,level['threshold'])
                        frame['candidate'],frame['fold'],frame['beta'],frame['alpha']=eid,fold,beta,level['alpha']
                        frames.append(frame)
                        summaries.append({'fold':fold,**level,**summarize_alarms(frame),
                            'secondary_smaller_model_ts_auc':original_score,'official_rank_parity':True})
            pd.concat(frames,ignore_index=True).to_csv(folder/'SERIES.csv',index=False)
            files=[p.name for p in folder.iterdir() if p.suffix in ('.npy','.joblib','.csv','.json') and p.name!='RESULTS.json']
            r={'status':'PASS','fold':fold,'created_utc':utc_now(),'summaries':summaries,'rank_checks':rank_checks,
                'new_fits':2,'training_series':len(item['training_ids']),'calibration_series':len(item['calibration_ids']),
                'validation_series':len(item['validation_ids']),'elapsed_seconds':time.perf_counter()-started,
                'seal_rows':0,'reduced_new_usage':0,'files_sha256':{name:sha256(folder/name) for name in files}}
            write_json(folder/'RESULTS.json',r)
            print('NEXT2 NESTED ALARM',fold,round(r['elapsed_seconds'],1),flush=True)
            checkpoint('Nested alarm outer fold '+str(fold),['Complete nested calibration','Compare false-alarm and detection tradeoff'],
                artifacts=[(folder/'RESULTS.json').relative_to(ROOT).as_posix()])
            del models,cal,outer
            gc.collect()
        all_summaries.extend(r['summaries'])
        all_series.append(pd.read_csv(folder/'SERIES.csv'))
    combined=pd.concat(all_series,ignore_index=True)
    aggregate=[{'candidate':eid,'beta':beta,'alpha':alpha,**summarize_alarms(frame)}
        for (eid,beta,alpha),frame in combined.groupby(['candidate','beta','alpha'])]
    pd.DataFrame(aggregate).to_csv(out/'SUMMARY.csv',index=False)
    pd.DataFrame(all_summaries).to_csv(out/'FOLDS.csv',index=False)
    write_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'policy_sha256':sha256(out/'POLICY.json'),
        'new_refits':10,'folds':5,'thresholds_selected_for_deployment':0,'summary':aggregate,
        'official_rank_parity':True,'seal_rows':0,'reduced_new_usage':0,
        'limitations':'Previously adaptively used DEV; heterogeneous lengths/regimes; calibration exchangeability not established; smaller training models differ from frozen complete-CV and all-DEV artifacts.'})
    print('NEXT2 NESTED ALARM COMPLETE',flush=True)


if __name__=='__main__':
    main()
