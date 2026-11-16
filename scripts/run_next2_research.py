"""Resumable prospective NEXT2 screening; failed screens never reach full CV."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
from pathlib import Path
import sys
import json
import argparse
import gc
import time
import traceback
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next2.io import design,fit,model_class,validate
from src.next.candidate_io import candidate_design
from src.next2.session import checkpoint
from src.next.fitting import fitting_hash
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import write_json,sha256,utc_now


def check_files(directory,receipt):
    for name,digest in receipt['files_sha256'].items():
        if sha256(directory/name)!=digest:
            raise RuntimeError('Saved NEXT2 fold identity changed: '+name)


def run_scope(data,candidate,scope):
    eid=candidate['experiment_id']
    out=ROOT/'artifacts/next2/experiments'/eid/scope
    out.mkdir(parents=True,exist_ok=True)
    x,meta,names,sources,global_rows=design(data,candidate,scope)
    identity={'candidate':candidate,'scope':scope,'feature_sources':sources,
        'policy_sha256':sha256(ROOT/'configs/next2/RESEARCH_POLICY_LOCK.json'),
        'fitting_sha256':fitting_hash(),'runner_sha256':sha256(Path(__file__)),
        'split_sha256':data.guard.split.digest,'seal_lock_sha256':data.guard.lock_sha}
    if (out/'POLICY.json').exists():
        if json.loads((out/'POLICY.json').read_text(encoding='utf-8'))!=identity:
            raise RuntimeError('NEXT2 resumed scope changed')
    else:
        write_json(out/'POLICY.json',identity)
    folds=[0,4] if scope=='screen' else list(range(5))
    scores,cv=[],[]
    for fold in folds:
        valid=np.flatnonzero(meta['fold']==fold)
        train=np.concatenate([np.flatnonzero(meta['fold']==k) for k in range(5) if k!=fold])
        path=out/f'fold_{fold}.json'
        if path.exists():
            receipt=json.loads(path.read_text(encoding='utf-8'))
            assert receipt['status']=='PASS'
            check_files(out,receipt)
            prediction=np.load(out/f'fold_{fold}.npy',allow_pickle=False)
            print('NEXT2 RESUME',eid,scope,fold,flush=True)
        else:
            tick=time.perf_counter()
            model,prediction,detail=fit(data,candidate,x,meta,names,train,valid)
            model.save(out/f'fold_{fold}.joblib')
            loaded=model_class(candidate).load(out/f'fold_{fold}.joblib')
            np.testing.assert_array_equal(loaded.predict(x[valid[:2048]]).astype(np.float32),prediction[:2048])
            for sid in np.unique(meta['dataset_id'][valid])[:2]:
                h,o,_=data.get(sid)
                state=loaded.make_state(h)
                live=np.asarray([loaded.predict_one(state.update(point)) for point in o],dtype=np.float32)
                np.testing.assert_array_equal(live,prediction[meta['dataset_id'][valid]==sid])
            np.save(out/f'fold_{fold}.npy',prediction,allow_pickle=False)
            np.save(out/f'fold_{fold}_global_rows.npy',global_rows[valid],allow_pickle=False)
            receipt={'status':'PASS','created_utc':utc_now(),'engineering':detail,
                'elapsed_seconds':time.perf_counter()-tick,'save_load_parity':True,'live_cached_probe_parity':True,
                'files_sha256':{n:sha256(out/n) for n in [f'fold_{fold}.joblib',f'fold_{fold}.npy',f'fold_{fold}_global_rows.npy']}}
            write_json(path,receipt)
            del loaded,model
        score=ts_auc(meta['target'][valid],prediction,meta['time_online'][valid])
        scores.append(score)
        cv.append({'experiment_id':eid,'scope':scope,'fold':fold,'ts_auc':score,
            'validation_rows':len(valid),'validation_series':len(np.unique(meta['dataset_id'][valid])),'seal_rows':0})
        print('NEXT2 SCORE',eid,scope,fold,score,flush=True)
        checkpoint(eid+'/'+scope+'/'+str(fold),['Finish prospective candidate queue','Finalist paired and alternate diagnostics'],
            artifacts=[path.relative_to(ROOT).as_posix()])
    result={'status':'COMPLETE','scope':scope,'fold_scores':scores,'mean':float(np.mean(scores)),
        'median':float(np.median(scores)),'worst_fold':min(scores),'feature_names':names,'cv_rows':cv,
        'feature_sources':sources,'created_utc':utc_now(),'seal_rows':0}
    write_json(out/'RESULTS.json',result)
    del x,meta
    gc.collect()
    return result


def export():
    records,cv=[],[]
    for path in sorted((ROOT/'artifacts/next2/experiments').glob('*/RESULTS.json')):
        r=json.loads(path.read_text(encoding='utf-8'))
        c=r['candidate']; full=r.get('full',{}); screen=r.get('screen',{})
        row={'experiment_id':c['experiment_id'],'parent':c.get('parent_id'),'generation':c.get('generation'),
            'hypothesis':c['hypothesis'],'component_change':str(c['change']),
            'counterexample_target':c.get('counterexample_target'),'feature_count':len(full.get('feature_names',screen.get('feature_names',[]))) or None,
            'state_size':None,'model':c['learner'],'screen_score':screen.get('mean'),
            'full_mean':full.get('mean'),'median':full.get('median'),'worst_fold':full.get('worst_fold'),
            'delta':r.get('delta'),'bootstrap_ci_low':None,'bootstrap_ci_high':None,'bootstrap_p':None,
            'alt_split_1':None,'alt_split_2':None,'alt_split_3':None,'regime_worst':None,
            'long_null_status':'PENDING','synthetic_status':'PENDING','latency':None,
            'status':r['status'],'reason':r['reason']}
        records.append(row)
        cv.extend(screen.get('cv_rows',[])); cv.extend(full.get('cv_rows',[]))
    pd.DataFrame(records).to_csv(ROOT/'MODEL_RESULTS_NEXT2.csv',index=False)
    pd.DataFrame(records).to_csv(ROOT/'EXPERIMENT_LEDGER.csv',index=False)
    pd.DataFrame(cv).to_csv(ROOT/'CV_RESULTS_NEXT2.csv',index=False)


def bridge(data,c):
    reference=c['parity_reference']
    prior=json.loads((ROOT/f'artifacts/next/experiments/{reference}/RESULTS.json').read_text(encoding='utf-8'))
    left,lmeta,lnames,lsources,lrows=candidate_design(data,prior['candidate'],'full')
    right,rmeta,rnames,rsources,rrows=design(data,c,'full')
    np.testing.assert_array_equal(lrows,rrows)
    np.testing.assert_array_equal(left,right)
    for key in lmeta:
        np.testing.assert_array_equal(lmeta[key],rmeta[key])
    out=ROOT/'artifacts/next2/experiments'/c['experiment_id']
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'RESULTS.json',{'candidate':c,'status':'EXACT_PARENT_REUSED','reason':'All DEV feature values bitwise equal to Y001; no duplicate refit',
        'full':prior['full'],'screen':prior['screen'],'delta':prior['full']['mean']-.62163175031248,
        'feature_entries_compared':int(left.size),'feature_max_abs_diff':0.,'left_sources':lsources,'right_sources':rsources,
        'reference_report_sha256':sha256(ROOT/f'artifacts/next/experiments/{reference}/RESULTS.json'),'new_fits':0})
    print('NEXT2 FULL BRIDGE EXACT',left.shape,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--queue',default='configs/next2/VARIANCE_CANDIDATES_V1.json')
    parser.add_argument('--candidate')
    parser.add_argument('--screen-only',action='store_true')
    args=parser.parse_args()
    policy_path=ROOT/'configs/next2/RESEARCH_POLICY_LOCK.json'
    policy=json.loads(policy_path.read_text(encoding='utf-8'))
    queue=json.loads((ROOT/args.queue).read_text(encoding='utf-8'))
    if queue['policy_sha256']!=sha256(policy_path):
        raise RuntimeError('Preregistered gate policy changed')
    data=prepare_dev(ROOT)
    ref=json.loads((ROOT/'artifacts/next/experiments/J102_SERIAL_VARIANCE_512/RESULTS.json').read_text(encoding='utf-8'))
    # Full-CV fold models must not be used as subset-trained screen references.
    screen_ref=np.asarray(ref['screen']['fold_scores'])
    for c in queue['candidates']:
        eid=c['experiment_id']
        if args.candidate and eid!=args.candidate:
            continue
        out=ROOT/'artifacts/next2/experiments'/eid
        out.mkdir(parents=True,exist_ok=True)
        path=out/'RESULTS.json'
        if path.exists():
            previous=json.loads(path.read_text(encoding='utf-8'))
            if previous['candidate']!=c:
                raise RuntimeError('Candidate changed after result creation')
            if previous['status'] not in ('BUG','SCREEN_PASS_FULL_PENDING'):
                print('NEXT2 RETAINED',eid,previous['status'],flush=True)
                continue
        started=time.perf_counter()
        r={'candidate':c,'status':'BUG','reason':'incomplete','created_utc':utc_now()}
        try:
            if c.get('parity_reference'):
                bridge(data,c)
                export()
                continue
            screen=run_scope(data,c,'screen')
            r['screen']=screen
            delta=np.asarray(screen['fold_scores'])-screen_ref
            pass_screen=bool(delta.mean()>=-.002 and min(delta)>=-.008)
            if not pass_screen:
                r.update(status='SCREEN_FAIL',reason='Prospective matched J102 screen gate failed; no full-CV claim')
            elif args.screen_only:
                r.update(status='SCREEN_PASS_FULL_PENDING',reason='Prospective screen passed')
            else:
                full=run_scope(data,c,'full')
                r['full']=full
                r['delta']=full['mean']-ref['full']['mean']
                finalist=r['delta']>=-.0005 and full['median']-ref['full']['median']>=-.001 and full['worst_fold']-ref['full']['worst_fold']>=-.001
                r.update(status='FINALIST_DIAGNOSTICS_PENDING' if finalist else 'FULL_GATE_FAIL',
                         reason='Paired bootstrap and alternate stress pending' if finalist else 'Fixed primary numeric finalist gate failed')
        except Exception as exc:
            r.update(status='BUG',reason=type(exc).__name__+': '+str(exc))
            (out/'FAILURE.log').write_text(traceback.format_exc(),encoding='utf-8')
            print('NEXT2 BUG',eid,r['reason'],flush=True)
        r['elapsed_seconds']=time.perf_counter()-started
        write_json(path,r)
        export()
        checkpoint('NEXT2 result '+eid,['Remaining fixed menu','Adaptive mechanistic backlog'],
            artifacts=[path.relative_to(ROOT).as_posix()],failed_tasks=[r['reason']] if r['status']=='BUG' else [])
        print('NEXT2 RESULT',eid,r['status'],r.get('full',{}).get('mean'),flush=True)


if __name__=='__main__':
    main()
