"""Frozen all-DEV models on disjoint fixed mechanistic and long-null cohorts."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import json
import time
import math
import argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next2.counterexamples import generate,policy,predict,source_hash as generator_hash
from src.next2.pruned_runtime import PreparedCompactPredictor,source_hash as runtime_hash
from src.next.candidate_io import model_class,validate_sources
from src.next2.session import checkpoint
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import write_json,sha256,utc_now


def background(spec):
    return (spec['phi_before'],spec['arch_before'],spec['t_df_before'])


def model_references():
    lock=json.loads((ROOT/'artifacts/next2/initialization/FROZEN_REFERENCES.json').read_text(encoding='utf-8'))
    models={}
    for role,ref in lock['references'].items():
        validate_sources(ref['candidate'])
        path=ROOT/ref['model_path']
        assert sha256(path)==ref['model_sha256']
        models[ref['experiment_id']]=(PreparedCompactPredictor(model_class(ref['candidate']).load(path)),ref)
    return models


def evaluate_job(lock,models,kind,scenario,n,horizon):
    out=ROOT/'artifacts/next2/counterexamples'/kind/scenario
    out.mkdir(parents=True,exist_ok=True)
    ids=list(models)
    identity={'kind':kind,'scenario':scenario,'series':n,'horizon':horizon,
        'counterexample_lock_sha256':sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'),
        'generator_sha256':generator_hash(),'runtime_sha256':runtime_hash(),'script_sha256':sha256(Path(__file__)),
        'model_sha256':{eid:ref['model_sha256'] for eid,(_,ref) in models.items()}}
    if (out/'RESULTS.json').exists():
        r=json.loads((out/'RESULTS.json').read_text(encoding='utf-8'))
        assert r['status']=='PASS' and r['identity']==identity
        for name,digest in r['files_sha256'].items():
            assert sha256(out/name)==digest
        print('NEXT2 SYNTHETIC REUSED',kind,scenario,flush=True)
        return out
    if (out/'POLICY.json').exists():
        assert json.loads((out/'POLICY.json').read_text(encoding='utf-8'))==identity
    else:
        write_json(out/'POLICY.json',identity)
    partition={'calibration':0,'counterexample':1,'long_null':2}[kind]
    paired=kind=='counterexample' and not scenario.startswith('N')
    count=n*2 if paired else n
    arrays={eid:np.lib.format.open_memmap(out/(eid+'.npy'),mode='w+',dtype=np.float32,shape=(count,horizon)) for eid in ids}
    taus=np.empty(count,dtype=np.int32)
    started=time.perf_counter()
    for j in range(count):
        control=kind=='calibration' or (paired and j<n)
        index=j%n
        h,o,tau=generate(scenario,index,partition=partition,horizon=horizon,control=control,lock=lock)
        taus[j]=tau
        for eid,(prepared,_) in models.items():
            arrays[eid][j]=predict(prepared,h,o)
        if (j+1)%128==0:
            print('NEXT2 SYNTHETIC',kind,scenario,j+1,'/',count,round(time.perf_counter()-started,1),flush=True)
    for a in arrays.values():
        a.flush()
    del arrays
    np.save(out/'synthetic_tau.npy',taus,allow_pickle=False)
    files=[eid+'.npy' for eid in ids]+['synthetic_tau.npy']
    write_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'identity':identity,
        'evaluated_series':count,'points_per_model':count*horizon,'paired_null_controls':paired,
        'elapsed_seconds':time.perf_counter()-started,'files_sha256':{name:sha256(out/name) for name in files},
        'seal_rows':0,'competition_quality_claim':False})
    checkpoint('Fixed synthetic '+kind+'/'+scenario,
        ['Remaining frozen scenarios','Directional and alarm uncertainty analysis'],
        artifacts=[(out/'RESULTS.json').relative_to(ROOT).as_posix()])
    return out


def summarize(lock,models,calibration_by_background):
    rows,long_rows=[] ,[]
    for spec in lock['scenarios']:
        scenario=spec['id']
        out=ROOT/'artifacts/next2/counterexamples/counterexample'/scenario
        if not (out/'RESULTS.json').exists():
            continue
        tau=np.load(out/'synthetic_tau.npy',allow_pickle=False)
        no=lock['online_length']
        times=np.broadcast_to(np.arange(no,dtype=np.int32),(len(tau),no)).ravel()
        y=((np.arange(no)[None,:]>=tau[:,None])&(tau[:,None]>=0)).astype(np.uint8)
        calib=calibration_by_background[background(spec)]
        for eid in models:
            p=np.load(out/(eid+'.npy'),mmap_mode='r',allow_pickle=False)
            cm=np.load(calib/(eid+'.npy'),mmap_mode='r',allow_pickle=False).max(axis=1)
            for alpha in lock['long_null']['threshold_alphas']:
                rank=math.ceil((len(cm)+1)*(1-alpha))
                threshold=float(np.sort(cm)[rank-1]) if rank<=len(cm) else 1.
                alarm=p>threshold
                crossed=alarm.any(axis=1)
                first=np.where(crossed,alarm.argmax(axis=1),-1)
                null=tau<0
                changed=tau>=0
                pre=(first>=0)&(first<tau)&changed
                detected=(first>=tau)&crossed&changed
                row={'candidate':eid,'scenario':scenario,'description':spec['description'],'alpha':alpha,
                    'ts_auc':ts_auc(y.ravel(),p.ravel(),times) if y.max()>0 else None,
                    'null_series':int(null.sum()),'break_series':int(changed.sum()),'threshold':threshold,
                    'null_eventual_alarm_rate':float(crossed[null].mean()) if null.any() else None,
                    'prebreak_alarm_rate':float(pre[changed].mean()) if changed.any() else None,
                    'postbreak_first_detection_rate':float(detected[changed].mean()) if changed.any() else None,
                    'conditional_detection_delay_median':float(np.median(first[detected]-tau[detected])) if detected.any() else None,
                    'score_q99':float(np.quantile(p,.99)),'mean_max_score':float(p.max(axis=1).mean()),
                    'diagnostic_only':True,'artifact_kind':'frozen all-DEV model'}
                if scenario=='O1':
                    n=len(p)//2
                    row['control_alarm_rate']=float(crossed[:n].mean())
                    row['outlier_alarm_rate']=float(crossed[n:].mean())
                    row['outlier_alarm_excess']=float(crossed[n:].mean()-crossed[:n].mean())
                rows.append(row)
    for scenario in lock['long_null']['regimes']:
        out=ROOT/'artifacts/next2/counterexamples/long_null'/scenario
        if not (out/'RESULTS.json').exists():
            continue
        spec=next(s for s in lock['scenarios'] if s['id']==scenario)
        calib=calibration_by_background[background(spec)]
        for eid in models:
            p=np.load(out/(eid+'.npy'),mmap_mode='r',allow_pickle=False)
            cm=np.load(calib/(eid+'.npy'),mmap_mode='r',allow_pickle=False).max(axis=1)
            for horizon in lock['long_null']['horizons']:
                scores=p[:,:horizon]
                maximum=scores.max(axis=1)
                for alpha in lock['long_null']['threshold_alphas']:
                    rank=math.ceil((len(cm)+1)*(1-alpha))
                    threshold=float(np.sort(cm)[rank-1]) if rank<=len(cm) else 1.
                    row={'candidate':eid,'regime':scenario,'horizon':horizon,'series':len(p),'alpha':alpha,
                        'threshold':threshold,'eventual_alarm_count':int((maximum>threshold).sum()),
                        'eventual_alarm_rate':float((maximum>threshold).mean()),
                        'mean_max_score':float(maximum.mean()),'median_max_score':float(np.median(maximum)),
                        'mean_max_score_growth_from_1024':float(maximum.mean()-p[:,:1024].max(axis=1).mean())}
                    row.update({'score_quantile_'+str(q):float(np.quantile(scores,q)) for q in lock['long_null']['quantiles']})
                    long_rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT/'COUNTEREXAMPLE_RESULTS.csv',index=False)
    pd.DataFrame(long_rows).to_csv(ROOT/'LONG_NULL_RESULTS_NEXT2.csv',index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['all','calibration','counterexample','long_null'],default='all')
    args=parser.parse_args()
    lock=policy(); models=model_references()
    backgrounds={}
    for s in lock['scenarios']:
        backgrounds.setdefault(background(s),s['id'])
    out=ROOT/'artifacts/next2/counterexamples'
    out.mkdir(parents=True,exist_ok=True)
    mapping={'backgrounds':[{'phi':b[0],'arch':b[1],'t_df':b[2],'calibration_scenario_seed':s} for b,s in backgrounds.items()],
        'rule':'Use each scenario pre-change DGP for disjoint null calibration; identical pre-change parameter tuples share one calibration cohort.',
        'created_utc':utc_now(),'diagnostic_only':True}
    if not (out/'CALIBRATION_MAPPING.json').exists():
        write_json(out/'CALIBRATION_MAPPING.json',mapping)
    paths={}
    for bg,scenario in backgrounds.items():
        paths[bg]=out/'calibration'/scenario
        if args.phase in ('all','calibration'):
            evaluate_job(lock,models,'calibration',scenario,lock['calibration_null_series_per_regime'],lock['online_length'])
        elif not (paths[bg]/'RESULTS.json').exists():
            raise RuntimeError('Disjoint calibration must finish first')
    if args.phase in ('all','counterexample'):
        for spec in lock['scenarios']:
            evaluate_job(lock,models,'counterexample',spec['id'],lock['evaluation_series_per_scenario'],lock['online_length'])
            summarize(lock,models,paths)
    if args.phase in ('all','long_null'):
        for scenario in lock['long_null']['regimes']:
            evaluate_job(lock,models,'long_null',scenario,lock['long_null']['series_per_regime'],max(lock['long_null']['horizons']))
            summarize(lock,models,paths)
    summarize(lock,models,paths)
    print('NEXT2 COUNTEREXAMPLE PHASE COMPLETE',args.phase,flush=True)


if __name__=='__main__':
    main()
