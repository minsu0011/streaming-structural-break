"""Evaluate the fixed C205 complement and 80/20 blend on the frozen DGP.

Reuse exactly aligned J102 synthetic predictions, verified by receipts. All
new stream calculations belong to the independently refitted complement.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import json
import argparse
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next2.counterexamples import generate,policy,predict,source_hash as generator_hash
from src.next2.io import model_class
from src.next2.runtime import prepare,source_hash as runtime_hash
from src.next2.blend import combine
from src.next2.synthetic_analysis import background,calibration_map,threshold,alarm_summary,score
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now

C='C205_DEPENDENCE'
B='BL2_C205_DEPENDENCE_80'
J='J102_SERIAL_VARIANCE_512'
OUT=ROOT/'artifacts/next2/finalist_counterexamples'


def evaluate(lock,engine,model_sha,kind,scenario,n,horizon):
    old=ROOT/'artifacts/next2/counterexamples'/kind/scenario
    reference=json.loads((old/'RESULTS.json').read_text())
    for name in (J+'.npy','synthetic_tau.npy'):
        if sha256(old/name)!=reference['files_sha256'][name]:
            raise RuntimeError('Frozen reference synthetic bytes changed')
    if reference['identity']['counterexample_lock_sha256']!=sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json') or reference['identity']['generator_sha256']!=generator_hash():
        raise RuntimeError('DGP alignment differs')
    folder=OUT/kind/scenario
    folder.mkdir(parents=True,exist_ok=True)
    identity={'kind':kind,'scenario':scenario,'series':n,'horizon':horizon,
        'reference_receipt_sha256':sha256(old/'RESULTS.json'),'complement_model_sha256':model_sha,
        'runtime_sha256':runtime_hash(),'script_sha256':sha256(Path(__file__)),
        'counterexample_lock_sha256':sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'),'blend_weight':.8}
    if (folder/'RESULTS.json').exists():
        r=json.loads((folder/'RESULTS.json').read_text())
        if r['status']!='PASS' or r['identity']!=identity or any(sha256(folder/name)!=digest for name,digest in r['files_sha256'].items()):
            raise RuntimeError('Synthetic finalist result changed')
        print('NEXT2 FINALIST SYNTHETIC REUSED',kind,scenario,flush=True)
        return
    write_json(folder/'POLICY.json',identity)
    original=np.load(old/(J+'.npy'),mmap_mode='r',allow_pickle=False)
    taus=np.load(old/'synthetic_tau.npy',allow_pickle=False)
    paired=kind=='counterexample' and not scenario.startswith('N')
    count=2*n if paired else n
    if original.shape!=(count,horizon):
        raise RuntimeError('Reference cohort dimensions differ')
    arrays={eid:np.lib.format.open_memmap(folder/(eid+'.npy'),mode='w+',dtype=np.float32,shape=original.shape) for eid in (C,B)}
    started=time.perf_counter()
    for j in range(count):
        h,o,tau=generate(scenario,j%n,partition={'calibration':0,'counterexample':1,'long_null':2}[kind],
            horizon=horizon,control=kind=='calibration' or (paired and j<n),lock=lock)
        if tau!=taus[j]:
            raise RuntimeError('Synthetic pairing/label mismatch')
        arrays[C][j]=predict(engine,h,o)
        arrays[B][j]=combine(original[j],arrays[C][j],.8)
        if (j+1)%256==0:
            print('NEXT2 FINALIST SYNTHETIC',kind,scenario,j+1,'/',count,round(time.perf_counter()-started,1),flush=True)
    for value in arrays.values():
        value.flush()
    np.save(folder/'synthetic_tau.npy',taus,allow_pickle=False)
    write_json(folder/'RESULTS.json',{'status':'PASS','identity':identity,'created_utc':utc_now(),
        'files_sha256':{name:sha256(folder/name) for name in (C+'.npy',B+'.npy','synthetic_tau.npy')},
        'points_per_model':count*horizon,'paired_null_controls':paired,'elapsed_seconds':time.perf_counter()-started,
        'seal_rows':0,'competition_quality_claim':False})
    checkpoint('Finalist synthetic '+kind+'/'+scenario,['Remaining fixed cohorts','Compare directional failures'],
        artifacts=[(folder/'RESULTS.json').relative_to(ROOT).as_posix()])


def summarize(lock):
    mapping=calibration_map(lock)
    rows,long_rows=[],[]
    for spec in lock['scenarios']:
        scenario=spec['id']
        folder=OUT/'counterexample'/scenario
        if not (folder/'RESULTS.json').exists():
            continue
        tau=np.load(folder/'synthetic_tau.npy',allow_pickle=False)
        for eid in (C,B):
            p=np.load(folder/(eid+'.npy'),mmap_mode='r',allow_pickle=False)
            cm=np.load(OUT/'calibration'/mapping[background(spec)]/(eid+'.npy'),mmap_mode='r',allow_pickle=False).max(axis=1)
            auc=score(p,tau)
            for alpha in lock['long_null']['threshold_alphas']:
                level=threshold(cm,alpha)
                summary,crossed,_=alarm_summary(p,tau,level)
                r={'candidate':eid,'scenario':scenario,'description':spec['description'],'alpha':alpha,
                    'ts_auc':auc,'threshold':level,**summary,'diagnostic_only':True,'artifact_kind':'frozen all-DEV model'}
                if scenario=='O1':
                    n=len(p)//2
                    r.update(control_alarm_rate=float(crossed[:n].mean()),outlier_alarm_rate=float(crossed[n:].mean()),
                        outlier_alarm_excess=float(crossed[n:].mean()-crossed[:n].mean()))
                rows.append(r)
    for scenario in lock['long_null']['regimes']:
        folder=OUT/'long_null'/scenario
        if not (folder/'RESULTS.json').exists():
            continue
        spec=next(s for s in lock['scenarios'] if s['id']==scenario)
        for eid in (C,B):
            p=np.load(folder/(eid+'.npy'),mmap_mode='r',allow_pickle=False)
            cm=np.load(OUT/'calibration'/mapping[background(spec)]/(eid+'.npy'),mmap_mode='r',allow_pickle=False).max(axis=1)
            for horizon in lock['long_null']['horizons']:
                scores=p[:,:horizon]
                maximum=scores.max(axis=1)
                for alpha in lock['long_null']['threshold_alphas']:
                    level=threshold(cm,alpha)
                    count=int((maximum>level).sum()) if level is not None else 0
                    lo,hi=__import__('src.next.alarm_calibration',fromlist=['wilson_interval']).wilson_interval(count,len(p))
                    r={'candidate':eid,'regime':scenario,'horizon':horizon,'series':len(p),'alpha':alpha,'threshold':level,
                        'eventual_alarm_count':count,'eventual_alarm_rate':count/len(p),'alarm_wilson95_low':lo,'alarm_wilson95_high':hi,
                        'mean_max_score':float(maximum.mean()),'median_max_score':float(np.median(maximum)),
                        'mean_max_score_growth_from_1024':float(maximum.mean()-p[:,:1024].max(axis=1).mean())}
                    r.update({'score_quantile_'+str(q):float(np.quantile(scores,q)) for q in lock['long_null']['quantiles']})
                    long_rows.append(r)
    pd.DataFrame(rows).to_csv(OUT/'COUNTEREXAMPLES.csv',index=False)
    pd.DataFrame(long_rows).to_csv(OUT/'LONG_NULL.csv',index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['all','calibration','counterexample','long_null'],default='all')
    args=parser.parse_args()
    lock=policy()
    refit=ROOT/'artifacts/next2/refits/C205_DEPENDENCE/cache_v1'
    receipt=json.loads((refit/'RESULTS.json').read_text())
    trained=json.loads((refit/'TRAINING.json').read_text())
    if receipt['status']!='PASS' or sha256(refit/'model.joblib')!=receipt['model_sha256']:
        raise RuntimeError('Finalist component identity changed')
    engine=prepare(model_class(trained['candidate']).load(refit/'model.joblib'),resident=False)
    OUT.mkdir(parents=True,exist_ok=True)
    nomination={'created_utc':utc_now(),'study':'Fixed DGP extension for scientifically eligible 80/20 dependence blend',
        'counterexample_lock_sha256':sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'),
        'new_model_sha256':receipt['model_sha256'],'fitted_on':'DEV only; no synthetic training or tuning',
        'model_selection':'Candidate passed original five-fold performance, bootstrap and all three ALT gates before this study.',
        'script_sha256':sha256(Path(__file__)),'runtime_sha256':runtime_hash()}
    if not (OUT/'NOMINATION.json').exists():
        write_json(OUT/'NOMINATION.json',nomination)
    for scenario in calibration_map(lock).values():
        if args.phase in ('all','calibration'):
            evaluate(lock,engine,receipt['model_sha256'],'calibration',scenario,lock['calibration_null_series_per_regime'],lock['online_length'])
        elif not (OUT/'calibration'/scenario/'RESULTS.json').exists():
            raise RuntimeError('Independent calibration missing')
    if args.phase in ('all','counterexample'):
        for spec in lock['scenarios']:
            evaluate(lock,engine,receipt['model_sha256'],'counterexample',spec['id'],lock['evaluation_series_per_scenario'],lock['online_length'])
    if args.phase in ('all','long_null'):
        for scenario in lock['long_null']['regimes']:
            evaluate(lock,engine,receipt['model_sha256'],'long_null',scenario,lock['long_null']['series_per_regime'],max(lock['long_null']['horizons']))
    summarize(lock)
    write_json(OUT/('PHASE_'+args.phase+'.json'),{'status':'PASS','created_utc':utc_now(),'phase':args.phase,'seal_rows':0})
    print('NEXT2 FINALIST SYNTHETIC PHASE COMPLETE',args.phase,flush=True)


if __name__=='__main__':
    main()
