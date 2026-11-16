"""Complete fixed CV0 component stress; artifact kind never mixed with all-DEV."""
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
from src.next2.synthetic_analysis import background,calibration_map,threshold,alarm_summary,score
from src.next.alarm_calibration import wilson_interval
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now

OUT=ROOT/'artifacts/next2/component_stress_cv0'


def evaluate(lock,engine,ref,kind,scenario,n,horizon,identity):
    folder=OUT/ref['experiment_id']/kind/scenario
    folder.mkdir(parents=True,exist_ok=True)
    job={**identity,'kind':kind,'scenario':scenario,'series':n,'horizon':horizon}
    if (folder/'RESULTS.json').exists():
        r=json.loads((folder/'RESULTS.json').read_text())
        if r['status']!='PASS' or r['identity']!=job or any(sha256(folder/name)!=digest for name,digest in r['files_sha256'].items()):
            raise RuntimeError('Component stress result changed')
        return
    write_json(folder/'POLICY.json',job)
    paired=kind=='counterexample' and not scenario.startswith('N')
    count=2*n if paired else n
    values=np.lib.format.open_memmap(folder/'prediction.npy',mode='w+',dtype=np.float32,shape=(count,horizon))
    taus=np.empty(count,dtype=np.int32)
    tick=time.perf_counter()
    for j in range(count):
        h,o,tau=generate(scenario,j%n,partition={'calibration':0,'counterexample':1,'long_null':2}[kind],
            horizon=horizon,control=kind=='calibration' or (paired and j<n),lock=lock)
        values[j]=predict(engine,h,o)
        taus[j]=tau
    values.flush()
    np.save(folder/'synthetic_tau.npy',taus,allow_pickle=False)
    write_json(folder/'RESULTS.json',{'status':'PASS','identity':job,'created_utc':utc_now(),
        'points':count*horizon,'paired_control_and_changed_same_seed':paired,
        'elapsed_seconds':time.perf_counter()-tick,'artifact_kind':'existing full-CV fold-0 model',
        'files_sha256':{name:sha256(folder/name) for name in ('prediction.npy','synthetic_tau.npy')},'seal_rows':0})
    print('NEXT2 COMPONENT STRESS',ref['experiment_id'],kind,scenario,round(time.perf_counter()-tick,1),flush=True)
    checkpoint('CV0 component stress '+ref['experiment_id']+'/'+kind+'/'+scenario,
        ['Remaining fixed component cohorts','Mechanistic synthesis without gate changes'])


def summarize(lock,ref,identity):
    folder=OUT/ref['experiment_id']
    mapping=calibration_map(lock)
    rows,long_rows=[],[]
    for spec in lock['scenarios']:
        scenario=spec['id']
        p=np.load(folder/'counterexample'/scenario/'prediction.npy',mmap_mode='r',allow_pickle=False)
        tau=np.load(folder/'counterexample'/scenario/'synthetic_tau.npy',allow_pickle=False)
        cm=np.load(folder/'calibration'/mapping[background(spec)]/'prediction.npy',mmap_mode='r',allow_pickle=False).max(axis=1)
        auc=score(p,tau)
        for alpha in lock['long_null']['threshold_alphas']:
            level=threshold(cm,alpha)
            metrics,crossed,_=alarm_summary(p,tau,level)
            row={'candidate':ref['experiment_id'],'scenario':scenario,'alpha':alpha,'ts_auc':auc,
                'threshold':level,**metrics,'artifact_kind':'existing full-CV fold-0 model','full_cv_status':ref['full_cv_status']}
            if scenario=='O1':
                n=len(p)//2
                row.update(control_alarm_rate=float(crossed[:n].mean()),outlier_alarm_rate=float(crossed[n:].mean()),
                    outlier_alarm_excess=float(crossed[n:].mean()-crossed[:n].mean()))
            rows.append(row)
    for scenario in lock['long_null']['regimes']:
        spec=next(s for s in lock['scenarios'] if s['id']==scenario)
        p=np.load(folder/'long_null'/scenario/'prediction.npy',mmap_mode='r',allow_pickle=False)
        cm=np.load(folder/'calibration'/mapping[background(spec)]/'prediction.npy',mmap_mode='r',allow_pickle=False).max(axis=1)
        for horizon in lock['long_null']['horizons']:
            maximum=p[:,:horizon].max(axis=1)
            for alpha in lock['long_null']['threshold_alphas']:
                level=threshold(cm,alpha)
                count=int((maximum>level).sum()) if level is not None else 0
                lo,hi=wilson_interval(count,len(p))
                long_rows.append({'candidate':ref['experiment_id'],'regime':scenario,'horizon':horizon,'alpha':alpha,
                    'series':len(p),'threshold':level,'eventual_alarm_count':count,'eventual_alarm_rate':count/len(p),
                    'alarm_wilson95_low':lo,'alarm_wilson95_high':hi,'mean_max_score':float(maximum.mean()),
                    'mean_max_score_growth_from_1024':float(maximum.mean()-p[:,:1024].max(axis=1).mean()),
                    'artifact_kind':'existing full-CV fold-0 model','full_cv_status':ref['full_cv_status']})
    pd.DataFrame(rows).to_csv(folder/'COUNTEREXAMPLES.csv',index=False)
    pd.DataFrame(long_rows).to_csv(folder/'LONG_NULL.csv',index=False)
    write_json(folder/'RESULTS.json',{'status':'PASS','identity':identity,'created_utc':utc_now(),'new_fits':0,
        'artifact_kind':'existing full-CV fold-0 model','synthetic_promotion_authority':False,
        'files_sha256':{name:sha256(folder/name) for name in ('COUNTEREXAMPLES.csv','LONG_NULL.csv')},'seal_rows':0})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--candidate')
    args=parser.parse_args()
    if not 0<=args.shard<args.shards:
        raise ValueError('Invalid fixed shard')
    menu_path=ROOT/'configs/next2/COMPONENT_STRESS_CV0_LOCK.json'
    menu=json.loads(menu_path.read_text())
    lock=policy()
    if menu['counterexample_lock_sha256']!=sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'):
        raise RuntimeError('Counterexample lock changed')
    for j,ref in enumerate(menu['models']):
        if args.candidate and args.candidate!=ref['experiment_id']:
            continue
        if not args.candidate and j%args.shards!=args.shard:
            continue
        path=ROOT/ref['model_path']
        if sha256(path)!=ref['model_sha256']:
            raise RuntimeError('Fixed CV0 component model changed')
        identity={'candidate':ref['experiment_id'],'model_sha256':ref['model_sha256'],
            'menu_sha256':sha256(menu_path),'generator_sha256':generator_hash(),'runtime_sha256':runtime_hash(),
            'script_sha256':sha256(Path(__file__)),'analysis_sha256':sha256(ROOT/'src/next2/synthetic_analysis.py')}
        done=OUT/ref['experiment_id']/'RESULTS.json'
        if done.exists():
            previous=json.loads(done.read_text())
            if previous['status']!='PASS' or previous['identity']!=identity:
                raise RuntimeError('Complete component identity changed')
            print('NEXT2 COMPONENT RETAINED',ref['experiment_id'],flush=True)
            continue
        engine=prepare(model_class(ref['candidate']).load(path),resident=False)
        for scenario in calibration_map(lock).values():
            evaluate(lock,engine,ref,'calibration',scenario,lock['calibration_null_series_per_regime'],lock['online_length'],identity)
        for spec in lock['scenarios']:
            evaluate(lock,engine,ref,'counterexample',spec['id'],lock['evaluation_series_per_scenario'],lock['online_length'],identity)
        for scenario in lock['long_null']['regimes']:
            evaluate(lock,engine,ref,'long_null',scenario,lock['long_null']['series_per_regime'],max(lock['long_null']['horizons']),identity)
        summarize(lock,ref,identity)
        print('NEXT2 COMPONENT COMPLETE',ref['experiment_id'],flush=True)


if __name__=='__main__':
    main()
