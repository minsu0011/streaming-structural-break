"""Fixed postprocessing diagnostic: rank invariance versus long-null alarms."""
from pathlib import Path
import sys
import json
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next2.counterexamples import policy
from src.next2.synthetic_analysis import background,calibration_map,threshold,alarm_summary,score,age_normalize
from src.next2.alarm_diagnostics import verify_within_time_ranks
from src.next.alarm_calibration import wilson_interval
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now

J='J102_SERIAL_VARIANCE_512'
Y='Y001_LONG_VARIANCE_ARCH1'
R='B104_LONG_VARIANCE_NO_GLR_MEMORY'
C='C205_DEPENDENCE'
B='BL2_C205_DEPENDENCE_80'
BETAS=(0.,.5,1.)


def main():
    lock=policy()
    out=ROOT/'artifacts/next2/age_alarm_diagnostics'
    out.mkdir(parents=True,exist_ok=True)
    roots={eid:ROOT/('artifacts/next2/counterexamples' if eid in (J,Y,R) else 'artifacts/next2/finalist_counterexamples') for eid in (J,Y,R,C,B)}
    identity={'script_sha256':sha256(Path(__file__)),'counterexample_lock_sha256':sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'),
        'normalization_sha256':sha256(ROOT/'src/next2/synthetic_analysis.py'),'rank_check_sha256':sha256(ROOT/'src/next2/alarm_diagnostics.py'),
        'hypothesis_sha256':sha256(ROOT/'docs/NEXT2_AGE_ALARM_HYPOTHESIS.md'),'betas':list(BETAS),'models':list(roots),
        'selection':'No exponent or deployment threshold selected; all fixed variants reported'}
    if (out/'POLICY.json').exists():
        if json.loads((out/'POLICY.json').read_text())!=identity:
            raise RuntimeError('Age alarm study policy changed')
    else:
        write_json(out/'POLICY.json',identity)
    mapping=calibration_map(lock)
    artifact_hashes={}
    def load(root,kind,scenario,eid):
        folder=root/kind/scenario
        r=json.loads((folder/'RESULTS.json').read_text())
        name=eid+'.npy'
        if r['status']!='PASS' or sha256(folder/name)!=r['files_sha256'][name]:
            raise RuntimeError('Synthetic study input changed')
        artifact_hashes[(folder/name).relative_to(ROOT).as_posix()]=r['files_sha256'][name]
        return np.load(folder/name,mmap_mode='r',allow_pickle=False)
    rows,long_rows,rank_rows=[],[],[]
    started=time.perf_counter()
    for eid,root in roots.items():
        levels={}
        for bg,scenario in mapping.items():
            p=load(root,'calibration',scenario,eid)
            for beta in BETAS:
                q=age_normalize(p,np.arange(p.shape[1]),beta)
                maxima=q.max(axis=1)
                for alpha in lock['long_null']['threshold_alphas']:
                    levels[bg,beta,alpha]=threshold(maxima,alpha)
        for spec in lock['scenarios']:
            scenario=spec['id']
            p=load(root,'counterexample',scenario,eid)
            tau=np.load(root/'counterexample'/scenario/'synthetic_tau.npy',allow_pickle=False)
            times=np.broadcast_to(np.arange(p.shape[1]),p.shape)
            original=score(p,tau)
            for beta in BETAS:
                q=age_normalize(p,np.arange(p.shape[1]),beta)
                checked=verify_within_time_ranks(p.ravel(),q.ravel(),times.ravel())
                mapped=score(q,tau)
                if original!=mapped:
                    raise AssertionError('Synthetic official TS-AUC changed under monotone map')
                rank_rows.append({'candidate':eid,'scenario':scenario,'beta':beta,'ts_auc_original':original,'ts_auc_transformed':mapped,**checked})
                for alpha in lock['long_null']['threshold_alphas']:
                    level=levels[background(spec),beta,alpha]
                    summary,crossed,_=alarm_summary(q,tau,level)
                    row={'candidate':eid,'scenario':scenario,'beta':beta,'alpha':alpha,'threshold':level,
                        'ts_auc':mapped,**summary,'diagnostic_only':True}
                    if scenario=='O1':
                        n=len(p)//2
                        row.update(control_alarm_rate=float(crossed[:n].mean()),outlier_alarm_rate=float(crossed[n:].mean()),
                            outlier_alarm_excess=float(crossed[n:].mean()-crossed[:n].mean()))
                    rows.append(row)
        for scenario in lock['long_null']['regimes']:
            p=load(root,'long_null',scenario,eid)
            spec=next(s for s in lock['scenarios'] if s['id']==scenario)
            for beta in BETAS:
                q=age_normalize(p,np.arange(p.shape[1]),beta)
                for horizon in lock['long_null']['horizons']:
                    maximum=q[:,:horizon].max(axis=1)
                    for alpha in lock['long_null']['threshold_alphas']:
                        level=levels[background(spec),beta,alpha]
                        crossed=maximum>level if level is not None else np.zeros(len(p),dtype=bool)
                        count=int(crossed.sum())
                        lo,hi=wilson_interval(count,len(p))
                        long_rows.append({'candidate':eid,'regime':scenario,'beta':beta,'alpha':alpha,'horizon':horizon,
                            'series':len(p),'threshold':level,'eventual_alarm_count':count,'eventual_alarm_rate':count/len(p),
                            'alarm_wilson95_low':lo,'alarm_wilson95_high':hi,'mean_max_score':float(maximum.mean()),
                            'mean_max_score_growth_from_1024':float(maximum.mean()-q[:,:1024].max(axis=1).mean())})
        pd.DataFrame(rows).to_csv(out/'COUNTEREXAMPLES.csv',index=False)
        pd.DataFrame(long_rows).to_csv(out/'LONG_NULL.csv',index=False)
        pd.DataFrame(rank_rows).to_csv(out/'RANK_PARITY.csv',index=False)
        checkpoint('Age/alarm diagnostic '+eid,['Remaining fixed models','Nested DEV alarm comparison'])
        print('NEXT2 AGE ALARM',eid,round(time.perf_counter()-started,1),flush=True)
    write_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'identity':identity,
        'rank_checks':len(rank_rows),'all_official_scores_exact':True,'elapsed_seconds':time.perf_counter()-started,
        'input_files_sha256':artifact_hashes,'seal_rows':0,'deployment_selection':None,
        'limitations':'No anytime bound; thresholds conditional on finite calibration data; a monotone same-time transformation can leave TS-AUC unchanged while changing first alarms and detection delays.'})
    print('NEXT2 AGE ALARM COMPLETE',flush=True)


if __name__=='__main__':
    main()
