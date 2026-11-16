"""Matched fitted-model synthetic diagnostics, with separately calibrated alarms."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys
import argparse
import time
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.synthetic import (sources,load_reference,load_candidate,cohort_series,candidate_prediction,
    direct_candidate_prediction,threshold,alarm_summary,background_mapping,scenario_background,paired_seed_counts)
from src.next2.counterexamples import policy as original_policy,predict
from src.next.bootstrap import PreparedAUC,paired_summary
from src.scoring.ts_auc import ts_auc


def verified(directory,identity):
    receipt=directory/'RESULTS.json'
    if not receipt.exists():return False
    r=read_json(receipt)
    if r['status']!='PASS' or r['identity']!=identity:raise RuntimeError('Stale synthetic cohort identity')
    for name,sha in r['files_sha256'].items():
        if digest(directory/name)!=sha:raise RuntimeError('Changed synthetic cohort bytes')
    return True


def reference_cohort(prepared,proof,training_scope,kind,scenario,n,horizon,lock,suite_path):
    out=BASE/'synthetic/reference'/training_scope/kind/scenario
    identity={'training_scope':training_scope,'kind':kind,'scenario':scenario,'n':n,'horizon':horizon,
        'suite_sha256':digest(suite_path),'model_proof':proof}
    if verified(out,identity):return out
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'identity':identity},immutable=True)
    paired=kind=='counterexample' and not scenario.startswith('N');count=2*n if paired else n
    arrays={key:np.lib.format.open_memmap(out/(key+'.npy'),mode='w+',dtype=np.float32,shape=(count,horizon)) for key in ('BL80','C205')}
    taus=np.empty(count,dtype=np.int32);started=time.perf_counter()
    for row in range(count):
        control=kind=='calibration' or (paired and row<n)
        h,o,tau=cohort_series(kind,scenario,row%n,horizon,control,lock);taus[row]=tau
        arrays['BL80'][row]=predict(prepared,h,o);arrays['C205'][row]=predict(prepared.complement,h,o)
        if (row+1)%32==0:
            progress('Matched synthetic reference replay',f'{training_scope} {kind} {scenario}: {row+1}/{count}')
            print('NEXT4 SYNTHETIC REFERENCE',training_scope,kind,scenario,row+1,count,flush=True)
    for array in arrays.values():array.flush()
    del arrays
    np.save(out/'tau.npy',taus,allow_pickle=False)
    files=['BL80.npy','C205.npy','tau.npy']
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'identity':identity,
        'elapsed_seconds':time.perf_counter()-started,'new_fits':0,'seal_rows':0,
        'files_sha256':{name:digest(out/name) for name in files}},immutable=True)
    return out


def candidate_cohort(model,nomination,kind,scenario,n,horizon,lock,suite_path,reference_path,out):
    identity={'nomination_sha256':digest(BASE/'synthetic_nominations'/(nomination['id']+'.json')),
        'suite_sha256':digest(suite_path),'kind':kind,'scenario':scenario,'n':n,'horizon':horizon,
        'reference_result_sha256':digest(reference_path/'RESULTS.json')}
    if verified(out,identity):return out
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'identity':identity},immutable=True)
    ref=np.load(reference_path/'BL80.npy',mmap_mode='r',allow_pickle=False)
    complement=np.load(reference_path/'C205.npy',mmap_mode='r',allow_pickle=False)
    tau=np.load(reference_path/'tau.npy',allow_pickle=False)
    scores=np.lib.format.open_memmap(out/'predictions.npy',mode='w+',dtype=np.float32,shape=ref.shape)
    paired=kind=='counterexample' and not scenario.startswith('N');started=time.perf_counter();checks=0
    for row in range(len(ref)):
        control=kind=='calibration' or (paired and row<n)
        h,o,t=cohort_series(kind,scenario,row%n,horizon,control,lock)
        assert t==tau[row]
        scores[row]=candidate_prediction(model,nomination,h,o,ref[row],complement[row])
        if row==0:
            live=direct_candidate_prediction(model,nomination,h,o[:min(1024,len(o))],ref[row,:1024],complement[row,:1024])
            np.testing.assert_array_equal(scores[row,:len(live)],live);checks+=1
        if paired and row>=n:
            stop=int(lock['tau_cycle'][(row-n)%len(lock['tau_cycle'])])
            np.testing.assert_array_equal(scores[row,:stop],scores[row-n,:stop]);checks+=1
        if (row+1)%32==0:
            scores.flush();progress('Fitted evidence-model synthetic replay',f'{nomination["id"]} {kind} {scenario}: {row+1}/{len(ref)}')
            print('NEXT4 SYNTHETIC CANDIDATE',nomination['id'],kind,scenario,row+1,len(ref),flush=True)
    scores.flush();del scores
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'identity':identity,
        'elapsed_seconds':time.perf_counter()-started,'prefix_and_streaming_checks':checks,
        'new_fits':0,'seal_rows':0,'files_sha256':{'predictions.npy':digest(out/'predictions.npy')}},immutable=True)
    return out


def outlier_metrics(p,tau_cycle,cutoff):
    n=len(p)//2;clean=p[:n];altered=p[n:];excess=altered-clean
    result={'control_alarm_rate':float((clean>cutoff).any(axis=1).mean()),
        'outlier_alarm_rate':float((altered>cutoff).any(axis=1).mean()),
        'outlier_alarm_excess':float((altered>cutoff).any(axis=1).mean()-(clean>cutoff).any(axis=1).mean())}
    for lag in (32,128,512):
        values=[]
        for index in range(n):
            t=int(tau_cycle[index%len(tau_cycle)])+lag
            if t<excess.shape[1]:values.append(excess[index,t])
        result['mean_paired_score_excess_lag_'+str(lag)]=float(np.mean(values)) if values else None
        result['fraction_paired_score_excess_gt_001_lag_'+str(lag)]=float(np.mean(np.asarray(values)>.01)) if values else None
    return result


def analyze_counterexample(nomination,scenario,p,ref,tau,calibration,reference_calibration,suite,out):
    rows=[];paired=None;no=p.shape[1]
    times=np.broadcast_to(np.arange(no,dtype=np.int32),p.shape).ravel()
    y=((np.arange(no)[None,:]>=tau[:,None])&(tau[:,None]>=0)).astype(np.uint8).ravel()
    score=ts_auc(y,p.ravel(),times) if y.max() else None
    reference=ts_auc(y,ref.ravel(),times) if y.max() else None
    if y.max():
        n=len(p)//2;counts=paired_seed_counts(n,suite['paired_bootstrap_replicates'],suite['paired_bootstrap_seed'])
        code=np.repeat(np.tile(np.arange(n),2),no)
        a=PreparedAUC.create(y,ref.ravel(),times,code,n).evaluate(counts)
        b=PreparedAUC.create(y,p.ravel(),times,code,n).evaluate(counts)
        np.testing.assert_allclose([a[0],b[0]],[reference,score],rtol=0,atol=1e-12)
        paired=paired_summary(a[:,None],b[:,None]);np.save(out/'PAIRED_DRAWS.npy',np.column_stack([a,b]),allow_pickle=False)
    for alpha in suite['threshold_alphas']:
        row={'candidate':nomination['candidate'],'nomination':nomination['id'],'training_scope':nomination['training_scope'],
            'scenario':scenario,'alpha':alpha,'ts_auc':score,'reference_ts_auc':reference,
            'delta':score-reference if score is not None else None,'diagnostic_only':True,'seal_rows':0}
        for prefix,array,cal in (('',p,calibration),('reference_',ref,reference_calibration)):
            cut=threshold(cal.max(axis=1),alpha)
            row.update({prefix+k:v for k,v in alarm_summary(array,tau,cut).items()})
            if scenario in ('O1','OB16'):row.update({prefix+k:v for k,v in outlier_metrics(array,suite['tau_cycle'],cut).items()})
        rows.append(row)
    result={'status':'PASS','scenario':scenario,'counterexample_rows':rows,'paired':paired,
        'statistical_scope':'Conditional fixed fitted-model synthetic cohort, paired generating seeds. Diagnostic intervals unadjusted across scenarios; no real DEV or promotion authority.'}
    if paired is not None:result['paired_draws_sha256']=digest(out/'PAIRED_DRAWS.npy')
    atomic_json(out/'ANALYSIS.json',result,immutable=True)
    print('NEXT4 SYNTHETIC SCORE',nomination['id'],scenario,score,reference,flush=True)
    return result


def analyze_long_null(nomination,scenario,p,ref,cal,refcal,suite,out):
    rows=[]
    for horizon in suite['long_null_horizons']:
        for alpha in suite['threshold_alphas']:
            row={'candidate':nomination['candidate'],'nomination':nomination['id'],'training_scope':nomination['training_scope'],
                'regime':scenario,'horizon':horizon,'series':len(p),'alpha':alpha,'diagnostic_only':True}
            for prefix,array,calibration in (('',p,cal),('reference_',ref,refcal)):
                scores=array[:,:horizon];maximum=scores.max(axis=1);cut=threshold(calibration.max(axis=1),alpha)
                count=int((maximum>cut).sum())
                detail={'threshold':cut,'eventual_alarm_count':count,'eventual_alarm_rate':count/len(p),
                    'mean_max_score':float(maximum.mean()),'median_max_score':float(np.median(maximum)),
                    'mean_max_score_growth_from_1024':float(maximum.mean()-array[:,:1024].max(axis=1).mean()),
                    **{'score_quantile_'+str(q):float(np.quantile(scores,q)) for q in suite['score_quantiles']}}
                row.update({prefix+k:v for k,v in detail.items()})
            row['alarm_rate_delta_vs_matched_reference']=row['eventual_alarm_rate']-row['reference_eventual_alarm_rate']
            row['tenfold_nominal_alarm_warning']=bool(row['eventual_alarm_rate']>10*alpha)
            rows.append(row)
    result={'status':'PASS','long_null_rows':rows,'thresholds_refitted_at_longer_horizons':False,
        'scope':'Separate model-specific 1024-point null calibration, then fixed thresholds at every horizon. Not a uniform anytime error guarantee.'}
    atomic_json(out/'ANALYSIS.json',result,immutable=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--nomination',required=True);args=parser.parse_args()
    set_num_threads(2)
    suite_path=BASE/'diagnostic_policy/SYNTHETIC_POLICY.json';suite=read_json(suite_path)
    assert suite['sources_sha256']==sources()
    nomination=read_json(BASE/'synthetic_nominations'/(args.nomination+'.json'))
    assert nomination['suite_sha256']==digest(suite_path)
    model=load_candidate(nomination);prepared,proof=load_reference(nomination['training_scope'])
    lock=original_policy();assert digest(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')==suite['original_generator_policy_sha256']
    out=BASE/'synthetic/candidates'/args.nomination
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'nomination_sha256':digest(BASE/'synthetic_nominations'/(args.nomination+'.json')),
        'reference_proof':proof,'source_sha256':sources(),'new_fits':0},immutable=True)
    sizes=suite['all_dev_sizes'] if nomination['training_scope']=='all_dev' else suite['fold0_sizes']
    mapping=background_mapping(lock);calibration_paths={};reference_paths={}
    for bg,scenario in mapping.items():
        rp=reference_cohort(prepared,proof,nomination['training_scope'],'calibration',scenario,sizes['calibration'],1024,lock,suite_path)
        cp=candidate_cohort(model,nomination,'calibration',scenario,sizes['calibration'],1024,lock,suite_path,rp,out/'calibration'/scenario)
        calibration_paths[bg]=cp;reference_paths[bg]=rp
    analyses=[]
    menu=[s['id'] for s in lock['scenarios']]+suite['additional_counterexamples']
    for scenario in menu:
        rp=reference_cohort(prepared,proof,nomination['training_scope'],'counterexample',scenario,sizes['evaluation'],1024,lock,suite_path)
        cp=candidate_cohort(model,nomination,'counterexample',scenario,sizes['evaluation'],1024,lock,suite_path,rp,out/'counterexample'/scenario)
        bg=scenario_background(scenario,lock)
        cal=np.load(calibration_paths[bg]/'predictions.npy',mmap_mode='r',allow_pickle=False)
        refcal=np.load(reference_paths[bg]/'BL80.npy',mmap_mode='r',allow_pickle=False)
        p=np.load(cp/'predictions.npy',mmap_mode='r',allow_pickle=False);ref=np.load(rp/'BL80.npy',mmap_mode='r',allow_pickle=False)
        tau=np.load(rp/'tau.npy',allow_pickle=False)
        analyses.append(analyze_counterexample(nomination,scenario,p,ref,tau,cal,refcal,suite,cp))
        progress('Synthetic diagnostic uncertainty',args.nomination+' '+scenario+' completed fixed paired seed bootstrap')
    for scenario in suite['long_null_regimes']:
        horizon=max(suite['long_null_horizons'])
        rp=reference_cohort(prepared,proof,nomination['training_scope'],'long_null',scenario,sizes['long_null'],horizon,lock,suite_path)
        cp=candidate_cohort(model,nomination,'long_null',scenario,sizes['long_null'],horizon,lock,suite_path,rp,out/'long_null'/scenario)
        bg=scenario_background(scenario,lock)
        cal=np.load(calibration_paths[bg]/'predictions.npy',mmap_mode='r',allow_pickle=False)
        refcal=np.load(reference_paths[bg]/'BL80.npy',mmap_mode='r',allow_pickle=False)
        p=np.load(cp/'predictions.npy',mmap_mode='r',allow_pickle=False);ref=np.load(rp/'BL80.npy',mmap_mode='r',allow_pickle=False)
        analyses.append(analyze_long_null(nomination,scenario,p,ref,cal,refcal,suite,cp))
        progress('Long-null diagnostic completed',args.nomination+' '+scenario+' all three horizons, fixed model-specific thresholds')
    counter_rows=[r for a in analyses for r in a.get('counterexample_rows',[])]
    long_rows=[r for a in analyses for r in a.get('long_null_rows',[])]
    paired_rows=[{'scenario':a['scenario'],**a['paired']} for a in analyses if a.get('paired') is not None]
    pd.DataFrame(counter_rows).to_csv(out/'COUNTEREXAMPLES.csv',index=False)
    pd.DataFrame(long_rows).to_csv(out/'LONG_NULL.csv',index=False)
    pd.DataFrame(paired_rows).to_csv(out/'PAIRED_SYNTHETIC.csv',index=False)
    diagnostics={r['scenario']:r['delta'] for r in counter_rows if r['alpha']==.05}
    files={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*.json')}
    files.update({name:digest(out/name) for name in ('COUNTEREXAMPLES.csv','LONG_NULL.csv','PAIRED_SYNTHETIC.csv')})
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'candidate':nomination['candidate'],
        'nomination':args.nomination,'training_scope':nomination['training_scope'],'new_fits':0,'seal_rows':0,
        'counterexample_scenarios':len(menu),'long_null_regimes':len(suite['long_null_regimes']),
        'long_null_horizons':suite['long_null_horizons'],'cohort_sizes':sizes,'paired_replicates':suite['paired_bootstrap_replicates'],
        'ARCH_decrease_each_delta_positive':all(diagnostics[s]>0 for s in ('V3','V4')),
        'heavy_tail_decrease_each_delta_positive':all(diagnostics[s]>0 for s in ('CB_N3_070','CB_N3_050')),
        'diagnostic_deltas':diagnostics,'promotion':False,'synthetic_only_promotion_allowed':False,
        'thresholds_fit_on_independent_1024_null_cohort':True,'files_sha256':files},immutable=True)
    print('NEXT4 SYNTHETIC SUITE COMPLETE',args.nomination,flush=True)


if __name__=='__main__':main()
