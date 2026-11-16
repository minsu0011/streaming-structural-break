"""Explain a completed synthetic rank/alarm gap without selecting new thresholds."""
import csv
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.synthetic import threshold,scenario_background,background_mapping
from src.next2.counterexamples import policy as original_policy


def main():
    policy_path=BASE/'diagnostic_policy/ALARM_RANK_GAP_POLICY.json';policy=read_json(policy_path)
    assert digest(Path(__file__))==policy['source_sha256']
    nomination=policy['nomination'];directory=BASE/'synthetic/candidates'/nomination
    result=read_json(directory/'RESULTS.json');assert result['status']=='PASS'
    assert result['training_scope']=='all_dev'
    lock=original_policy();mapping=background_mapping(lock);rows=[];evidence={}
    def verify(directory):
        r=read_json(directory/'RESULTS.json');assert r['status']=='PASS'
        evidence[(directory/'RESULTS.json').relative_to(ROOT).as_posix()]=digest(directory/'RESULTS.json')
        for name,value in r['files_sha256'].items():assert digest(directory/name)==value
    for scenario in policy['scenarios']:
        background=mapping[scenario_background(scenario,lock)]
        cp=directory/'counterexample'/scenario;rp=BASE/'synthetic/reference/all_dev/counterexample'/scenario
        cc=directory/'calibration'/background;rc=BASE/'synthetic/reference/all_dev/calibration'/background
        for folder in (cp,rp,cc,rc):verify(folder)
        analysis=read_json(cp/'ANALYSIS.json');evidence[(cp/'ANALYSIS.json').relative_to(ROOT).as_posix()]=digest(cp/'ANALYSIS.json')
        tau=np.load(rp/'tau.npy',allow_pickle=False);n=len(tau)//2;assert (tau[:n]<0).all() and (tau[n:]>=0).all()
        for model,path,calpath in [('B2',cp/'predictions.npy',cc/'predictions.npy'),('BL80',rp/'BL80.npy',rc/'BL80.npy')]:
            p=np.load(path,mmap_mode='r',allow_pickle=False);cal=np.load(calpath,mmap_mode='r',allow_pickle=False)
            maxima=np.asarray(cal.max(axis=1),dtype=np.float64);sorted_maxima=np.sort(maxima)
            changed=p[n:];post_mask=np.arange(p.shape[1])[None,:]>=tau[n:,None]
            postmax=np.max(np.where(post_mask,changed,-np.inf),axis=1)
            premax=np.max(np.where(~post_mask,changed,-np.inf),axis=1)
            percentile=np.searchsorted(sorted_maxima,postmax,side='right')/len(sorted_maxima)
            assert np.isfinite(premax).all() and np.isfinite(postmax).all()
            for alpha in policy['alphas']:
                cutoff=threshold(maxima,alpha);prior=premax>cutoff;later=postmax>cutoff
                a=next(row for row in analysis['counterexample_rows'] if row['alpha']==alpha)
                prefix='' if model=='B2' else 'reference_'
                detection=float(np.mean(~prior&later));prealarm=float(prior.mean())
                assert detection==a[prefix+'postbreak_first_detection_rate']
                assert prealarm==a[prefix+'prebreak_alarm_rate']
                rows.append({'scenario':scenario,'model':model,'training_scope':'all_dev',
                    'alpha':alpha,'calibration_series':len(maxima),'changed_series':n,
                    'existing_ts_auc':a[prefix+'ts_auc'],'unchanged_1024_null_max_threshold':cutoff,
                    'changed_prebreak_alarm_rate':prealarm,
                    'changed_postbreak_any_crossing_rate':float(later.mean()),
                    'changed_postbreak_first_detection_rate':detection,
                    'prebreak_alarm_preempted_postbreak_crossing_rate':float(np.mean(prior&later)),
                    'changed_postbreak_max_q10':float(np.quantile(postmax,.1)),
                    'changed_postbreak_max_median':float(np.median(postmax)),
                    'changed_postbreak_max_q90':float(np.quantile(postmax,.9)),
                    'median_postbreak_max_minus_threshold':float(np.median(postmax-cutoff)),
                    'median_postbreak_max_empirical_null_max_percentile':float(np.median(percentile)),
                    'fixed_threshold_or_prediction_modified':False})
        progress('Synthetic ranking/alarm boundary diagnosed',scenario+' fixed calibration maximum, post-change score maximum and pre-alarm preemption; no threshold tuning')
    out=BASE/'alarm_rank_gap/B2_all_dev';out.mkdir(parents=True,exist_ok=False)
    with (out/'FIXED_THRESHOLD_GAPS.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'nomination':nomination,
        'policy_sha256':digest(policy_path),'synthetic_result_sha256':digest(directory/'RESULTS.json'),
        'scenarios':policy['scenarios'],'new_fits':0,'new_thresholds':0,'new_predictions':0,
        'new_random_seeds_or_bootstrap_draws':0,'seal_rows':0,'promotion_authority':False,
        'existing_alarm_metrics_exactly_reconciled':True,
        'interpretation':'Cross-sectional ranking and a first crossing of an independent null-trajectory maximum threshold are different targets. Empirical null-max percentiles are descriptive, not p-values or anytime guarantees.',
        'case_selection':'Post-hoc mechanistic follow-up to the observed B2 ranking/alarm gap, retaining the existing decrease scenarios. Not an independently preselected hypothesis test.',
        'source_sha256':digest(Path(__file__)),'evidence_sha256':evidence,
        'files_sha256':{'FIXED_THRESHOLD_GAPS.csv':digest(out/'FIXED_THRESHOLD_GAPS.csv')}},immutable=True)
    print('NEXT4 B2 ALARM/RANK GAP RECONCILIATION COMPLETE',len(rows),'rows',flush=True)


if __name__=='__main__':main()
