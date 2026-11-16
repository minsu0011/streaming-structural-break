"""Same frozen memory statistics, bound to a nominated final-artifact policy."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.synthetic import threshold,paired_seed_counts,background_mapping,scenario_background
from src.next2.counterexamples import policy as generator_policy
from src.next4.memory_diagnostics import local_recovery,aligned_outlier_excess,null_segments,paired_mean_interval


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--nomination',required=True);args=parser.parse_args()
    path=BASE/'diagnostic_policy'/('MEMORY_FINALIST_'+args.nomination+'.json');policy=read_json(path)
    assert args.nomination in policy['nominations']
    for name,sha in policy['sources_sha256'].items():assert digest(ROOT/name)==sha
    suite_path=BASE/'diagnostic_policy/SYNTHETIC_POLICY.json';assert digest(suite_path)==policy['synthetic_policy_sha256']
    suite=read_json(suite_path);lock=generator_policy();mapping=background_mapping(lock)
    nomination_path=BASE/'synthetic_nominations'/(args.nomination+'.json')
    assert digest(nomination_path)==policy['nominations'][args.nomination]
    nomination=read_json(nomination_path);source=BASE/'synthetic/candidates'/args.nomination
    completed=read_json(source/'RESULTS.json');assert completed['status']=='PASS'
    reference=BASE/'synthetic/reference'/nomination['training_scope']
    out=BASE/'memory_diagnostics'/args.nomination;out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'policy_sha256':digest(path),
        'synthetic_result_sha256':digest(source/'RESULTS.json')},immutable=True)
    evidence={};verified=set()
    def array(directory,name):
        if directory not in verified:
            receipt=directory/'RESULTS.json';result=read_json(receipt);assert result['status']=='PASS'
            identity=result['identity'];assert identity['suite_sha256']==digest(suite_path)
            if 'nomination_sha256' in identity:assert identity['nomination_sha256']==digest(nomination_path)
            else:assert identity['training_scope']==nomination['training_scope']
            for filename,sha in result['files_sha256'].items():assert digest(directory/filename)==sha
            evidence[receipt.relative_to(ROOT).as_posix()]=digest(receipt);verified.add(directory)
        return np.load(directory/name,mmap_mode='r',allow_pickle=False)
    def calibration(scenario):
        bg=mapping[scenario_background(scenario,lock)]
        return array(source/'calibration'/bg,'predictions.npy'),array(reference/'calibration'/bg,'BL80.npy')
    curve_rows=[];recovery_rows=[];effect_rows=[];segment_rows=[];null_rows=[]
    common={'candidate':nomination['candidate'],'nomination':args.nomination,'training_scope':nomination['training_scope']}
    for scenario,burst in policy['outlier_scenarios'].items():
        p=array(source/'counterexample'/scenario,'predictions.npy')
        ref=array(reference/'counterexample'/scenario,'BL80.npy')
        assert p.shape==ref.shape
        _,_,a=aligned_outlier_excess(p,suite['tau_cycle'],burst_length=burst,window=policy['post_intervention_window'])
        _,_,b=aligned_outlier_excess(ref,suite['tau_cycle'],burst_length=burst,window=policy['post_intervention_window'])
        counts=paired_seed_counts(len(a),suite['paired_bootstrap_replicates'],suite['paired_bootstrap_seed'])
        for label,excess in [('candidate',a),('reference',b)]:
            time,event,limit=local_recovery(excess,sustain=policy['quiet_run_length'],tolerance=policy['positive_excess_tolerance'])
            recovery_rows.append({**common,'scenario':scenario,'model':label,'series':len(excess),
                'post_intervention_window':excess.shape[1],'quiet_run_length':policy['quiet_run_length'],
                'censoring_time':limit,'observed_local_recovery_fraction':float(event.mean()),
                'restricted_mean_quiet_run_start':float(time.mean()),
                'conditional_median_quiet_run_start':float(np.median(time[event])) if event.any() else None,
                'mean_signed_score_excess':float(excess.mean()),
                'mean_positive_score_excess':float(np.maximum(excess,0).mean()),
                'mean_score_excess_final_64':float(excess[:,-64:].mean())})
            for age in range(excess.shape[1]):
                e=excess[:,age]
                curve_rows.append({**common,'scenario':scenario,'model':label,'age_after_last_intervention':age+1,
                    'series':len(e),'mean_score_excess':float(e.mean()),'median_score_excess':float(np.median(e)),
                    'score_excess_q95':float(np.quantile(e,.95)),
                    'fraction_positive_excess_gt_001':float((e>policy['positive_excess_tolerance']).mean())})
        for metric,aa,bb in [('mean_signed_excess',a.mean(axis=1),b.mean(axis=1)),
                ('mean_positive_excess',np.maximum(a,0).mean(axis=1),np.maximum(b,0).mean(axis=1)),
                ('mean_signed_excess_final_64',a[:,-64:].mean(axis=1),b[:,-64:].mean(axis=1))]:
            effect_rows.append({**common,'scenario':scenario,'metric':metric,'direction_better':'smaller_positive_contamination',
                **paired_mean_interval(aa,bb,counts)})
        progress('Outlier memory diagnosis',args.nomination+' '+scenario+' fixed post-intervention curves and paired excess intervals')
    for scenario in suite['long_null_regimes']:
        p=array(source/'long_null'/scenario,'predictions.npy');ref=array(reference/'long_null'/scenario,'BL80.npy')
        assert p.shape==ref.shape
        cal,refcal=calibration(scenario)
        counts=paired_seed_counts(len(p),suite['paired_bootstrap_replicates'],suite['paired_bootstrap_seed'])
        for alpha in suite['threshold_alphas']:
            cut=threshold(cal.max(axis=1),alpha);refcut=threshold(refcal.max(axis=1),alpha)
            for label,scores,cutoff in [('candidate',p,cut),('reference',ref,refcut)]:
                segment_rows.extend([{**common,'regime':scenario,'model':label,'alpha':alpha,'threshold':cutoff,**r}
                    for r in null_segments(scores,cutoff,segment_length=policy['null_segment_length'])])
            for horizon in suite['long_null_horizons']:
                a=(p[:,:horizon]>cut).any(axis=1).astype(float)
                b=(ref[:,:horizon]>refcut).any(axis=1).astype(float)
                null_rows.append({**common,'regime':scenario,'alpha':alpha,'horizon':horizon,'series':len(p),
                    'candidate_alarm_fraction':float(a.mean()),'reference_alarm_fraction':float(b.mean()),
                    **paired_mean_interval(a,b,counts)})
        progress('Long-null memory diagnosis',args.nomination+' '+scenario+' local segment drift, risk sets and paired fixed-threshold uncertainty')
    tables={'OUTLIER_RECOVERY.csv':recovery_rows,'OUTLIER_AGE_CURVES.csv':curve_rows,
        'PAIRED_OUTLIER_EXCESS.csv':effect_rows,'NULL_SEGMENT_HAZARDS.csv':segment_rows,'PAIRED_NULL_ALARMS.csv':null_rows}
    for filename,rows in tables.items():pd.DataFrame(rows).to_csv(out/filename,index=False)
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),**common,
        'new_fits':0,'new_predictions':0,'new_generating_seeds':0,'seal_rows':0,'gate_changes':False,
        'input_receipt_sha256':evidence,'table_rows':{name:len(rows) for name,rows in tables.items()},
        'interpretation':'Retrospective paired score contamination after the last isolated or burst intervention, plus per-1024 null risk sets. A rising eventual alarm probability alone does not establish score drift. Local quiet-run recovery is finite-window positive-excess recovery, not a causal stopping rule or guaranteed permanent reset. Paired intervals condition on fitted models and independently fitted thresholds and do not propagate calibration or model-selection uncertainty.',
        'files_sha256':{name:digest(out/name) for name in tables}},immutable=True)
    print('NEXT4 MEMORY DIAGNOSIS COMPLETE',args.nomination,flush=True)


if __name__=='__main__':main()
