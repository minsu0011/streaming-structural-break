"""One authorized same-recipe continuation after preserved native infrastructure failure."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.synthetic import sources,load_candidate,load_reference,background_mapping,scenario_background
from src.next2.counterexamples import policy as original_policy
from scripts.stress_next4_candidate import reference_cohort,candidate_cohort,analyze_counterexample,analyze_long_null


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recovery-policy',required=True);args=parser.parse_args()
    recovery_path=(ROOT/args.recovery_policy).resolve()
    assert recovery_path.is_relative_to(BASE/'maintenance')
    recovery=read_json(recovery_path);assert recovery['continuation_source_sha256']==digest(Path(__file__))
    for name,sha in recovery['preserved_candidate_files_sha256'].items():assert digest(ROOT/name)==sha
    for name,sha in recovery['unchanged_scientific_source_sha256'].items():assert digest(ROOT/name)==sha
    nomination_id=recovery['nomination'];set_num_threads(2)
    suite_path=BASE/'diagnostic_policy/SYNTHETIC_POLICY.json';suite=read_json(suite_path)
    assert suite['sources_sha256']==sources()
    nomination=read_json(BASE/'synthetic_nominations'/(nomination_id+'.json'))
    assert nomination['suite_sha256']==digest(suite_path) and digest(BASE/'synthetic_nominations'/(nomination_id+'.json'))==recovery['nomination_sha256']
    out=BASE/'synthetic/candidates'/nomination_id
    assert (out/'START.json').exists() and not (out/'RESULTS.json').exists()
    atomic_json(out/'RECOVERY_START.json',{'created_utc':utc_now(),'recovery_policy_sha256':digest(recovery_path),
        'original_failed_job':recovery['original_failed_job'],'new_fits':0,'new_recipe_or_seed':False},immutable=True)
    model=load_candidate(nomination);prepared,proof=load_reference(nomination['training_scope'])
    assert read_json(out/'START.json')['reference_proof']==proof
    lock=original_policy();assert digest(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')==suite['original_generator_policy_sha256']
    sizes=suite['all_dev_sizes'] if nomination['training_scope']=='all_dev' else suite['fold0_sizes']
    mapping=background_mapping(lock);calibration_paths={};reference_paths={}
    for bg,scenario in mapping.items():
        rp=reference_cohort(prepared,proof,nomination['training_scope'],'calibration',scenario,sizes['calibration'],1024,lock,suite_path)
        cp=candidate_cohort(model,nomination,'calibration',scenario,sizes['calibration'],1024,lock,suite_path,rp,out/'calibration'/scenario)
        calibration_paths[bg]=cp;reference_paths[bg]=rp
    analyses=[];reused=0;menu=[s['id'] for s in lock['scenarios']]+suite['additional_counterexamples']
    def previous(path):
        if not path.exists():return None
        r=read_json(path);assert r['status']=='PASS'
        if 'paired_draws_sha256' in r:assert digest(path.parent/'PAIRED_DRAWS.npy')==r['paired_draws_sha256']
        return r
    for scenario in menu:
        rp=reference_cohort(prepared,proof,nomination['training_scope'],'counterexample',scenario,sizes['evaluation'],1024,lock,suite_path)
        cp=candidate_cohort(model,nomination,'counterexample',scenario,sizes['evaluation'],1024,lock,suite_path,rp,out/'counterexample'/scenario)
        old=previous(cp/'ANALYSIS.json')
        if old is not None:
            analyses.append(old);reused+=1;continue
        bg=scenario_background(scenario,lock)
        cal=np.load(calibration_paths[bg]/'predictions.npy',mmap_mode='r',allow_pickle=False)
        refcal=np.load(reference_paths[bg]/'BL80.npy',mmap_mode='r',allow_pickle=False)
        p=np.load(cp/'predictions.npy',mmap_mode='r',allow_pickle=False);ref=np.load(rp/'BL80.npy',mmap_mode='r',allow_pickle=False)
        tau=np.load(rp/'tau.npy',allow_pickle=False)
        analyses.append(analyze_counterexample(nomination,scenario,p,ref,tau,cal,refcal,suite,cp))
        progress('Resumed fixed synthetic diagnostic',nomination_id+' '+scenario+' completed with original models, predictions and seed policy')
    for scenario in suite['long_null_regimes']:
        horizon=max(suite['long_null_horizons'])
        rp=reference_cohort(prepared,proof,nomination['training_scope'],'long_null',scenario,sizes['long_null'],horizon,lock,suite_path)
        cp=candidate_cohort(model,nomination,'long_null',scenario,sizes['long_null'],horizon,lock,suite_path,rp,out/'long_null'/scenario)
        old=previous(cp/'ANALYSIS.json')
        if old is not None:
            analyses.append(old);reused+=1;continue
        bg=scenario_background(scenario,lock)
        cal=np.load(calibration_paths[bg]/'predictions.npy',mmap_mode='r',allow_pickle=False)
        refcal=np.load(reference_paths[bg]/'BL80.npy',mmap_mode='r',allow_pickle=False)
        p=np.load(cp/'predictions.npy',mmap_mode='r',allow_pickle=False);ref=np.load(rp/'BL80.npy',mmap_mode='r',allow_pickle=False)
        analyses.append(analyze_long_null(nomination,scenario,p,ref,cal,refcal,suite,cp))
        progress('Resumed long-null diagnostic',nomination_id+' '+scenario+' all original fixed horizons completed')
    # Earlier complete cohorts, predictions, analyses and score draws stay byte-identical.
    for name,sha in recovery['preserved_candidate_files_sha256'].items():assert digest(ROOT/name)==sha
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
        'nomination':nomination_id,'training_scope':nomination['training_scope'],'new_fits':0,'seal_rows':0,
        'counterexample_scenarios':len(menu),'long_null_regimes':len(suite['long_null_regimes']),
        'long_null_horizons':suite['long_null_horizons'],'cohort_sizes':sizes,'paired_replicates':suite['paired_bootstrap_replicates'],
        'ARCH_decrease_each_delta_positive':all(diagnostics[s]>0 for s in ('V3','V4')),
        'heavy_tail_decrease_each_delta_positive':all(diagnostics[s]>0 for s in ('CB_N3_070','CB_N3_050')),
        'diagnostic_deltas':diagnostics,'promotion':False,'synthetic_only_promotion_allowed':False,
        'thresholds_fit_on_independent_1024_null_cohort':True,'files_sha256':files,
        'recovery_policy_sha256':digest(recovery_path),'original_native_failure_preserved':True,
        'completed_analyses_reused_without_rescoring':reused,'all_prior_candidate_bytes_unchanged':True,
        'scientific_recipe_thresholds_models_seeds_changed':False},immutable=True)
    print('NEXT4 SYNTHETIC CONTINUATION COMPLETE',nomination_id,'prior analyses retained',reused,flush=True)


if __name__=='__main__':main()
