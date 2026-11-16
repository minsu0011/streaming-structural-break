"""Fixed-seed diagnostic decomposition of already evaluated outlier predictions."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.synthetic import load_candidate,cohort_series
from src.next2.counterexamples import policy as generator_policy
from src.next2.blend import combine
from src.next4.outlier_attribution import group_name,allocate_probability_change


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--nomination',required=True);args=parser.parse_args()
    pp=BASE/'diagnostic_policy/OUTLIER_ATTRIBUTION_POLICY.json';policy=read_json(pp)
    for name,sha in policy['sources_sha256'].items():assert digest(ROOT/name)==sha
    npth=BASE/'synthetic_nominations'/(args.nomination+'.json')
    assert digest(npth)==policy['nominations'][args.nomination]
    nomination=read_json(npth);model=load_candidate(nomination);candidate=nomination['candidate']
    own_weight=.8 if nomination['family']=='B' else .1;partner_weight=1-own_weight
    partner_name='C205' if nomination['family']=='B' else 'BL80'
    selected_names=[model.names[int(i)] for i in model.columns]
    groups=[group_name(candidate,name) for name in selected_names]+['expected_margin']
    unique=sorted(set(groups));positions={group:np.flatnonzero(np.array(groups)==group) for group in unique}
    source=BASE/'synthetic/candidates'/args.nomination;reference=BASE/'synthetic/reference'/nomination['training_scope']
    completed=read_json(source/'RESULTS.json');assert completed['status']=='PASS'
    suite=read_json(BASE/'diagnostic_policy/SYNTHETIC_POLICY.json');lock=generator_policy()
    assert digest(BASE/'diagnostic_policy/SYNTHETIC_POLICY.json')==policy['synthetic_policy_sha256']
    assert digest(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')==suite['original_generator_policy_sha256']
    out=BASE/'outlier_attribution'/args.nomination;out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'policy_sha256':digest(pp),'nomination_sha256':digest(npth),
        'synthetic_result_sha256':digest(source/'RESULTS.json'),'model_sha256':nomination['model_sha256']},immutable=True)
    evidence={};summary=[];curves=[];parity=[]
    def array(directory,name):
        receipt=read_json(directory/'RESULTS.json');assert receipt['status']=='PASS'
        path=directory/name;assert digest(path)==receipt['files_sha256'][name]
        evidence[(directory/'RESULTS.json').relative_to(ROOT).as_posix()]=digest(directory/'RESULTS.json')
        return np.load(path,mmap_mode='r',allow_pickle=False)
    for scenario,burst in policy['scenarios'].items():
        cached=array(source/'counterexample'/scenario,'predictions.npy')
        partner=array(reference/'counterexample'/scenario,partner_name+'.npy')
        assert cached.shape==partner.shape and len(cached)%2==0
        n=len(cached)//2;seeds=policy['paired_seed_count'];assert seeds<=n
        window=policy['post_intervention_window'];group_arrays={g:np.empty((seeds,window)) for g in unique}
        group_arrays['frozen_partner']=np.empty((seeds,window));group_arrays['float32_rounding_residual']=np.empty((seeds,window))
        total=np.empty((seeds,window));max_shap=max_native=max_round=0.
        for seed in range(seeds):
            stop=suite['tau_cycle'][seed%len(suite['tau_cycle'])]+burst
            columns=np.arange(stop,stop+window);assert columns[-1]<cached.shape[1]
            contributions=[]
            for control,row in ((True,seed),(False,seed+n)):
                h,o,tau=cohort_series('counterexample',scenario,seed,cached.shape[1],control,lock);assert tau==-1
                x=model.make_state(h).replay(o);expert=np.asarray(model.predict(x),dtype=np.float32)
                expected=combine(expert,partner[row],.8) if nomination['family']=='B' else combine(partner[row],expert,.9)
                np.testing.assert_array_equal(expected,cached[row])
                matrix=np.asarray(x[columns],dtype=np.float32)[:,model.columns]
                booster=model.estimator.booster_;shap=np.asarray(booster.predict(matrix,pred_contrib=True,num_threads=1),dtype=np.float64)
                raw=np.asarray(booster.predict(matrix,raw_score=True,num_threads=1),dtype=np.float64)
                native=np.asarray(booster.predict(matrix,num_threads=1),dtype=np.float64)
                assert shap.shape==(window,len(model.columns)+1)
                error=float(np.max(np.abs(shap.sum(axis=1)-raw)));max_shap=max(max_shap,error)
                np.testing.assert_allclose(shap.sum(axis=1),raw,rtol=0,atol=1e-9)
                error=float(np.max(np.abs(native-expert[columns])));max_native=max(max_native,error)
                np.testing.assert_allclose(native,expert[columns],rtol=0,atol=1e-7)
                contributions.append(shap)
            allocated,own_delta=allocate_probability_change(contributions[0],contributions[1],expert_weight=own_weight)
            for group,index in positions.items():group_arrays[group][seed]=allocated[:,index].sum(axis=1)
            group_arrays['frozen_partner'][seed]=partner_weight*(partner[seed+n,columns].astype(float)-partner[seed,columns].astype(float))
            total[seed]=cached[seed+n,columns].astype(float)-cached[seed,columns].astype(float)
            reconstructed=allocated.sum(axis=1)+group_arrays['frozen_partner'][seed]
            residual=total[seed]-reconstructed;max_round=max(max_round,float(np.max(np.abs(residual))))
            np.testing.assert_allclose(reconstructed,total[seed],rtol=0,atol=2e-7)
            group_arrays['float32_rounding_residual'][seed]=residual
            if (seed+1)%16==0:
                progress('Fitted outlier attribution',args.nomination+' '+scenario+' '+str(seed+1)+'/128 paired seeds; raw-margin and cached-score checks')
                print('NEXT4 OUTLIER ATTRIBUTION',args.nomination,scenario,seed+1,seeds,flush=True)
        np.testing.assert_allclose(sum(group_arrays.values()),total,rtol=0,atol=1e-14)
        for group,values in group_arrays.items():
            per_seed=values.mean(axis=1)
            summary.append({'candidate':candidate,'nomination':args.nomination,'training_scope':nomination['training_scope'],
                'scenario':scenario,'group':group,'paired_seeds':seeds,'post_intervention_points':window,
                'mean_allocated_score_excess':float(per_seed.mean()),'median_seed_mean':float(np.median(per_seed)),
                'seed_mean_q10':float(np.quantile(per_seed,.1)),'seed_mean_q90':float(np.quantile(per_seed,.9)),
                'mean_allocated_excess_final_64':float(values[:,-64:].mean())})
            for age in range(window):
                curves.append({'candidate':candidate,'nomination':args.nomination,'scenario':scenario,'group':group,
                    'age_after_last_intervention':age+1,'paired_seeds':seeds,'mean_allocated_score_excess':float(values[:,age].mean())})
        parity.append({'scenario':scenario,'paired_seeds':seeds,'cached_predictions_exact':True,
            'native_compact_probability_max_error':max_native,'Tree_SHAP_raw_margin_additivity_max_error':max_shap,
            'maximum_float32_rounding_residual':max_round,'group_decomposition_closes_with_rounding_residual':True})
    pd.DataFrame(summary).to_csv(out/'GROUP_CONTRIBUTIONS.csv',index=False)
    pd.DataFrame(curves).to_csv(out/'AGE_GROUP_CURVES.csv',index=False)
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'candidate':candidate,'nomination':args.nomination,
        'training_scope':nomination['training_scope'],'paired_seed_count':policy['paired_seed_count'],'groups':unique,
        'parity':parity,'new_fits':0,'new_prediction_recipes':0,'new_quality_scores':0,'new_random_seeds':0,'seal_rows':0,
        'source_receipt_sha256':evidence,'gate_changes':False,
        'interpretation':'LightGBM native Tree-SHAP margin changes between paired clean and perturbed inputs are allocated through the sigmoid secant, then the unchanged expert blend weight. This is an algebraic probability-change allocation, not probability-space SHAP, causal feature attribution, model ablation or a repair recipe. The first 128 original generating seeds were fixed before synthetic scores; no selection of only alarming paths. Quantiles describe seed heterogeneity, not confidence intervals.',
        'method_sources':['https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html','https://arxiv.org/abs/1802.03888'],
        'files_sha256':{p.name:digest(p) for p in out.glob('*.csv')}},immutable=True)
    print('NEXT4 OUTLIER ATTRIBUTION COMPLETE',args.nomination,flush=True)


if __name__=='__main__':main()
