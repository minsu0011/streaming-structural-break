"""Finish finite menus only after all original folds are complete."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from src.next6_extension.common import *
from src.next6_extension.synthetic import CASES
from summarize_next6_extension import rows,baselines,bootstrap,wilson

def synthetic():
    assert all((BASE/f'extended_synthetic/fold_{f}/DONE.json').exists() for f in range(5))
    if not (BASE/'FRESH_VS_FROZEN_PAIRED_SYNTHETIC.csv').exists():baselines()
    ff=rows(BASE/'FRESH_SYNTHETIC_SUMMARY.csv');bb=rows(BASE/'FROZEN_BASELINE_SYNTHETIC_SUMMARY.csv');out=[]
    for case in CASES:
        for horizon in (1024,4096):
            for method in ('FRESH','M0','M3'):
                rr=[r for r in (ff if method=='FRESH' else bb) if r['scenario']==case and int(r['horizon'])==horizon and r['method']==method];assert len(rr)==(1 if method=='FRESH' else 5)
                for metric in ('any_alarm_rate','observed_null_alarm_rate','prealarm_rate','first_detection_rate','any_post_detection_rate','restricted_normalized_delay_mean','recovery_censor_rate'):
                    values=[float(r[metric]) for r in rr if r.get(metric,'')!='']
                    if not values:continue
                    out.append({'scenario':case,'horizon':horizon,'method':method,'metric':metric,'mean_across_fixed_fold_models':float(np.mean(values)),'minimum_fixed_fold_model':min(values),'maximum_fixed_fold_model':max(values),'models':len(values),'paired_seeds':512,'range_is_not_confidence_interval':True,'not_an_ensemble_prediction':True})
    csv_write('COUNTEREXAMPLE_RATE_AND_DELAY_COMPARISON.csv',out)
    # Validate every completed row and its same-stream prefix summaries.
    audit=[]
    for name in ['fresh']+[f'fold_{f}' for f in range(5)]:
        for case in CASES:
            rr=rows(BASE/f'extended_synthetic/{name}/{case}_SERIES.csv');methods=('FRESH',) if name=='fresh' else ('M0','M3');assert len(rr)==512*2*len(methods);lookup={(int(r['seed']),r['method'],int(r['horizon'])):r for r in rr};assert len(lookup)==len(rr)
            for seed in range(512):
                for method in methods:
                    a=lookup[seed,method,1024];b=lookup[seed,method,4096];first=int(a['first_alarm']);longfirst=int(b['first_alarm']);assert float(a['paired_prechange_max_error'])==float(b['paired_prechange_max_error'])==0
                    assert (longfirst==first if first>=0 else longfirst==-1 or longfirst>=1024)
                    assert float(b['score_max'])>=float(a['score_max']);assert all(np.isfinite(float(r['score_max'])) for r in (a,b))
                    for r in (a,b):
                        changed=int(r['tau'])>=0 and int(r['tau'])<int(r['horizon']);assert (r['observed_break']=='True')==changed
                        assert (r['postbreak_first_detection']=='True')==(changed and int(r['first_alarm'])>=int(r['tau']))
            audit.append({'cohort':name,'scenario':case,'rows':len(rr),'unique_seed_method_horizon_keys':True,'same_stream_prefix_first_alarm':True,'official_first_change_target':True})
    csv_write('SYNTHETIC_METRIC_INTEGRITY.csv',audit);save('SYNTHETIC_FINAL_STATISTICS_DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'cohorts':len(audit),'rows_audited':sum(r['rows'] for r in audit),'all_five_models_preserved':True,'paired_CI_unit':'seed, carrying five fixed models together','fold_ranges_are_descriptive_not_independent_CIs':True})

def graph():
    assert all((BASE/f'graph_bootstrap/fold_{f}/DONE.json').exists() and (BASE/f'bootstrap_signs/fold_{f}/DONE.json').exists() for f in range(5))
    if not (BASE/'BOOTSTRAP_EDGE_SELECTION_FREQUENCY.csv').exists():bootstrap()
    from src.next6.nodes import FAMILIES
    out=[];family_rows=[]
    for fold in range(5):
        d=BASE/f'bootstrap_signs/fold_{fold}';original={(e['source'],e['target'],e['lag']):e for e in read_json(d/'ORIGINAL_SIGNS.json')['edges']};rr=[read_json(p) for p in sorted(d.glob('replicate_*.json'))];assert len(rr)==200;keys=set(original)|{(e['source'],e['target'],e['lag']) for r in rr for e in r['edges']}
        for key in sorted(keys):
            selected=[e for r in rr for e in r['edges'] if (e['source'],e['target'],e['lag'])==key];n=len(selected);old=original.get(key);positive=sum(e['sign']>0 for e in selected);phase0=sum(e['selection_phase']==0 for e in selected)
            out.append({'fold':fold,'source':key[0],'target':key[1],'lag':key[2],'selected_resamples':n,'resamples':200,'positive_coefficient_given_selected':positive/n if n else None,'negative_coefficient_given_selected':1-positive/n if n else None,'normal_selection_phase_given_selected':phase0/n if n else None,'original_sign':old['sign'] if old else None,'original_phase':old['selection_phase'] if old else None,'same_sign_as_original_given_selected':sum(e['sign']==old['sign'] for e in selected)/n if old and n else None,'selection_and_phase_conditioning_explicit':True})
        graphs=[read_json(p) for p in sorted((BASE/f'graph_bootstrap/fold_{fold}').glob('replicate_*.json'))];assert len(graphs)==200
        for source_family in sorted(set(FAMILIES)):
            for target_family in sorted(set(FAMILIES)):
                selected=[[e for e in r['selected'] if FAMILIES[e['source']]==source_family and FAMILIES[e['target']]==target_family] for r in graphs];present=[bool(e) for e in selected];p,lo,hi=wilson(present);total=sum(len(e) for e in selected);mechanical=sum(e['mechanical']=='MECHANICAL_DERIVED' for edges in selected for e in edges)
                family_rows.append({'fold':fold,'source_family':source_family,'target_family':target_family,'resamples':200,'at_least_one_family_link_frequency':p,'MC_binomial_lo':lo,'MC_binomial_hi':hi,'mean_family_links':total/200,'mechanical_share_of_selected_links':mechanical/total if total else None,'aggregation':'any admitted lag and either selection phase; fixed source taxonomy','not_independent_causal_support':True})
    csv_write('BOOTSTRAP_SELECTED_COEFFICIENT_SIGNS.csv',out);csv_write('BOOTSTRAP_FAMILY_SELECTION_FREQUENCY.csv',family_rows);save('GRAPH_BOOTSTRAP_FINAL_STATISTICS_DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'graphs':1000,'sign_reconstructions':1000,'same_bootstrap_multiplicities_verified':True,'conditional_training_sensitivity_not_true_edge_probability':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--section',choices=['synthetic','graph'],required=True);a=p.parse_args();globals()[a.section]()
