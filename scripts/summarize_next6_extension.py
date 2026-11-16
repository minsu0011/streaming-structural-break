from pathlib import Path
import sys,csv,math,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.next6_extension.common import *
from src.next6_extension.synthetic import CASES

def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def wilson(values):
    x=np.asarray(values,dtype=float);n=len(x)
    if not n:return None,None,None
    p=x.mean();z=1.959963984540054;den=1+z*z/n;c=(p+z*z/(2*n))/den;d=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(p),float(max(0,c-d)),float(min(1,c+d))
def grouped_summary(allrows):
    out=[]
    for case in CASES:
        for horizon in (1024,4096):
            rr=[r for r in allrows if r['scenario']==case and int(r['horizon'])==horizon]
            if not rr:continue
            assert len(rr)==512
            changed=[r for r in rr if r['observed_break']=='True'];null=[r for r in rr if r['observed_break']!='True'];result={'scenario':case,'horizon':horizon,'series':512,'observed_break_series':len(changed),'observed_no_break_series':len(null),'kind':rr[0]['kind'],'method':rr[0]['method'],'fold':rr[0]['fold']}
            for name,items,key in [('any_alarm',rr,'any_alarm'),('observed_null_alarm',null,'any_alarm'),('prealarm',changed,'prebreak_alarm'),('first_detection',changed,'postbreak_first_detection'),('any_post_detection',changed,'postbreak_any_detection')]:
                p,lo,hi=wilson([r[key]=='True' for r in items]);result.update({name+'_rate':p,name+'_lo':lo,name+'_hi':hi})
            delay=[float(r['first_detection_delay']) for r in changed if r['first_detection_delay']];result['conditional_first_detection_delay_median']=float(np.median(delay)) if delay else None
            result['restricted_normalized_delay_mean']=float(np.mean([float(r['first_detection_delay'])/(horizon-int(r['tau'])) if r['first_detection_delay'] else 1. for r in changed])) if changed else None
            recovered=[r for r in rr if r.get('recovery_censored','')!='']
            if recovered:
                p,lo,hi=wilson([r['recovery_censored']=='True' for r in recovered]);result.update({'recovery_censor_rate':p,'recovery_censor_lo':lo,'recovery_censor_hi':hi});d=[float(r['recovery_delay']) for r in recovered if r['recovery_delay']];result['conditional_recovery_delay_median']=float(np.median(d)) if d else None
                result['max_paired_effect_after_128']=max(float(r['paired_effect_after_128_max']) for r in recovered)
            out.append(result)
    return out

def fresh():
    if not (BASE/'FRESH_SYNTHETIC_SUMMARY.csv').exists():
        allrows=[r for p in (BASE/'extended_synthetic/fresh').glob('*_SERIES.csv') for r in rows(p)];csv_write('FRESH_SYNTHETIC_SUMMARY.csv',grouped_summary(allrows))
    if not (BASE/'INTERACTION_POWER_SUMMARY.csv').exists():
        assert (BASE/'INTERACTION_POWER_DONE.json').exists();out=[]
        for d in sorted((BASE/'interaction_power').iterdir()):
            if not d.is_dir():continue
            rr=[read_json(p) for p in d.glob('replicate_*.json')];assert len(rr)==40
            p,lo,hi=wilson([r['true_link_selected'] for r in rr]);fp,flo,fhi=wilson([r['false_edges']>0 for r in rr]);r=rr[0]
            out.append({k:r[k] for k in ('groups','occupancy','kind','coupling')}|{'replicates':40,'true_edge_selection_rate':p,'true_edge_selection_lo':lo,'true_edge_selection_hi':hi,'any_false_edge_rate':fp,'any_false_edge_lo':flo,'any_false_edge_hi':fhi,'mean_selected_edges':float(np.mean([r['selected_edges'] for r in rr]))})
        csv_write('INTERACTION_POWER_SUMMARY.csv',out)
    if not (BASE/'CALIBRATION_TRANSPORT_SUMMARY.csv').exists():
        rr=rows(BASE/'CALIBRATION_TRANSPORT_SERIES.csv');out=[]
        for nh in (1000,1500,3000,5000):
            for case in ('N0','N1','N2','N3'):
                for horizon in (1024,4096):
                    for fraction in (0.,.01):
                        pp=[r for r in rr if int(r['H_length'])==nh and r['background']==case and int(r['horizon'])==horizon and float(r['H_contamination_fraction'])==fraction];assert len(pp)==512;p,lo,hi=wilson([r['alarm']=='True' for r in pp]);out.append({'H_length':nh,'background':case,'horizon':horizon,'H_contamination_fraction':fraction,'series':512,'alarm_rate':p,'ci_lo':lo,'ci_hi':hi})
        csv_write('CALIBRATION_TRANSPORT_SUMMARY.csv',out)
    checkpoint('extension preliminary summaries','Independent synthetic, power and transport tables complete')

def baselines():
    allrows=[];summary=[]
    for fold in range(5):
        assert (BASE/f'extended_synthetic/fold_{fold}/DONE.json').exists()
        rr=[r for p in (BASE/f'extended_synthetic/fold_{fold}').glob('*_SERIES.csv') for r in rows(p)];allrows.extend(rr)
        for method in ('M0','M3'):summary.extend(grouped_summary([r for r in rr if r['method']==method]))
    csv_write('FROZEN_BASELINE_SYNTHETIC_SUMMARY.csv',summary)
    ff=[r for p in (BASE/'extended_synthetic/fresh').glob('*_SERIES.csv') for r in rows(p)];out=[]
    for ordinal,case in enumerate(CASES):
        for horizon in (1024,4096):
            f={int(r['seed']):r for r in ff if r['scenario']==case and int(r['horizon'])==horizon};assert len(f)==512
            for method in ('M0','M3'):
                bb={(int(r['fold']),int(r['seed'])):r for r in allrows if r['scenario']==case and int(r['horizon'])==horizon and r['method']==method};assert len(bb)==5*512
                for metric,column,positive_only in [('alarm','any_alarm',False),('first_detection','postbreak_first_detection',True),('prealarm','prebreak_alarm',True)]:
                    seeds=[s for s in range(512) if not positive_only or f[s]['observed_break']=='True']
                    if not seeds:continue
                    effects=np.array([float(f[s][column]=='True')-np.mean([bb[fold,s][column]=='True' for fold in range(5)]) for s in seeds]);rng=np.random.default_rng(np.random.SeedSequence([2026091305,ordinal,horizon,len(seeds)]));n=len(seeds);boot=rng.multinomial(n,np.full(n,1/n),size=1000)@effects/n;lo,hi=np.quantile(boot,[.025,.975])
                    direction=True if metric=='first_detection' else False
                    if metric=='alarm' and any(f[s]['observed_break']=='True' for s in seeds):direction=None
                    out.append({'scenario':case,'horizon':horizon,'reference':method,'metric':metric,'paired_seeds':n,'reference_fold_models':5,'FRESH_minus_reference_mean':float(effects.mean()),'ci_lo':float(lo),'ci_hi':float(hi),'resampling':'whole seed, carrying all five fold predictions together','higher_is_better':direction,'direction_note':'Any-path alarm mixes premature and post-change alarms; use first_detection and prealarm separately' if direction is None else 'Higher first detection is favorable' if direction else 'Lower null-path or premature alarm is favorable','fixed_prediction_conditional':True})
    csv_write('FRESH_VS_FROZEN_PAIRED_SYNTHETIC.csv',out)

def bootstrap():
    out=[];counts=[]
    from src.next6.nodes import NAMES
    import joblib
    for fold in range(5):
        d=BASE/f'graph_bootstrap/fold_{fold}';assert (d/'DONE.json').exists();rr=[read_json(p) for p in d.glob('replicate_*.json')];assert len(rr)==200
        original=joblib.load(ROOT/f'artifacts/next6/fold_{fold}/graphs/outer_full/graph.joblib');original_keys={(e['source'],e['target'],e['lag']) for e in original.selected};keys=original_keys|{(e['source'],e['target'],e['lag']) for r in rr for e in r['selected']}
        for key in sorted(keys):
            included=[any((e['source'],e['target'],e['lag'])==key for e in r['selected']) for r in rr];p,lo,hi=wilson(included);out.append({'fold':fold,'source':key[0],'target':key[1],'lag':key[2],'source_name':NAMES[key[0]],'target_name':NAMES[key[1]],'original_selected':key in original_keys,'resamples':200,'selection_frequency':p,'MC_binomial_lo':lo,'MC_binomial_hi':hi,'interpretation':'conditional bootstrap selection frequency; interval is Monte Carlo uncertainty, not probability edge is true','group_bootstrap_fixed_internal_halves':True})
        for r in rr:counts.append({'fold':fold,'replicate':r['replicate'],'edges':len(r['selected']),'mechanical_edges':sum(e['mechanical']=='MECHANICAL_DERIVED' for e in r['selected']),'wall_seconds':r['wall_seconds']})
    csv_write('BOOTSTRAP_EDGE_SELECTION_FREQUENCY.csv',out);csv_write('BOOTSTRAP_GRAPH_SIZE_DISTRIBUTION.csv',counts)

def main():
    p=argparse.ArgumentParser();p.add_argument('--section',choices=['fresh','baselines','bootstrap'],required=True);a=p.parse_args();globals()[a.section]()
if __name__=='__main__':main()
