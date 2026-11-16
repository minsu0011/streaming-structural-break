"""Summaries of repaired-null checks and frozen-head memory attribution."""
from pathlib import Path
import sys,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from src.next6_extension.common import *
from summarize_next6_extension import rows,wilson

def main():
    assert (BASE/'COHERENT_POWER_DONE.json').exists();assert (BASE/'COHERENT_RAW_NULL_DONE.json').exists()
    out=[]
    for d in sorted((BASE/'coherent_power').iterdir()):
        if not d.is_dir():continue
        rr=[read_json(p) for p in d.glob('replicate_*.json')];assert len(rr)==40
        p,lo,hi=wilson([r['true_link_selected'] for r in rr]);fp,flo,fhi=wilson([r['false_edges']>0 for r in rr]);r=rr[0]
        gain=[v['mean_heldout_gain'] for a in rr for v in a['heldout_gains'] if a['true_changed_link'] and v['source']==0 and v['target']==4]
        out.append({k:r[k] for k in ('groups','occupancy','kind','coupling')}|{'replicates':40,'true_edge_selection_rate':p,'true_edge_selection_lo':lo,'true_edge_selection_hi':hi,'any_false_edge_rate':fp,'any_false_edge_lo':flo,'any_false_edge_hi':fhi,'mean_selected_edges':float(np.mean([r['selected_edges'] for r in rr])),'selected_true_edge_heldout_gain_mean':float(np.mean(gain)) if gain else None,'implementation':'coherent whole-own-context backoff'})
    csv_write('COHERENT_INTERACTION_POWER_SUMMARY.csv',out)
    raw=[]
    for case in ('N0','N1','N2','N3'):
        rr=[read_json(p) for p in (BASE/'coherent_raw_null'/case).glob('replicate_*.json')];assert len(rr)==40;p,lo,hi=wilson([r['selected_edges']>0 for r in rr])
        raw.append({'background':case,'graph_replications':40,'series_per_graph':512,'any_false_edge_rate':p,'ci_lo':lo,'ci_hi':hi,'total_selected_edges':sum(r['selected_edges'] for r in rr),'horizon':4096,'H_length':2048,'coherent_null':True})
    csv_write('COHERENT_RAW_NULL_SUMMARY.csv',raw)
    old=rows(BASE/'INTERACTION_POWER_SUMMARY.csv');before={(int(r['groups']),r['occupancy'],r['kind'],float(r['coupling'])):r for r in old};comparison=[]
    for r in out:
        a=before[r['groups'],r['occupancy'],r['kind'],r['coupling']]
        comparison.append({k:r[k] for k in ('groups','occupancy','kind','coupling')}|{'pre_repair_true_selection_rate':float(a['true_edge_selection_rate']),'coherent_true_selection_rate':r['true_edge_selection_rate'],'pre_repair_false_edge_rate':float(a['any_false_edge_rate']),'coherent_false_edge_rate':r['any_false_edge_rate']})
    csv_write('COHERENT_REPAIR_POWER_COMPARISON.csv',comparison)
    memory=[]
    for case in ('O1','OB16','V2','D2'):
        rr=rows(BASE/f'head_memory/{case}/SERIES_WINDOW_ATTRIBUTION.csv');windows=sorted(set(int(r['relative_start']) for r in rr));groups=sorted(set(r['group'] for r in rr));index={(int(r['seed']),int(r['fold']),int(r['relative_start']),r['group']):r for r in rr}
        rng=np.random.default_rng([2026091308,('O1','OB16','V2','D2').index(case)]);weights=rng.multinomial(64,np.full(64,1/64),size=2000)/64
        for low in windows:
            for group in groups:
                for metric in ('mean_signed_contribution','mean_absolute_contribution','actual_mean_prediction_effect'):
                    values=np.array([np.mean([float(index[s,f,low,group][metric]) for f in range(5)]) for s in range(64)]);boot=weights@values;lo,hi=np.quantile(boot,[.025,.975]);sample=index[0,0,low,group]
                    memory.append({'scenario':case,'relative_start':low,'relative_end':sample['relative_end'],'group':group,'metric':metric,'mean':float(values.mean()),'ci_lo':float(lo),'ci_hi':float(hi),'independent_seeds':64,'frozen_fold_models':5,'bootstrap_replicates':2000,'resampling':'whole seed, all fold/window/group predictions carried together','interpretation':'off-manifold paired group-replacement sensitivity; not causal attribution'})
    csv_write('FROZEN_HEAD_MEMORY_SUMMARY.csv',memory)
    save('COHERENT_AND_MEMORY_SUMMARIES_DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'power_settings':len(out),'raw_null_graphs':160,'head_memory_seed_unit':64,'all_five_RP_coherent_graphs_empty':True,'earlier_power_results_retained_as_pre_repair':True})
    for case in ('O1','OB16'):
        print(case,[(r['group'],round(r['mean'],6)) for r in memory if r['scenario']==case and r['relative_start']==512 and r['metric']=='mean_signed_contribution'])
    print('Raw stationary null:',raw)
if __name__=='__main__':main()
