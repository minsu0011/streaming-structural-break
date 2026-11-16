"""Affected rechecks after the sparse-null mathematical repair; no detector tuning."""
from pathlib import Path
import sys,os,time,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np,joblib
from src.next6_extension.common import *
from src.next6_extension.factorized_coherent import discover_and_fit,evaluate_pair
from src.next6_extension.categorical_controls import generated
from src.next6_extension.fresh_runtime import SafeState
from src.next6_extension.fresh_state import categorize
from src.next2.counterexamples import generate

def power():
    lock=read_json(BASE/'INTERACTION_POWER_LOCK.json');menu=[]
    for n in lock['groups']:
        for occupancy in lock['occupancy']:
            menu.extend((n,occupancy,'changed_link',c) for c in lock['changed_coupling']);menu.extend((n,occupancy,k,.6) for k in ('stable_link','marginal_shift'))
    for ordinal,(n,occupancy,kind,coupling) in enumerate(menu):
        key=f'{n}_{occupancy}_{kind}_{coupling:.2f}';dest=BASE/'coherent_power'/key;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'DONE.json').exists():continue
        with activity('repaired_interaction_power_'+key,'AFFECTED_RECHECK'):
            for repeat in range(40):
                receipt=dest/f'replicate_{repeat:02}.json'
                if receipt.exists():continue
                start=time.perf_counter();cur,prev,phase,ids=generated(n,64,[2026091303,ordinal,repeat,0],coupling,occupancy,kind);groups=set(map(int,ids));graph,records=discover_and_fit(cur,prev,phase,ids,groups,{g:g for g in groups})
                truth=kind=='changed_link' and coupling>0;true_selected=any(e['source']==0 and e['target']==4 for e in graph.selected);false=sum(not(truth and e['source']==0 and e['target']==4) for e in graph.selected);evaluated=[]
                if graph.selected:
                    a,b,p,sid=generated(n,64,[2026091303,ordinal,repeat,1],coupling,occupancy,kind)
                    for edge in graph.selected:
                        j=edge['target'];ev=evaluate_pair((graph.full[:,j],graph.null[:,j],graph.support[:,j],{}),a,b,p,sid,edge['source'],j);evaluated.append({'source':edge['source'],'target':j,'mean_heldout_gain':float(np.mean([e['mean_gain'] for e in ev]))})
                save(str(receipt.relative_to(BASE)),{'groups':n,'occupancy':occupancy,'kind':kind,'coupling':coupling,'repeat':repeat,'true_changed_link':truth,'true_link_selected':true_selected,'false_edges':false,'selected_edges':len(graph.selected),'heldout_gains':evaluated,'wall_seconds':time.perf_counter()-start,'selected':graph.selected,'null_coherence_validated':True})
                if repeat==0 and coupling==.9 and occupancy=='uniform' and n==256:joblib.dump(graph,BASE/'coherent_power/KNOWN_POSITIVE_GRAPH.joblib')
            save(str((dest/'DONE.json').relative_to(BASE)),{'status':'COMPLETE','repeats':40,'created_utc':utc_now()});checkpoint('repaired power condition complete',key)
    save('COHERENT_POWER_DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'settings':len(menu),'replicates':len(menu)*40,'purpose':'Necessary affected recheck of mathematical repair, same locked menu; not adaptive model tuning.'})

def raw_null():
    original=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')
    for case in ('N0','N1','N2','N3'):
        dest=BASE/'coherent_raw_null'/case;dest.mkdir(parents=True,exist_ok=True)
        with activity('coherent_raw_stationary_'+case,'RAW_LEVEL_NEGATIVE_CONTROL'):
            for replicate in range(40):
                receipt=dest/f'replicate_{replicate:02}.json'
                if receipt.exists():continue
                cur=[];prev=[];phase=[];ids=[]
                for seed in range(512):
                    h,o,tau=generate(case,replicate*512+seed,partition=35,horizon=4096,control=True,lock=original);state=SafeState(h);initial=state.H_previous_state.copy();out=state.replay(o);end=np.flatnonzero(out[:,4]);s=categorize(out[end,12:]);p=np.full(len(end),-1,dtype=np.int8);p[end<2048]=0;p[end-31>=2048]=1
                    cur.append(s);prev.append(np.vstack((initial[None,:],s[:-1])));phase.append(p);ids.append(np.full(len(s),1000000+seed,dtype=np.int32))
                cc=np.concatenate(cur);pp=np.concatenate(prev);yy=np.concatenate(phase);ii=np.concatenate(ids);groups=set(map(int,ii));graph,records=discover_and_fit(cc,pp,yy,ii,groups,{g:g for g in groups})
                save(str(receipt.relative_to(BASE)),{'replicate':replicate,'background':case,'series':512,'actual_change':False,'artificial_phase_split':2048,'phase_label_used_only_for_false_discovery_control':True,'selected_edges':len(graph.selected),'selected':graph.selected,'same_fresh_feature_engine':True,'null_coherence_validated':True})
                if replicate==0:csv_write(f'coherent_raw_null/{case}/FIRST_DISCOVERY.csv',records)
                if (replicate+1)%5==0:checkpoint('same-engine stationary raw null',case+' '+str(replicate+1)+'/40')
        save(f'coherent_raw_null/{case}/DONE.json',{'status':'COMPLETE','replicates':40,'created_utc':utc_now()})
    save('COHERENT_RAW_NULL_DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'backgrounds':4,'replicates_per_background':40,'series_per_replicate':512})

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['power','raw-null'],required=True);a=p.parse_args();lock=BASE/'COHERENT_RECHECK_LOCK.json'
    if not lock.exists():save(lock.name,{'created_utc':utc_now(),'coherent_source_sha256':digest(ROOT/'src/next6_extension/factorized_coherent.py'),'runner_sha256':digest(Path(__file__)),
        'power':'Repeat all42 previously locked conditions x40, same seeds, because fallback changed. Earlier results remain preserved and labeled pre-repair.',
        'raw_null':'4 stationary raw backgrounds x40 graph replications x512 series; partition35; same SafeState node engine;4096 horizon; artificial phase split2048; no true structural break.',
        'no_new_candidate_or_threshold_search':True})
    assert read_json(lock)['runner_sha256']==digest(Path(__file__))
    power() if a.stage=='power' else raw_null()
if __name__=='__main__':main()
