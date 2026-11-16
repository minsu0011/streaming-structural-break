from pathlib import Path
import sys,os,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np,joblib
from src.next6_extension.common import *
from src.next6_extension.categorical_controls import generated
from src.next6_extension.factorized import discover_and_fit,evaluate_pair

def main():
    path=BASE/'INTERACTION_POWER_LOCK.json'
    if not path.exists():save(path.name,{'created_utc':utc_now(),'groups':[64,256,1024],'blocks_per_group':64,'occupancy':['uniform','central80percent'],'changed_coupling':[0.,.15,.30,.60,.90],
        'null_controls':['stable_link_coupling.6','marginal_shift_without_link'],'repeats':40,'seed':2026091303,'known_true_changed_edge':[0,4,16],
        'purpose':'Separate absence of evidence from lack of sensitivity; quantify false inclusion under stable predictive edges and changed marginals. No RP tuning or extra detector recipe.',
        'minimum_support_note':'64 groups can fail32 retained training/validation series if allocation differs; record INSUFFICIENT_EVIDENCE, never lower support.',
        'source_sha256':digest(ROOT/'src/next6_extension/categorical_controls.py'),'graph_source_sha256':digest(ROOT/'src/next6_extension/factorized.py')})
    lock=read_json(path);assert lock['graph_source_sha256']==digest(ROOT/'src/next6_extension/factorized.py')
    menu=[]
    for n in lock['groups']:
        for occupancy in lock['occupancy']:
            menu.extend((n,occupancy,'changed_link',c) for c in lock['changed_coupling']);menu.extend((n,occupancy,k,.6) for k in ('stable_link','marginal_shift'))
    for ordinal,(n,occupancy,kind,coupling) in enumerate(menu):
        key=f'{n}_{occupancy}_{kind}_{coupling:.2f}';dest=BASE/'interaction_power'/key;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'DONE.json').exists():continue
        with activity('interaction_graph_power_'+key,'SENSITIVITY_AND_FALSE_DISCOVERY'):
            for repeat in range(40):
                receipt=dest/f'replicate_{repeat:02}.json'
                if receipt.exists():continue
                start=time.perf_counter();cur,prev,phase,ids=generated(n,64,[2026091303,ordinal,repeat,0],coupling,occupancy,kind);groups=set(map(int,ids));graph,records=discover_and_fit(cur,prev,phase,ids,groups,{g:g for g in groups})
                truth=kind=='changed_link' and coupling>0;true_selected=any(e['source']==0 and e['target']==4 for e in graph.selected);false=sum(not(truth and e['source']==0 and e['target']==4) for e in graph.selected)
                evaluated=[]
                if graph.selected:
                    a,b,p,sid=generated(n,64,[2026091303,ordinal,repeat,1],coupling,occupancy,kind)
                    for edge in graph.selected:
                        j=edge['target'];table=(graph.full[:,j],graph.null[:,j],graph.support[:,j],{})
                        ev=evaluate_pair(table,a,b,p,sid,edge['source'],j)
                        evaluated.append({'source':edge['source'],'target':j,'mean_heldout_gain':float(np.mean([e['mean_gain'] for e in ev]))})
                info={'groups':n,'occupancy':occupancy,'kind':kind,'coupling':coupling,'repeat':repeat,'true_changed_link':truth,'true_link_selected':true_selected,
                    'false_edges':false,'selected_edges':len(graph.selected),'heldout_gains':evaluated,'wall_seconds':time.perf_counter()-start,'selected':graph.selected}
                save(str(receipt.relative_to(BASE)),info)
                if repeat==0 and coupling==.9 and occupancy=='uniform' and n==256:
                    joblib.dump(graph,BASE/'interaction_power/KNOWN_POSITIVE_GRAPH.joblib');csv_write('interaction_power/KNOWN_POSITIVE_DISCOVERY.csv',records)
                if (repeat+1)%10==0:checkpoint('interaction sensitivity',key+' '+str(repeat+1)+'/40')
            save(str((dest/'DONE.json').relative_to(BASE)),{'status':'COMPLETE','repeats':40,'created_utc':utc_now()})
    save('INTERACTION_POWER_DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'settings':len(menu),'replicates':len(menu)*40})
if __name__=='__main__':main()
