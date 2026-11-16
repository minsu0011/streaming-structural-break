"""Export actual fitted graph and conditional state tables for faithful figures."""
from pathlib import Path
import sys,os,argparse,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
import joblib,numpy as np
from src.next6.common import BASE,save,read_json,digest,activity,ids_hash
from src.next6.graph import pack_sample,state_contexts
from src.next6.data import arrays
from src.next6.nodes import NAMES,KOREAN

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',default='pair_0');a=p.parse_args()
    output=f'STATE_TRANSITION_TABLES/fold0_{a.tag}.json'
    if (BASE/output).exists():print('Existing graph visualization values retained');return
    with activity('graph_visual_data_'+a.tag,'RELATION_ANALYSIS'):
        directory=BASE/'fold_0/graphs'/a.tag;g=joblib.load(directory/'graph.joblib');data,meta,v=arrays()
        ids=set(g.provenance['training_ids']);sample=pack_sample(data,meta,v,ids);cur,own,ctx=state_contexts(data,v,sample['rows'],g.parents)
        rng=np.random.default_rng(20260913);transitions=[]
        for j in (2,5,7,12):
            for phase in (0,1):
                normal_parent_offset=sum(3**(k+1) for k in range(len(g.parents[j])))
                selected=(sample['phase']==phase)&(cur[:,j]<3)&(own[:,j]<3)&((ctx[:,j]-own[:,j])==normal_parent_offset)
                sids=sample['ids'][selected];unique,codes=np.unique(sids,return_inverse=True)
                counts=np.zeros((len(unique),3,3))
                np.add.at(counts,(codes,own[selected,j],cur[selected,j]),1.)
                fractions=counts/np.maximum(counts.sum(axis=(1,2),keepdims=True),1.)*64
                point=np.sum(fractions,axis=0)+.5;point/=point.sum(axis=1,keepdims=True)
                draws=[]
                for _ in range(200):
                    b=fractions[rng.integers(len(unique),size=len(unique))].sum(axis=0)+.5 if len(unique) else np.full((3,3),.5)
                    draws.append(b/b.sum(axis=1,keepdims=True))
                low,high=np.quantile(draws,[.025,.975],axis=0)
                transitions.append({'node':j,'name':NAMES[j],'korean':KOREAN[j],'phase':'normal' if phase==0 else 'post',
                    'parents':g.parents[j],'condition':'all selected lagged parents in normal state',
                    'table':g.tables[phase,j,normal_parent_offset:normal_parent_offset+3].tolist(),
                    'table_is_actual_model':True,'descriptive_series_equal_conditional_table':point.tolist(),
                    'descriptive_bootstrap_CI_low':low.tolist(),'descriptive_bootstrap_CI_high':high.tolist(),
                    'CI_note':'200 training-series bootstrap of conditional transition frequencies, descriptive; fitted table uses full-series pseudo-count weights and support backoff',
                    'context_independent_series':g.context_support[phase,j,normal_parent_offset:normal_parent_offset+3].tolist(),
                    'conditioned_series':len(unique),'conditioned_sample_rows':int(selected.sum()),'mechanical':'DERIVED_FROM_SHARED_RAW_STREAM'})
        edges=[]
        with (directory/'EDGE_DISCOVERY.csv').open(encoding='utf-8-sig',newline='') as f:discovery=list(csv.DictReader(f))
        for e in g.selected:
            r=[x for x in discovery if int(x['source'])==e['source'] and int(x['target'])==e['target'] and int(x['lag'])==e['lag']]
            phases={}
            for label in (0,1):
                values=[x for x in r if int(x['phase'])==label]
                phases[str(label)]=[{'heldout_series':int(x['support']),'relative_mse_gain':float(x['gain']) if x['gain'] else None,
                    'positive_series_fraction':float(x['positive_fraction']) if x['positive_fraction'] else None} for x in values]
            edges.append({**e,'source_name':NAMES[e['source']],'source_ko':KOREAN[e['source']],
                'target_name':NAMES[e['target']],'target_ko':KOREAN[e['target']],'inner_half_evidence':phases})
        save(output,{'scope':('OUTER_FOLD0_TRAIN / FINAL_TRAINING_GRAPH' if a.tag=='outer_full' else 'OUTER_FOLD0_TRAIN / INNER_DISCOVERY '+a.tag),'training_series':len(ids),
            'training_ids_sha256':ids_hash(ids),'graph_sha256':digest(directory/'graph.joblib'),
            'nodes':[{'id':j,'feature_name':n,'korean':k} for j,(n,k) in enumerate(zip(NAMES,KOREAN))],
            'edges':edges,'transitions':transitions,'inference_uses_no_tau':True,
            'warning':'Predictive lead-lag, not economic causality. Actual model probabilities, not a generative illustration.'})

if __name__=='__main__':main()
