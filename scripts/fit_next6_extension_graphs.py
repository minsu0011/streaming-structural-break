from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np,joblib
from src.next6_extension.common import *
from src.next6_extension.factorized import discover_and_fit,evaluate_pair
from src.next6.data import source

def main():
    data=source();m=read_json(BASE/'block_cache/MANIFEST.json');assert m['fresh_source_sha256']==digest(ROOT/'src/next6_extension/fresh_state.py')
    for name,sha in m['files_sha256'].items():assert digest(BASE/'block_cache'/name)==sha
    v={Path(name).stem:np.load(BASE/'block_cache'/name,mmap_mode='r',allow_pickle=False) for name in m['files_sha256']}
    lock=BASE/'REFINEMENT_IMPLEMENTATION_LOCK.json';sources={p:digest(ROOT/p) for p in ('src/next6_extension/fresh_state.py','src/next6_extension/factorized.py','scripts/fit_next6_extension_graphs.py')}
    if not lock.exists():save(lock.name,{'created_utc':utc_now(),'sources':sources,'design_lock_sha256':digest(BASE/'FRESH_STATE_DESIGN_LOCK.json'),'no_outer_refinement_results_opened':True})
    assert read_json(lock)['sources']==sources
    for fold in range(5):
        dest=BASE/f'fresh_graphs/fold_{fold}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'DONE.json').exists():continue
        tr=v['fold']!=fold;train=data.ids[data.series['fold']!=fold];valid=data.ids[data.series['fold']==fold];data.guard.partition(set(map(int,train)),set(map(int,valid)))
        with activity(f'fit_marginal_controlled_graph_fold{fold}','RELATION_DISCOVERY'):
            graph,records=discover_and_fit(v['states'][tr],v['previous'][tr],v['phase_clean'][tr],v['dataset_id'][tr],train,data.guard.split.groups)
            joblib.dump(graph,dest/'graph.joblib');csv_write(f'fresh_graphs/fold_{fold}/INNER_DISCOVERY.csv',records)
            save(f'fresh_graphs/fold_{fold}/DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'graph_sha256':digest(dest/'graph.joblib'),'selected':graph.selected,'training_ids_sha256':graph.provenance['training_ids_sha256'],'source_lock_sha256':digest(lock),'outer_labels_used':False})
    save('FRESH_GRAPHS_FROZEN.json',{'created_utc':utc_now(),'status':'ALL_FIVE_FROZEN','graphs':{str(f):digest(BASE/f'fresh_graphs/fold_{f}/graph.joblib') for f in range(5)}})
    data.guard.check_state()
if __name__=='__main__':main()
