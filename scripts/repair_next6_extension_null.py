from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np,joblib
from src.next6_extension.common import *
from src.next6_extension.factorized_coherent import discover_and_fit
from src.next6.data import source

def main():
    lock=BASE/'COHERENT_NULL_REPAIR_LOCK.json'
    if not lock.exists():save(lock.name,{'created_utc':utc_now(),'diagnosed_by':'SPARSE_NULL_COHERENCE_COVERAGE.json','original_sources_and_results_preserved':True,
        'repair':'If ANY phase/parent context within an own-state stratum has <8 unique series, both phases and all parent states in that stratum use the same own-only probability. Returned null retains no phase-parent-current interaction.',
        'unchanged':['8 nodes','raw lag16','gain>=.001','positive series fraction>=.55','32 train/validation groups each phase','original group halves','macrostate thresholds and logic','all existing classifier outputs'],
        'not_an_additional_classifier_recipe':True,'old_RP_scores_not_used_for_repair':'Mathematical invariant failed on independently constructed sparse state table; no thresholds relaxed.',
        'source_sha256':digest(ROOT/'src/next6_extension/factorized_coherent.py'),'old_source_sha256':digest(ROOT/'src/next6_extension/factorized.py'),
        'required_rechecks':['all5 train-only graph fits frozen before replay','128 sparse table invariant probes','known positive relation','affected synthetic graph power menu','raw stationary controls through same fresh engine'],
        'old_empirical_power_results_not_confirmatory_for_new_repair':True})
    assert read_json(lock)['source_sha256']==digest(ROOT/'src/next6_extension/factorized_coherent.py')
    data=source();v={k:np.load(BASE/f'block_cache/{k}.npy',mmap_mode='r') for k in ('states','previous','phase_clean','dataset_id','fold')}
    for fold in range(5):
        dest=BASE/f'coherent_graphs/fold_{fold}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'DONE.json').exists():continue
        tr=v['fold']!=fold;train=data.ids[data.series['fold']!=fold];valid=data.ids[data.series['fold']==fold];data.guard.partition(set(map(int,train)),set(map(int,valid)))
        with activity(f'coherent_null_graph_refit_fold{fold}','MATHEMATICAL_REPAIR'):
            graph,records=discover_and_fit(v['states'][tr],v['previous'][tr],v['phase_clean'][tr],v['dataset_id'][tr],train,data.guard.split.groups);joblib.dump(graph,dest/'graph.joblib');csv_write(f'coherent_graphs/fold_{fold}/INNER_DISCOVERY.csv',records)
            save(f'coherent_graphs/fold_{fold}/DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'graph_sha256':digest(dest/'graph.joblib'),'selected':graph.selected,'training_ids_sha256':graph.provenance['training_ids_sha256'],'repair_lock_sha256':digest(lock),'outer_labels_selection':False})
    save('COHERENT_GRAPHS_FROZEN.json',{'created_utc':utc_now(),'status':'ALL_FIVE_FROZEN','graphs':{str(f):digest(BASE/f'coherent_graphs/fold_{f}/graph.joblib') for f in range(5)}});data.guard.check_state()
if __name__=='__main__':main()
