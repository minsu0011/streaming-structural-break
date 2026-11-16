from pathlib import Path
import sys,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,joblib
from src.next6_extension.common import *
from src.next6_extension.observer import ObservableModel

def main():
    lock=BASE/'OBSERVABLE_NETWORK_EXPORT_LOCK.json'
    if not lock.exists():save(lock.name,{'created_utc':utc_now(),'purpose':'Export already fitted outer-training macro transition tables as an actual streaming observer. No new threshold, classifier fitting or blend.',
        'models':'Both normal/post P(current observable state | previous observable state) every observed transition; first point has transition_available=0 and neutral probabilities1.',
        'support':'Report unique training series for both conditions; fewer8 flagged unsupported. Do not confuse transition probabilities with breakOccurred probability.',
        'source_tables':'rp_replay/macro_probabilities.npy and MACRO_STATE_TRANSITIONS.csv, fitted before the fixed RP score release','coherent_graphs_frozen':digest(BASE/'COHERENT_GRAPHS_FROZEN.json'),
        'source_sha256':digest(ROOT/'src/next6_extension/observer.py')})
    assert read_json(lock)['source_sha256']==digest(ROOT/'src/next6_extension/observer.py')
    p=np.load(BASE/'rp_replay/macro_probabilities.npy');cut=read_json(BASE/'calibration/fresh/DONE.json')['thresholds']['FRESH']
    with (BASE/'MACRO_STATE_TRANSITIONS.csv').open(encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
    sources={name:digest(ROOT/name) for name in ('src/next6_extension/fresh_state.py','src/next6_extension/fresh_runtime.py','src/next6_extension/factorized.py','src/next6_extension/factorized_coherent.py','src/next6_extension/observer.py')}
    with activity('export_learned_observable_networks','MODEL_EXPORT'):
        for fold in range(5):
            support=np.zeros((2,7,7),dtype=np.int32)
            for r in rr:
                if int(r['fold'])==fold:support[int(r['phase']),int(r['previous']),int(r['current'])]=int(r['unique_training_series'])
            graph=joblib.load(BASE/f'coherent_graphs/fold_{fold}/graph.joblib');model=ObservableModel(cut,graph,p[fold],support,{'fold':fold,'training_ids_sha256':graph.provenance['training_ids_sha256'],'training_series':5120,
                'macro_tables_sha256':digest(BASE/'MACRO_STATE_TRANSITIONS.csv'),'smoothing':.5,'series_phase_total_count':64,'graph_repair_lock_sha256':digest(BASE/'COHERENT_NULL_REPAIR_LOCK.json'),'not_a_break_probability_classifier':True},sources);model.validate()
            dest=BASE/f'observable_models/fold_{fold}';dest.mkdir(parents=True,exist_ok=True);assert not (dest/'model.joblib').exists();joblib.dump(model,dest/'model.joblib');save(f'observable_models/fold_{fold}/DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'model_sha256':digest(dest/'model.joblib'),'training_ids_sha256':model.provenance['training_ids_sha256'],'export_only_no_refit':True})
    save('OBSERVABLE_NETWORKS_FROZEN.json',{'status':'COMPLETE','created_utc':utc_now(),'models':{str(f):digest(BASE/f'observable_models/fold_{f}/model.joblib') for f in range(5)}})
if __name__=='__main__':main()
