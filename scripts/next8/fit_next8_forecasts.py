"""Run preregistered auxiliary comparators; never release heldout scores."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,read_json,atomic,digest,utc_now,note
from src.next8.cache import FeatureCache
from src.next8.data import FitScope
from src.next8.screening import procedure_scope
from src.next8.forecasting import fit_full_comparators,fit_cross_state_forecasts
from src.next8.transitions import fit_transition_comparators


def start(directory,receipt):
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'START.json'
    if path.exists():
        previous=read_json(path)
        for key,value in receipt.items():assert previous[key]==value,key
    else:atomic(path,{'timestamp':utc_now(),**receipt})


def main(fold,meta_inner,early,mode):
    assert read_json(BASE/'FORECAST_TRANSITION_TEST_GATE_NEXT8.json')['status']=='PASS'
    lock_path=BASE/'FORECAST_TRANSITION_IMPLEMENTATION_LOCK_NEXT8.json'
    lock=read_json(lock_path)
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    if mode=='full' and meta_inner is not None:raise ValueError('Full comparators only outer scopes; no redundant meta copies')
    cache=FeatureCache();scope=procedure_scope(cache,fold,meta_inner)
    if early:scope=FitScope(scope.training_ids,scope.validation_ids,scope.outer_excluded_ids,scope.name+'_EARLY_T199')
    gp=BASE/'parent_selection'/scope.name
    gd=read_json(gp/'DONE.json');assert gd['status']=='COMPLETE'
    assert digest(gp/'GRAPH.json')==gd['graph_sha256']
    graph=read_json(gp/'GRAPH.json')
    receipt={'status':'FROZEN','scope':scope.provenance(),'early_training_cutoff':199 if early else None,
        'graph_sha256':digest(gp/'GRAPH.json'),'implementation_lock_sha256':digest(lock_path),
        'outer_or_meta_score_released':False}
    donor=None
    if mode=='full':
        sp=BASE/'h_context_selection'/scope.name
        sd=read_json(sp/'DONE.json');assert sd['status']=='COMPLETE'
        assert digest(sp/'SELECTION.json')==sd['selection_sha256']
        selection=read_json(sp/'SELECTION.json')
        full=BASE/'forecasts'/scope.name;donor=full
        start(full,{**receipt,'selection_sha256':digest(sp/'SELECTION.json')})
        if not (full/'DONE.json').exists():fit_full_comparators(cache,scope,selection,graph,full,early=early)
        else:
            done=read_json(full/'DONE.json');assert done['scope']==scope.provenance()
            for p in full.glob('F*/DONE.json'):
                r=read_json(p)
                for name,sha in r['files_sha256'].items():assert digest(p.parent/name)==sha
        tp=BASE/'transitions'/scope.name;start(tp,receipt)
        if not (tp/'DONE.json').exists():fit_transition_comparators(cache,scope,graph,tp,early=early)
        else:
            done=read_json(tp/'DONE.json');assert done['scope']==scope.provenance()
            for name,sha in done['files_sha256'].items():assert digest(tp/name)==sha
    output=BASE/'state_forecasts'/scope.name;start(output,receipt)
    fit_cross_state_forecasts(cache,scope,graph,output,early=early,reuse_directory=donor)
    note('Auxiliary forecasts and state summaries complete',scope.name+' '+mode+'; heldout predictions unscored','B2_STATE_OBSERVABILITY')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True)
    p.add_argument('--meta-inner',type=int,choices=range(3));p.add_argument('--early',action='store_true')
    p.add_argument('--mode',choices=('full','state-only'),required=True)
    a=p.parse_args();main(a.fold,a.meta_inner,a.early,a.mode)
