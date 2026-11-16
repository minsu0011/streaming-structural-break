"""Actual T3 state outputs and necessary nested training inputs, without scores."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,read_json,atomic,digest,utc_now,note
from src.next8.cache import FeatureCache
from src.next8.screening import procedure_scope
from src.next8.graph_states import fit_graph_states
from src.next8.nested_auxiliary import early_scope,fit_meta_leaf_states,fit_early_outer_state_parts


def main(fold,inner,early,mode):
    assert read_json(BASE/'GRAPH_STATE_TEST_GATE_NEXT8.json')['status']=='PASS'
    lock=read_json(BASE/'GRAPH_STATE_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    cache=FeatureCache();scope=early_scope(procedure_scope(cache,fold,inner),early)
    p=BASE/'parent_selection'/scope.name;done=read_json(p/'DONE.json')
    assert digest(p/'GRAPH.json')==done['graph_sha256'];graph=read_json(p/'GRAPH.json')
    if mode=='inference':
        donor=BASE/'transitions'/scope.name if inner is None else None
        output=BASE/'graph_states'/scope.name
        output.mkdir(parents=True,exist_ok=True)
        contract={'scope':scope.provenance(),'graph_sha256':digest(p/'GRAPH.json'),
            'implementation_lock_sha256':digest(BASE/'GRAPH_STATE_IMPLEMENTATION_LOCK_NEXT8.json')}
        if (output/'START.json').exists():
            old=read_json(output/'START.json')
            for k,v in contract.items():assert old[k]==v
        else:atomic(output/'START.json',{'status':'FROZEN','timestamp':utc_now(),**contract})
        fit_graph_states(cache,scope,graph,output,early=early,reuse_directory=donor)
    else:
        s=BASE/'h_context_selection'/scope.name;receipt=read_json(s/'DONE.json')
        assert digest(s/'SELECTION.json')==receipt['selection_sha256']
        selection=read_json(s/'SELECTION.json')
        if mode=='meta-leaf':
            if inner is None:raise ValueError('Meta leaf requires inner model')
            fit_meta_leaf_states(cache,fold,inner,selection,graph,BASE/'meta_state_inputs'/scope.name,early=early)
        else:
            if inner is not None or not early:raise ValueError('Early outer OOF only')
            fit_early_outer_state_parts(cache,fold,selection,graph,BASE/'early_state_inputs'/scope.name)
    note('Complete graph state task',scope.name+' '+mode+'; no heldout scores released','B9_NESTED_SELECTION')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True)
    p.add_argument('--meta-inner',type=int,choices=range(3));p.add_argument('--early',action='store_true')
    p.add_argument('--mode',choices=('inference','meta-leaf','early-outer-oof'),required=True)
    a=p.parse_args();main(a.fold,a.meta_inner,a.early,a.mode)
