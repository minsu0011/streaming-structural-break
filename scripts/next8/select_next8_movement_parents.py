"""Select a sparse predictive parent graph inside one complete training scope."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,read_json,atomic,digest,utc_now,note
from src.next8.cache import FeatureCache
from src.next8.data import FitScope
from src.next8.movement import parent_selection_split,select_parent_graph
from src.next8.screening import procedure_scope,selection_splits


def main(fold,meta_inner,early=False):
    assert read_json(BASE/'MOVEMENT_PARENT_TEST_GATE_NEXT8.json')['status']=='PASS'
    lock=read_json(BASE/'MOVEMENT_PARENT_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    cache=FeatureCache();parent=procedure_scope(cache,fold,meta_inner)
    if early:parent=FitScope(parent.training_ids,parent.validation_ids,parent.outer_excluded_ids,parent.name+'_EARLY_T199')
    directory=BASE/'parent_selection'/parent.name;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'DONE.json').exists():raise RuntimeError('Do not rerun completed parent selection')
    if not (directory/'START.json').exists():atomic(directory/'START.json',{
        'status':'FROZEN','timestamp':utc_now(),'scope':parent.provenance(),'implementation_lock_sha256':digest(BASE/'MOVEMENT_PARENT_IMPLEMENTATION_LOCK_NEXT8.json'),
        'breakOccurred_labels_used':False,'outer_or_meta_validation_excluded':True})
    panels=[];corrections=[]
    for k,scope in enumerate(selection_splits(cache,fold,meta_inner)):
        if early:scope=FitScope(scope.training_ids,scope.validation_ids,scope.outer_excluded_ids,scope.name+'_EARLY_T199')
        p=directory/f'split_{k}'
        if (p/'DONE.json').exists():
            done=read_json(p/'DONE.json')
            for name,sha in done['records_sha256'].items():assert digest(p/name)==sha
            records=[read_json(p/name) for name in done['records_sha256']]
            adjustment=read_json(p/'MULTIPLICITY.json')['tests']
        else:records,adjustment=parent_selection_split(cache,scope,p,cutoff=199 if early else None)
        panels.append(records);corrections.append(adjustment)
    graph=select_parent_graph(panels,corrections,parent)
    atomic(directory/'GRAPH.json',graph)
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':parent.provenance(),
        'graph_sha256':digest(directory/'GRAPH.json'),'new_outer_score':False,
        'implementation_lock_sha256':digest(BASE/'MOVEMENT_PARENT_IMPLEMENTATION_LOCK_NEXT8.json')})
    note('Training-scoped mechanical parent selection complete',parent.name+'; all tested candidates and support retained','B3_SELF_HISTORY_SUFFICIENCY')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True)
    p.add_argument('--meta-inner',type=int,choices=range(3));p.add_argument('--early',action='store_true')
    a=p.parse_args();main(a.fold,a.meta_inner,a.early)
