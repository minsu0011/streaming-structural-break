"""Freeze single-feature direction comparisons separately from model scores."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,read_json,atomic,digest,utc_now,note
from src.next8.cache import FeatureCache
from src.next8.data import rows_for
from src.next8.screening import procedure_scope
from src.next8.nested_auxiliary import early_scope
from src.next8.orientation_refits import fit_orientation_refits
from scripts.next8.fit_next8_models import prepare_inputs


def main(fold,early):
    assert read_json(BASE/'ORIENTATION_REFIT_TEST_GATE_NEXT8.json')['status']=='PASS'
    lock_path=BASE/'ORIENTATION_REFIT_IMPLEMENTATION_LOCK_NEXT8.json';lock=read_json(lock_path)
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    cache=FeatureCache();scope=early_scope(procedure_scope(cache,fold),early)
    sp=BASE/'h_context_selection'/scope.name;gp=BASE/'parent_selection'/scope.name
    assert digest(sp/'SELECTION.json')==read_json(sp/'DONE.json')['selection_sha256']
    assert digest(gp/'GRAPH.json')==read_json(gp/'DONE.json')['graph_sha256']
    selection=read_json(sp/'SELECTION.json');graph=read_json(gp/'GRAPH.json')
    tr,va,provenance=prepare_inputs(cache,fold,None,early,scope,selection,graph)
    rows=rows_for(cache.meta,scope.validation_ids)
    if early:rows=rows[cache.meta['time_online'][rows]>=400]
    va=va[cache.meta['time_online'][rows]%8==0]
    p=BASE/'orientation_refits'/scope.name;p.mkdir(parents=True,exist_ok=True)
    if (p/'DONE.json').exists():raise RuntimeError('Do not silently rerun completed orientation refits')
    if not (p/'START.json').exists():atomic(p/'START.json',{
        'status':'FROZEN','timestamp':utc_now(),'scope':scope.provenance(),
        'selection_sha256':digest(sp/'SELECTION.json'),'graph_sha256':digest(gp/'GRAPH.json'),
        'state_input_provenance':provenance,'implementation_lock_sha256':digest(lock_path)})
    fit_orientation_refits(cache,scope,selection,tr,va,p,early=early)
    note('Single-feature direction refits complete',scope.name+'; common training domains and heldout predictions frozen without heldout labels','B1_CONDITIONAL_DIRECTION')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True);p.add_argument('--early',action='store_true')
    a=p.parse_args();main(a.fold,a.early)
