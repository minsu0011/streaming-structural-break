"""Supplement H-background coverage within the same untouched outer boundary."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,read_json,atomic,digest,utc_now,note
from src.next8.cache import FeatureCache
from src.next8.data import FitScope
from src.next8.screening import procedure_scope,selection_splits
from src.next8.h_context import audit_h_split,combine_selection,STATE_ORDER


def main(fold,meta_inner,early=False):
    assert read_json(BASE/'H_CONTEXT_TEST_GATE_NEXT8.json')['status']=='PASS'
    lock=read_json(BASE/'H_CONTEXT_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    cache=FeatureCache();scope=procedure_scope(cache,fold,meta_inner)
    if early:scope=FitScope(scope.training_ids,scope.validation_ids,scope.outer_excluded_ids,scope.name+'_EARLY_T199')
    old_dir=BASE/'selection'/scope.name
    old_done=read_json(old_dir/'DONE.json');assert old_done['status']=='COMPLETE'
    assert digest(old_dir/'SELECTION.json')==old_done['selection_sha256']
    initial=read_json(old_dir/'SELECTION.json')
    directory=BASE/'h_context_selection'/scope.name;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'DONE.json').exists():raise RuntimeError('Do not rerun completed H-context selection')
    if not (directory/'START.json').exists():atomic(directory/'START.json',{
        'status':'FROZEN','timestamp':utc_now(),'scope':scope.provenance(),
        'initial_selection_sha256':digest(old_dir/'SELECTION.json'),
        'supplement_sha256':digest(BASE/'NEXT8_PROTOCOL_SUPPLEMENT_H_CONTEXT.json'),
        'implementation_lock_sha256':digest(BASE/'H_CONTEXT_IMPLEMENTATION_LOCK_NEXT8.json')})
    all_rows=[];all_branches=[];all_corrections=[]
    for k,inner in enumerate(selection_splits(cache,fold,meta_inner)):
        if early:inner=FitScope(inner.training_ids,inner.validation_ids,inner.outer_excluded_ids,inner.name+'_EARLY_T199')
        p=directory/f'split_{k}';previous=old_dir/f'split_{k}'
        prior_receipt=read_json(previous/'DONE.json')
        for name,sha in prior_receipt['records_sha256'].items():assert digest(previous/name)==sha,name
        if (p/'DONE.json').exists():
            done=read_json(p/'DONE.json')
            for name,sha in done['records_sha256'].items():assert digest(p/name)==sha
            rows=[read_json(p/f'F{j:03d}.json') for j in range(153)]
            branches={kind:[] for kind in STATE_ORDER}
            import copy
            for j in range(153):
                old=read_json(previous/f'F{j:03d}.json');branches['OWN_F'].append(old)
                for kind in STATE_ORDER[1:]:
                    r=copy.deepcopy(old);r['conditional']=rows[j]['conditional'][kind];branches[kind].append(r)
            corrections=read_json(p/'MULTIPLICITY.json')['branches']
        else:
            rows,branches,corrections=audit_h_split(cache,inner,previous,p,initial['stable_features'],199 if early else None)
        all_rows.append(rows);all_branches.append(branches);all_corrections.append(corrections)
    result=combine_selection(cache,scope,initial,all_rows,all_branches,all_corrections)
    result['parent_full_initial_selection_sha256']=digest(old_dir/'SELECTION.json')
    atomic(directory/'SELECTION.json',result)
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),
        'selection_sha256':digest(directory/'SELECTION.json'),
        'initial_selection_preserved':True,'outer_or_meta_validation_used':False,
        'supplement_sha256':digest(BASE/'NEXT8_PROTOCOL_SUPPLEMENT_H_CONTEXT.json')})
    note('Complete combined own-history/H-background selector',scope.name+'; original results preserved and joint multiplicity included','B5_DOMAIN_GENERALIZATION')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True)
    p.add_argument('--meta-inner',type=int,choices=range(3));p.add_argument('--early',action='store_true')
    a=p.parse_args();main(a.fold,a.meta_inner,a.early)
